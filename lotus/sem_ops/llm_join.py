import argparse
import asyncio
import concurrent.futures
import configparser
import os
import pandas as pd
import json
from typing import Any
import random
import time  
import atexit  

import sys
sys.path.append('fasttree')
from fasttree import load_fasttree_plugin

SGLANG_URL = "http://localhost:30000"   
MODEL_NAME = "Undi95/Meta-Llama-3-8B-hf"
magic_id = 0

config = configparser.ConfigParser()
config.read('config/decode.ini', encoding='utf-8')
enable_fasttree = config.getboolean('decode', 'enable_fasttree')
annotate_breakdown = config.getboolean('decode', 'annotate_breakdown')
if enable_fasttree or annotate_breakdown:
    load_fasttree_plugin(enable_fasttree, annotate_breakdown)

import sglang as sgl

runtime = None

def init_runtime(args):
    global runtime
    runtime = sgl.Runtime(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        disable_cuda_graph=args.disable_cuda_graph,
        disable_flashinfer=args.disable_flashinfer,
        tp_size=args.tp_size,
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=args.max_running_requests,
        max_num_reqs=args.max_num_reqs,
        chunked_prefill_size=args.chunked_prefill_size,
        schedule_policy=args.schedule_policy,
        schedule_conservativeness=args.schedule_conservativeness,
        dtype=args.dtype,
    )
    sgl.set_default_backend(runtime)

@sgl.function
def single_generate(s, prompt, max_tokens, temperature=0.0):
    s += prompt
    s += sgl.gen("response", max_tokens=max_tokens, temperature=temperature)


def evaluate_with_sgl(prompt_parts: tuple[str, str], left_num: int) -> str:
    system_prompt, user_prompt = prompt_parts
    full_prompt = system_prompt + "\n\n" + user_prompt

    max_tokens = left_num + 1

    state = single_generate.run(
        prompt=full_prompt,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    raw_output = state["response"].strip()

    filtered_output = "".join(ch for ch in raw_output if ch in "01")
    if len(filtered_output) > left_num:
        filtered_output = filtered_output[:left_num]

    return filtered_output

def _read_file_sync(file_path: str) -> list[str]:
    """Synchronously read prompt data parts from a JSONL file.
    Each line is a JSON string representing one prompt_data_part.
    """
    prompt_parts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    prompt_part = json.loads(line)
                    prompt_parts.append(prompt_part)
    except FileNotFoundError:
        print("[WARN] File {} not found.".format(file_path))
        return []
    return prompt_parts

def parse_parts(prompt_part: dict) -> tuple[str, dict]:
    data_part = prompt_part.get("data_part", "")
    row_id_pair_part = prompt_part.get("row_id_pair_part", {})
    assert isinstance(data_part, str), "data_part must be a string"
    assert isinstance(row_id_pair_part, dict), "row_id_pair_part must be a dict"
    assert "left" in row_id_pair_part and "right" in row_id_pair_part, "row_id_pair_part must contain 'left' and 'right' keys"
    assert isinstance(row_id_pair_part["left"], list), "row_id_pair_part['left'] must be a list"
    assert isinstance(row_id_pair_part["right"], list), "row_id_pair_part['right'] must be a list"
    assert data_part, "data_part must be a non-empty string"
    return data_part, row_id_pair_part

def output_valid_pairs(row_id_pair_part: dict, output_bitmap: str) -> list[tuple[int, int]]:

    assert len(row_id_pair_part["left"]) > 0, "row_id_pair_part['left'] must contain at least one row ID"
    assert len(row_id_pair_part["right"]) == 1, "row_id_pair_part['right'] must contain exactly one row ID"

    if output_bitmap:
        output_bitmap = output_bitmap.strip()

    if not output_bitmap or not all(c in "01" for c in output_bitmap):
        raise ValueError(
            "Output must be a binary string, got: {}".format(repr(output_bitmap)))
    if len(output_bitmap) != len(row_id_pair_part["left"]):
        raise ValueError(
            "Output length {} != batch size {}, output={}".format(len(output_bitmap), len(row_id_pair_part["left"]), repr(output_bitmap)))

    result = [index for index, char in enumerate(output_bitmap) if char == "1"]

    valid_pairs = []
    for idx in result:
        left_id = row_id_pair_part["left"][idx]
        right_id = row_id_pair_part["right"][0]
        valid_pairs.append((left_id, right_id))

    if len(output_bitmap) != len(row_id_pair_part["left"]):
        return []
    return valid_pairs

def parse_data_part(data_part: str) -> tuple[dict, dict]:
    """
    Parse the data part string into left/right column dicts for display.
    Format: {col1:left} -> ["v1","v2"]  /  {col2:right} -> ["v3"]
    """
    left_data = {}
    right_data = {}

    blocks = data_part.strip().split('\n\n')

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if ":left} -> " in block:
            col_name, list_str = block.split(":left} -> ", 1)
            left_data[col_name.strip()[1:]] = json.loads(list_str)
        elif ":right} -> " in block:
            col_name, list_str = block.split(":right} -> ", 1)
            right_data[col_name.strip()[1:]] = json.loads(list_str)

    return left_data, right_data

async def process_one_batch(
        prompt_part: str,
        prompt_prefix: tuple[str, str, str],
        executor: concurrent.futures.ThreadPoolExecutor
    ) -> list[tuple[int, int]]:
    """Process one prompt by calling the LLM and return matched indices."""
    data_part, row_id_pair_part = parse_parts(prompt_part)

    left_data, right_data = parse_data_part(data_part)
    left_values = list(left_data.values())
    right_values = list(right_data.values())
    left_len = len(left_values[0])
    right_len = len(right_values[0])
    assert all(len(v) == left_len for v in left_values), "Inconsistent number of rows in left columns"
    assert all(len(v) == right_len for v in right_values), "Inconsistent number of rows in right columns"
    assert len(row_id_pair_part["right"]) == right_len, "Number of right row_ids does not match number of rows in right columns"
    assert len(row_id_pair_part["left"]) == left_len, "Number of left row_ids does not match number of rows in left columns"

    system_prompt = prompt_prefix[0]
    user_prompt = prompt_prefix[1] + "\n\n" + data_part + prompt_prefix[2]

    loop = asyncio.get_running_loop()
    output_bitmap = await loop.run_in_executor(
        executor,
        evaluate_with_sgl,
        (system_prompt, user_prompt),
        len(row_id_pair_part["left"]),  
    )
    valid_pairs = output_valid_pairs(row_id_pair_part, output_bitmap)
    return valid_pairs


async def optimized_join_run(
        prompt_prefix: tuple[str, str, str],
        concurrency: int,
        host: str,
        port: int
    ) -> list[dict]:
    """Socket-driven optimized task execution with deserialization."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)

    start_time = time.perf_counter()
    processed_pairs = 0               
    last_print_pairs = 0             

    valid_pairs_list = []
    error_count = 0
    total_tasks = 0

    task_queue = asyncio.Queue(maxsize=concurrency * 3)
    
    fetch_done = asyncio.Event()

    async def fetcher():
        nonlocal total_tasks
        fetch_round = 0

        try:
            while not fetch_done.is_set():
                fetch_round += 1
                available = task_queue.maxsize - task_queue.qsize()
                available = min(available, concurrency) 

                if available >= concurrency: 
                    try:
                        reader, writer = await asyncio.open_connection(host, port)

                        writer.write(available.to_bytes(4, 'big'))
                        await writer.drain()

                        data = await reader.read()
                        response = data.decode('utf-8').strip()

                        writer.close()
                        await writer.wait_closed()

                        if response == "already end":
                            print("[DEBUG fetcher] Server says 'already end', stopping fetcher")
                            break
                        elif response == "not start":
                            print("[DEBUG fetcher] Server says 'not start', waiting 5s...")
                            await asyncio.sleep(5.0)
                            continue
                        elif response == "return zero":
                            print("[DEBUG fetcher] Server says 'return zero', waiting 0.1s...")
                            await asyncio.sleep(0.1)
                            continue

                        file_path = response
                        if not file_path:
                            print("[WARN] Received empty response from server, skipping...")
                            continue

                        loop = asyncio.get_running_loop()
                        prompt_parts = await loop.run_in_executor(
                            None, _read_file_sync, file_path)

                        if not prompt_parts:
                            print("[DEBUG fetcher] File read returned 0 parts, waiting 0.1s...")
                            await asyncio.sleep(0.1)
                            continue

                        for i, prompt_part in enumerate(prompt_parts):
                            total_tasks += 1
                            if i < 3 or i == len(prompt_parts) - 1:
                                pass
                            await task_queue.put((prompt_part, prompt_prefix))

                        print("[DEBUG fetcher] All {} parts enqueued, total_tasks so far={}".format(
                            len(prompt_parts), total_tasks))
                        os.remove(file_path)  

                    except ConnectionRefusedError:
                        print("[WARN] Connection refused by {}:{}, retrying in 0.1s...".format(host, port))
                        await asyncio.sleep(0.1)
                        continue
                    except Exception as e:
                        print("[ERROR] fetcher exception: {}".format(e))
                        import traceback
                        traceback.print_exc()
                        continue
                else:
                    await asyncio.sleep(0.01) 
        finally:
            fetch_done.set()
            print("[DEBUG fetcher] Fetching done, sending {} sentinel values to workers".format(concurrency))
            for _ in range(concurrency):
                await task_queue.put(None)

    async def call_and_collect(prompt_part: str, prompt_prefix: tuple[str, str, str]):
        nonlocal error_count, processed_pairs, last_print_pairs 
        try:
            result = await process_one_batch(prompt_part, prompt_prefix, executor)
            assert isinstance(result, list), "output_valid_pairs must be a list"
            assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in result), "output_valid_pairs must be a list of (left_id, right_id) tuples"
            

            data_part, row_id_pair_part = parse_parts(prompt_part)
            batch_pairs = len(row_id_pair_part["left"])  
            processed_pairs += batch_pairs

            
            valid_pairs_list.extend(result)
            

            if processed_pairs // 10000 > last_print_pairs // 10000:
                now = time.perf_counter()
                elapsed = now - start_time
                print("[TIMER] Processed {} pairs in {:.2f} seconds".format(
                    processed_pairs, elapsed), flush=True)
                last_print_pairs = processed_pairs


        except Exception as e:
            error_count += 1
            print("[ERROR call_and_collect] Exception #{}: {}".format(error_count, e))
            import traceback
            traceback.print_exc()


    async def worker(worker_id: int):
        processed = 0
        while True:
            try:

                item = await asyncio.wait_for(task_queue.get(), timeout=2.0)
            except asyncio.TimeoutError:

                if fetch_done.is_set() and task_queue.empty():
                    print("[DEBUG worker-{}] fetch_done + queue empty, exiting (processed={})".format(
                        worker_id, processed))
                    break

                continue

            if item is None:
                print("[DEBUG worker-{}] Got sentinel, stopping (processed {} tasks)".format(
                    worker_id, processed))
                assert fetch_done.is_set(), "Received sentinel but fetcher not marked done"
                break
            prompt_part, prompt_prefix = item
            processed += 1
            if processed <= 3 or processed % 50 == 0:
                pass
            await call_and_collect(prompt_part, prompt_prefix)

    print("[DEBUG] Launching fetcher and {} workers...".format(concurrency))
    fetcher_task = asyncio.create_task(fetcher())
    worker_tasks = [
        asyncio.create_task(worker(wid))
        for wid in range(concurrency)
    ]


    try:
        await fetcher_task
        print("[DEBUG] Fetcher task completed, waiting for workers...")
        await asyncio.gather(*worker_tasks)
        print("[DEBUG] All workers completed")
    except KeyboardInterrupt:
        print("\n[WARN] KeyboardInterrupt received! Cancelling all tasks...", flush=True)

        fetcher_task.cancel()
        for t in worker_tasks:
            t.cancel()

        await asyncio.gather(fetcher_task, *worker_tasks, return_exceptions=True)
        print("[WARN] All tasks cancelled.", flush=True)

    if error_count > 0:
        print("[WARN] {}/{} batches failed, results may be incomplete".format(
            error_count, total_tasks))

    print("[DEBUG optimized_join_run] total_tasks={}, valid_pairs found={}, errors={}".format(
        total_tasks, len(valid_pairs_list), error_count))


    total_elapsed = time.perf_counter() - start_time
    print("[TIMER] Total time: {:.2f} seconds for {} pairs".format(
        total_elapsed, processed_pairs), flush=True)


    executor.shutdown(wait=True)
    return valid_pairs_list

def _cleanup_runtime():

    global runtime
    if runtime is not None:
        print("\n[INFO] Process exiting, shutting down SGLang Runtime...", flush=True)
        try:
            runtime.shutdown()
        except Exception as e:
            print("[WARN] Error during runtime.shutdown(): {}".format(e), flush=True)
        runtime = None


@pd.api.extensions.register_dataframe_accessor("llm_join")
class LLMJoinDataframe:
    def __init__(self, pandas_obj: Any):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    def start_runtime(self):

        global runtime
        if runtime is not None:
            print("[WARN] Runtime already started, skipping.", flush=True)
            return
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-path", type=str, default="Undi95/Meta-Llama-3-8B-hf")
        parser.add_argument("--tokenizer-path", type=str, default="Undi95/Meta-Llama-3-8B-hf")
        parser.add_argument("--tp-size", type=int, default=1)
        parser.add_argument("--mem-fraction-static", type=float, default=0.80)
        parser.add_argument("--max-running-requests", type=int, default=256)
        parser.add_argument("--max-num-reqs", type=int, default=256)
        parser.add_argument("--chunked-prefill-size", type=int, default=-1)
        parser.add_argument("--schedule-policy", type=str, default="fcfs")
        parser.add_argument("--schedule-conservativeness", type=float, default=1.3)
        parser.add_argument("--dtype", type=str, default="auto")
        parser.add_argument("--disable-cuda-graph", type=bool, default=True)
        parser.add_argument("--disable-flashinfer", type=bool, default=False)
        args = parser.parse_args()

        init_runtime(args)
        print("[INFO] SGLang Runtime started.", flush=True)
        

        atexit.register(_cleanup_runtime)

    def shutdown_runtime(self):

        global runtime
        if runtime is None:
            return
        print("[INFO] Shutting down runtime...", flush=True)
        try:
            runtime.shutdown()
        except Exception as e:
            print("[WARN] Error during runtime.shutdown(): {}".format(e), flush=True)
        runtime = None
        print("[INFO] Runtime shut down, GPU resources released.", flush=True)

    async def __call__(
        self,
        right_df: pd.DataFrame,
        prompt_prefix: tuple[str, str, str],
        concurrency: int = 32,
        how: str = "inner",
        host: str = "localhost",
        port: int = 9999,
        auto_start: bool = True,    
        auto_shutdown: bool = True,  
    ) -> pd.DataFrame:
        left_df = self._obj

        if not isinstance(left_df, pd.DataFrame):
            raise ValueError("Left must be a DataFrame")
        if not isinstance(right_df, pd.DataFrame):
            raise ValueError("Right must be a DataFrame")

        if how != "inner":
            raise NotImplementedError("Only inner join is currently supported")
        if concurrency <= 0:
            raise ValueError("concurrency must be >= 1")

        if auto_start and runtime is None:
            self.start_runtime()

        try:
            valid_pairs_list = await optimized_join_run(
                prompt_prefix, concurrency, host, port)

            overlap_cols = set(left_df.columns) & set(right_df.columns)

            rename_left = {col: f"{col}_1" for col in overlap_cols}
            left_df = left_df.rename(columns=rename_left)

            rename_right = {col: f"{col}_2" for col in overlap_cols}
            right_df = right_df.rename(columns=rename_right)

            if len(valid_pairs_list) == 0:
                print("[DEBUG __call__] No valid pairs, returning empty DataFrame")
                empty_cols = list(left_df.columns) + list(right_df.columns)
                return pd.DataFrame(columns=empty_cols)

            concat_rows = []
            for i, (left_id, right_id) in enumerate(valid_pairs_list):
                if i < 5 or i == len(valid_pairs_list) - 1 or i % 1000 == 0:
                    pass
                left_row = left_df.loc[int(left_id)]
                right_row = right_df.loc[int(right_id)]
                concat_row = pd.concat([left_row, right_row])
                concat_rows.append(concat_row)

            joined_df = pd.DataFrame(concat_rows)
            print("[DEBUG __call__] Joined DataFrame shape={}, columns={}".format(
                joined_df.shape, list(joined_df.columns)))

            return joined_df
        finally:
            if auto_shutdown:
                self.shutdown_runtime()