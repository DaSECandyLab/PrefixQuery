import argparse
import asyncio
import configparser
import json

import concurrent
import os
import atexit  

import pandas as pd

import random   
import time     

import sys
sys.path.append('fasttree')
from fasttree import load_fasttree_plugin

MODEL_NAME = "Undi95/Meta-Llama-3-8B-hf"
magic_id = 0

# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

config = configparser.ConfigParser()
config.read('config/decode.ini', encoding='utf-8')
enable_fasttree = config.getboolean('decode', 'enable_fasttree')
annotate_breakdown = config.getboolean('decode', 'annotate_breakdown')
mock_llm = True
if (enable_fasttree or annotate_breakdown) and not mock_llm:
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


def evaluate_with_sgl(full_prompt: str, max_tokens: int) -> str:
    if mock_llm:
        time.sleep(random.uniform(0.6, 1.0))
        return random.choice(['yes', 'no'])
    state = single_generate.run(
        prompt=full_prompt,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return state["response"].strip()


def _read_file_sync(file_path: str) -> list[str]:
    """Synchronously read prompt data parts from a JSONL file."""
    prompt_parts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    prompt_part = json.loads(line)
                    prompt_parts.append(prompt_part)
    except FileNotFoundError:
        print("[WARN] File {} not found.".format(file_path), flush=True)
        return []
    return prompt_parts


def parse_parts(prompt_part: dict) -> tuple[str, dict]:
    data_part = prompt_part.get("data_part", "")
    row_id_pair_part = prompt_part.get("row_id_pair_part", {})
    assert isinstance(data_part, str), "data_part must be a string"
    assert isinstance(row_id_pair_part, dict), "row_id_pair_part must be a dict"
    assert "row_id" in row_id_pair_part, "row_id_pair_part must contain 'row_id'"
    assert data_part, "data_part must be a non-empty string"
    return data_part, row_id_pair_part


def parse_data_part(data_part: str) -> dict:
    """Parse the data part string into left/right column dicts for display."""
    data = {}
    blocks = data_part.strip().split('\n\n')
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        col_name, list_str = block.split("} -> ", 1)
        data[col_name.strip()[1:]] = json.loads(list_str)
    return data

async def process_one_batch(
        prompt_part: str,
        prompt_prefix: str,
        executor: concurrent.futures.ThreadPoolExecutor,
        max_tokens: int
    ) -> any:
    """Process one prompt by calling the LLM and return matched indices."""
    data_part, row_id_pair_part = parse_parts(prompt_part)
    full_prompt = prompt_prefix + "\n\n" + data_part

    data = parse_data_part(data_part)
    print(f"Processing row_id pair: {row_id_pair_part['row_id']} with data: {data}", flush=True)

    loop = asyncio.get_running_loop()
    output = await loop.run_in_executor(executor, evaluate_with_sgl, full_prompt, max_tokens)
    row_id = row_id_pair_part["row_id"]
    return int(row_id), output


async def optimized_eval_run(
        prompt_prefix: str,
        concurrency: int,
        host: str,
        port: int,
        max_tokens: int
    ) -> dict[int, any]:
    """Socket-driven optimized task execution with deserialization."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)

    output_dict = {}
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
                            print("[DEBUG fetcher] Server says 'already end', stopping fetcher", flush=True)
                            break

                        file_path = response
                        if not file_path:
                            print("[WARN] Received empty response from server, skipping...", flush=True)
                            continue

                        loop = asyncio.get_running_loop()
                        prompt_parts = await loop.run_in_executor(
                            None, _read_file_sync, file_path)

                        if not prompt_parts:
                            print("[DEBUG fetcher] File read returned 0 parts, waiting 0.1s...", flush=True)
                            await asyncio.sleep(0.1)
                            continue

                        for i, prompt_part in enumerate(prompt_parts):
                            total_tasks += 1
                            await task_queue.put((prompt_part, prompt_prefix))

                        print("[DEBUG fetcher] All {} parts enqueued, total_tasks so far={}".format(
                            len(prompt_parts), total_tasks), flush=True)
                        os.remove(file_path)

                    except ConnectionRefusedError:
                        print("[WARN] Connection refused by {}:{}, retrying in 1s...".format(host, port), flush=True)
                        await asyncio.sleep(1)
                        continue
                    except Exception as e:
                        print("[ERROR] fetcher exception: {}".format(e), flush=True)
                        import traceback
                        traceback.print_exc()
                        continue
                else:
                    await asyncio.sleep(0.01)
        finally:
            fetch_done.set()
            print("[DEBUG fetcher] Fetching done, sending {} sentinel values to workers".format(concurrency), flush=True)
            for _ in range(concurrency):
                await task_queue.put(None)

    async def call_and_collect(prompt_part: str, prompt_prefix: str):
        nonlocal error_count
        try:
            row_id, output = await process_one_batch(prompt_part, prompt_prefix, executor, max_tokens)
            assert row_id not in output_dict, "Duplicate row_id received: {}".format(row_id)
            output_dict[row_id] = output
        except Exception as e:
            error_count += 1
            print("[ERROR call_and_collect] Exception #{}: {}".format(error_count, e), flush=True)
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
                        worker_id, processed), flush=True)
                    break
                continue

            if item is None:
                print("[DEBUG worker-{}] Got sentinel, stopping (processed {} tasks)".format(
                    worker_id, processed), flush=True)
                assert fetch_done.is_set(), "Received sentinel but fetcher not marked done"
                break

            prompt_part, pp_prefix = item
            processed += 1
            await call_and_collect(prompt_part, pp_prefix)

    print("[DEBUG] Launching fetcher and {} workers...".format(concurrency), flush=True)
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
            error_count, total_tasks), flush=True)

    executor.shutdown(wait=True)
    return output_dict

def _cleanup_runtime():
    global runtime
    if runtime is not None:
        print("\n[INFO] Process exiting, shutting down SGLang Runtime...", flush=True)
        try:
            runtime.shutdown()
        except Exception as e:
            print("[WARN] Error during runtime.shutdown(): {}".format(e), flush=True)
        runtime = None


@pd.api.extensions.register_dataframe_accessor("llm_eval")
class LLMEvalDataframe:
    def __init__(self, pandas_obj: any):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    def start_runtime(self):
        global runtime
        if runtime is not None:
            print("[WARN] Runtime already started, skipping.", flush=True)
            return
        
        if mock_llm:
            print("[INFO] mock_llm 已启用，跳过 SGLang Runtime 初始化"
                  "（模拟处理模式，不调用 GPU）。", flush=True)
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
        prompt_prefix: str,
        concurrency: int = 32,
        host: str = "localhost",
        port: int = 9999,
        max_tokens: int = 128,
        auto_start: bool = True,    
        auto_shutdown: bool = True,  
    ) -> dict[int, any]:
        df = self._obj

        if not isinstance(df, pd.DataFrame):
            raise ValueError("must be a DataFrame")

        if concurrency <= 0:
            raise ValueError("concurrency must be >= 1")

        if auto_start and runtime is None:
            self.start_runtime()

        try:
            row_id_to_result_dict = await optimized_eval_run(
                prompt_prefix, concurrency, host, port, max_tokens
            )
            assert len(df) == len(row_id_to_result_dict), \
                "Output row count mismatch: {} vs {}".format(len(df), len(row_id_to_result_dict))
            return row_id_to_result_dict
        finally:
            if auto_shutdown:
                self.shutdown_runtime()

