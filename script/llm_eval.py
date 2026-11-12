import asyncio
import configparser
import sys
import time

sys.path.append('lotus')
from lotus import lotus
import pandas as pd

async def test_eval(dataset_path: str,
                    prompt_prefix: str,
                    concurrency: int,
                    host: str,
                    port: int,
                    max_tokens: int,
                    output_path: str):
    total_start_time = time.perf_counter()

    df = pd.read_csv(dataset_path, encoding="utf-8-sig")

    start_time = time.perf_counter()
    
    output_dict = await df.llm_eval(max_tokens = max_tokens,
                                    prompt_prefix=prompt_prefix,
                                    concurrency=concurrency, 
                                    host=host, port=port)
    
    end_time = time.perf_counter()
    print(f"LLM Eval elapsed time: {end_time - start_time:.4f} seconds")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("row_id\toutput\n")
    
        for row_id in sorted(output_dict.keys()):
            value = output_dict[row_id]
            safe_value = str(value).replace("\n", "\\n") 
            f.write(f"{row_id}\t{safe_value}\n")
    
    print(f"Eval operation completed. Output saved to {output_path}.")
    
    total_end_time = time.perf_counter()
    print(f"Total elapsed time: {total_end_time - total_start_time:.4f} seconds")
    
if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config/tree_eval.ini', encoding='utf-8')
    
    dataset_path = config.get('backend', 'dataset_path')
    output_path = config.get('backend', 'output_prefix') + "_output.tsv"
    concurrency = config.getint('backend', 'concurrency')
    max_tokens = config.getint('backend', 'max_tokens')

    host = config.get('backend', 'host')
    port = config.getint('backend', 'port')
    
    prefix_path = config.get('backend', 'prefix_path')
    with open(prefix_path, "r", encoding="utf-8") as f:
        prompt_prefix = f.read()

    asyncio.run(test_eval(dataset_path = dataset_path,
                           prompt_prefix = prompt_prefix,
                           concurrency = concurrency,
                           host = host,
                           port = port,
                           max_tokens = max_tokens,
                           output_path = output_path))