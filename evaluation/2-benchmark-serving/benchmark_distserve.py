"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    (vLLM backend)
    python -m vllm.entrypoints.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_tgi_server.sh <your_model> <max_batch_total_tokens>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> --dataset <target_dataset> \
        --request-rate <request_rate>
"""

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, List, Tuple

import numpy as np
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer

from backend_request_func import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
)


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p90_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p90_tpot_ms: float
    p99_tpot_ms: float
    p50_ttft_ms: float
    p30_ttft_ms: float


def sample_hunman_eval_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    from human_eval.data import write_jsonl, read_problems
    import random

    problems = read_problems()
    samples = [problems[task_id]["prompt"] for task_id in problems for i in range(20)]
    import random

    random.seed(777)
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt_data in samples:
        pred = tokenizer.encode(prompt_data)
        output_len = random.randint(128, 256)
        if len(pred) + output_len > 4096:
            continue
        filtered_dataset.append((prompt_data, len(pred), output_len))
    sampled_requests = filtered_dataset[0:num_requests]
    out_len = []
    inp_len = []
    for i in sampled_requests:
        out_len.append(i[1])
        inp_len.append(i[2])
    print(out_len)
    print(inp_len)
    print(sum(out_len) / len(sampled_requests), sum(inp_len) / len(sampled_requests))
    # exit(0)
    print("human_eval =======", sampled_requests[0], "\nlen=", len(sampled_requests))
    # breakpoint()
    return sampled_requests


def sample_alpaca_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
):
    import pandas as pd

    # # 读取 .parquet 文件
    # df = pd.read_parquet(
    #     "/share/datasets/public_datasets/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"
    # )
    df = pd.read_parquet(dataset_path)
    filtered_df = df[df["output"].str.split().apply(len) > 160]
    filtered_dataset: List[Tuple[str, int, int]] = []
    for index, row in filtered_df.iterrows():
        prefill = row["instruction"]
        output = row["output"]
        max_gen = output.split(" ").__len__()
        pred = tokenizer.encode(prefill)
        filtered_dataset.append((prefill, len(pred), max_gen))
    sampled_requests = filtered_dataset[0:num_requests]

    print(
        "sample_alpaca_requests ======= prefill_len: %s, out_len: %s \n%s"
        % (sampled_requests[0][1], sampled_requests[0][2], sampled_requests[0][0])
    )
    return sampled_requests


def sample_long_bench_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    filtered_dataset: List[Tuple[str, int, int]] = []
    with open(dataset_path, "r") as lb:
        for line in lb:
            data = json.loads(line)
            pred = tokenizer.encode(data["prompt"])
            max_gen = data["max_gen"]
            filtered_dataset.append((data["prompt"], len(pred), max_gen))
    sampled_requests = filtered_dataset[0:num_requests]

    # sorted_data = sorted(sampled_requests, key=lambda x: x[1])
    # mid_id = len(sorted_data) // 2
    
    # print(sorted_data[mid_id])
    # return [sorted_data[mid_id]]
    # exit(0)
    # len_arr = np.array([i[1] for i in filtered_dataset])
    # mid = np.median(len_arr)
    # print(mid)
    # median_index = np.where(len_arr == mid)
    # print(median_index)
    # # print(mid)
    # exit(0)
    # print(filtered_dataset[median_index])
    # exit(0)
    # return [filtered_dataset[median_index]]

    print("sample_long_bench_requests=======", sampled_requests[0])
    return sampled_requests


def sample_arixv_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    filtered_dataset: List[Tuple[str, int, int]] = []
    with open(dataset_path, "r") as lb:
        for line in lb:
            data = json.loads(line)
            prompt = "Please summerize this given article \nArticle: "
            txt = prompt + " ".join(data["article_text"])
            pred = tokenizer.encode(txt)
            output_len = len(" ".join(data["abstract_text"]).split(" "))
            prefill_len = len(pred)
            if prefill_len < 10000 or prefill_len > 32000:
                continue
            filtered_dataset.append((txt, len(pred), output_len))
    sampled_requests = filtered_dataset[0:num_requests]

    print("sample_arixv_requests=======", sampled_requests[0])
    return sampled_requests


def sample_codeparrot_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    filtered_dataset: List[Tuple[str, int, int]] = []
    count = 0
    with open(dataset_path, "r") as lb:
        for line in lb:
            data = json.loads(line)
            question = data["question"]
            solutions = data["solutions"]
            len_prefill = tokenizer.encode(question).__len__()
            len_solutions = tokenizer.encode(solutions).__len__()
            filtered_dataset.append((question, len_prefill, len_solutions))
            count += 1
            if count > num_requests:
                break
    sampled_requests = filtered_dataset[0:num_requests]

    print("sample_codeparrot_requests=======", sampled_requests[0])
    return sampled_requests

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]
    

    import random

    random.seed(777)
    # some of these will be filtered out, so sample more than we need
    # sampled_indices = random.sample(range(len(dataset)),
    #                                 int(num_requests * 1.2))
    # dataset = [dataset[i] for i in sampled_indices]
    # len_arr = np.array([i[1] for i in dataset])
    # mid = np.median(len_arr)
    # median_index = np.where(len_arr == mid)[0][0]
    # print(median_index)
    # print(dataset[int(median_index)])
    # print(mid)
    
    # median_index = 860
    
    # prompt = dataset[int(median_index)][0]
    # prompt_len = len(tokenizer(prompt).input_ids)
    # completion = dataset[int(median_index)][1]
    # completion_token_ids = tokenizer(completion).input_ids
    # output_len = len(completion_token_ids)
    # print(prompt, prompt_len, output_len)

    # return 
    dataset = random.sample(dataset, num_requests * 3)

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    outlier_count = 0

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        # if prompt_len == 106:
        #     print((prompt, prompt_len, output_len))
        #     return [(prompt, prompt_len, output_len)]
        filtered_dataset.append((prompt, prompt_len, output_len))

    # len_arr = np.array([i[1] for i in filtered_dataset])
    # mid = np.median(len_arr)
    # median_index = np.where(len_arr == mid)[0][0]
    # print(median_index)
    # print(dataset[int(median_index)])
    # print(mid)
    
    # exit(0)

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    
    out_len = []
    inp_len = []
    for i in sampled_requests:
        out_len.append(i[1])
        inp_len.append(i[2])
    # print(out_len)
    # print(inp_len)
    print(sum(out_len) / len(sampled_requests), sum(inp_len) / len(sampled_requests))
    # exit(0)
    sum_len = 0
    for e in sampled_requests:
        sum_len += e[1] + e[2]
    print("total tokens:", sum_len)
    print("outliers:", outlier_count)
    return sampled_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    import random

    random.seed(0)
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    request_rate,
    iter,
) -> BenchmarkMetrics:
    total_output = 0
    total_input = 0
    completed = 0
    per_token_latencies = []
    ttfts = []
    max_ttft_idx = 0
    for i in range(len(outputs)):
        if outputs[i].success:
            # output_len = len(tokenizer.encode(outputs[i].generated_text))
            # print(output_len)
            output_len = outputs[i].output_len
            total_output += output_len
            total_input += input_requests[i][1]
            if output_len > 1:
                per_token_latencies.append(
                    (outputs[i].latency - outputs[i].ttft) / (output_len - 1)
                )
                ttfts.append(outputs[i].ttft)
            completed += 1

            if outputs[i].ttft > outputs[max_ttft_idx].ttft:
                max_ttft_idx = i

    # path = "/nvme_data/chenlufang/workspace/multi_nodes_research/experiment/llama3_8b_distserve_base"
    # path = "/nvme_data/chenlufang/workspace/multi_nodes_research/experiment/llama3_8b_vllm_splitfuse_chunk_1024"
    path = "/nvme_data/chenlufang/workspace/multi_nodes_research/experiment/llama3_8b_mps_wo_dynamic_schedule"
    # path = "/nvme_data/chenlufang/workspace/multi_nodes_research/experiment/llama3_8b_mps_wo_dynamic_schedule_tp2"
    # path = "/nvme_data/chenlufang/workspace/multi_nodes_research/experiment/llama3_8b_mps_wo_dynamic_schedule_tp2_60_100"
    # path = "/nvme_data/chenlufang/workspace/multi_nodes_research/experiment/llama3_8b_mps_with_dynamic_schedule_slo_500_100"


    import os

    # file_name = "input_rate_" + str(request_rate) + f"_{str(iter)}_" + ".npy"
    file_name = f"input_rate_{str(request_rate)}_{str(iter)}.npy"
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, file_name)
    print(path)
    result = np.array([per_token_latencies, ttfts])
    # np.save(path, result)

    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=total_output,
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=total_output / dur_s,
        mean_ttft_ms=np.mean(ttfts) * 1000,
        median_ttft_ms=np.median(ttfts) * 1000,
        p90_ttft_ms=np.percentile(ttfts, 90) * 1000,
        p99_ttft_ms=np.percentile(ttfts, 99) * 1000,
        mean_tpot_ms=np.mean(per_token_latencies) * 1000,
        median_tpot_ms=np.median(per_token_latencies) * 1000,
        p90_tpot_ms=np.percentile(per_token_latencies, 90) * 1000,
        p99_tpot_ms=np.percentile(per_token_latencies, 99) * 1000,
        p50_ttft_ms=np.percentile(per_token_latencies, 50) * 1000,
        p30_ttft_ms=np.percentile(per_token_latencies, 30) * 1000,
    )

    print(f"max ttft request: {0}, time: {outputs[0].ttft}")
    print(f"max ttft request: {max_ttft_idx}, time: {outputs[max_ttft_idx].ttft}")
    # np.save("results/tpot.npy", per_token_latencies)
    # np.save("results/ttft.npy", ttfts)

    return metrics


async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    disable_tqdm: bool,
    iter: int,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS.get(backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print(f"Traffic request rate: {request_rate}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()
    tasks = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
        )
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )
    outputs = await asyncio.gather(*tasks)

    if not disable_tqdm:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        request_rate=request_rate,
        iter=iter,
    )

    print(f"Successful requests: {metrics.completed}")
    print(f"Benchmark duration: {benchmark_duration:2f} s")
    print(f"Total input tokens: {metrics.total_input}")
    print(f"Total generated tokens: {metrics.total_output}")
    print(f"Request throughput: {metrics.request_throughput:.2f} requests/s")
    print(f"Input token throughput: {metrics.input_throughput:.2f} tokens/s")
    print(f"Output token throughput: {metrics.output_throughput:.2f} tokens/s")
    print(f"Mean TTFT: {metrics.mean_ttft_ms:.2f} ms")
    print(f"Median TTFT: {metrics.median_ttft_ms:.2f} ms")
    print(f"P90 TTFT: {metrics.p90_ttft_ms:.2f} ms")
    print(f"P99 TTFT: {metrics.p99_ttft_ms:.2f} ms")
    print(f"Mean TPOT: {metrics.mean_tpot_ms:.2f} ms")
    print(f"Median TPOT: {metrics.median_tpot_ms:.2f} ms")
    print(f"P90 TPOT: {metrics.p90_tpot_ms:.2f} ms")
    print(f"P99 TPOT: {metrics.p99_tpot_ms:.2f} ms")

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_inthroughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "p50_ttft_ms": metrics.p50_ttft_ms,
        "p30_ttft_ms": metrics.p30_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
    }
    return result


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"

    tokenizer = get_tokenizer(tokenizer_id, trust_remote_code=args.trust_remote_code)
    if args.dataset_name == "sharegpt":
        input_requests = sample_requests(args.dataset, args.num_prompts, tokenizer)
    elif args.dataset_name == "long_bench":
        input_requests = sample_long_bench_requests(
            args.dataset, args.num_prompts, tokenizer
        )
    elif args.dataset_name == "arxiv":
        input_requests = sample_arixv_requests(args.dataset, args.num_prompts, tokenizer)
    # exit(0)
    # input_requests = sample_hunman_eval_requests(args.dataset, args.num_prompts, tokenizer)
    # input_requests = sample_arixv_requests(args.dataset, args.num_prompts, tokenizer)
    # input_requests = sample_alpaca_requests(args.dataset, args.num_prompts, tokenizer)
    # input_requests = sample_codeparrot_requests(args.dataset, args.num_prompts, tokenizer)

    # for args.request_rate in [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4s,1.5,1.6,1.7,1.8,1.9,2.0]:
    # for args.request_rate in [0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1]:
    # for args.request_rate in [0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1]:
    # for args.request_rate in [0.12,0.14,0.16,0.18,0.2, 0.22, 0.24, 0.26,0.28]:
    # for args.request_rate in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]:
    # for i in range(0,15):
    # for j in range(1, 10):
    #     args.request_rate  = 19 + j * 0.5
    # # for j in range(0, 10):
    # #     args.request_rate  = 40 + j * 2
    #     for i in range(1):
    #         benchmark_result = asyncio.run(
    #             benchmark(
    #                 backend=backend,
    #                 api_url=api_url,
    #                 model_id=model_id,
    #                 tokenizer=tokenizer,
    #                 input_requests=input_requests,
    #                 best_of=args.best_of,
    #                 use_beam_search=args.use_beam_search,
    #                 request_rate=args.request_rate,
    #                 disable_tqdm=args.disable_tqdm,
    #                 iter=i,
    #             )
    #         )

    #         # Save config and results to json
    #         if args.save_result:
    #             result_json = {}

    #             # Setup
    #             current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    #             result_json["date"] = current_dt
    #             result_json["backend"] = backend
    #             result_json["version"] = args.version
    #             result_json["model_id"] = model_id
    #             result_json["tokenizer_id"] = tokenizer_id
    #             result_json["best_of"] = args.best_of
    #             result_json["use_beam_search"] = args.use_beam_search
    #             result_json["num_prompts"] = args.num_prompts

    #             # Traffic
    #             result_json["request_rate"] = (
    #                 args.request_rate if args.request_rate < float("inf") else "inf"
    #             )

    #             # Merge with benchmark result
    #             result_json = {**result_json, **benchmark_result}

    #             # Save to file
    #             base_model_id = model_id.split("/")[-1]
    #             file_name = f"{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
    #             with open(file_name, "w") as outfile:
    #                 json.dump(result_json, outfile)
    
    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            best_of=args.best_of,
            use_beam_search=args.use_beam_search,
            request_rate=args.request_rate,
            disable_tqdm=args.disable_tqdm,
            iter=0,
        )
    )

    # Save config and results to json
    if args.save_result:
        result_json = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["version"] = args.version
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["best_of"] = args.best_of
        result_json["use_beam_search"] = args.use_beam_search
        result_json["num_prompts"] = args.num_prompts

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf"
        )

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = model_id.split("/")[-1]
        file_name = f"{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--version",
        type=str,
        default="N/A",
        help="Version of the serving backend/engine.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/generate",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer, if not using the default tokenizer.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and " "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=["sharegpt", "long_bench", "arxiv"],
        default="sharegpt",
        help="Specify to save benchmark results to a json file",
    )

    args = parser.parse_args()
    main(args)

    # for args.request_rate in [1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5]:
    # for args.request_rate in [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]:
    #     main(args)
