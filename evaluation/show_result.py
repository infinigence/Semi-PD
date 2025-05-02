import numpy as np
import os

import pandas as pd

def show_metrics(path, idx = 0):
    print(f"#########################{path} processed#########################")
    params = path.rsplit("/", 3)[-1]
    input_len = 0
    output_len = 0
    range_ratio = 0.9
    input_request_rate = 0
    concurrency = "None"
    name, num_prompt, input_request_rate = params.split("_")
    import json
    if os.path.exists(f"{path}/metrics_{idx}.json"):
        with open(f"{path}/metrics_{idx}.json", "r") as f:
            metrics_json = json.load(f)
    else:
        metrics_json = {"output_throughput": 0}
    data_dict = {
        "input_len": int(input_len),
        "output_len": int(output_len),
        "range_ratio": float(range_ratio),
        "input_request_rate": float(input_request_rate),
        "concurrency": int(concurrency) if concurrency != "None" else 0,
        "ttft": np.load(f"{path}/ttft_{idx}.npy"),
        "itl": np.load(f"{path}/itl_{idx}.npy"),
        "output_throughput": metrics_json["output_throughput"],
    }

    ttft = data_dict["ttft"]

    globel_itl = data_dict["itl"].flatten()
    globel_itl = globel_itl[globel_itl >= 0]
    # print("total reqs num: ", ttft.shape[0])
    # print("total token num: ", globel_itl.flatten().shape[0])
    tpot_list = []
    for row in  data_dict["itl"]:
        record = row[row > 0]
        if record.shape == (0,):
            continue
        tpot = record.mean()
        tpot_list.append(tpot)
    tpot_list = np.array(tpot_list)
    
    metrics = {}  # Initialize a dictionary to store the metrics

    metrics['mean_ttft'] = ttft.mean()
    metrics['p50_ttft'] = np.percentile(ttft, 50)
    metrics['p80_ttft'] = np.percentile(ttft, 80)
    metrics['p90_ttft'] = np.percentile(ttft, 90)
    metrics['p95_ttft'] = np.percentile(ttft, 95)
    metrics['p99_ttft'] = np.percentile(ttft, 99)

    metrics['mean_tpot'] = tpot_list.mean()
    metrics['p90_tpot'] = np.percentile(tpot_list, 90)
    metrics['p95_tpot'] = np.percentile(tpot_list, 95)
    metrics['p99_tpot'] = np.percentile(tpot_list, 99)

    metrics['p90_global_itl'] = np.percentile(globel_itl, 90)
    metrics['p95_global_itl'] = np.percentile(globel_itl, 95)
    metrics['p99_global_itl'] = np.percentile(globel_itl, 99)
    metrics['p99.8_global_itl'] = np.percentile(globel_itl, 99.8)

    e2e = ttft.sum() + globel_itl.sum()
    metrics['e2e'] = e2e / ttft.shape[0]
    metrics['output_throughput'] = data_dict["output_throughput"]

    return metrics, input_len, output_len, concurrency, input_request_rate, name

if __name__ == "__main__":
    # get args from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./result_v2_semi_pd")
    args = parser.parse_args()
    path = args.path
    
    print(f"result_path: {path}")

    total_metrics = {}
    for inner_path in os.listdir(path):
        inner_path = os.path.join(path, inner_path)
        if not os.listdir(inner_path):
            print(f"Empty dir: {inner_path}")
            continue
        metrics, input_len, output_len, concurrency, input_request_rate, name = show_metrics(inner_path)
        # io = f"{input_len}_{output_len}"
        io = name
        io_metrics = total_metrics.get(io, {})
        io_metrics[input_request_rate] = metrics
        total_metrics[io] = io_metrics

    # sorted_io_keys = sorted(total_metrics.keys(), key=lambda x: tuple(map(int, x.split('_'))))
    # Convert the nested dictionary into a DataFrame
    data = []
    for io, rr_dict in total_metrics.items():
        for rr, metrics in rr_dict.items():
            row = {"io": io, "rr": rr}  # Add 'io' and 'rr' as columns
            row.update(metrics)  # Add all metrics
            data.append(row)

    df = pd.DataFrame(data)

    # check whether this col all zero
    if (df["output_throughput"] == 0).all():
        # remove this col
        df = df.drop(columns=["output_throughput"])

    # Print the DataFrame in a clean tabular format
    print(df.to_string(index=False))  # `index=False` removes the default row numbers
