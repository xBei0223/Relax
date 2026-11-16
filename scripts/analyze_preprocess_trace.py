import argparse
import json
from pathlib import Path


def relative_event(event: dict, origin: float) -> dict:
    return {
        "name": event["name"],
        "category": event["cat"],
        "start_ms": (event["ts"] - origin) / 1000,
        "end_ms": (event["ts"] + event["dur"] - origin) / 1000,
        "duration_ms": event["dur"] / 1000,
        "args": event["args"],
    }


def overlaps(event: dict, interval: dict) -> bool:
    return event["ts"] < interval["ts"] + interval["dur"] and event["ts"] + event["dur"] > interval["ts"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("events", type=Path)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.events.read_text())
    capture = json.loads(args.samples.read_text())[str(args.rank)]
    samples = capture["samples"][str(args.rank)]
    results = []
    for sample, window in zip(samples, source["windows"], strict=True):
        function = window["function"]
        origin = function["ts"]
        (copy_api,) = [event for event in window["host_events"] if event["cat"] == "cuda_runtime" and event["name"] == "cudaMemcpyAsync"]
        (synchronize,) = [event for event in window["host_events"] if event["cat"] == "cuda_runtime" and event["name"] == "cudaStreamSynchronize"]
        (copy_gpu,) = [event for event in window["gpu_events"] if event["cat"] == "gpu_memcpy" and event["args"]["correlation"] == copy_api["args"]["correlation"]]
        if copy_api["args"]["External id"] != copy_gpu["args"]["External id"]:
            raise ValueError("CPU and GPU copy events have different external identifiers")
        if abs(sample["target_preprocess_ms"] - function["dur"] / 1000) > 1:
            raise ValueError("Tracker and trace function durations do not match")
        (array_conversion,) = [event for event in window["host_events"] if event["cat"] == "python_function" and event["name"] == "<built-in function array>"]
        function_communication = [event for event in window["gpu_events"] if event["cat"] == "kernel" and "ncclDevKernel_AllGather" in event["name"] and overlaps(event, function)]
        function_copy_out = [event for event in window["gpu_events"] if event["cat"] == "kernel" and "split_with_sizes_copy_out" in event["name"] and overlaps(event, function)]
        communication = [event for event in window["gpu_events"] if event["cat"] == "kernel" and "ncclDevKernel_AllGather" in event["name"] and overlaps(event, copy_api)]
        copy_out = [event for event in window["gpu_events"] if event["cat"] == "kernel" and "split_with_sizes_copy_out" in event["name"] and overlaps(event, copy_api)]
        external_ids = {event["args"]["External id"] for event in [*communication, *copy_out]}
        host_ops = [event for event in window["host_prelude_events"] if event["cat"] == "cpu_op" and event["args"].get("External id") in external_ids]
        annotations = [event for event in window["host_prelude_events"] if event["cat"] == "user_annotation" and event["name"].startswith("FSDP::")]
        results.append({
            "sample": sample,
            "function_duration_ms": function["dur"] / 1000,
            "array_conversion": relative_event(array_conversion, origin),
            "communication_overlapping_function": [relative_event(event, origin) for event in function_communication],
            "copy_out_overlapping_function": [relative_event(event, origin) for event in function_copy_out],
            "copy_api": relative_event(copy_api, origin),
            "synchronize_api": relative_event(synchronize, origin),
            "copy_gpu": relative_event(copy_gpu, origin),
            "api_start_to_gpu_copy_start_ms": (copy_gpu["ts"] - copy_api["ts"]) / 1000,
            "communication_overlapping_copy_api": [relative_event(event, origin) for event in communication],
            "copy_out_overlapping_copy_api": [relative_event(event, origin) for event in copy_out],
            "matching_host_ops": [relative_event(event, origin) for event in host_ops],
            "preceding_fsdp_annotations": [relative_event(event, origin) for event in annotations],
        })
    report = {"run": capture["run"], "rank": args.rank, "trace": source["trace"], "events": str(args.events), "windows": results}
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"run": capture["run"], "rank": args.rank, "windows": [{"sample": row["sample"], "function_ms": row["function_duration_ms"], "copy_api_ms": row["copy_api"]["duration_ms"], "synchronize_api_ms": row["synchronize_api"]["duration_ms"], "gpu_copy_ms": row["copy_gpu"]["duration_ms"], "bytes": row["copy_gpu"]["args"]["bytes"], "overlapping_all_gathers": len(row["communication_overlapping_copy_api"])} for row in results]}, indent=1))


if __name__ == "__main__":
    main()
