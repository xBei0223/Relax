import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import ijson


def trace_events(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        yield from ijson.items(stream, "traceEvents.item", use_float=True)


def compact(event: dict) -> dict:
    result = {key: event[key] for key in ("cat", "name", "pid", "tid", "ts", "dur")}
    result["args"] = {key: value for key, value in event.get("args", {}).items() if key not in {"Call stack", "Concrete Inputs", "Input Strides"}}
    return result


def totals(events: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for event in events:
        grouped[(event["cat"], event["name"])].append(event["dur"])
    return sorted(
        ({"category": category, "name": name, "count": len(values), "inclusive_duration_us": sum(values), "maximum_duration_us": max(values)} for (category, name), values in grouped.items()),
        key=lambda item: item["inclusive_duration_us"], reverse=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--function", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--gpu-prelude-ms", type=float, default=500)
    parser.add_argument("--host-prelude-ms", type=float, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    categories = Counter()
    spans = []
    for event in trace_events(args.trace):
        categories[event.get("cat", "<metadata>")] += 1
        if event.get("ph") == "X" and event.get("cat") == "python_function" and args.function in event["name"]:
            spans.append(compact(event))
    spans.sort(key=lambda span: span["ts"])
    if len(spans) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} function calls, observed {len(spans)}")
    windows = [{"function": span, "host_events": [], "host_prelude_events": [], "gpu_events": []} for span in spans]
    for event in trace_events(args.trace):
        if event.get("ph") != "X":
            continue
        category = event.get("cat")
        if category not in {"cpu_op", "cuda_runtime", "cuda_driver", "python_function", "user_annotation", "kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        start, end = event["ts"], event["ts"] + event["dur"]
        for window in windows:
            span = window["function"]
            span_start, span_end = span["ts"], span["ts"] + span["dur"]
            if category in {"kernel", "gpu_memcpy", "gpu_memset"}:
                if start <= span_end and end >= span_start - args.gpu_prelude_ms * 1000:
                    window["gpu_events"].append(compact(event))
            elif event["pid"] == span["pid"] and event["tid"] == span["tid"]:
                if start >= span_start and end <= span_end + 0.001:
                    window["host_events"].append(compact(event))
                elif start < span_start and end >= span_start - args.host_prelude_ms * 1000:
                    window["host_prelude_events"].append(compact(event))
    for window in windows:
        window["host_events"].sort(key=lambda event: (event["ts"], -event["dur"]))
        window["host_prelude_events"].sort(key=lambda event: (event["ts"], -event["dur"]))
        window["gpu_events"].sort(key=lambda event: (event["ts"], -event["dur"]))
        window["host_totals"] = totals(window["host_events"])
        window["gpu_totals"] = totals(window["gpu_events"])
    report = {"trace": str(args.trace), "trace_bytes": args.trace.stat().st_size, "function": args.function, "categories": dict(categories), "gpu_prelude_ms": args.gpu_prelude_ms, "host_prelude_ms": args.host_prelude_ms, "windows": windows}
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"output": str(args.output), "categories": dict(categories), "windows": [{"duration_ms": window["function"]["dur"] / 1000, "host_events": len(window["host_events"]), "host_prelude_events": len(window["host_prelude_events"]), "gpu_events": len(window["gpu_events"])} for window in windows]}, indent=1))


if __name__ == "__main__":
    main()
