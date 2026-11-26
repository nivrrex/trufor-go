#!/usr/bin/env python3
"""Fixed-input ONNX Runtime CPU benchmark, mirroring tools/bench_inference.go."""

import argparse
import json
import os
import platform
import resource
import struct
import time

import numpy as np
import onnxruntime as ort


def checksum(values: np.ndarray) -> int:
    h = 14695981039346656037
    for value in values.astype("<f4", copy=False).ravel():
        for byte in struct.pack("<f", float(value)):
            h ^= byte
            h = (h * 1099511628211) & ((1 << 64) - 1)
    return h


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if min(args.width, args.height, args.threads, args.repeat) <= 0 or args.warmup < 0:
        parser.error("尺寸/线程/采样必须为正，warmup 不能为负")

    y, x = np.indices((args.height, args.width), dtype=np.int32)
    rgb = np.stack(
        ((x * 17 + y * 31 + 11) & 0xFF,
         (x * 29 + y * 7 + 23) & 0xFF,
         (x * 13 + y * 19 + 37) & 0xFF),
        axis=0,
    ).astype(np.uint8)
    tensor = rgb[None].astype(np.float32) / np.float32(256.0)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        args.weights, options, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    def run() -> tuple[np.ndarray, np.ndarray, float]:
        out, conf, det = session.run(None, {input_name: tensor})
        logits = out[0]
        exp = np.exp(logits - logits.max(axis=0, keepdims=True))
        pred = (exp / exp.sum(axis=0, keepdims=True))[1].astype(np.float32)
        conf_out = (1.0 / (1.0 + np.exp(-conf[0, 0]))).astype(np.float32)
        score = float(1.0 / (1.0 + np.exp(-float(det.ravel()[0]))))
        return pred, conf_out, score

    for _ in range(args.warmup):
        run()

    samples = []
    for index in range(args.repeat):
        start = time.perf_counter()
        pred, conf_out, score = run()
        elapsed = time.perf_counter() - start
        samples.append({
            "run": index + 1,
            "seconds": elapsed,
            "score": score,
            "map_checksum": checksum(pred),
            "conf_checksum": checksum(conf_out),
        })

    report = {
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "cpu_model": cpu_model(),
        "num_cpu": os.cpu_count(),
        "threads": args.threads,
        "width": args.width,
        "height": args.height,
        "warmup": args.warmup,
        "maxrss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "samples": samples,
    }
    data = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(data)
    print(data, end="")


if __name__ == "__main__":
    main()
