"""CLI utility to measure streaming latency from the OpenAI Chat Completions API."""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Iterable, List

from openai import OpenAI


@dataclass
class LatencyMetrics:
    ttft: float
    tbt: float
    total_latency: float
    token_count: int


def compute_metrics(start: float, timestamps: List[float], last: float) -> LatencyMetrics:
    """Compute latency metrics based on recorded timestamps.

    Args:
        start: The timestamp captured before sending the request.
        timestamps: A list of timestamps for each token received.
        last: The timestamp captured after the stream completes.
    """

    token_count = len(timestamps)
    ttft = timestamps[0] - start if token_count else 0.0

    if token_count <= 1:
        tbt = 0.0
    else:
        gaps = [timestamps[i] - timestamps[i - 1] for i in range(1, token_count)]
        tbt = sum(gaps) / len(gaps)

    total_latency = last - start

    return LatencyMetrics(ttft=ttft, tbt=tbt, total_latency=total_latency, token_count=token_count)


def measure_latency(prompt: str) -> LatencyMetrics:
    """Stream a response from the OpenAI API and return latency metrics."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    start = time.perf_counter()
    timestamps: List[float] = []

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        if delta is None:
            continue

        has_content = delta.content is not None and delta.content != ""
        has_token = delta.tool_calls is not None and len(delta.tool_calls) > 0
        if has_content or has_token:
            timestamps.append(time.perf_counter())

    last = time.perf_counter()

    return compute_metrics(start=start, timestamps=timestamps, last=last)


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure streaming latency for an OpenAI Chat Completion.")
    parser.add_argument("prompt", help="Prompt to send to the model.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv or [])
    metrics = measure_latency(args.prompt)
    print(json.dumps(asdict(metrics), indent=2))


if __name__ == "__main__":
    main()
