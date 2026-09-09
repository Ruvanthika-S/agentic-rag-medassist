#!/usr/bin/env python3
"""Extract the PubMedQA and BioASQ subsets from MIRAGE's benchmark file."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="mirage_repo/benchmark.json")
    parser.add_argument("--output-dir", default="data/benchmark_subsets")
    args = parser.parse_args()

    with Path(args.benchmark).open("r", encoding="utf-8") as benchmark_file:
        benchmark = json.load(benchmark_file)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subset_name in ("pubmedqa", "bioasq"):
        output_path = output_dir / f"{subset_name}.json"
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(benchmark[subset_name], output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        print(f"{subset_name}: {len(benchmark[subset_name]):,} entries -> {output_path}")


if __name__ == "__main__":
    main()