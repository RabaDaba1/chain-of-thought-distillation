from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

from src.config import RAW_DATA_DIR
from src.dataset_generator.generator import run_generation
from src.dataset_generator.models import GenerationSettings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Chain-of-Thought examples for GSM8K."
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=-1,
        help="Number of questions to process (-1 for all).",
    )
    parser.add_argument(
        "--samples-per-question",
        type=int,
        default=30,
        help="Number of CoT samples per question.",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=25, help="Max concurrent API requests."
    )
    parser.add_argument(
        "--data-split", type=str, default="train", help="GSM8K dataset split to use."
    )
    args = parser.parse_args()

    settings = GenerationSettings(
        num_questions=args.num_questions,
        samples_per_question=args.samples_per_question,
        max_concurrency=args.max_concurrency,
        data_split=args.data_split,
        output_file=RAW_DATA_DIR / "dataset_test.jsonl",
        failed_log_file=RAW_DATA_DIR / "dataset_test_failed.jsonl",
        meta_file=RAW_DATA_DIR / "dataset_test_meta.json",
    )
    settings.ensure_paths()

    asyncio.run(run_generation(settings))


if __name__ == "__main__":
    main()
