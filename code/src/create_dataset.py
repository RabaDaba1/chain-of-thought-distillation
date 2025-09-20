from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from sys import path

import httpx
import tqdm
import tqdm.asyncio
from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()
path.append(str(Path(__file__).parents[1]))

from src.config import (  # noqa: E402
    GSM8K_PATH,
    RAW_DATA_DIR,
    TEACHER_SYSTEM_PROMPT,
    TEACHER_USER_PROMPT,
)
from src.logs import get_logger  # noqa: E402


class State(BaseModel):
    start_time: float = Field(default_factory=time.time)
    successful_samples: int = 0
    failed_samples: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    correct_samples: int = 0

    @property
    def total_samples(self) -> int:
        return self.successful_samples + self.failed_samples


class GenerationConfig(BaseModel):
    num_questions: int
    samples_per_question: int
    max_concurrency: int
    model_name: str = "deepseek/deepseek-r1-distill-qwen-32b"
    output_file: Path = RAW_DATA_DIR / "dataset.jsonl"
    failed_log_file: Path = RAW_DATA_DIR / "dataset_failed.jsonl"
    meta_file: Path = RAW_DATA_DIR / "dataset_meta.json"
    request_timeout: int = 120
    max_retries: int = 5
    temperature: float = 1.0
    top_p: float = 1.0
    cost_per_m_in_usd: float = 0.27
    cost_per_m_out_usd: float = 0.27
    state: State = Field(default_factory=State)

    def __post_init__(self):
        self.output_file.parent.mkdir(exist_ok=True, parents=True)


logger = get_logger("dataset_generator")

NUMBER_REGEX = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def to_number(s: str):
    m = NUMBER_REGEX.findall(s.replace(",", ""))
    if not m:
        return None
    val = m[-1]
    if "e" in val.lower() or "." in val:
        return float(val)
    return int(val)


def parse_gsm8k_answer_number(ans: str) -> float | int | None:
    try:
        answer_part = ans.split("####")[-1].strip()
        return to_number(answer_part)
    except (IndexError, ValueError) as e:
        raise ValueError(f"Could not parse GSM8K answer: '{ans}'") from e


def parse_teacher_final_answer(content: str) -> float | int | None:
    try:
        for line in content.splitlines():
            if line.lower().startswith("final answer:"):
                return to_number(line.split(":", 1)[-1].strip())
        raise ValueError("No 'Final Answer:' line found")
    except (IndexError, ValueError) as e:
        raise ValueError(f"Could not parse teacher final answer: '{content}'") from e


@retry(
    retry=retry_if_exception_type(httpx.RequestError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    before_sleep=lambda rs: logger.warning(
        f"Retrying API call: {rs.outcome.exception()}"
    ),
)
async def get_teacher_response(
    question: str, client: httpx.AsyncClient, config: GenerationConfig
):
    payload = {
        "model": config.model_name,
        "messages": [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": TEACHER_USER_PROMPT.format(question=question)},
        ],
        "provider": {"only": ["deepinfra/fp8"], "allow_fallbacks": False},
        "temperature": config.temperature,
        "top_p": config.top_p,
        "tool_choice": "none",
    }
    resp = await client.post("/chat/completions", json=payload)
    resp.raise_for_status()
    return resp.json()


async def process_question(
    idx: int,
    row: dict,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    config: GenerationConfig,
    pbar: tqdm.tqdm,
    pbar_lock: asyncio.Lock,
    existing_sample_ids: set[int],
):
    question = row["question"]
    gold_answer = row["answer"]
    base_record = {
        "question_id": idx,
        "question": question,
        "gold_answer_text": gold_answer,
        "gold_answer_number": parse_gsm8k_answer_number(gold_answer),
    }

    for sample_id in set(range(config.samples_per_question)) - existing_sample_ids:
        async with semaphore:
            try:
                start_time = time.perf_counter()
                response = await get_teacher_response(question, client, config)
                latency_ms = int((time.perf_counter() - start_time) * 1000)

                choice = response["choices"][0]
                content = choice["message"]["content"]
                teacher_answer = parse_teacher_final_answer(content)

                sample_record = {
                    **base_record,
                    "sample_id": sample_id,
                    "teacher_answer_text": content,
                    "teacher_answer_number": teacher_answer,
                    "is_correct": teacher_answer == base_record["gold_answer_number"],
                    "finish_reason": choice.get("finish_reason"),
                    "usage": response.get("usage"),
                    "latency_ms": latency_ms,
                    "model": response.get("model"),
                    "request_id": response.get("id"),
                }

                with config.output_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(sample_record, ensure_ascii=False) + "\n")

                config.state.successful_samples += 1
                if response.get("usage"):
                    config.state.total_prompt_tokens += response["usage"].get(
                        "prompt_tokens", 0
                    )
                    config.state.total_completion_tokens += response["usage"].get(
                        "completion_tokens", 0
                    )
                if sample_record["is_correct"]:
                    config.state.correct_samples += 1

            except Exception as e:
                logger.error(
                    f"Failed to process sample {sample_id} for question {idx}: {e}"
                )
                config.state.failed_samples += 1
            finally:
                # Update progress bar per-sample with live stats
                cost = (
                    config.state.total_prompt_tokens / 1_000_000
                ) * config.cost_per_m_in_usd + (
                    config.state.total_completion_tokens / 1_000_000
                ) * config.cost_per_m_out_usd

                def _fmt_int(n: int) -> str:
                    if n >= 1_000_000:
                        return f"{n / 1_000_000:.2f}M"
                    if n >= 1_000:
                        return f"{n / 1_000:.1f}k"
                    return str(n)

                acc = (
                    (config.state.correct_samples / config.state.total_samples)
                    if config.state.total_samples > 0
                    else 0.0
                )

                postfix = {
                    "cost": f"${cost:.4f}",
                    "tokens_in": _fmt_int(config.state.total_prompt_tokens),
                    "tokens_out": _fmt_int(config.state.total_completion_tokens),
                    "acc": f"{acc * 100:.1f}%",
                    "ok/err": f"{config.state.successful_samples}/{config.state.failed_samples}",
                }

                async with pbar_lock:
                    pbar.update(1)
                    pbar.set_postfix(postfix, refresh=True)


def create_meta_file(config: GenerationConfig, num_questions: int):
    meta = {
        "model": config.model_name,
        "system_prompt": TEACHER_SYSTEM_PROMPT,
        "user_prompt_template": TEACHER_USER_PROMPT,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "num_questions": num_questions,
        "samples_per_question": config.samples_per_question,
        "max_concurrency": config.max_concurrency,
        "request_timeout": config.request_timeout,
        "timestamp": time.time(),
    }
    with config.meta_file.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved to {config.meta_file}")


def log_summary_report(config: GenerationConfig):
    end_time = time.time()
    duration_s = end_time - config.state.start_time
    cost = (config.state.total_prompt_tokens / 1_000_000 * config.cost_per_m_in_usd) + (
        config.state.total_completion_tokens / 1_000_000 * config.cost_per_m_out_usd
    )

    logger.info("--- Generation Complete ---")
    logger.info(f"Duration: {duration_s:.2f} seconds")
    logger.info(f"Successful samples: {config.state.successful_samples}")
    logger.info(f"Failed samples: {config.state.failed_samples}")
    logger.info(f"Total prompt tokens: {config.state.total_prompt_tokens}")
    logger.info(f"Total completion tokens: {config.state.total_completion_tokens}")
    logger.info(f"Estimated cost: ${cost:.4f}")
    logger.info(f"Raw data saved to: {config.output_file}")
    if config.state.failed_samples > 0:
        logger.warning(f"Failed attempts logged to: {config.failed_log_file}")


def get_generated_dataset(file_path: Path) -> list[dict]:
    if not file_path.exists():
        return []

    with file_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_existing_ids(existing_data: list[dict]) -> dict[int, set[int]]:
    existing_sample_ids: dict[int, set[int]] = defaultdict(set)
    for rec in existing_data:
        existing_sample_ids[rec["question_id"]].add(int(rec["sample_id"]))
    return existing_sample_ids


def get_processed_counts(existing_data: list[dict]) -> defaultdict[int, int]:
    counts = defaultdict(int)
    for record in existing_data:
        question_id = record["question_id"]
        counts[question_id] += 1
    return counts


def get_questions_to_process(
    dataset, config: GenerationConfig
) -> list[tuple[int, dict, int, set[int]]]:
    """
    Returns a list of tuples containing:
    (question index, question row, number of samples to generate, set of existing sample IDs)
    """
    existing = get_generated_dataset(config.output_file)
    counts = get_processed_counts(existing)
    existing_sample_ids_map = get_existing_ids(existing)

    num_questions_to_process = (
        len(dataset)
        if config.num_questions == -1
        else min(config.num_questions, len(dataset))
    )
    to_process: list[tuple[int, dict, int, set[int]]] = []
    for i in range(num_questions_to_process):
        if counts[i] < config.samples_per_question:
            to_process.append(
                (
                    i,
                    dataset[i],
                    config.samples_per_question - counts[i],
                    existing_sample_ids_map[i],
                )
            )
    return to_process


async def launch_generation_tasks(
    questions: list[tuple[int, dict, int, set[int]]],
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    config: GenerationConfig,
):
    total_remaining = sum(n for _, _, n, _ in questions)
    pbar = tqdm.tqdm(total=total_remaining, desc="Generating samples", unit="sample")
    pbar_lock = asyncio.Lock()

    tasks = []
    async with client:
        for idx, row, _, existing_sample_ids in questions:
            tasks.append(
                asyncio.create_task(
                    process_question(
                        idx,
                        row,
                        client,
                        semaphore,
                        config,
                        pbar,
                        pbar_lock,
                        existing_sample_ids,
                    )
                )
            )
        await asyncio.gather(*tasks)
    pbar.close()


async def main(config: GenerationConfig):
    logger.info(f"Starting CoT generation with config: {config}")

    client = httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        timeout=config.request_timeout,
    )
    semaphore = asyncio.Semaphore(config.max_concurrency)
    dataset = load_dataset(GSM8K_PATH, "main", split="train")

    questions_to_process = get_questions_to_process(dataset, config)

    if not questions_to_process:
        logger.info("All questions have already been processed. Exiting.")
        return

    create_meta_file(config, len(questions_to_process))
    await launch_generation_tasks(questions_to_process, client, semaphore, config)

    logger.info(f"Total questions to process: {len(questions_to_process)}")

    log_summary_report(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Chain-of-Thought examples for GSM8K."
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=1000,
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
    args = parser.parse_args()

    config = GenerationConfig(
        num_questions=args.num_questions,
        samples_per_question=args.samples_per_question,
        max_concurrency=args.max_concurrency,
    )

    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Exiting.")
