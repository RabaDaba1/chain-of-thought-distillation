from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Callable

from datasets import load_dataset

from dataset_generator.helpers.answers import (
    parse_gold_answer_number,
    parse_teacher_final_answer,
)
from dataset_generator.helpers.metrics import final_summary, live_progress
from dataset_generator.lib.teacher_client import TeacherClient
from src.config import (
    GSM8K_PATH,
    TEACHER_SYSTEM_PROMPT,
    TEACHER_USER_PROMPT,
)
from src.dataset_generator.io.jsonl import append_jsonl, read_jsonl
from src.dataset_generator.io.meta import write_meta
from src.dataset_generator.models import GenerationSettings, Question, SampleRecord
from src.logs import get_logger

logger = get_logger("dataset_generator")


def _existing_index(
    settings: GenerationSettings,
) -> tuple[dict[int, set[int]], dict[int, int]]:
    existing_ids: dict[int, set[int]] = defaultdict(set)
    counts: dict[int, int] = defaultdict(int)
    for rec in read_jsonl(settings.output_file):
        qid = int(rec["question_id"])
        sid = int(rec["sample_id"])
        existing_ids[qid].add(sid)
        counts[qid] += 1
    return existing_ids, counts


def _plan_questions(raw_dataset, settings: GenerationSettings):
    n = len(raw_dataset)
    limit = n if settings.num_questions == -1 else min(settings.num_questions, n)
    existing_ids, counts = _existing_index(settings)
    plan = []
    for i in range(limit):
        if counts.get(i, 0) < settings.samples_per_question:
            plan.append((i, raw_dataset[i], existing_ids.get(i, set())))
    return plan


def _to_question(idx: int, row: dict) -> Question:
    gold_text = row["answer"]
    return Question(
        question_id=idx,
        question=row["question"],
        gold_answer_text=gold_text,
        gold_answer_number=parse_gold_answer_number(gold_text),
    )


async def _produce_sample(
    *,
    settings: GenerationSettings,
    client: TeacherClient,
    question: Question,
    sample_id: int,
    lock: asyncio.Lock,
    update: Callable[[], None],
) -> None:
    try:
        resp = await client.generate_cot(question.question)
        parsed = parse_teacher_final_answer(resp.answer)

        record = SampleRecord(
            question_id=question.question_id,
            question=question.question,
            gold_answer_text=question.gold_answer_text,
            gold_answer_number=question.gold_answer_number,
            sample_id=sample_id,
            teacher_answer_text=resp.answer,
            teacher_answer_number=parsed,
            is_correct=parsed == question.gold_answer_number,
            finish_reason=resp.finish_reason,
            usage=resp.usage,
            latency_ms=resp.latency_ms,
            model=resp.model,
            request_id=resp.request_id,
        )

        append_jsonl(settings.output_file, record.model_dump())

        async with lock:
            settings.stats.successful_samples += 1
            if resp.usage:
                settings.stats.total_prompt_tokens += int(
                    resp.usage.get("prompt_tokens", 0)
                )
                settings.stats.total_completion_tokens += int(
                    resp.usage.get("completion_tokens", 0)
                )
            if record.is_correct:
                settings.stats.correct_samples += 1
    except Exception as e:
        logger.error(
            f"Failed to process sample {sample_id} for question {question.question_id}: {e}"
        )
        async with lock:
            settings.stats.failed_samples += 1

    update()


async def run_generation(settings: GenerationSettings) -> None:
    logger.info(f"Starting CoT generation with settings: {settings}")
    settings.ensure_paths()

    raw_dataset = load_dataset(GSM8K_PATH, "main", split="train")
    plan = _plan_questions(raw_dataset, settings)

    if not plan:
        logger.info("All questions have already been processed. Exiting.")
        return

    write_meta(
        settings.meta_file,
        {
            "model": settings.model_name,
            "system_prompt": TEACHER_SYSTEM_PROMPT,
            "user_prompt_template": TEACHER_USER_PROMPT,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "num_questions": len(plan),
            "samples_per_question": settings.samples_per_question,
            "max_concurrency": settings.max_concurrency,
            "request_timeout": settings.request_timeout,
        },
    )

    client = TeacherClient(
        model_name=settings.model_name,
        timeout=settings.request_timeout,
        temperature=settings.temperature,
        top_p=settings.top_p,
        base_url=settings.base_url,
        allow_fallbacks=settings.allow_fallbacks,
        provider_only=settings.provider_only,
    )

    sem = asyncio.Semaphore(settings.max_concurrency)
    lock = asyncio.Lock()
    update = live_progress(settings)

    async with client._client:
        tasks: list[asyncio.Task] = []
        for idx, row, existing_ids in plan:
            q = _to_question(idx, row)
            missing_sample_ids = set(range(settings.samples_per_question)) - set(
                existing_ids
            )
            for sid in missing_sample_ids:

                async def _task(q=q, sid=sid):
                    async with sem:
                        await _produce_sample(
                            settings=settings,
                            client=client,
                            question=q,
                            sample_id=sid,
                            lock=lock,
                            update=update,
                        )

                tasks.append(asyncio.create_task(_task()))

        if tasks:
            await asyncio.gather(*tasks)

    logger.info(f"Total questions processed: {len(plan)}")
    summary = final_summary(settings)
    logger.info(
        "Summary | duration=%s ok=%s failed=%s tokens_in=%s tokens_out=%s cost=%s output=%s",
        summary["duration"],
        summary["ok"],
        summary["failed"],
        summary["tokens_in"],
        summary["tokens_out"],
        summary["cost"],
        summary["output"],
    )
