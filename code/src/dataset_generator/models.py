from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class RunStats(BaseModel):
    start_time: float = Field(default_factory=time.time)
    successful_samples: int = 0
    failed_samples: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    correct_samples: int = 0

    @property
    def total_samples(self) -> int:
        return self.successful_samples + self.failed_samples


class GenerationSettings(BaseModel):
    num_questions: int
    samples_per_question: int
    max_concurrency: int

    model_name: str = "deepseek/deepseek-r1-distill-qwen-32b"

    # OpenRouter clinet
    base_url: str = "https://openrouter.ai/api/v1"
    allow_fallbacks: bool = False
    provider_only: list[str] = ["deepinfra/fp8"]

    # IO
    output_file: Path
    failed_log_file: Path
    meta_file: Path

    # Runtime
    request_timeout: int = 120
    max_retries: int = 5
    temperature: float = 1.0
    top_p: float = 1.0

    # Cost
    cost_per_m_in_usd: float = 0.27
    cost_per_m_out_usd: float = 0.27

    # Mutable run-time stats
    stats: RunStats = Field(default_factory=RunStats)

    def ensure_paths(self) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.failed_log_file.parent.mkdir(parents=True, exist_ok=True)
        self.meta_file.parent.mkdir(parents=True, exist_ok=True)


class Question(BaseModel):
    question_id: int
    question: str
    gold_answer_text: str
    gold_answer_number: Optional[float | int]


class TeacherResponse(BaseModel):
    answer: str
    parsed_number: Optional[float | int]
    finish_reason: Optional[str] = None
    usage: Optional[dict[str, Any]] = None
    latency_ms: int
    model: Optional[str] = None
    request_id: Optional[str] = None


class SampleRecord(BaseModel):
    # Mirrors the JSONL schema currently emitted
    question_id: int
    question: str
    gold_answer_text: str
    gold_answer_number: Optional[float | int]

    sample_id: int
    teacher_answer_text: str
    teacher_answer_number: Optional[float | int]
    is_correct: bool

    finish_reason: Optional[str] = None
    usage: Optional[dict[str, Any]] = None
    latency_ms: int
    model: Optional[str] = None
    request_id: Optional[str] = None
