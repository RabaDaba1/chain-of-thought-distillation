from __future__ import annotations

import time
from typing import Callable

import tqdm

from src.dataset_generator.models import GenerationSettings


def _fmt_int(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def live_progress(settings: GenerationSettings) -> Callable[[None], None]:
    pbar = tqdm.tqdm(desc="Generating samples", unit="sample")

    def update() -> None:
        stats = settings.stats
        cost = (stats.total_prompt_tokens / 1_000_000) * settings.cost_per_m_in_usd + (
            stats.total_completion_tokens / 1_000_000
        ) * settings.cost_per_m_out_usd
        acc = (
            (stats.correct_samples / stats.total_samples)
            if stats.total_samples
            else 0.0
        )
        pbar.total = stats.total_samples
        pbar.n = stats.total_samples
        pbar.set_postfix(
            {
                "cost": f"${cost:.4f}",
                "tokens_in": _fmt_int(stats.total_prompt_tokens),
                "tokens_out": _fmt_int(stats.total_completion_tokens),
                "acc": f"{acc * 100:.1f}%",
                "ok/err": f"{stats.successful_samples}/{stats.failed_samples}",
            },
            refresh=True,
        )

    return update


def final_summary(settings: GenerationSettings) -> dict[str, str]:
    end_time = time.time()
    duration_s = end_time - settings.stats.start_time
    cost = (
        settings.stats.total_prompt_tokens / 1_000_000 * settings.cost_per_m_in_usd
    ) + (
        settings.stats.total_completion_tokens / 1_000_000 * settings.cost_per_m_out_usd
    )
    return {
        "duration": f"{duration_s:.2f}s",
        "ok": str(settings.stats.successful_samples),
        "failed": str(settings.stats.failed_samples),
        "tokens_in": str(settings.stats.total_prompt_tokens),
        "tokens_out": str(settings.stats.total_completion_tokens),
        "cost": f"${cost:.4f}",
        "output": str(settings.output_file),
    }
