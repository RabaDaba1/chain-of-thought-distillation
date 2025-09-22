from __future__ import annotations

import re
from typing import Optional


class ParsingError(ValueError):
    pass


NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_number(text: str) -> int | float | None:
    match = NUMBER_PATTERN.findall(text.replace(",", ""))
    if not match:
        raise ParsingError(f"No number found in text: {text}")
    val = match[-1]
    if "e" in val.lower() or "." in val:
        return float(val)
    return int(val)


def parse_gold_answer_number(answer: str) -> Optional[int | float]:
    part = answer.split("####")[-1].strip()
    return parse_number(part)


def parse_teacher_final_answer(answer: str) -> Optional[int | float]:
    for line in answer.splitlines():
        if line.lower().startswith("final answer:"):
            return parse_number(line.split(":", 1)[-1].strip())
    raise ParsingError(f"No 'Final Answer:' line found in teacher answer: {answer}")
