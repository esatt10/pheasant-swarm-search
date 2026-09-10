"""``C0`` - the prior-only control.

No Pheasant, no research trace, no memory, no internet. It exists to separate
what the model already knew from what the region contributed, which is the
difference ``P0 - C0`` is named for.

Under the deterministic ``replay`` provider this arm has **no prior at all**
and abstains everywhere. That makes ``P0 - C0`` a floor on the corpus's
contribution rather than an estimate of it, and every report that prints the
delta says so.
"""

from __future__ import annotations

import time
from typing import Any

from .base import Answer, Arm


class PriorControlArm(Arm):
    arm_id = "C0"
    role = "control"

    def answer(self, question: Any, *, repetition: int) -> Answer:
        answer = self._new_answer(question, repetition)
        started = time.monotonic()
        with self.tracer.span(
            "arm.answer", arm_id=self.arm_id, question_id=question.question_id, stage="answer"
        ):
            # No retrieval, and therefore no passages. The answerer is given
            # the question and nothing else, which is the whole design.
            self._compose(question, [], answer)
        answer.latency_ms = (time.monotonic() - started) * 1000.0
        return self._record(answer)
