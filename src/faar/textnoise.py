"""Shared character-noise measurement for OCR text.

The quality gate and the correction gate intentionally use different
punctuation whitelists, so they share one mechanism but keep distinct
character sets:

- ``GATE_ALLOWED_PUNCTUATION`` tolerates ``=`` and ``+`` because table- and
  formula-heavy pages legitimately contain them; the gate has separate layout
  signals for structural damage.
- ``CORRECTION_ALLOWED_PUNCTUATION`` tolerates prose punctuation
  (``! ? ' " ; &``) because the corrector runs on running text and must not
  count ordinary sentences as noise, which would make corrections look like
  they "reduced" noise they never touched.

Changing either set changes gate/correction behavior and therefore the
research result; tests/test_textnoise.py pins the current values.
"""

from __future__ import annotations

GATE_ALLOWED_PUNCTUATION = ".,:%$()-/=+"
CORRECTION_ALLOWED_PUNCTUATION = ".,;:!?$%()-/'\"&"


def char_noise_ratio(text: str, allowed_punctuation: str) -> float:
    """Fraction of characters that are neither alphanumeric, whitespace, nor allowed."""
    if not text:
        return 0.0
    weird = sum(
        1
        for char in text
        if not (char.isalnum() or char.isspace() or char in allowed_punctuation)
    )
    return weird / max(len(text), 1)
