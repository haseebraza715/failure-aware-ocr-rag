from faar.quality import weird_char_ratio as gate_ratio
from faar.recovery import _weird_char_ratio as correction_ratio
from faar.settings import CorrectionSettings
from faar.textnoise import (
    CORRECTION_ALLOWED_PUNCTUATION,
    GATE_ALLOWED_PUNCTUATION,
    char_noise_ratio,
)


def test_correction_defaults_pin_phase3_behavior() -> None:
    defaults = CorrectionSettings()
    assert defaults.min_weird_char_ratio == 0.08
    assert defaults.min_length_ratio == 0.6
    assert defaults.max_length_ratio == 1.4
    assert defaults.min_token_overlap == 0.5
    assert defaults.max_noise_increase == 0.01


def test_punctuation_whitelists_are_pinned() -> None:
    assert GATE_ALLOWED_PUNCTUATION == ".,:%$()-/=+"
    assert CORRECTION_ALLOWED_PUNCTUATION == ".,;:!?$%()-/'\"&"


def test_metrics_diverge_only_on_whitelist_differences() -> None:
    # Prose punctuation counts as noise for the gate but not the corrector.
    prose = 'she said "wait!" and left?'
    assert correction_ratio(prose) == 0.0
    assert gate_ratio(prose) > 0.0

    # Formula characters count as noise for the corrector but not the gate's
    # character ratio (the gate handles structure via layout signals instead).
    formula = "a=b+c"
    assert char_noise_ratio(formula, GATE_ALLOWED_PUNCTUATION) == 0.0
    assert correction_ratio(formula) > 0.0

    # Truly weird characters count for both.
    garbled = "inv¢oice tØtal §6"
    assert correction_ratio(garbled) > 0.0
    assert char_noise_ratio(garbled, GATE_ALLOWED_PUNCTUATION) > 0.0


def test_clean_text_is_quiet_for_both_metrics() -> None:
    clean = "installation requires certified engineers."
    assert correction_ratio(clean) == 0.0
    assert char_noise_ratio(clean, GATE_ALLOWED_PUNCTUATION) == 0.0
