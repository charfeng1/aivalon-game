"""Leak filter utilities (currently no-op to allow unrestricted speech)."""

from typing import Tuple


def sanitize_speech(seat: int, text: str) -> Tuple[str, bool]:
    """Return speech unchanged; leak filtering disabled."""

    return text, False


def filter_speech_for_phase(seat: int, text: str, is_speech_phase: bool) -> Tuple[str, bool]:
    """Return speech unchanged regardless of phase."""

    return text, False
