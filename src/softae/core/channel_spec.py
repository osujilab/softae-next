"""Parse page-selection-style channel specs like ``"2,4,5-10"``.

Shared by the GUI (the manual EIS "Channel(s)" field) and anywhere a compact
multi-channel selection is entered as text.  Accepts comma-separated single
channels and ``lo-hi`` inclusive ranges; whitespace is ignored.  Returns a
sorted, de-duplicated list of channels, validated against ``[min_ch, max_ch]``.

:func:`format_channel_spec` is the inverse, and it lives here rather than in any
one tool because the display need recurs everywhere the parse need does — and
because the obvious shortcut, printing ``f"{channels[0]}-{channels[-1]}"``, is
wrong: it renders ``1-3,8-16`` as ``1-16`` and tells the operator that seven
channels are in a run when they are not.
"""

from __future__ import annotations

from typing import Iterable

__all__ = ["parse_channel_spec", "format_channel_spec", "ChannelSpecError"]


class ChannelSpecError(ValueError):
    """Raised when a channel spec is empty, malformed, or out of range."""


def parse_channel_spec(spec: str, *, min_ch: int = 1, max_ch: int = 32) -> list[int]:
    """Parse ``spec`` (e.g. ``"2,4,5-10"``) into a sorted, unique channel list.

    Parameters
    ----------
    spec :
        Comma-separated channels and ``lo-hi`` inclusive ranges, e.g.
        ``"1, 3, 5-8"``.  Whitespace around tokens and hyphens is ignored.
    min_ch, max_ch :
        Inclusive bounds every parsed channel must fall within.

    Raises
    ------
    ChannelSpecError
        If the spec is empty, a token is non-numeric, a range is reversed, or a
        channel falls outside ``[min_ch, max_ch]``.
    """
    if spec is None or not str(spec).strip():
        raise ChannelSpecError("no channels given")

    channels: set[int] = set()
    for raw in str(spec).split(","):
        token = raw.strip()
        if not token:
            continue  # tolerate trailing/duplicate commas ("2,,4", "2,")
        if "-" in token.lstrip("-"):  # a range (leave a bare leading '-' to fail as non-numeric)
            lo_str, _, hi_str = token.partition("-")
            lo, hi = _as_int(lo_str, token), _as_int(hi_str, token)
            if lo > hi:
                raise ChannelSpecError(f"reversed range '{token}' (low > high)")
            for ch in range(lo, hi + 1):
                _check_bounds(ch, min_ch, max_ch)
                channels.add(ch)
        else:
            ch = _as_int(token, token)
            _check_bounds(ch, min_ch, max_ch)
            channels.add(ch)

    if not channels:
        raise ChannelSpecError("no channels given")
    return sorted(channels)


def format_channel_spec(channels: Iterable[int]) -> str:
    """Collapse a channel list into the compact spec that would re-parse to it.

    ``[1, 2, 3, 8, 9, 10]`` → ``"1-3,8-10"``; ``[5]`` → ``"5"``; ``[]`` → ``""``.
    Contiguous runs of two are written out in full (``"4,5"``, not ``"4-5"``),
    which is what a reader expects and is no longer.

    Round-trips with :func:`parse_channel_spec` for any valid input, and that is
    the property worth having: a printed spec an operator can paste straight back
    into ``--channels``.
    """
    ordered = sorted({int(ch) for ch in channels})
    if not ordered:
        return ""
    parts: list[str] = []
    start = previous = ordered[0]
    for ch in ordered[1:] + [None]:            # sentinel closes the final run
        if ch is not None and ch == previous + 1:
            previous = ch
            continue
        span = previous - start
        if span == 0:
            parts.append(str(start))
        elif span == 1:
            parts.extend((str(start), str(previous)))
        else:
            parts.append(f"{start}-{previous}")
        if ch is not None:
            start = previous = ch
    return ",".join(parts)


def _as_int(text: str, token: str) -> int:
    try:
        return int(text.strip())
    except (TypeError, ValueError):
        raise ChannelSpecError(f"invalid channel token '{token}'") from None


def _check_bounds(ch: int, min_ch: int, max_ch: int) -> None:
    if ch < min_ch or ch > max_ch:
        raise ChannelSpecError(f"channel {ch} out of range [{min_ch}, {max_ch}]")
