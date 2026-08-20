"""Parse page-selection-style channel specs like ``"2,4,5-10"``.

Shared by the GUI (the manual EIS "Channel(s)" field) and anywhere a compact
multi-channel selection is entered as text.  Accepts comma-separated single
channels and ``lo-hi`` inclusive ranges; whitespace is ignored.  Returns a
sorted, de-duplicated list of channels, validated against ``[min_ch, max_ch]``.
Two keywords relax that default for callers that need it — ``on_invalid="drop"``
for a forgiving free-text field, and ``order="as-written"`` where channel order
is measurement order — and both default to the strict contract above.

:func:`format_channel_spec` is the inverse, and it lives here rather than in any
one tool because the display need recurs everywhere the parse need does — and
because the obvious shortcut, printing ``f"{channels[0]}-{channels[-1]}"``, is
wrong: it renders ``1-3,8-16`` as ``1-16`` and tells the operator that seven
channels are in a run when they are not.
"""

from __future__ import annotations

from typing import Iterable, Literal

__all__ = ["parse_channel_spec", "format_channel_spec", "ChannelSpecError"]


class ChannelSpecError(ValueError):
    """Raised when a channel spec is empty, malformed, or out of range."""


def parse_channel_spec(
    spec: str,
    *,
    min_ch: int = 1,
    max_ch: int = 32,
    on_invalid: Literal["raise", "drop"] = "raise",
    order: Literal["sorted", "as-written"] = "sorted",
) -> list[int]:
    """Parse ``spec`` (e.g. ``"2,4,5-10"``) into a unique channel list.

    Parameters
    ----------
    spec :
        Comma-separated channels and ``lo-hi`` inclusive ranges, e.g.
        ``"1, 3, 5-8"``.  Whitespace around tokens and hyphens is ignored.
    min_ch, max_ch :
        Inclusive bounds every parsed channel must fall within.
    on_invalid :
        ``"raise"`` (default) refuses anything malformed or out of range.
        ``"drop"`` is the *forgiving* mode a free-text GUI field wants: an
        empty spec yields ``[]`` rather than raising, non-numeric tokens are
        skipped, a **single** channel outside the bounds is skipped, and a
        **range** that straddles a bound is **clamped** to it rather than
        dropped.  That clamp/drop asymmetry is deliberate — it is the
        long-standing Experiment-tab behaviour, where ``"0-99"`` at
        ``max_ch=16`` means "all sixteen", not "nothing".
    order :
        ``"sorted"`` (default) returns ascending channels.  ``"as-written"``
        preserves entry order with first-wins de-duplication, for the callers
        where channel order *is* measurement order (the Arrhenius sweep and
        the live BO campaign both drive channels in the order given).

    Raises
    ------
    ChannelSpecError
        Under ``on_invalid="raise"``: if the spec is empty, a token is
        non-numeric, a range is reversed, or a channel falls outside
        ``[min_ch, max_ch]``.  Never raised under ``"drop"``.
    """
    if on_invalid not in ("raise", "drop"):
        raise ValueError(f"on_invalid must be 'raise' or 'drop', not {on_invalid!r}")
    if order not in ("sorted", "as-written"):
        raise ValueError(f"order must be 'sorted' or 'as-written', not {order!r}")
    dropping = on_invalid == "drop"

    if spec is None or not str(spec).strip():
        if dropping:
            return []
        raise ChannelSpecError("no channels given")

    channels: list[int] = []
    seen: set[int] = set()

    def _keep(ch: int) -> None:
        if ch not in seen:
            seen.add(ch)
            channels.append(ch)

    for raw in str(spec).split(","):
        token = raw.strip()
        if not token:
            continue  # tolerate trailing/duplicate commas ("2,,4", "2,")
        try:
            if "-" in token.lstrip("-"):  # a range (bare leading '-' fails as non-numeric)
                lo_str, _, hi_str = token.partition("-")
                lo, hi = _as_int(lo_str, token), _as_int(hi_str, token)
                if dropping:
                    # Clamp to the bounds; a reversed range yields an empty
                    # range() and so contributes nothing, without erroring.
                    for ch in range(max(min_ch, lo), min(max_ch, hi) + 1):
                        _keep(ch)
                    continue
                if lo > hi:
                    raise ChannelSpecError(f"reversed range '{token}' (low > high)")
                for ch in range(lo, hi + 1):
                    _check_bounds(ch, min_ch, max_ch)
                    _keep(ch)
            else:
                ch = _as_int(token, token)
                if dropping:
                    if min_ch <= ch <= max_ch:
                        _keep(ch)
                    continue
                _check_bounds(ch, min_ch, max_ch)
                _keep(ch)
        except ChannelSpecError:
            if not dropping:
                raise  # "drop" swallows the malformed token and moves on

    if not channels:
        if dropping:
            return []
        raise ChannelSpecError("no channels given")
    return sorted(channels) if order == "sorted" else channels


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
