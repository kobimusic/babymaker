"""babymaker: N-way DCT-domain morphing of instrument samples from different soundfonts.

Given N source notes (rendered from .sf2 presets through the bundled TinySoundFont
renderer, or loaded from WAV), a `Morpher` aligns them in pitch, level and time, then
renders a "baby" sample for any weight vector w in R^N (normally on the simplex).
See README.md in this directory.
"""
from .morph import Morpher, MorphConfig  # noqa: F401
from .sources import Source, parse_source_spec, load_source  # noqa: F401

__all__ = ["Morpher", "MorphConfig", "Source", "parse_source_spec", "load_source"]
