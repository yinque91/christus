"""WPHR - World-Practice-History Recursive Architecture.

Non-Transformer, non-JEPA cognitive core: sparse modular continuous-time
recurrent network (SM-CTRN) plus explicit memory, concept graph and learning
rules.

Terminology note: "subject", "experience" and "affect" are engineering
shorthands. Nothing in this package claims or demonstrates consciousness,
subjective experience, or human-level intelligence.
"""

from wphr.version import __version__, SPEC_VERSION, SCHEMA_VERSION

__all__ = ["__version__", "SPEC_VERSION", "SCHEMA_VERSION"]
