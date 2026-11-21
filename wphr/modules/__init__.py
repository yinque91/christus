"""The seven logical modules (MASTER SPEC 2.1)."""

from wphr.modules.world import WorldModule
from wphr.modules.practice import PracticeModule
from wphr.modules.affect import AffectModule
from wphr.modules.temporal import TemporalModule
from wphr.modules.concept_language import ConceptLanguageModule
from wphr.modules.social import SocialModule
from wphr.modules.volition import VolitionModule

MODULE_CLASSES = (
    WorldModule,
    PracticeModule,
    AffectModule,
    TemporalModule,
    ConceptLanguageModule,
    SocialModule,
    VolitionModule,
)

__all__ = [
    "WorldModule",
    "PracticeModule",
    "AffectModule",
    "TemporalModule",
    "ConceptLanguageModule",
    "SocialModule",
    "VolitionModule",
    "MODULE_CLASSES",
]
