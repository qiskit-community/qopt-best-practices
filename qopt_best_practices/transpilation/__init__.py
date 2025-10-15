"""Module with transpiler methods for QAOA like circuits."""

from .generate_preset_qaoa_pass_manager import generate_preset_qaoa_pass_manager
from .backend_utils import remove_instruction_from_target
from .annotated_transpilation_passes import (
    AnnotatedPrepareCostLayer,
    AnnotatedCommuting2qGateRouter,
    AnnotatedSwapToFinalMapping,
    SynthesizeAndSimplifyCostLayer,
    UnrollBoxes,
)
