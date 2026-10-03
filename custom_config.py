"""
custom_config.py

User-specific overrides for OCR training and synthetic dataset generation.
Inherits all default configurations from config_defaults.py.
"""
import copy
from config_defaults import OCR_TRAINING_CONFIG as _DEFAULT_CONFIG

# Deep copy ensures user overrides do not mutate fallback defaults
OCR_TRAINING_CONFIG = copy.deepcopy(_DEFAULT_CONFIG)

# Custom user overrides can be specified below:
# OCR_TRAINING_CONFIG["debug_mode"] = False

# >>> managed by manage_domains — do not edit >>>
# <<< managed by manage_domains — do not edit <<<
