"""SAGE: Self-Adapting Group of Experts for multi-agent reasoning."""

from .answer.extractor import RegexExtractor, XFinderExtractor
from .config import ModelConfig, SageConfig, load_model_config, load_sage_config
from .llm.client import OpenAICompatClient
from .sage import SAGE, SageResult

__all__ = [
    "SAGE",
    "ModelConfig",
    "OpenAICompatClient",
    "RegexExtractor",
    "SageConfig",
    "SageResult",
    "XFinderExtractor",
    "load_model_config",
    "load_sage_config",
]
__version__ = "1.0.0"
