"""
Dog Whistle Detection Research Package

This package contains modules for detecting and analyzing dog whistle language
in social media content, including integration with the Allen AI Dog Whistle Glossary.
"""

from .preprocessing import TextPreprocessor, PlatformSpecificPreprocessor, create_sample_data
from .semantic_drift import CombinedDriftAnalyzer
from .glossary_builder import DogWhistleGlossary, GlossaryEntry
from .allen_ai_loader import AllenAIGlossaryLoader, DogWhistleEntry, load_allen_ai_glossary

try:
    from .llm_vetting import LLMVetter, ResultsProcessor, TermAnalysis, ClassificationResult
except ImportError:
    # LLM vetting may not be available without API keys
    LLMVetter = None
    ResultsProcessor = None
    TermAnalysis = None
    ClassificationResult = None

__version__ = "0.1.0"
__author__ = "Dog Whistle Detection Research Team"

__all__ = [
    'TextPreprocessor',
    'PlatformSpecificPreprocessor', 
    'create_sample_data',
    'CombinedDriftAnalyzer',
    'DogWhistleGlossary',
    'GlossaryEntry',
    'AllenAIGlossaryLoader',
    'DogWhistleEntry',
    'load_allen_ai_glossary',
    'LLMVetter',
    'ResultsProcessor',
    'TermAnalysis',
    'ClassificationResult'
]
