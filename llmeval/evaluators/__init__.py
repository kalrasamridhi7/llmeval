"""
LLMEval evaluators subpackage.

Provides four evaluation metrics for LLM output quality assessment,
all built on the LLM-as-judge pattern.
"""

from typing import Dict, Type

from llmeval.evaluators.llm_judge import LLMJudge
from llmeval.evaluators.faithfulness import FaithfulnessEvaluator
from llmeval.evaluators.relevance import RelevanceEvaluator
from llmeval.evaluators.hallucination import HallucinationEvaluator
from llmeval.evaluators.coherence import CoherenceEvaluator

# Registry mapping metric name strings to evaluator classes.
METRICS_REGISTRY = {
    "faithfulness": FaithfulnessEvaluator,
    "relevance": RelevanceEvaluator,
    "hallucination": HallucinationEvaluator,
    "coherence": CoherenceEvaluator,
}  # type: Dict[str, Type[LLMJudge]]

__all__ = [
    "LLMJudge",
    "FaithfulnessEvaluator",
    "RelevanceEvaluator",
    "HallucinationEvaluator",
    "CoherenceEvaluator",
    "METRICS_REGISTRY",
]
