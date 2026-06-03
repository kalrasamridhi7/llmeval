"""
LLMEval — Open-Source LLM Output Evaluation Framework.

Evaluate the output quality of any LLM-powered application using
an LLM-as-judge approach across four key metrics: Faithfulness,
Relevance, Hallucination, and Coherence.
"""

__version__ = "0.1.0"

from typing import Dict, List, Optional, Any, Union

from llmeval.evaluators import (
    METRICS_REGISTRY,
    FaithfulnessEvaluator,
    RelevanceEvaluator,
    HallucinationEvaluator,
    CoherenceEvaluator,
)


def evaluate(
    question: str,
    context: str,
    answer: str,
    metrics: Optional[List[str]] = None,
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """Evaluate a single LLM output across specified metrics.

    Args:
        question: The input question that was asked.
        context: The retrieved context or documents provided to the LLM.
        answer: The LLM's generated answer to evaluate.
        metrics: List of metric names to evaluate. Defaults to all four metrics.
            Valid values: 'faithfulness', 'relevance', 'hallucination', 'coherence'.
        model: The LLM model to use as judge. Defaults to 'gpt-4o-mini'.

    Returns:
        Dictionary containing per-metric scores and an overall score:
        {
            "scores": {"faithfulness": {"score": 0.9, "explanation": "...", "details": {...}}, ...},
            "overall": 0.85
        }
    """
    if metrics is None or metrics == ["all"] or "all" in metrics:
        metrics = list(METRICS_REGISTRY.keys())

    scores = {}  # type: Dict[str, Any]
    for metric_name in metrics:
        metric_name = metric_name.lower().strip()
        if metric_name not in METRICS_REGISTRY:
            raise ValueError(
                "Unknown metric '{}'. Valid metrics: {}".format(
                    metric_name, list(METRICS_REGISTRY.keys())
                )
            )
        evaluator = METRICS_REGISTRY[metric_name](model=model)
        result = evaluator.evaluate(question=question, context=context, answer=answer)
        scores[metric_name] = result

    score_values = [s["score"] for s in scores.values() if isinstance(s.get("score"), (int, float))]
    overall = sum(score_values) / len(score_values) if score_values else 0.0

    return {"scores": scores, "overall": round(overall, 4)}


def evaluate_batch(
    rows: List[Dict[str, str]],
    metrics: Optional[List[str]] = None,
    model: str = "gpt-4o-mini",
) -> List[Dict[str, Any]]:
    """Evaluate a batch of LLM outputs across specified metrics.

    Args:
        rows: List of dicts, each containing 'question', 'context', and 'answer' keys.
        metrics: List of metric names to evaluate. Defaults to all four metrics.
        model: The LLM model to use as judge. Defaults to 'gpt-4o-mini'.

    Returns:
        List of result dictionaries, one per input row. Each contains:
        {
            "input": {"question": "...", "context": "...", "answer": "..."},
            "scores": {...},
            "overall": 0.85
        }
    """
    results = []  # type: List[Dict[str, Any]]
    for row in rows:
        result = evaluate(
            question=row.get("question", ""),
            context=row.get("context", ""),
            answer=row.get("answer", ""),
            metrics=metrics,
            model=model,
        )
        result["input"] = {
            "question": row.get("question", ""),
            "context": row.get("context", "")[:200] + "..." if len(row.get("context", "")) > 200 else row.get("context", ""),
            "answer": row.get("answer", "")[:200] + "..." if len(row.get("answer", "")) > 200 else row.get("answer", ""),
        }
        results.append(result)
    return results


__all__ = [
    "__version__",
    "evaluate",
    "evaluate_batch",
    "FaithfulnessEvaluator",
    "RelevanceEvaluator",
    "HallucinationEvaluator",
    "CoherenceEvaluator",
    "METRICS_REGISTRY",
]
