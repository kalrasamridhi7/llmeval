"""
Faithfulness Evaluator.

Evaluates whether every claim in the LLM's answer is supported by
the provided context. Uses the LLM-as-judge pattern to extract claims
and verify each one against the source material.
"""

from typing import Any, Dict

from llmeval.evaluators.llm_judge import LLMJudge

# ---------------------------------------------------------------------------
# Prompt constants — the heart of the faithfulness evaluation.
# ---------------------------------------------------------------------------

FAITHFULNESS_SYSTEM_PROMPT = """You are an expert evaluation judge specialising in \
factual faithfulness assessment. Your task is to determine whether an AI-generated \
answer is fully grounded in the provided context.

You MUST respond with ONLY a valid JSON object — no additional text, no markdown \
formatting, no code blocks. Just the raw JSON object.

JSON schema you MUST follow:
{
  "score": <float between 0.0 and 1.0>,
  "explanation": "<string summarising the faithfulness assessment>",
  "details": {
    "claims": [
      {"claim": "<extracted claim text>", "supported": <true or false>}
    ],
    "unsupported_claims": ["<list of unsupported claim texts>"]
  }
}

Scoring rules:
- Extract every distinct factual claim from the answer.
- For each claim, determine if it is SUPPORTED by the context (true) or NOT (false).
- score = (number of supported claims) / (total number of claims).
- If the answer contains no factual claims, score = 1.0.
- If the context is empty but the answer makes factual claims, score = 0.0.
- Be strict: a claim is only supported if the context explicitly or clearly implies it.
"""

FAITHFULNESS_USER_PROMPT_TEMPLATE = """Evaluate the faithfulness of the following answer.

QUESTION:
{question}

CONTEXT (source of truth):
{context}

ANSWER (to evaluate):
{answer}

Extract all factual claims from the ANSWER, check each against the CONTEXT, \
and return the evaluation as JSON."""


class FaithfulnessEvaluator(LLMJudge):
    """Evaluates whether the answer is faithfully grounded in the context.

    The faithfulness score represents the ratio of claims in the answer
    that are supported by the provided context. A score of 1.0 means all
    claims are supported; 0.0 means none are.
    """

    def judge(
        self, question: str, context: str, answer: str
    ) -> Dict[str, Any]:
        """Judge the faithfulness of the answer against the context.

        Args:
            question: The input question that was asked.
            context: The retrieved context or documents (source of truth).
            answer: The LLM's generated answer to evaluate.

        Returns:
            Dictionary with 'score', 'explanation', and 'details' containing
            'claims' list and 'unsupported_claims' list.
        """
        # Handle edge case: empty context.
        if not context or not context.strip():
            return {
                "score": 0.0,
                "explanation": (
                    "No context was provided. Cannot verify faithfulness "
                    "of the answer without a source of truth."
                ),
                "details": {
                    "claims": [],
                    "unsupported_claims": [],
                },
            }

        user_prompt = FAITHFULNESS_USER_PROMPT_TEMPLATE.format(
            question=question,
            context=context,
            answer=answer,
        )

        result = self._call_llm(
            system_prompt=FAITHFULNESS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # Ensure expected detail fields exist.
        details = result.get("details", {})
        if "claims" not in details:
            details["claims"] = []
        if "unsupported_claims" not in details:
            details["unsupported_claims"] = [
                c["claim"] for c in details.get("claims", [])
                if not c.get("supported", True)
            ]
        result["details"] = details

        return result
