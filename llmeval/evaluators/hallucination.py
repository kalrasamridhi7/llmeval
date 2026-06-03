"""
Hallucination Evaluator.

Identifies factual claims in the LLM's answer that contradict or go
beyond the provided context. Uses chain-of-thought prompting to
systematically identify hallucinated content.
"""

from typing import Any, Dict

from llmeval.evaluators.llm_judge import LLMJudge

# ---------------------------------------------------------------------------
# Prompt constants — chain-of-thought prompting for hallucination detection.
# ---------------------------------------------------------------------------

HALLUCINATION_SYSTEM_PROMPT = """You are an expert evaluation judge specialising in \
hallucination detection. Your task is to identify any claims in an AI-generated \
answer that contradict or go beyond what is stated in the provided context.

You MUST respond with ONLY a valid JSON object — no additional text, no markdown \
formatting, no code blocks. Just the raw JSON object.

JSON schema you MUST follow:
{
  "score": <float between 0.0 and 1.0>,
  "explanation": "<string summarising the hallucination assessment>",
  "details": {
    "hallucinated_spans": ["<list of hallucinated text spans from the answer>"]
  }
}

Use the following chain-of-thought process:
1. Extract all factual claims from the answer.
2. For each claim, check if the context SUPPORTS it, CONTRADICTS it, or is SILENT on it.
3. A claim is hallucinated if it:
   - Contradicts information in the context, OR
   - Introduces specific facts, numbers, names, or details NOT present in the context.
4. General knowledge statements (e.g., "the sky is blue") are NOT hallucinations \
unless they contradict the context.

Scoring rules:
- score = 1.0 means NO hallucination detected — the answer is fully grounded.
- score = 0.0 means HEAVY hallucination — most claims are fabricated.
- score = (non-hallucinated claims) / (total claims).
- If the answer contains no factual claims, score = 1.0.
- If the context is empty but the answer makes specific factual claims, \
those are likely hallucinations — score low.
- Include the exact text spans from the answer that are hallucinated \
in the hallucinated_spans list.
"""

HALLUCINATION_USER_PROMPT_TEMPLATE = """Detect hallucinations in the following answer. \
Use chain-of-thought reasoning to identify each claim, verify it against the context, \
and flag any hallucinated content.

QUESTION:
{question}

CONTEXT (source of truth):
{context}

ANSWER (to check for hallucinations):
{answer}

Analyse the ANSWER step by step, identify any hallucinated spans, and return \
the evaluation as JSON."""


class HallucinationEvaluator(LLMJudge):
    """Evaluates whether the answer contains hallucinated content.

    The hallucination score is inverted: 1.0 means NO hallucination
    (the answer is fully grounded), while 0.0 means the answer is
    heavily hallucinated with fabricated claims.
    """

    def judge(
        self, question: str, context: str, answer: str
    ) -> Dict[str, Any]:
        """Judge the answer for hallucinated content.

        Args:
            question: The input question that was asked.
            context: The retrieved context or documents (source of truth).
            answer: The LLM's generated answer to check.

        Returns:
            Dictionary with 'score', 'explanation', and 'details' containing
            'hallucinated_spans' list.
        """
        # Handle edge case: empty context.
        if not context or not context.strip():
            return {
                "score": 0.0,
                "explanation": (
                    "No context was provided. Any factual claims in the answer "
                    "cannot be verified and are treated as potential hallucinations."
                ),
                "details": {
                    "hallucinated_spans": [],
                },
            }

        user_prompt = HALLUCINATION_USER_PROMPT_TEMPLATE.format(
            question=question,
            context=context,
            answer=answer,
        )

        result = self._call_llm(
            system_prompt=HALLUCINATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # Ensure expected detail fields exist.
        details = result.get("details", {})
        if "hallucinated_spans" not in details:
            details["hallucinated_spans"] = []
        result["details"] = details

        return result
