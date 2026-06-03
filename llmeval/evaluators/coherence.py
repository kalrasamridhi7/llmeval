"""
Coherence Evaluator.

Evaluates whether the LLM's answer is logically structured and
internally consistent. This evaluator does NOT require context —
it assesses the answer as a standalone piece of text.
"""

from typing import Any, Dict

from llmeval.evaluators.llm_judge import LLMJudge

# ---------------------------------------------------------------------------
# Prompt constants — standalone answer quality assessment.
# ---------------------------------------------------------------------------

COHERENCE_SYSTEM_PROMPT = """You are an expert evaluation judge specialising in \
textual coherence assessment. Your task is to evaluate whether an AI-generated \
answer is logically structured, internally consistent, and well-organised.

You MUST respond with ONLY a valid JSON object — no additional text, no markdown \
formatting, no code blocks. Just the raw JSON object.

JSON schema you MUST follow:
{
  "score": <float between 0.0 and 1.0>,
  "explanation": "<string summarising the coherence assessment>",
  "details": {}
}

Evaluate the answer on these dimensions:
1. **Logical Flow**: Do ideas follow a logical progression? Are transitions smooth?
2. **Internal Consistency**: Does the answer contradict itself anywhere?
3. **Structure**: Is the answer well-organised with clear paragraphs or sections?
4. **Clarity**: Is the language clear and unambiguous?
5. **Completeness**: Does the answer feel complete, or does it trail off or leave gaps?

Scoring guidelines:
- 1.0: Perfectly coherent — logical flow, no contradictions, well-structured.
- 0.8-0.9: Mostly coherent with very minor structural issues.
- 0.5-0.7: Somewhat coherent but has noticeable logical gaps or structural problems.
- 0.2-0.4: Poor coherence — contradictions, unclear logic, or disorganised.
- 0.0-0.1: Incoherent — the text does not make logical sense.

IMPORTANT: You are evaluating ONLY the answer text. Do not consider the question \
or context — focus solely on whether the answer reads well as a standalone text.
"""

COHERENCE_USER_PROMPT_TEMPLATE = """Evaluate the coherence of the following answer. \
Focus on logical flow, internal consistency, structure, and clarity.

QUESTION (for reference only — do not factor into scoring):
{question}

ANSWER (to evaluate for coherence):
{answer}

Assess the coherence of the ANSWER as a standalone text and return \
the evaluation as JSON."""


class CoherenceEvaluator(LLMJudge):
    """Evaluates the logical coherence of the answer.

    Unlike other evaluators, Coherence does not use the context —
    it assesses the answer as a standalone piece of text for logical
    structure, internal consistency, and clarity.
    """

    def judge(
        self, question: str, context: str, answer: str
    ) -> Dict[str, Any]:
        """Judge the coherence of the answer.

        Note: The context parameter is accepted for API consistency but
        is NOT included in the evaluation prompt — coherence is assessed
        on the answer standalone.

        Args:
            question: The input question (included for reference only).
            context: Not used in coherence evaluation (ignored).
            answer: The LLM's generated answer to evaluate.

        Returns:
            Dictionary with 'score', 'explanation', and 'details' keys.
        """
        user_prompt = COHERENCE_USER_PROMPT_TEMPLATE.format(
            question=question,
            answer=answer,
        )

        result = self._call_llm(
            system_prompt=COHERENCE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # Ensure details exists.
        if "details" not in result:
            result["details"] = {}

        return result
