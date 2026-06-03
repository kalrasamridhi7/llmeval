"""
Relevance Evaluator.

Evaluates whether the LLM's answer directly addresses the question
that was asked. Uses the LLM-as-judge pattern to assess how well
the answer responds to the user's intent.
"""

from typing import Any, Dict

from llmeval.evaluators.llm_judge import LLMJudge

# ---------------------------------------------------------------------------
# Prompt constants — the heart of the relevance evaluation.
# ---------------------------------------------------------------------------

RELEVANCE_SYSTEM_PROMPT = """You are an expert evaluation judge specialising in \
answer relevance assessment. Your task is to determine whether an AI-generated \
answer directly and sufficiently addresses the question that was asked.

You MUST respond with ONLY a valid JSON object — no additional text, no markdown \
formatting, no code blocks. Just the raw JSON object.

JSON schema you MUST follow:
{
  "score": <float between 0.0 and 1.0>,
  "explanation": "<string summarising the relevance assessment>",
  "details": {}
}

Scoring guidelines:
- 1.0: The answer directly and completely addresses the question with no irrelevant content.
- 0.8-0.9: The answer addresses the question well but includes minor tangential information.
- 0.5-0.7: The answer partially addresses the question but misses key aspects or includes \
significant irrelevant content.
- 0.2-0.4: The answer barely addresses the question; mostly off-topic or tangential.
- 0.0-0.1: The answer does not address the question at all; completely irrelevant.

Focus on:
1. Does the answer address the core intent of the question?
2. Does the answer cover all key aspects the question asks about?
3. Is the answer free from excessive irrelevant information?
4. Would the questioner be satisfied that their question was answered?
"""

RELEVANCE_USER_PROMPT_TEMPLATE = """Evaluate the relevance of the following answer \
to the given question.

QUESTION:
{question}

CONTEXT (provided to the answering system):
{context}

ANSWER (to evaluate):
{answer}

Assess how directly and completely the ANSWER addresses the QUESTION, \
and return the evaluation as JSON."""


class RelevanceEvaluator(LLMJudge):
    """Evaluates whether the answer is relevant to the question.

    The relevance score represents how directly and completely the answer
    addresses the user's question. A score of 1.0 means the answer is
    perfectly on-topic; 0.0 means it is completely irrelevant.
    """

    def judge(
        self, question: str, context: str, answer: str
    ) -> Dict[str, Any]:
        """Judge the relevance of the answer to the question.

        Args:
            question: The input question that was asked.
            context: The retrieved context or documents.
            answer: The LLM's generated answer to evaluate.

        Returns:
            Dictionary with 'score', 'explanation', and 'details' keys.
        """
        user_prompt = RELEVANCE_USER_PROMPT_TEMPLATE.format(
            question=question,
            context=context,
            answer=answer,
        )

        result = self._call_llm(
            system_prompt=RELEVANCE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # Ensure details exists.
        if "details" not in result:
            result["details"] = {}

        return result
