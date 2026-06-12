"""
LLM-as-Judge base class.

Provides the foundation for all LLMEval evaluators. Handles:
- Dual provider support (OpenAI and Anthropic)
- Structured prompt dispatch and JSON response parsing
- Retry logic with exponential backoff
- Input truncation for very long inputs
"""

import json
import os
import re
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Maximum character length for context and answer inputs before truncation.
MAX_INPUT_LENGTH = 8000

# Retry configuration for rate-limited API calls.
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds


class LLMJudge(ABC):
    """Abstract base class for LLM-as-judge evaluators.

    All four LLMEval metrics (Faithfulness, Relevance, Hallucination,
    Coherence) extend this class. It handles API communication, JSON
    parsing, retries, and input validation.

    Args:
        model: The LLM model identifier to use as the judge.
            Defaults to 'gpt-4o-mini'. Models starting with 'claude'
            will use the Anthropic API; all others use OpenAI.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        """Initialize the LLM judge with the specified model.

        Args:
            model: The LLM model to use for evaluation. Defaults to 'gpt-4o-mini'.
        """
        self.model = model
        self._provider = self._detect_provider(model)
        self._client = None  # type: Any

    @staticmethod
    def _detect_provider(model: str) -> str:
        """Detect the API provider based on the model name.

        Args:
            model: The model identifier string.

        Returns:
            'anthropic' if the model name starts with 'claude',
            'gemini' if the model name starts with 'gemini',
            otherwise 'openai'.
        """
        if model.lower().startswith("claude"):
            return "anthropic"
        elif model.lower().startswith("gemini"):
            return "gemini"
        return "openai"

    def _get_client(self) -> Any:
        """Lazily initialise and return the API client.

        Returns:
            An initialised OpenAI, Anthropic, or Gemini client instance.

        Raises:
            EnvironmentError: If the required API key is not set.
        """
        if self._client is not None:
            return self._client

        if self._provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise EnvironmentError(
                    "ANTHROPIC_API_KEY environment variable is not set. "
                    "Please add it to your .env file."
                )
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        elif self._provider == "gemini":
            api_key = os.environ.get("GOOGLE_API_KEY", "")
            if not api_key:
                raise EnvironmentError(
                    "GOOGLE_API_KEY environment variable is not set. "
                    "Please add it to your .env file."
                )
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._client = genai
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY environment variable is not set. "
                    "Please add it to your .env file."
                )
            import openai
            self._client = openai.OpenAI(api_key=api_key)

        return self._client

    @staticmethod
    def _truncate_input(text: str, max_length: int = MAX_INPUT_LENGTH) -> str:
        """Truncate text to the maximum allowed length.

        If the text exceeds max_length, it is truncated and a note is
        appended indicating that truncation occurred.

        Args:
            text: The input text to potentially truncate.
            max_length: Maximum allowed character count. Defaults to MAX_INPUT_LENGTH.

        Returns:
            The original text if under the limit, or truncated text with a note.
        """
        if len(text) <= max_length:
            return text
        truncated = text[:max_length]
        return truncated + "\n\n[NOTE: Input was truncated from {} to {} characters]".format(
            len(text), max_length
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Send a prompt to the LLM and parse the JSON response.

        Implements retry logic with exponential backoff for transient errors
        and rate limiting. Extracts JSON from the response, handling cases
        where the LLM wraps the JSON in markdown code blocks.

        Args:
            system_prompt: The system-level instruction for the LLM.
            user_prompt: The user-level prompt containing the evaluation data.

        Returns:
            Parsed JSON dictionary from the LLM response.

        Raises:
            RuntimeError: If all retry attempts fail.
            ValueError: If the response cannot be parsed as JSON.
        """
        client = self._get_client()
        last_exception = None  # type: Optional[Exception]

        for attempt in range(MAX_RETRIES + 1):
            try:
                if self._provider == "anthropic":
                    response = client.messages.create(
                        model=self.model,
                        max_tokens=2048,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                    )
                    raw_text = response.content[0].text
                elif self._provider == "gemini":
                    # Use Google Gemini API
                    # Map gemini-1.5-flash to gemini-2.5-flash for current API compatibility
                    model_name = self.model
                    if model_name == "gemini-1.5-flash":
                        model_name = "gemini-2.5-flash"
                    model = client.GenerativeModel(
                        model_name=model_name,
                        system_instruction=system_prompt,
                    )
                    response = model.generate_content(
                        user_prompt,
                        generation_config=client.types.GenerationConfig(
                            response_mime_type="application/json",
                            temperature=0.0
                        )
                    )
                    raw_text = response.text
                else:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.0,
                        max_tokens=2048,
                    )
                    raw_text = response.choices[0].message.content

                return self._parse_json_response(raw_text)

            except (json.JSONDecodeError, ValueError) as e:
                # Don't retry parse errors — re-raise immediately.
                raise ValueError(
                    "Failed to parse LLM response as JSON: {}".format(str(e))
                ) from e
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                is_rate_limit = "rate" in error_str and "limit" in error_str
                is_overloaded = "overloaded" in error_str or "capacity" in error_str
                is_transient = "timeout" in error_str or "connection" in error_str

                if attempt < MAX_RETRIES and (is_rate_limit or is_overloaded or is_transient):
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "LLM call failed (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1, MAX_RETRIES + 1, delay, str(e)
                    )
                    time.sleep(delay)
                else:
                    raise RuntimeError(
                        "LLM call failed after {} attempts: {}".format(
                            attempt + 1, str(e)
                        )
                    ) from last_exception

        raise RuntimeError(
            "LLM call failed after {} attempts: {}".format(
                MAX_RETRIES + 1, str(last_exception)
            )
        )

    @staticmethod
    def _parse_json_response(raw_text: str) -> Dict[str, Any]:
        """Extract and parse JSON from an LLM response string.

        Handles cases where the LLM wraps JSON in markdown code blocks
        (```json ... ```) or returns plain JSON.

        Args:
            raw_text: The raw text response from the LLM.

        Returns:
            Parsed JSON as a dictionary.

        Raises:
            ValueError: If no valid JSON can be extracted from the response.
        """
        if not raw_text or not raw_text.strip():
            raise ValueError("Empty response from LLM")

        text = raw_text.strip()

        # Try to extract JSON from markdown code blocks.
        code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if code_block_match:
            text = code_block_match.group(1).strip()

        # Try direct JSON parse.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find a JSON object in the text.
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(
            "Could not extract valid JSON from LLM response. Raw text: {}".format(
                raw_text[:500]
            )
        )

    @abstractmethod
    def judge(
        self, question: str, context: str, answer: str
    ) -> Dict[str, Any]:
        """Perform the evaluation judgement. Subclasses must override this.

        Args:
            question: The input question that was asked.
            context: The retrieved context or documents.
            answer: The LLM's generated answer.

        Returns:
            Dictionary with 'score' (float 0.0-1.0), 'explanation' (str),
            and 'details' (dict) keys.
        """
        ...

    def evaluate(
        self, question: str, context: str, answer: str
    ) -> Dict[str, Any]:
        """Evaluate an LLM output and return validated results.

        Calls the subclass's judge() method and validates the output,
        ensuring the score is within the valid 0.0-1.0 range.

        Args:
            question: The input question that was asked.
            context: The retrieved context or documents.
            answer: The LLM's generated answer.

        Returns:
            Validated dictionary with 'score', 'explanation', and 'details' keys.
        """
        # Handle edge case: empty answer.
        if not answer or not answer.strip():
            return {
                "score": 0.0,
                "explanation": "The answer is empty. No evaluation can be performed.",
                "details": {},
            }

        # Truncate long inputs.
        context = self._truncate_input(context)
        answer = self._truncate_input(answer)

        result = self.judge(question=question, context=context, answer=answer)

        # Validate and clamp the score.
        score = result.get("score", 0.0)
        if not isinstance(score, (int, float)):
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0
        score = max(0.0, min(1.0, float(score)))
        result["score"] = round(score, 4)

        # Ensure explanation exists.
        if "explanation" not in result or not result["explanation"]:
            result["explanation"] = "No explanation provided by the judge."

        # Ensure details exists.
        if "details" not in result:
            result["details"] = {}

        return result
