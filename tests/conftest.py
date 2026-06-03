import os
import pytest
from unittest.mock import MagicMock

# Set up mock API keys in environment before anything loads
os.environ["OPENAI_API_KEY"] = "mock-openai-api-key-value"
os.environ["ANTHROPIC_API_KEY"] = "mock-anthropic-api-key-value"


@pytest.fixture
def sample_inputs():
    """Provides a standard set of question, context, and answer inputs for tests."""
    return {
        "question": "What is the capital of France?",
        "context": "France is a country in Western Europe. Its capital and most populous city is Paris.",
        "answer": "The capital of France is Paris."
    }


@pytest.fixture
def mock_openai_response():
    """Fixture that returns a function to create a mock OpenAI chat completion response.

    The response returns a JSON string in its content attribute containing the
    specified score, explanation, and optional details.
    """
    def _create_mock(score: float, explanation: str, details: dict = None):
        if details is None:
            details = {}
            
        import json
        response_data = {
            "score": score,
            "explanation": explanation,
            "details": details
        }
        
        # Build the mock structure
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        mock_message.content = json.dumps(response_data)
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]
        
        return mock_completion

    return _create_mock
