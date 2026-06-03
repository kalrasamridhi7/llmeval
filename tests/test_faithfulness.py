from unittest.mock import patch, MagicMock
from llmeval.evaluators.faithfulness import FaithfulnessEvaluator


@patch("openai.OpenAI")
def test_high_faithfulness(mock_openai_class, sample_inputs, mock_openai_response):
    """Test case where all claims are supported by the context."""
    # Set up mock response
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    details = {
        "claims": [
            {"claim": "The capital of France is Paris.", "supported": True}
        ],
        "unsupported_claims": []
    }
    mock_client.chat.completions.create.return_value = mock_openai_response(
        score=1.0,
        explanation="The answer is fully supported by the context.",
        details=details
    )
    
    evaluator = FaithfulnessEvaluator(model="gpt-4o-mini")
    result = evaluator.evaluate(
        question=sample_inputs["question"],
        context=sample_inputs["context"],
        answer=sample_inputs["answer"]
    )
    
    assert isinstance(result["score"], float)
    assert result["score"] == 1.0
    assert result["explanation"] == "The answer is fully supported by the context."
    assert "claims" in result["details"]
    assert len(result["details"]["unsupported_claims"]) == 0


@patch("openai.OpenAI")
def test_low_faithfulness(mock_openai_class, sample_inputs, mock_openai_response):
    """Test case where some claims are unsupported by the context."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    details = {
        "claims": [
            {"claim": "The capital of France is Paris.", "supported": True},
            {"claim": "Paris has a population of 50 million people.", "supported": False}
        ],
        "unsupported_claims": ["Paris has a population of 50 million people."]
    }
    mock_client.chat.completions.create.return_value = mock_openai_response(
        score=0.5,
        explanation="The answer contains a claim about population which is unsupported.",
        details=details
    )
    
    evaluator = FaithfulnessEvaluator(model="gpt-4o-mini")
    result = evaluator.evaluate(
        question=sample_inputs["question"],
        context=sample_inputs["context"],
        answer="The capital of France is Paris, which has a population of 50 million people."
    )
    
    assert isinstance(result["score"], float)
    assert result["score"] == 0.5
    assert len(result["details"]["unsupported_claims"]) == 1
    assert result["details"]["unsupported_claims"][0] == "Paris has a population of 50 million people."


def test_empty_answer(sample_inputs):
    """Test how the evaluator handles an empty answer string (should evaluate immediately without calling API)."""
    evaluator = FaithfulnessEvaluator(model="gpt-4o-mini")
    result = evaluator.evaluate(
        question=sample_inputs["question"],
        context=sample_inputs["context"],
        answer=""
    )
    
    assert result["score"] == 0.0
    assert "empty" in result["explanation"].lower()
    assert result["details"] == {}


def test_empty_context(sample_inputs):
    """Test how the evaluator handles an empty context (should return 0.0 immediately without calling API)."""
    evaluator = FaithfulnessEvaluator(model="gpt-4o-mini")
    result = evaluator.evaluate(
        question=sample_inputs["question"],
        context="",
        answer=sample_inputs["answer"]
    )
    
    assert result["score"] == 0.0
    assert "no context" in result["explanation"].lower()
    assert "claims" in result["details"]


@patch("openai.OpenAI")
def test_long_input_truncation(mock_openai_class, sample_inputs, mock_openai_response):
    """Test that inputs exceeding the character limit are truncated without throwing an error."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    mock_client.chat.completions.create.return_value = mock_openai_response(
        score=0.9,
        explanation="Evaluated with truncated context.",
        details={}
    )
    
    # Create a context that is well over 8000 characters
    long_context = "France is a country. " * 500  # ~10,000 characters
    
    evaluator = FaithfulnessEvaluator(model="gpt-4o-mini")
    result = evaluator.evaluate(
        question=sample_inputs["question"],
        context=long_context,
        answer=sample_inputs["answer"]
    )
    
    assert isinstance(result["score"], float)
    assert result["score"] == 0.9
    
    # Verify mock was called with the truncated string
    args, kwargs = mock_client.chat.completions.create.call_args
    user_prompt = kwargs["messages"][1]["content"]
    assert "[NOTE: Input was truncated" in user_prompt
