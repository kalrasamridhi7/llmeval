from unittest.mock import patch, MagicMock
from llmeval.evaluators.hallucination import HallucinationEvaluator


@patch("openai.OpenAI")
def test_no_hallucination(mock_openai_class, sample_inputs, mock_openai_response):
    """Test when the answer contains no hallucinated claims and is fully grounded."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    details = {
        "hallucinated_spans": []
    }
    mock_client.chat.completions.create.return_value = mock_openai_response(
        score=1.0,
        explanation="The answer contains no hallucinations and is fully supported.",
        details=details
    )
    
    evaluator = HallucinationEvaluator(model="gpt-4o-mini")
    result = evaluator.evaluate(
        question=sample_inputs["question"],
        context=sample_inputs["context"],
        answer=sample_inputs["answer"]
    )
    
    assert isinstance(result["score"], float)
    assert result["score"] == 1.0
    assert len(result["details"]["hallucinated_spans"]) == 0


@patch("openai.OpenAI")
def test_heavy_hallucination(mock_openai_class, sample_inputs, mock_openai_response):
    """Test when the answer contains fabricated facts not present in the context."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    details = {
        "hallucinated_spans": [
            "Paris has a population of 50 million people.",
            "Paris is located in Germany."
        ]
    }
    mock_client.chat.completions.create.return_value = mock_openai_response(
        score=0.1,
        explanation="The answer contains severe hallucinations about location and population.",
        details=details
    )
    
    evaluator = HallucinationEvaluator(model="gpt-4o-mini")
    result = evaluator.evaluate(
        question=sample_inputs["question"],
        context=sample_inputs["context"],
        answer="Paris is the capital of France, but it is located in Germany and has 50 million people."
    )
    
    assert isinstance(result["score"], float)
    assert result["score"] == 0.1
    assert len(result["details"]["hallucinated_spans"]) == 2
    assert "located in Germany" in result["details"]["hallucinated_spans"][1]


def test_empty_context(sample_inputs):
    """Test handling of an empty context (should default to 0.0 hallucination score, indicating potential hallucinations since it cannot verify claims)."""
    evaluator = HallucinationEvaluator(model="gpt-4o-mini")
    result = evaluator.evaluate(
        question=sample_inputs["question"],
        context="",
        answer=sample_inputs["answer"]
    )
    
    assert result["score"] == 0.0
    assert "no context" in result["explanation"].lower()
    assert "hallucinated_spans" in result["details"]


@patch("openai.OpenAI")
def test_partial_hallucination(mock_openai_class, sample_inputs, mock_openai_response):
    """Test when the answer has a mixture of grounded and ungrounded statements."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    details = {
        "hallucinated_spans": [
            "and was founded by Julius Caesar"
        ]
    }
    mock_client.chat.completions.create.return_value = mock_openai_response(
        score=0.7,
        explanation="Most claims are grounded, but the historical claim about Julius Caesar is hallucinated.",
        details=details
    )
    
    evaluator = HallucinationEvaluator(model="gpt-4o-mini")
    result = evaluator.evaluate(
        question=sample_inputs["question"],
        context=sample_inputs["context"],
        answer="The capital of France is Paris and was founded by Julius Caesar."
    )
    
    assert isinstance(result["score"], float)
    assert result["score"] == 0.7
    assert len(result["details"]["hallucinated_spans"]) == 1
