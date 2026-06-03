from unittest.mock import patch, MagicMock
from llmeval.evaluators.coherence import CoherenceEvaluator


@patch("openai.OpenAI")
def test_highly_coherent(mock_openai_class, sample_inputs, mock_openai_response):
    """Test when the answer is logical and well-structured."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    mock_client.chat.completions.create.return_value = mock_openai_response(
        score=1.0,
        explanation="The text is extremely well-structured and logical."
    )
    
    evaluator = CoherenceEvaluator(model="gpt-4o-mini")
    result = evaluator.evaluate(
        question=sample_inputs["question"],
        context=sample_inputs["context"],
        answer=sample_inputs["answer"]
    )
    
    assert isinstance(result["score"], float)
    assert result["score"] == 1.0
    assert result["explanation"] == "The text is extremely well-structured and logical."
    assert isinstance(result["details"], dict)


@patch("openai.OpenAI")
def test_incoherent(mock_openai_class, sample_inputs, mock_openai_response):
    """Test when the answer is logical nonsense or self-contradicting."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    mock_client.chat.completions.create.return_value = mock_openai_response(
        score=0.2,
        explanation="The text contains severe self-contradictions and illogical statements."
    )
    
    evaluator = CoherenceEvaluator(model="gpt-4o-mini")
    result = evaluator.evaluate(
        question=sample_inputs["question"],
        context=sample_inputs["context"],
        answer="I love Paris because it is a city. But it is not a city. It is a dog."
    )
    
    assert isinstance(result["score"], float)
    assert result["score"] == 0.2
    assert "contradiction" in result["explanation"].lower()


def test_empty_answer(sample_inputs):
    """Test handling of an empty answer string."""
    evaluator = CoherenceEvaluator(model="gpt-4o-mini")
    result = evaluator.evaluate(
        question=sample_inputs["question"],
        context=sample_inputs["context"],
        answer=""
    )
    
    assert result["score"] == 0.0
    assert "empty" in result["explanation"].lower()
    assert result["details"] == {}
