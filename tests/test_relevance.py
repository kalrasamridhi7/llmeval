from unittest.mock import patch, MagicMock
from llmeval.evaluators.relevance import RelevanceEvaluator


@patch("openai.OpenAI")
def test_highly_relevant(mock_openai_class, sample_inputs, mock_openai_response):
    """Test when the answer is highly relevant and directly answers the question."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    mock_client.chat.completions.create.return_value = mock_openai_response(
        score=1.0,
        explanation="The answer directly and fully addresses the question."
    )
    
    evaluator = RelevanceEvaluator(model="gpt-4o-mini")
    result = evaluator.evaluate(
        question=sample_inputs["question"],
        context=sample_inputs["context"],
        answer=sample_inputs["answer"]
    )
    
    assert isinstance(result["score"], float)
    assert result["score"] == 1.0
    assert result["explanation"] == "The answer directly and fully addresses the question."
    assert isinstance(result["details"], dict)


@patch("openai.OpenAI")
def test_irrelevant(mock_openai_class, sample_inputs, mock_openai_response):
    """Test when the answer is completely off-topic or irrelevant."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    mock_client.chat.completions.create.return_value = mock_openai_response(
        score=0.1,
        explanation="The answer discusses pizza, which is irrelevant to France's capital."
    )
    
    evaluator = RelevanceEvaluator(model="gpt-4o-mini")
    result = evaluator.evaluate(
        question=sample_inputs["question"],
        context=sample_inputs["context"],
        answer="I love eating pizza on Fridays."
    )
    
    assert isinstance(result["score"], float)
    assert result["score"] == 0.1
    assert "pizza" in result["explanation"]


@patch("openai.OpenAI")
def test_partially_relevant(mock_openai_class, sample_inputs, mock_openai_response):
    """Test when the answer is only partially relevant or contains too much tangential info."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    mock_client.chat.completions.create.return_value = mock_openai_response(
        score=0.6,
        explanation="The answer mentions Paris but focuses too much on other European capitals."
    )
    
    evaluator = RelevanceEvaluator(model="gpt-4o-mini")
    result = evaluator.evaluate(
        question=sample_inputs["question"],
        context=sample_inputs["context"],
        answer="Paris is a city, but London is the capital of the UK and Berlin is the capital of Germany."
    )
    
    assert isinstance(result["score"], float)
    assert result["score"] == 0.6
    assert isinstance(result["details"], dict)


def test_empty_answer(sample_inputs):
    """Test handling of an empty answer string."""
    evaluator = RelevanceEvaluator(model="gpt-4o-mini")
    result = evaluator.evaluate(
        question=sample_inputs["question"],
        context=sample_inputs["context"],
        answer=""
    )
    
    assert result["score"] == 0.0
    assert "empty" in result["explanation"].lower()
    assert result["details"] == {}
