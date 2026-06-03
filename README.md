# 🔍 LLMEval

[![PyPI version](https://badge.fury.io/py/llmeval.svg)](https://badge.fury.io/py/llmeval)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**LLMEval** is a powerful, open-source Python evaluation framework designed to score the output quality of any LLM-powered application — RAG pipelines, chatbots, or autonomous agents. 

Using an **LLM-as-judge** pattern, LLMEval assesses generated answers across four core dimension-level metrics, delivering not just numerical scores but detailed, transparent step-by-step explanations and structured failure details (e.g. specific unsupported claims or hallucinated text spans).

---

## ✨ Features

- **4 Core Evaluation Metrics**: Faithfulness, Relevance, Hallucination, and Coherence.
- **Dual Provider Support**: Out-of-the-box integration with OpenAI (GPT models) and Anthropic (Claude models).
- **Interactive Web Dashboard**: A modern, dark-themed single-page browser UI with live score visualization, animated SVG gauges, and batch evaluations.
- **Robust CLI Tool**: Evaluate individual outputs or run batch jobs directly from your terminal with rich, color-coded tables.
- **Self-Contained Portable Reports**: Generates elegant, dark-mode HTML reports with inlined CSS/JS that you can share with your team.
- **Developer First**: Fully typed, dependency-minimized, and designed to work seamlessly in unit tests or CI/CD pipelines.

---

## 🚀 Quick Start

### Installation

Install LLMEval in editable mode or from source:

```bash
# Clone the repository
git clone https://github.com/your-username/llmeval.git
cd llmeval

# Install package with development dependencies
pip install -e .
```

### Environment Configuration

LLMEval requires API credentials for the LLM judge. Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Add your keys inside `.env`:
```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

---

## 📖 Python API Usage

You can evaluate outputs programmatically in just a few lines of code.

### Single Output Evaluation

```python
from llmeval import evaluate

# Define the inputs to evaluate
question = "How do I configure automatic backup retention for Amazon RDS PostgreSQL?"
context = "Amazon RDS automatically creates backups and saves them according to the retention period (0 to 35 days) that you specify."
answer = "You can set backup retention to between 0 and 35 days. Setting it to 0 disables automated backups."

# Run the evaluation
result = evaluate(
    question=question,
    context=context,
    answer=answer,
    metrics=["faithfulness", "relevance"],
    model="gpt-4o-mini"
)

# Print results
import json
print(json.dumps(result, indent=2))
```

### Batch Evaluation

```python
from llmeval import evaluate_batch

rows = [
    {
        "question": "What is the capital of France?",
        "context": "France is a country in Western Europe. Its capital is Paris.",
        "answer": "Paris is the capital of France."
    },
    {
        "question": "Is Python compiled or interpreted?",
        "context": "Python is dynamically typed and interpreted, though it compiles source to bytecode (.pyc).",
        "answer": "Python is purely compiled directly to machine code."
    }
]

results = evaluate_batch(rows, metrics=["all"], model="gpt-4o-mini")
```

---

## 🔧 CLI Reference

LLMEval provides a command-line interface using the `llmeval` command.

### Single Evaluation

Run a single evaluation and output an HTML report:

```bash
llmeval evaluate \
  --question "What is RAG?" \
  --context "Retrieval-Augmented Generation (RAG) is a technique that grants LLMs access to external facts." \
  --answer "RAG stands for Retrieval-Augmented Generation. It enhances LLMs with external data." \
  --metrics faithfulness \
  --metrics relevance \
  --output report.html \
  --model gpt-4o-mini
```

### Batch Evaluation

Evaluate a JSON dataset (array of objects with `question`, `context`, `answer` keys) and export a report:

```bash
llmeval batch \
  --input examples/sample_rag_output.json \
  --metrics all \
  --output batch_report.html \
  --model gpt-4o-mini
```

---

## 📊 Web Dashboard

LLMEval ships with an interactive FastAPI-powered dashboard.

To launch the dashboard server:

```bash
# Start uvicorn
uvicorn dashboard.app:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000` in your web browser.

### Features:
- **Interactive Playground**: Paste a question, context, and answer, choose your metrics, and hit **Evaluate**.
- **Score Gauges**: Premium, animated SVG score rings color-coded by performance thresholds (Green: $\ge 80\%$, Yellow: $50-79\%$, Red: $< 50\%$).
- **Explanation Accordions**: Click on any metric card to expand and read the judge's step-by-step reasoning.
- **Structured Failure Analysis**: Displays lists of specific unsupported statements or hallucinated phrases extracted by the model.
- **Batch Upload**: Drag and drop `.json` or `.csv` files, view progress bars, sort the results table dynamically, and download combined HTML reports.

---

## 🔍 Metrics Reference

| Metric | What it Measures | Score Range | Input Requirements | Failure Details Captured |
|:---|:---|:---|:---|:---|
| **Faithfulness** | Is the answer strictly grounded *only* in the retrieved context? | `0.0 - 1.0` | Question, Context, Answer | Lists of specific unsupported claims |
| **Relevance** | Does the answer directly and completely address the user's question? | `0.0 - 1.0` | Question, Context, Answer | Qualitative critique of gaps/tangents |
| **Hallucination** | Does the answer introduce fabricated facts or contradict the context? | `0.0 - 1.0` | Question, Context, Answer | Precise spans/phrases that are hallucinated |
| **Coherence** | Is the answer logically structured, consistent, and clear (standalone)? | `0.0 - 1.0` | Question, Answer (Context ignored) | Qualitative critique of logical structure |

---

## 🏗 How It Works (LLM-as-Judge)

Evaluating text quality is notoriously difficult for traditional metrics (like ROUGE or BLEU) which only measure n-gram overlap. LLMEval solves this by framing evaluation as structured logical tasks executed by highly capable LLMs:

1. **Information Extraction**: The judge extracts distinct factual assertions (claims) from the answer.
2. **NLI Verification**: For each claim, the judge performs Natural Language Inference against the context to determine if it is supported, refuted, or silent.
3. **Structured Scoring**: The final score is computed deterministically (e.g. supported claims divided by total claims), eliminating subjective variations.
4. **JSON Schema Enforcement**: Prompts enforce exact JSON output formatting, which is parsed, validated, and clamped automatically by the framework.

---

## 🧪 Running Automated Tests

We use `pytest` for all unit testing. To run the tests:

```bash
# Install package with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

Tests use `unittest.mock` to mock LLM provider APIs, meaning you can run the suite offline without spending API credits.

---

## 🤝 Contributing

Contributions are welcome! Whether you are proposing a new evaluator metric, optimizing prompt engineering, or enhancing dashboard animations:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
