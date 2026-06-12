"""
LLMEval Dashboard — FastAPI web application.

Provides a REST API and serves the web dashboard for interactive
LLM output evaluation. Supports single and batch evaluation modes.
"""

import json
import os
import tempfile
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Load .env at startup.
load_dotenv()

# Resolve paths.
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(DASHBOARD_DIR, "static")

app = FastAPI(
    title="LLMEval Dashboard",
    description="Interactive LLM Output Quality Evaluation Dashboard",
    version="0.1.0",
)

# CORS middleware for development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models.
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    """Request model for single evaluation."""

    question: str = Field(..., description="The input question that was asked")
    context: str = Field(..., description="The retrieved context or documents")
    answer: str = Field(..., description="The LLM's generated answer")
    metrics: List[str] = Field(
        default=["faithfulness", "relevance", "hallucination", "coherence"],
        description="List of metrics to evaluate",
    )
    model: str = Field(
        default="gemini-1.5-flash",
        description="LLM model to use as judge",
    )


class BatchRow(BaseModel):
    """A single row in a batch evaluation request."""

    question: str
    context: str
    answer: str


class BatchEvaluateRequest(BaseModel):
    """Request model for batch evaluation."""

    rows: List[BatchRow] = Field(..., description="List of evaluation inputs")
    metrics: List[str] = Field(
        default=["faithfulness", "relevance", "hallucination", "coherence"],
        description="List of metrics to evaluate",
    )
    model: str = Field(
        default="gemini-1.5-flash",
        description="LLM model to use as judge",
    )


# ---------------------------------------------------------------------------
# API endpoints.
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint.

    Returns:
        Dictionary with status 'ok'.
    """
    return {"status": "ok"}


@app.post("/evaluate")
async def evaluate_single(request: EvaluateRequest) -> Dict[str, Any]:
    """Evaluate a single LLM output.

    Args:
        request: EvaluateRequest with question, context, answer, metrics, model.

    Returns:
        Dictionary with per-metric scores, explanations, and overall score.
    """
    try:
        from llmeval import evaluate
        results = evaluate(
            question=request.question,
            context=request.context,
            answer=request.answer,
            metrics=request.metrics,
            model=request.model,
        )
        return results
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Evaluation failed: {}".format(str(e)))


@app.post("/evaluate/batch")
async def evaluate_batch_endpoint(request: BatchEvaluateRequest) -> Dict[str, Any]:
    """Evaluate a batch of LLM outputs.

    Args:
        request: BatchEvaluateRequest with rows, metrics, model.

    Returns:
        Dictionary with per-row results and summary statistics.
    """
    try:
        from llmeval import evaluate

        results = []  # type: List[Dict[str, Any]]
        for idx, row in enumerate(request.rows):
            try:
                result = evaluate(
                    question=row.question,
                    context=row.context,
                    answer=row.answer,
                    metrics=request.metrics,
                    model=request.model,
                )
                result["input"] = {
                    "question": row.question[:200],
                    "context": row.context[:200],
                    "answer": row.answer[:200],
                }
                results.append(result)
            except Exception as e:
                results.append({
                    "scores": {},
                    "overall": 0.0,
                    "input": {"question": row.question[:200], "context": "", "answer": ""},
                    "error": str(e),
                })

        # Compute summary statistics.
        all_metrics = set()  # type: set
        for r in results:
            all_metrics.update(r.get("scores", {}).keys())

        summary = {}  # type: Dict[str, Dict[str, float]]
        for m in sorted(all_metrics):
            values = [
                r["scores"][m]["score"]
                for r in results
                if m in r.get("scores", {})
                and isinstance(r["scores"][m].get("score"), (int, float))
            ]
            if values:
                summary[m] = {
                    "mean": round(sum(values) / len(values), 4),
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                }

        return {"results": results, "summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail="Batch evaluation failed: {}".format(str(e)))


@app.post("/evaluate/batch/report")
async def batch_report(request: BatchEvaluateRequest) -> FileResponse:
    """Evaluate a batch and return an HTML report file.

    Args:
        request: BatchEvaluateRequest with rows, metrics, model.

    Returns:
        HTML file download response.
    """
    try:
        from llmeval import evaluate
        from llmeval.report import generate_batch_report

        results = []  # type: List[Dict[str, Any]]
        for row in request.rows:
            try:
                result = evaluate(
                    question=row.question,
                    context=row.context,
                    answer=row.answer,
                    metrics=request.metrics,
                    model=request.model,
                )
                result["input"] = {
                    "question": row.question[:200],
                    "context": row.context[:200],
                    "answer": row.answer[:200],
                }
                results.append(result)
            except Exception as e:
                results.append({
                    "scores": {},
                    "overall": 0.0,
                    "input": {"question": row.question[:200], "context": "", "answer": ""},
                    "error": str(e),
                })

        # Generate report to a temp file.
        tmp = tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w", encoding="utf-8"
        )
        tmp.close()
        generate_batch_report(results=results, output_path=tmp.name)

        return FileResponse(
            path=tmp.name,
            filename="llmeval_batch_report.html",
            media_type="text/html",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail="Report generation failed: {}".format(str(e)))


# ---------------------------------------------------------------------------
# Serve the frontend.
# ---------------------------------------------------------------------------

# Mount static files.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_dashboard() -> FileResponse:
    """Serve the main dashboard HTML page.

    Returns:
        The index.html file from the static directory.
    """
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
