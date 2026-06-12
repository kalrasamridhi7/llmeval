"""
LLMEval CLI — Command-line interface for LLM output evaluation.

Provides two commands:
- `llmeval evaluate` — evaluate a single LLM output
- `llmeval batch` — evaluate a batch of LLM outputs from a JSON file

Uses click for argument parsing and rich for formatted terminal output.
"""

import json
import os
import sys

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from llmeval import evaluate, evaluate_batch
from llmeval.evaluators import METRICS_REGISTRY
from llmeval.report import generate_single_report, generate_batch_report

console = Console()

# Available model choices.
MODEL_CHOICES = ["gemini-1.5-flash", "gpt-4o-mini", "gpt-4o", "claude-sonnet-4-6"]

# Available metric choices.
METRIC_CHOICES = list(METRICS_REGISTRY.keys()) + ["all"]


def _score_style(score: float) -> str:
    """Return a rich markup colour based on the score threshold.

    Args:
        score: Evaluation score between 0.0 and 1.0.

    Returns:
        Rich colour string for styling.
    """
    if score >= 0.8:
        return "green"
    elif score >= 0.5:
        return "yellow"
    else:
        return "red"


def _print_results(results: dict) -> None:
    """Print evaluation results as a formatted rich table.

    Args:
        results: Dictionary with 'scores' and 'overall' keys.
    """
    scores = results.get("scores", {})
    overall = results.get("overall", 0.0)

    table = Table(
        title="LLMEval Results",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        padding=(0, 1),
    )
    table.add_column("Metric", style="bold", min_width=16)
    table.add_column("Score", justify="center", min_width=8)
    table.add_column("Explanation", max_width=60, no_wrap=False)

    for metric_name, metric_result in scores.items():
        score = metric_result.get("score", 0.0)
        explanation = metric_result.get("explanation", "")
        style = _score_style(score)
        table.add_row(
            metric_name.title(),
            "[{}]{:.2f}[/{}]".format(style, score, style),
            explanation[:120] + ("..." if len(explanation) > 120 else ""),
        )

    # Add separator and overall row.
    table.add_section()
    overall_style = _score_style(overall)
    pf = "✓ PASS" if overall >= 0.7 else "✗ FAIL"
    pf_style = "green" if overall >= 0.7 else "red"
    table.add_row(
        "[bold]OVERALL[/bold]",
        "[bold {}]{:.2f}[/bold {}]".format(overall_style, overall, overall_style),
        "[{}]{}[/{}]".format(pf_style, pf, pf_style),
    )

    console.print()
    console.print(table)
    console.print()


@click.group()
@click.version_option(version="0.1.0", prog_name="llmeval")
def cli() -> None:
    """LLMEval — Evaluate the output quality of any LLM-powered application.

    Uses an LLM-as-judge approach to score outputs on Faithfulness,
    Relevance, Hallucination, and Coherence.
    """
    # Load environment variables from .env file.
    load_dotenv()


@cli.command()
@click.option(
    "--question", "-q",
    required=True,
    help="The input question that was asked.",
)
@click.option(
    "--context", "-c",
    required=True,
    help="The retrieved context or documents provided to the LLM.",
)
@click.option(
    "--answer", "-a",
    required=True,
    help="The LLM's generated answer to evaluate.",
)
@click.option(
    "--metrics", "-m",
    multiple=True,
    default=["all"],
    type=click.Choice(METRIC_CHOICES, case_sensitive=False),
    help="Metrics to evaluate. Can specify multiple. Default: all.",
)
@click.option(
    "--output", "-o",
    default="report.html",
    help="Output path for the HTML report. Default: report.html.",
)
@click.option(
    "--model",
    default="gemini-1.5-flash",
    type=click.Choice(MODEL_CHOICES, case_sensitive=False),
    help="LLM model to use as judge. Default: gemini-1.5-flash.",
)
@click.argument('extra_args', nargs=-1, required=False)
def evaluate_cmd(
    question: str,
    context: str,
    answer: str,
    metrics: tuple,
    output: str,
    model: str,
    extra_args: tuple = (),
) -> None:
    """Evaluate a single LLM output across specified metrics."""
    metrics_list = list(metrics)
    if extra_args:
        metrics_list.extend(extra_args)

    # Validate metrics
    valid_metrics = []
    for m in metrics_list:
        m_clean = m.lower().strip()
        if m_clean == "all":
            continue
        if m_clean not in METRICS_REGISTRY:
            raise click.BadParameter(
                f"Unknown metric '{m}'. Valid metrics: {list(METRICS_REGISTRY.keys())}"
            )
        valid_metrics.append(m_clean)

    if not valid_metrics or "all" in [m.lower().strip() for m in metrics_list]:
        metrics_list = list(METRICS_REGISTRY.keys())
    else:
        metrics_list = valid_metrics

    console.print(
        Panel(
            "[bold cyan]LLMEval[/bold cyan] — Evaluating with [bold]{}[/bold] using model [bold]{}[/bold]".format(
                ", ".join(m.title() for m in metrics_list), model
            ),
            border_style="cyan",
        )
    )

    try:
        with console.status("[bold cyan]Evaluating...", spinner="dots"):
            results = evaluate(
                question=question,
                context=context,
                answer=answer,
                metrics=metrics_list,
                model=model,
            )
    except Exception as e:
        console.print("[bold red]Error:[/bold red] {}".format(str(e)))
        sys.exit(1)

    _print_results(results)

    # Generate HTML report.
    try:
        generate_single_report(
            results=results,
            output_path=output,
            question=question,
            context=context,
            answer=answer,
        )
        console.print(
            "[dim]HTML report saved to:[/dim] [bold cyan]{}[/bold cyan]".format(output)
        )
    except Exception as e:
        console.print("[yellow]Warning:[/yellow] Could not save report: {}".format(str(e)))


@cli.command()
@click.option(
    "--input", "-i",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to a JSON file with evaluation data.",
)
@click.option(
    "--metrics", "-m",
    multiple=True,
    default=["all"],
    type=click.Choice(METRIC_CHOICES, case_sensitive=False),
    help="Metrics to evaluate. Can specify multiple. Default: all.",
)
@click.option(
    "--output", "-o",
    default="batch_report.html",
    help="Output path for the HTML report. Default: batch_report.html.",
)
@click.option(
    "--model",
    default="gemini-1.5-flash",
    type=click.Choice(MODEL_CHOICES, case_sensitive=False),
    help="LLM model to use as judge. Default: gemini-1.5-flash.",
)
@click.argument('extra_args', nargs=-1, required=False)
def batch(
    input_file: str,
    metrics: tuple,
    output: str,
    model: str,
    extra_args: tuple = (),
) -> None:
    """Evaluate a batch of LLM outputs from a JSON file."""
    metrics_list = list(metrics)
    if extra_args:
        metrics_list.extend(extra_args)

    # Validate metrics
    valid_metrics = []
    for m in metrics_list:
        m_clean = m.lower().strip()
        if m_clean == "all":
            continue
        if m_clean not in METRICS_REGISTRY:
            raise click.BadParameter(
                f"Unknown metric '{m}'. Valid metrics: {list(METRICS_REGISTRY.keys())}"
            )
        valid_metrics.append(m_clean)

    if not valid_metrics or "all" in [m.lower().strip() for m in metrics_list]:
        metrics_list = list(METRICS_REGISTRY.keys())
    else:
        metrics_list = valid_metrics

    # Load input data.
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        console.print("[bold red]Error loading input file:[/bold red] {}".format(str(e)))
        sys.exit(1)

    if not isinstance(data, list):
        console.print("[bold red]Error:[/bold red] Input JSON must be an array of objects.")
        sys.exit(1)

    console.print(
        Panel(
            "[bold cyan]LLMEval Batch[/bold cyan] — Evaluating [bold]{}[/bold] rows with [bold]{}[/bold] using model [bold]{}[/bold]".format(
                len(data),
                ", ".join(m.title() for m in metrics_list),
                model,
            ),
            border_style="cyan",
        )
    )

    results = []
    with console.status("[bold cyan]Evaluating...", spinner="dots") as status:
        for idx, row in enumerate(data):
            status.update(
                "[bold cyan]Evaluating row {}/{}...".format(idx + 1, len(data))
            )
            try:
                result = evaluate(
                    question=row.get("question", ""),
                    context=row.get("context", ""),
                    answer=row.get("answer", ""),
                    metrics=metrics_list,
                    model=model,
                )
                result["input"] = {
                    "question": row.get("question", "")[:80],
                    "context": row.get("context", "")[:80],
                    "answer": row.get("answer", "")[:80],
                }
                results.append(result)
            except Exception as e:
                console.print(
                    "[yellow]Warning:[/yellow] Row {} failed: {}".format(idx + 1, str(e))
                )
                results.append({
                    "scores": {},
                    "overall": 0.0,
                    "input": {"question": row.get("question", "")[:80], "context": "", "answer": ""},
                    "error": str(e),
                })

    # Print summary table.
    table = Table(
        title="Batch Results ({} rows)".format(len(results)),
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Row", justify="center", min_width=5)
    table.add_column("Question", max_width=40, no_wrap=True)
    for m in metrics_list:
        table.add_column(m.title(), justify="center", min_width=10)
    table.add_column("Overall", justify="center", min_width=10, style="bold")

    for idx, r in enumerate(results):
        row_data = [str(idx + 1)]
        q = r.get("input", {}).get("question", "")[:35]
        row_data.append(q + ("..." if len(q) >= 35 else ""))
        for m in metrics_list:
            score = r.get("scores", {}).get(m, {}).get("score", 0.0)
            style = _score_style(score)
            row_data.append("[{}]{:.2f}[/{}]".format(style, score, style))
        ov = r.get("overall", 0.0)
        ov_style = _score_style(ov)
        row_data.append("[{}]{:.2f}[/{}]".format(ov_style, ov, ov_style))
        table.add_row(*row_data)

    console.print()
    console.print(table)
    console.print()

    # Generate HTML report.
    try:
        generate_batch_report(results=results, output_path=output)
        console.print(
            "[dim]HTML report saved to:[/dim] [bold cyan]{}[/bold cyan]".format(output)
        )
    except Exception as e:
        console.print("[yellow]Warning:[/yellow] Could not save report: {}".format(str(e)))


if __name__ == "__main__":
    cli()
