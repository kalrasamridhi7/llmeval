"""
HTML Report Generator.

Generates self-contained HTML reports for LLMEval evaluation results.
All CSS and JavaScript are inlined so the HTML file is fully portable
with no external dependencies.
"""

import json
import html
from typing import Any, Dict, List, Optional


def _score_color(score: float) -> str:
    """Return a CSS colour based on the score threshold.

    Args:
        score: Evaluation score between 0.0 and 1.0.

    Returns:
        Hex colour string: green for >= 0.8, yellow for 0.5-0.79, red for < 0.5.
    """
    if score >= 0.8:
        return "#00e676"
    elif score >= 0.5:
        return "#ffab00"
    else:
        return "#ff5252"


def _score_label(score: float) -> str:
    """Return a human-readable label for the score.

    Args:
        score: Evaluation score between 0.0 and 1.0.

    Returns:
        'Excellent', 'Good', 'Fair', or 'Poor'.
    """
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.7:
        return "Good"
    elif score >= 0.5:
        return "Fair"
    else:
        return "Poor"


def _pass_fail(score: float) -> str:
    """Return pass/fail status based on score.

    Args:
        score: Overall evaluation score.

    Returns:
        'PASS' if score >= 0.7, otherwise 'FAIL'.
    """
    return "PASS" if score >= 0.7 else "FAIL"


def _report_css() -> str:
    """Return the complete CSS for the HTML report.

    Returns:
        CSS string with dark theme, score bars, and responsive layout.
    """
    return """
    :root {
        --bg-primary: #0f0f13;
        --bg-secondary: #1a1a24;
        --bg-card: #1e1e2e;
        --border-color: rgba(255,255,255,0.08);
        --text-primary: #e8e8ed;
        --text-secondary: #8b8b9e;
        --text-muted: #555568;
        --accent: #6c63ff;
        --score-green: #00e676;
        --score-yellow: #ffab00;
        --score-red: #ff5252;
        --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: var(--font-family);
        background: var(--bg-primary);
        color: var(--text-primary);
        line-height: 1.6;
        padding: 2rem;
        max-width: 960px;
        margin: 0 auto;
    }
    h1 {
        font-size: 1.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6c63ff, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
    }
    h2 { font-size: 1.25rem; font-weight: 600; margin: 1.5rem 0 0.75rem; }
    h3 { font-size: 1rem; font-weight: 600; margin: 1rem 0 0.5rem; }
    .subtitle { color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 2rem; }
    .summary-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    .overall-score {
        font-size: 2.5rem;
        font-weight: 800;
        min-width: 80px;
        text-align: center;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .badge-pass { background: rgba(0,230,118,0.15); color: #00e676; }
    .badge-fail { background: rgba(255,82,82,0.15); color: #ff5252; }
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
    }
    .metric-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        cursor: pointer;
    }
    .metric-name { font-weight: 600; font-size: 0.95rem; }
    .metric-score {
        font-weight: 700;
        font-size: 1.1rem;
        font-variant-numeric: tabular-nums;
    }
    .score-bar-container {
        background: rgba(255,255,255,0.05);
        border-radius: 6px;
        height: 8px;
        margin: 0.75rem 0;
        overflow: hidden;
    }
    .score-bar {
        height: 100%;
        border-radius: 6px;
        transition: width 0.6s ease-out;
    }
    .explanation {
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-top: 0.75rem;
        padding-top: 0.75rem;
        border-top: 1px solid var(--border-color);
        display: none;
    }
    .explanation.open { display: block; }
    .expand-icon {
        color: var(--text-muted);
        font-size: 0.8rem;
        transition: transform 0.2s;
    }
    .expand-icon.open { transform: rotate(180deg); }
    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        margin: 1rem 0;
    }
    th {
        text-align: left;
        padding: 0.75rem;
        border-bottom: 2px solid var(--border-color);
        color: var(--text-secondary);
        font-weight: 600;
        cursor: pointer;
        user-select: none;
    }
    th:hover { color: var(--text-primary); }
    td {
        padding: 0.75rem;
        border-bottom: 1px solid var(--border-color);
    }
    .worst-section {
        background: rgba(255,82,82,0.05);
        border: 1px solid rgba(255,82,82,0.15);
        border-radius: 12px;
        padding: 1.25rem;
        margin-top: 1.5rem;
    }
    .worst-section h3 { color: var(--score-red); }
    .btn {
        display: inline-block;
        padding: 0.5rem 1.25rem;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        background: var(--bg-card);
        color: var(--text-primary);
        font-size: 0.85rem;
        font-weight: 500;
        cursor: pointer;
        margin: 0.5rem 0.25rem 0.5rem 0;
        transition: background 0.2s, border-color 0.2s;
    }
    .btn:hover { background: rgba(108,99,255,0.15); border-color: var(--accent); }
    .question-preview { max-width: 250px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    @media (max-width: 640px) {
        body { padding: 1rem; }
        .summary-card { flex-direction: column; text-align: center; }
        table { font-size: 0.75rem; }
    }
    """


def _report_js(json_data: str) -> str:
    """Return the JavaScript for the HTML report.

    Args:
        json_data: JSON-serialised evaluation results for the export button.

    Returns:
        JavaScript string for accordion toggling and JSON export.
    """
    return """
    var reportData = {json_data};

    document.querySelectorAll('.metric-header').forEach(function(header) {{
        header.addEventListener('click', function() {{
            var card = this.closest('.metric-card');
            var explanation = card.querySelector('.explanation');
            var icon = card.querySelector('.expand-icon');
            explanation.classList.toggle('open');
            icon.classList.toggle('open');
        }});
    }});

    document.querySelectorAll('.sort-header').forEach(function(th) {{
        th.addEventListener('click', function() {{
            var table = this.closest('table');
            var tbody = table.querySelector('tbody');
            var rows = Array.from(tbody.querySelectorAll('tr'));
            var col = parseInt(this.dataset.col);
            var asc = this.dataset.asc !== 'true';
            this.dataset.asc = asc;
            rows.sort(function(a, b) {{
                var av = parseFloat(a.cells[col].textContent) || 0;
                var bv = parseFloat(b.cells[col].textContent) || 0;
                return asc ? av - bv : bv - av;
            }});
            rows.forEach(function(row) {{ tbody.appendChild(row); }});
        }});
    }});

    function exportJSON() {{
        var blob = new Blob([JSON.stringify(reportData, null, 2)], {{type: 'application/json'}});
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = 'llmeval_results.json';
        a.click();
        URL.revokeObjectURL(url);
    }}
    """.format(json_data=json_data)


def _render_metric_card(metric_name: str, result: Dict[str, Any]) -> str:
    """Render a single metric card as HTML.

    Args:
        metric_name: Display name for the metric.
        result: Evaluation result dict with score, explanation, details.

    Returns:
        HTML string for the metric card.
    """
    score = result.get("score", 0.0)
    explanation = html.escape(result.get("explanation", "No explanation available."))
    color = _score_color(score)
    width_pct = int(score * 100)

    details_html = ""
    details = result.get("details", {})

    # Render unsupported claims if present.
    unsupported = details.get("unsupported_claims", [])
    if unsupported:
        items = "".join(
            "<li>{}</li>".format(html.escape(c)) for c in unsupported
        )
        details_html += '<div style="margin-top:0.5rem"><strong>Unsupported claims:</strong><ul style="margin:0.25rem 0 0 1.25rem">{}</ul></div>'.format(items)

    # Render hallucinated spans if present.
    hallucinated = details.get("hallucinated_spans", [])
    if hallucinated:
        items = "".join(
            "<li>{}</li>".format(html.escape(s)) for s in hallucinated
        )
        details_html += '<div style="margin-top:0.5rem"><strong>Hallucinated spans:</strong><ul style="margin:0.25rem 0 0 1.25rem">{}</ul></div>'.format(items)

    return """
    <div class="metric-card">
        <div class="metric-header">
            <span class="metric-name">{name}</span>
            <div style="display:flex;align-items:center;gap:0.75rem">
                <span class="metric-score" style="color:{color}">{score:.2f}</span>
                <span class="expand-icon">▼</span>
            </div>
        </div>
        <div class="score-bar-container">
            <div class="score-bar" style="width:{width}%;background:{color}"></div>
        </div>
        <div class="explanation">
            <p>{explanation}</p>
            {details}
        </div>
    </div>
    """.format(
        name=html.escape(metric_name.title()),
        color=color,
        score=score,
        width=width_pct,
        explanation=explanation,
        details=details_html,
    )


def generate_single_report(
    results: Dict[str, Any],
    output_path: str,
    question: str = "",
    context: str = "",
    answer: str = "",
) -> str:
    """Generate a self-contained HTML report for a single evaluation.

    Args:
        results: Dictionary from llmeval.evaluate() with 'scores' and 'overall'.
        output_path: File path to write the HTML report to.
        question: The original question (for display in report).
        context: The original context (for display in report).
        answer: The original answer (for display in report).

    Returns:
        The output_path string.
    """
    overall = results.get("overall", 0.0)
    scores = results.get("scores", {})
    pf = _pass_fail(overall)
    badge_class = "badge-pass" if pf == "PASS" else "badge-fail"

    metrics_html = ""
    for metric_name, metric_result in scores.items():
        metrics_html += _render_metric_card(metric_name, metric_result)

    json_data = json.dumps(results, indent=2, default=str)

    report_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLMEval Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>{css}</style>
</head>
<body>
    <h1>LLMEval Report</h1>
    <p class="subtitle">LLM Output Quality Evaluation</p>

    <div class="summary-card">
        <div class="overall-score" style="color:{overall_color}">{overall:.2f}</div>
        <div>
            <span class="badge {badge_class}">{pf}</span>
            <p style="color:var(--text-secondary);font-size:0.85rem;margin-top:0.5rem">
                Overall score across {n_metrics} metric{plural}
            </p>
        </div>
    </div>

    <h2>Metric Scores</h2>
    {metrics}

    <div style="margin-top:1.5rem">
        <button class="btn" onclick="exportJSON()">📥 Export JSON</button>
    </div>

    <script>{js}</script>
</body>
</html>""".format(
        css=_report_css(),
        overall=overall,
        overall_color=_score_color(overall),
        badge_class=badge_class,
        pf=pf,
        n_metrics=len(scores),
        plural="s" if len(scores) != 1 else "",
        metrics=metrics_html,
        js=_report_js(json_data),
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_html)

    return output_path


def generate_batch_report(
    results: List[Dict[str, Any]],
    output_path: str,
) -> str:
    """Generate a self-contained HTML report for a batch evaluation.

    Args:
        results: List of result dictionaries from llmeval.evaluate_batch().
        output_path: File path to write the HTML report to.

    Returns:
        The output_path string.
    """
    if not results:
        # Write an empty report.
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("<html><body><h1>No results</h1></body></html>")
        return output_path

    # Compute summary statistics.
    all_metrics = set()  # type: set
    for r in results:
        all_metrics.update(r.get("scores", {}).keys())
    metric_names = sorted(all_metrics)

    summary = {}  # type: Dict[str, Dict[str, float]]
    for m in metric_names:
        values = [
            r["scores"][m]["score"]
            for r in results
            if m in r.get("scores", {}) and isinstance(r["scores"][m].get("score"), (int, float))
        ]
        if values:
            summary[m] = {
                "mean": round(sum(values) / len(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }

    overall_scores = [
        r["overall"] for r in results if isinstance(r.get("overall"), (int, float))
    ]
    overall_mean = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    pf = _pass_fail(overall_mean)
    badge_class = "badge-pass" if pf == "PASS" else "badge-fail"

    # Build table header.
    th_cells = '<th class="sort-header" data-col="0">Row</th>'
    th_cells += '<th>Question</th>'
    for i, m in enumerate(metric_names):
        th_cells += '<th class="sort-header" data-col="{}">{}</th>'.format(
            i + 2, html.escape(m.title())
        )
    th_cells += '<th class="sort-header" data-col="{}">Overall</th>'.format(
        len(metric_names) + 2
    )

    # Build table rows.
    rows_html = ""
    for idx, r in enumerate(results):
        q_preview = html.escape(
            r.get("input", {}).get("question", "Row {}".format(idx + 1))[:80]
        )
        row_cells = "<td>{}</td>".format(idx + 1)
        row_cells += '<td class="question-preview" title="{}">{}</td>'.format(
            q_preview, q_preview
        )
        for m in metric_names:
            score = r.get("scores", {}).get(m, {}).get("score", 0.0)
            color = _score_color(score)
            row_cells += '<td style="color:{};font-weight:600">{:.2f}</td>'.format(
                color, score
            )
        ov = r.get("overall", 0.0)
        row_cells += '<td style="color:{};font-weight:700">{:.2f}</td>'.format(
            _score_color(ov), ov
        )
        rows_html += "<tr>{}</tr>".format(row_cells)

    # Summary stats row.
    summary_html = '<tr style="border-top:2px solid var(--border-color);font-weight:600">'
    summary_html += "<td></td><td>Mean / Min / Max</td>"
    for m in metric_names:
        s = summary.get(m, {"mean": 0, "min": 0, "max": 0})
        summary_html += "<td>{:.2f} / {:.2f} / {:.2f}</td>".format(
            s["mean"], s["min"], s["max"]
        )
    summary_html += "<td>{:.2f}</td>".format(overall_mean)
    summary_html += "</tr>"

    # Worst performing examples (bottom 3).
    sorted_by_overall = sorted(results, key=lambda x: x.get("overall", 0.0))
    worst_3 = sorted_by_overall[:3]
    worst_html = ""
    for idx, r in enumerate(worst_3):
        original_idx = results.index(r) + 1
        q = html.escape(r.get("input", {}).get("question", "")[:120])
        ov = r.get("overall", 0.0)
        worst_html += """
        <div style="margin-bottom:0.75rem;padding:0.75rem;background:var(--bg-card);border-radius:8px;border:1px solid var(--border-color)">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-weight:600">Row {row} — <span style="font-weight:400;color:var(--text-secondary)">{question}</span></span>
                <span style="color:{color};font-weight:700">{score:.2f}</span>
            </div>
        </div>
        """.format(row=original_idx, question=q, color=_score_color(ov), score=ov)

    json_data = json.dumps(
        {"results": results, "summary": summary},
        indent=2,
        default=str,
    )

    report_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLMEval Batch Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>{css}</style>
</head>
<body>
    <h1>LLMEval Batch Report</h1>
    <p class="subtitle">{n_rows} examples evaluated across {n_metrics} metrics</p>

    <div class="summary-card">
        <div class="overall-score" style="color:{overall_color}">{overall:.2f}</div>
        <div>
            <span class="badge {badge_class}">{pf}</span>
            <p style="color:var(--text-secondary);font-size:0.85rem;margin-top:0.5rem">
                Mean overall score across {n_rows} examples
            </p>
        </div>
    </div>

    <h2>Results Table</h2>
    <div style="overflow-x:auto">
        <table>
            <thead><tr>{th}</tr></thead>
            <tbody>{rows}{summary_row}</tbody>
        </table>
    </div>

    <div class="worst-section">
        <h3>⚠ Worst Performing Examples</h3>
        <div style="margin-top:0.75rem">{worst}</div>
    </div>

    <div style="margin-top:1.5rem">
        <button class="btn" onclick="exportJSON()">📥 Export JSON</button>
    </div>

    <script>{js}</script>
</body>
</html>""".format(
        css=_report_css(),
        n_rows=len(results),
        n_metrics=len(metric_names),
        overall=overall_mean,
        overall_color=_score_color(overall_mean),
        badge_class=badge_class,
        pf=pf,
        th=th_cells,
        rows=rows_html,
        summary_row=summary_html,
        worst=worst_html,
        js=_report_js(json_data),
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_html)

    return output_path
