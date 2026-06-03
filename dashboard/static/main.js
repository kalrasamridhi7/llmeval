/**
 * LLMEval Dashboard — Main Application Logic
 *
 * Vanilla JavaScript handling:
 * - Tab switching
 * - Single evaluation form submission
 * - Score ring rendering and animation
 * - Explanation accordion
 * - Batch file upload (JSON/CSV)
 * - Batch evaluation with progress
 * - Table sorting
 * - JSON/Report download
 * - Toast notifications
 */

(function () {
    'use strict';

    // =========================================================================
    // Constants & Configuration
    // =========================================================================

    /** Base URL for API endpoints. */
    var API_BASE = '';

    /** SVG circle circumference for score rings. */
    var RING_CIRCUMFERENCE_LARGE = 2 * Math.PI * 52;  // r=52 for large ring
    var RING_CIRCUMFERENCE_SMALL = 2 * Math.PI * 52;  // r=52 for small ring

    /** Parsed batch data stored after file upload. */
    var batchData = null;

    /** Last evaluation results for copy/download. */
    var lastSingleResult = null;
    var lastBatchResult = null;

    // =========================================================================
    // Utility Functions
    // =========================================================================

    /**
     * Get a colour based on score thresholds.
     * @param {number} score - Score between 0 and 1.
     * @returns {string} CSS colour string.
     */
    function getScoreColor(score) {
        if (score >= 0.8) return '#00e676';
        if (score >= 0.5) return '#ffab00';
        return '#ff5252';
    }

    /**
     * Get a background colour based on score thresholds.
     * @param {number} score - Score between 0 and 1.
     * @returns {string} CSS rgba colour string.
     */
    function getScoreBg(score) {
        if (score >= 0.8) return 'rgba(0, 230, 118, 0.1)';
        if (score >= 0.5) return 'rgba(255, 171, 0, 0.1)';
        return 'rgba(255, 82, 82, 0.1)';
    }

    /**
     * Display a toast notification.
     * @param {string} message - The message to display.
     * @param {string} type - 'success', 'error', or 'info'.
     */
    function showToast(message, type) {
        type = type || 'info';
        var container = document.getElementById('toast-container');
        var toast = document.createElement('div');
        toast.className = 'toast toast-' + type;
        toast.textContent = message;
        container.appendChild(toast);
        setTimeout(function () {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 3000);
    }

    /**
     * Set loading state on a button.
     * @param {HTMLElement} btn - The button element.
     * @param {boolean} loading - Whether to show loading state.
     */
    function setButtonLoading(btn, loading) {
        var text = btn.querySelector('.btn-text');
        var spinner = btn.querySelector('.btn-spinner');
        if (loading) {
            btn.disabled = true;
            if (text) text.hidden = true;
            if (spinner) spinner.hidden = false;
        } else {
            btn.disabled = false;
            if (text) text.hidden = false;
            if (spinner) spinner.hidden = true;
        }
    }

    /**
     * Capitalise the first letter of a string.
     * @param {string} str - Input string.
     * @returns {string} Capitalised string.
     */
    function capitalise(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    // =========================================================================
    // Tab Switching
    // =========================================================================

    /**
     * Initialise tab switching behaviour.
     */
    function initTabs() {
        var tabs = document.querySelectorAll('.tab');
        var indicator = document.getElementById('tab-indicator');

        tabs.forEach(function (tab) {
            tab.addEventListener('click', function () {
                var targetTab = this.getAttribute('data-tab');

                // Update tab active states.
                tabs.forEach(function (t) {
                    t.classList.remove('active');
                    t.setAttribute('aria-selected', 'false');
                });
                this.classList.add('active');
                this.setAttribute('aria-selected', 'true');

                // Move indicator.
                if (targetTab === 'batch') {
                    indicator.style.transform = 'translateX(100%)';
                } else {
                    indicator.style.transform = 'translateX(0)';
                }

                // Show/hide panels.
                document.querySelectorAll('.tab-panel').forEach(function (panel) {
                    panel.classList.remove('active');
                });
                document.getElementById('panel-' + targetTab).classList.add('active');
            });
        });
    }

    // =========================================================================
    // Score Ring Rendering
    // =========================================================================

    /**
     * Animate a score ring SVG element.
     * @param {SVGCircleElement} circle - The SVG circle element.
     * @param {number} score - Score between 0 and 1.
     * @param {number} circumference - The circle's circumference.
     * @param {number} delay - Animation delay in ms.
     */
    function animateRing(circle, score, circumference, delay) {
        var offset = circumference - (score * circumference);
        circle.style.strokeDasharray = circumference;
        circle.style.strokeDashoffset = circumference;
        circle.style.stroke = getScoreColor(score);

        setTimeout(function () {
            circle.style.strokeDashoffset = offset;
        }, delay || 100);
    }

    /**
     * Animate a score value counter.
     * @param {HTMLElement} elem - The element to update.
     * @param {number} target - Target value (0-100).
     * @param {number} duration - Animation duration in ms.
     */
    function animateValue(elem, target, duration) {
        duration = duration || 1200;
        var start = 0;
        var startTime = null;

        function step(timestamp) {
            if (!startTime) startTime = timestamp;
            var progress = Math.min((timestamp - startTime) / duration, 1);
            var eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
            var current = Math.round(start + (target - start) * eased);
            elem.textContent = current + '%';
            if (progress < 1) {
                requestAnimationFrame(step);
            }
        }
        requestAnimationFrame(step);
    }

    /**
     * Render the overall score ring.
     * @param {number} score - Overall score between 0 and 1.
     */
    function renderOverallRing(score) {
        var ring = document.getElementById('overall-ring');
        var value = document.getElementById('overall-value');

        animateRing(ring, score, RING_CIRCUMFERENCE_LARGE, 200);
        value.style.color = getScoreColor(score);
        animateValue(value, Math.round(score * 100));
    }

    /**
     * Render individual metric score rings.
     * @param {Object} scores - Dictionary of metric name to result object.
     */
    function renderMetricRings(scores) {
        var container = document.getElementById('metric-rings');
        container.innerHTML = '';

        var metricNames = Object.keys(scores);
        var delay = 300;

        metricNames.forEach(function (metric) {
            var result = scores[metric];
            var score = result.score || 0;
            var color = getScoreColor(score);

            var card = document.createElement('div');
            card.className = 'metric-ring-card';
            card.setAttribute('data-metric', metric);

            card.innerHTML =
                '<div class="score-ring-wrapper score-ring-small">' +
                '  <svg class="score-ring" viewBox="0 0 120 120">' +
                '    <circle class="ring-bg" cx="60" cy="60" r="52" />' +
                '    <circle class="ring-progress" cx="60" cy="60" r="52" />' +
                '  </svg>' +
                '  <div class="ring-label">' +
                '    <span class="ring-value" style="color:' + color + '">0%</span>' +
                '    <span class="ring-caption">' + capitalise(metric) + '</span>' +
                '  </div>' +
                '</div>';

            container.appendChild(card);

            var ring = card.querySelector('.ring-progress');
            var valueElem = card.querySelector('.ring-value');

            animateRing(ring, score, RING_CIRCUMFERENCE_SMALL, delay);
            animateValue(valueElem, Math.round(score * 100));

            // Click to scroll to explanation.
            card.addEventListener('click', function () {
                var expCard = document.querySelector('.explanation-card[data-metric="' + metric + '"]');
                if (expCard) {
                    expCard.classList.add('open');
                    expCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            });

            delay += 150;
        });
    }

    // =========================================================================
    // Explanations Accordion
    // =========================================================================

    /**
     * Render the explanations accordion for each metric.
     * @param {Object} scores - Dictionary of metric name to result object.
     */
    function renderExplanations(scores) {
        var container = document.getElementById('explanations');
        container.innerHTML = '';

        Object.keys(scores).forEach(function (metric) {
            var result = scores[metric];
            var score = result.score || 0;
            var explanation = result.explanation || 'No explanation available.';
            var details = result.details || {};
            var color = getScoreColor(score);

            var card = document.createElement('div');
            card.className = 'explanation-card';
            card.setAttribute('data-metric', metric);

            var detailsHtml = '';

            // Unsupported claims.
            var unsupported = details.unsupported_claims || [];
            if (unsupported.length > 0) {
                detailsHtml += '<div class="explanation-details"><strong>Unsupported claims:</strong><ul>';
                unsupported.forEach(function (c) {
                    detailsHtml += '<li>' + escapeHtml(c) + '</li>';
                });
                detailsHtml += '</ul></div>';
            }

            // Hallucinated spans.
            var hallucinated = details.hallucinated_spans || [];
            if (hallucinated.length > 0) {
                detailsHtml += '<div class="explanation-details"><strong>Hallucinated spans:</strong><ul>';
                hallucinated.forEach(function (s) {
                    detailsHtml += '<li>' + escapeHtml(s) + '</li>';
                });
                detailsHtml += '</ul></div>';
            }

            card.innerHTML =
                '<div class="explanation-header">' +
                '  <div style="display:flex;align-items:center;gap:0.75rem">' +
                '    <span class="explanation-metric">' + capitalise(metric) + '</span>' +
                '    <span class="explanation-score" style="color:' + color + '">' + score.toFixed(2) + '</span>' +
                '  </div>' +
                '  <span class="explanation-toggle">▼</span>' +
                '</div>' +
                '<div class="explanation-body">' +
                '  <div class="explanation-content">' +
                '    <p>' + escapeHtml(explanation) + '</p>' +
                detailsHtml +
                '  </div>' +
                '</div>';

            // Toggle accordion.
            card.querySelector('.explanation-header').addEventListener('click', function () {
                card.classList.toggle('open');
            });

            container.appendChild(card);
        });
    }

    /**
     * Escape HTML special characters.
     * @param {string} str - Raw string.
     * @returns {string} Escaped string.
     */
    function escapeHtml(str) {
        var div = document.createElement('div');
        div.appendChild(document.createTextNode(str));
        return div.innerHTML;
    }

    // =========================================================================
    // Single Evaluation
    // =========================================================================

    /**
     * Handle single evaluation form submission.
     */
    function initSingleEval() {
        var form = document.getElementById('single-eval-form');
        var btn = document.getElementById('btn-evaluate');
        var copyBtn = document.getElementById('btn-copy-json');

        form.addEventListener('submit', function (e) {
            e.preventDefault();

            var question = document.getElementById('input-question').value.trim();
            var context = document.getElementById('input-context').value.trim();
            var answer = document.getElementById('input-answer').value.trim();
            var model = document.getElementById('model-select').value;

            if (!question || !answer) {
                showToast('Please fill in the Question and Answer fields.', 'error');
                return;
            }

            // Collect selected metrics.
            var metrics = [];
            document.querySelectorAll('#metrics-checkboxes input:checked').forEach(function (cb) {
                metrics.push(cb.value);
            });

            if (metrics.length === 0) {
                showToast('Please select at least one metric.', 'error');
                return;
            }

            setButtonLoading(btn, true);

            fetch(API_BASE + '/evaluate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: question,
                    context: context,
                    answer: answer,
                    metrics: metrics,
                    model: model
                })
            })
            .then(function (res) {
                if (!res.ok) {
                    return res.json().then(function (data) {
                        throw new Error(data.detail || 'Evaluation failed');
                    });
                }
                return res.json();
            })
            .then(function (data) {
                lastSingleResult = data;
                displaySingleResults(data);
                showToast('Evaluation complete!', 'success');
            })
            .catch(function (err) {
                showToast('Error: ' + err.message, 'error');
            })
            .finally(function () {
                setButtonLoading(btn, false);
            });
        });

        // Copy JSON button.
        copyBtn.addEventListener('click', function () {
            if (lastSingleResult) {
                navigator.clipboard.writeText(JSON.stringify(lastSingleResult, null, 2))
                    .then(function () { showToast('JSON copied to clipboard!', 'success'); })
                    .catch(function () { showToast('Failed to copy to clipboard.', 'error'); });
            }
        });
    }

    /**
     * Display single evaluation results.
     * @param {Object} data - API response with scores and overall.
     */
    function displaySingleResults(data) {
        var section = document.getElementById('results-section');
        section.hidden = false;

        renderOverallRing(data.overall || 0);
        renderMetricRings(data.scores || {});
        renderExplanations(data.scores || {});

        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // =========================================================================
    // Batch Evaluation
    // =========================================================================

    /**
     * Parse a CSV string into an array of objects.
     * @param {string} csv - Raw CSV string.
     * @returns {Array<Object>} Array of row objects.
     */
    function parseCSV(csv) {
        var lines = csv.trim().split('\n');
        if (lines.length < 2) return [];

        var headers = lines[0].split(',').map(function (h) {
            return h.trim().replace(/^["']|["']$/g, '').toLowerCase();
        });

        var rows = [];
        for (var i = 1; i < lines.length; i++) {
            var values = [];
            var current = '';
            var inQuotes = false;

            for (var j = 0; j < lines[i].length; j++) {
                var ch = lines[i][j];
                if (ch === '"') {
                    inQuotes = !inQuotes;
                } else if (ch === ',' && !inQuotes) {
                    values.push(current.trim());
                    current = '';
                } else {
                    current += ch;
                }
            }
            values.push(current.trim());

            var row = {};
            headers.forEach(function (h, idx) {
                row[h] = values[idx] || '';
            });
            rows.push(row);
        }
        return rows;
    }

    /**
     * Initialise batch evaluation functionality.
     */
    function initBatchEval() {
        var uploadZone = document.getElementById('upload-zone');
        var fileInput = document.getElementById('file-input');
        var fileInfo = document.getElementById('file-info');
        var fileName = document.getElementById('file-name');
        var fileRows = document.getElementById('file-rows');
        var clearBtn = document.getElementById('btn-clear-file');
        var batchOptions = document.getElementById('batch-options');
        var batchBtn = document.getElementById('btn-batch-evaluate');
        var downloadJsonBtn = document.getElementById('btn-download-json');
        var downloadReportBtn = document.getElementById('btn-download-report');

        // Upload zone click.
        uploadZone.addEventListener('click', function () {
            fileInput.click();
        });

        // Drag & drop.
        uploadZone.addEventListener('dragover', function (e) {
            e.preventDefault();
            uploadZone.classList.add('drag-over');
        });

        uploadZone.addEventListener('dragleave', function () {
            uploadZone.classList.remove('drag-over');
        });

        uploadZone.addEventListener('drop', function (e) {
            e.preventDefault();
            uploadZone.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        // File input change.
        fileInput.addEventListener('change', function () {
            if (this.files.length > 0) {
                handleFile(this.files[0]);
            }
        });

        // Clear file.
        clearBtn.addEventListener('click', function () {
            batchData = null;
            fileInput.value = '';
            fileInfo.hidden = true;
            batchOptions.hidden = true;
            uploadZone.style.display = '';
            document.getElementById('batch-results').hidden = true;
            document.getElementById('progress-container').hidden = true;
        });

        /**
         * Handle an uploaded file.
         * @param {File} file - The uploaded file.
         */
        function handleFile(file) {
            var reader = new FileReader();
            reader.onload = function (e) {
                var content = e.target.result;
                try {
                    if (file.name.endsWith('.csv')) {
                        batchData = parseCSV(content);
                    } else {
                        batchData = JSON.parse(content);
                    }

                    if (!Array.isArray(batchData) || batchData.length === 0) {
                        showToast('File must contain an array of objects.', 'error');
                        batchData = null;
                        return;
                    }

                    // Validate structure.
                    var first = batchData[0];
                    if (!first.question && !first.answer) {
                        showToast('Each row must have "question" and "answer" fields.', 'error');
                        batchData = null;
                        return;
                    }

                    fileName.textContent = file.name;
                    fileRows.textContent = batchData.length + ' rows';
                    fileInfo.hidden = false;
                    batchOptions.hidden = false;
                    uploadZone.style.display = 'none';
                    showToast('File loaded: ' + batchData.length + ' rows', 'success');

                } catch (err) {
                    showToast('Failed to parse file: ' + err.message, 'error');
                    batchData = null;
                }
            };
            reader.readAsText(file);
        }

        // Batch evaluate.
        batchBtn.addEventListener('click', function () {
            if (!batchData || batchData.length === 0) {
                showToast('No data loaded. Please upload a file.', 'error');
                return;
            }

            var metrics = [];
            document.querySelectorAll('#batch-metrics-checkboxes input:checked').forEach(function (cb) {
                metrics.push(cb.value);
            });

            if (metrics.length === 0) {
                showToast('Please select at least one metric.', 'error');
                return;
            }

            var model = document.getElementById('batch-model-select').value;
            setButtonLoading(batchBtn, true);

            // Show progress.
            var progressContainer = document.getElementById('progress-container');
            var progressFill = document.getElementById('progress-fill');
            var progressText = document.getElementById('progress-text');
            progressContainer.hidden = false;
            progressFill.style.width = '10%';
            progressText.textContent = 'Evaluating ' + batchData.length + ' rows...';

            // Simulate progress while waiting.
            var progressInterval = setInterval(function () {
                var current = parseFloat(progressFill.style.width) || 10;
                if (current < 85) {
                    progressFill.style.width = (current + Math.random() * 5) + '%';
                }
            }, 1000);

            fetch(API_BASE + '/evaluate/batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    rows: batchData,
                    metrics: metrics,
                    model: model
                })
            })
            .then(function (res) {
                if (!res.ok) {
                    return res.json().then(function (data) {
                        throw new Error(data.detail || 'Batch evaluation failed');
                    });
                }
                return res.json();
            })
            .then(function (data) {
                clearInterval(progressInterval);
                progressFill.style.width = '100%';
                progressText.textContent = 'Complete!';

                lastBatchResult = data;
                displayBatchResults(data, metrics);
                showToast('Batch evaluation complete!', 'success');

                setTimeout(function () {
                    progressContainer.hidden = true;
                }, 1000);
            })
            .catch(function (err) {
                clearInterval(progressInterval);
                progressContainer.hidden = true;
                showToast('Error: ' + err.message, 'error');
            })
            .finally(function () {
                setButtonLoading(batchBtn, false);
            });
        });

        // Download JSON.
        downloadJsonBtn.addEventListener('click', function () {
            if (lastBatchResult) {
                var blob = new Blob([JSON.stringify(lastBatchResult, null, 2)], { type: 'application/json' });
                var url = URL.createObjectURL(blob);
                var a = document.createElement('a');
                a.href = url;
                a.download = 'llmeval_batch_results.json';
                a.click();
                URL.revokeObjectURL(url);
                showToast('JSON downloaded!', 'success');
            }
        });

        // Download report.
        downloadReportBtn.addEventListener('click', function () {
            if (!batchData || !lastBatchResult) {
                showToast('No results to export.', 'error');
                return;
            }

            var metrics = [];
            document.querySelectorAll('#batch-metrics-checkboxes input:checked').forEach(function (cb) {
                metrics.push(cb.value);
            });
            var model = document.getElementById('batch-model-select').value;

            setButtonLoading(downloadReportBtn, true);

            fetch(API_BASE + '/evaluate/batch/report', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    rows: batchData,
                    metrics: metrics,
                    model: model
                })
            })
            .then(function (res) {
                if (!res.ok) throw new Error('Failed to generate report');
                return res.blob();
            })
            .then(function (blob) {
                var url = URL.createObjectURL(blob);
                var a = document.createElement('a');
                a.href = url;
                a.download = 'llmeval_batch_report.html';
                a.click();
                URL.revokeObjectURL(url);
                showToast('Report downloaded!', 'success');
            })
            .catch(function (err) {
                showToast('Error: ' + err.message, 'error');
            })
            .finally(function () {
                setButtonLoading(downloadReportBtn, false);
            });
        });
    }

    /**
     * Display batch evaluation results.
     * @param {Object} data - API response with results and summary.
     * @param {Array<string>} metrics - List of evaluated metrics.
     */
    function displayBatchResults(data, metrics) {
        var resultsSection = document.getElementById('batch-results');
        resultsSection.hidden = false;

        // Render summary stats.
        var statsContainer = document.getElementById('summary-stats');
        statsContainer.innerHTML = '';

        var summary = data.summary || {};
        metrics.forEach(function (metric) {
            var stat = summary[metric] || { mean: 0, min: 0, max: 0 };
            var color = getScoreColor(stat.mean);
            var card = document.createElement('div');
            card.className = 'stat-card';
            card.innerHTML =
                '<div class="stat-label">' + capitalise(metric) + '</div>' +
                '<div class="stat-value" style="color:' + color + '">' + (stat.mean || 0).toFixed(2) + '</div>' +
                '<div class="stat-range">Min: ' + (stat.min || 0).toFixed(2) + ' · Max: ' + (stat.max || 0).toFixed(2) + '</div>';
            statsContainer.appendChild(card);
        });

        // Render table.
        var thead = document.getElementById('results-table-head');
        var tbody = document.getElementById('results-table-body');
        thead.innerHTML = '';
        tbody.innerHTML = '';

        // Header row.
        var headerRow = document.createElement('tr');
        headerRow.innerHTML = '<th data-col="0"># <span class="sort-arrow">↕</span></th>' +
            '<th>Question</th>';
        metrics.forEach(function (metric, idx) {
            headerRow.innerHTML += '<th data-col="' + (idx + 2) + '">' + capitalise(metric) + ' <span class="sort-arrow">↕</span></th>';
        });
        headerRow.innerHTML += '<th data-col="' + (metrics.length + 2) + '">Overall <span class="sort-arrow">↕</span></th>';
        thead.appendChild(headerRow);

        // Data rows.
        var results = data.results || [];
        results.forEach(function (result, idx) {
            var row = document.createElement('tr');
            var q = (result.input && result.input.question) || ('Row ' + (idx + 1));
            var qShort = q.length > 60 ? q.substring(0, 60) + '...' : q;

            var cells = '<td>' + (idx + 1) + '</td>';
            cells += '<td class="question-cell" title="' + escapeHtml(q) + '">' + escapeHtml(qShort) + '</td>';

            metrics.forEach(function (metric) {
                var score = (result.scores && result.scores[metric]) ? result.scores[metric].score : 0;
                var color = getScoreColor(score);
                cells += '<td class="score-cell" style="color:' + color + '">' + score.toFixed(2) + '</td>';
            });

            var overall = result.overall || 0;
            var overallColor = getScoreColor(overall);
            cells += '<td class="score-cell" style="color:' + overallColor + ';font-weight:700">' + overall.toFixed(2) + '</td>';

            row.innerHTML = cells;
            tbody.appendChild(row);
        });

        // Table sorting.
        thead.querySelectorAll('th[data-col]').forEach(function (th) {
            th.addEventListener('click', function () {
                var col = parseInt(this.getAttribute('data-col'));
                var ascending = this.getAttribute('data-asc') !== 'true';
                this.setAttribute('data-asc', ascending);

                // Reset other sorted headers.
                thead.querySelectorAll('th').forEach(function (h) { h.classList.remove('sorted'); });
                this.classList.add('sorted');

                var rows = Array.from(tbody.querySelectorAll('tr'));
                rows.sort(function (a, b) {
                    var aVal = parseFloat(a.cells[col].textContent) || 0;
                    var bVal = parseFloat(b.cells[col].textContent) || 0;
                    return ascending ? aVal - bVal : bVal - aVal;
                });
                rows.forEach(function (row) { tbody.appendChild(row); });
            });
        });

        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // =========================================================================
    // Initialisation
    // =========================================================================

    /**
     * Initialise all dashboard functionality on DOM ready.
     */
    function init() {
        initTabs();
        initSingleEval();
        initBatchEval();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
