/*
 * Results page: everything shown comes from GET /api/training-jobs/{id}/results.
 * Cross-validation scores (training data) and held-out test scores are kept visibly separate.
 */
(function () {
  "use strict";
  const { api, el, render, errorBox, alertBox, empty, chip, fmt, loading, syntheticFlag, disclaimer, param } = window.DRP;

  const root = document.getElementById("results-root");
  const picker = document.getElementById("job-picker");
  const pickerWrap = document.getElementById("job-picker-wrap");
  const charts = [];

  const COLORS = { logistic_regression: "#1d5faf", random_forest: "#0f7f79", decision_tree: "#6b4fa0" };
  const TEST_METRICS = ["roc_auc", "f1", "recall", "precision", "accuracy"];
  const hasChart = () => typeof window.Chart === "function";

  function destroyCharts() {
    while (charts.length) charts.pop().destroy();
  }

  function refBar(value, kind) {
    if (typeof value !== "number" || !isFinite(value)) return el("span", { class: "ref-na" }, "Not defined");
    const fill = el("span", { class: "ref-fill" });
    fill.style.width = Math.max(0, Math.min(1, value)) * 100 + "%";
    return el("div", { class: "ref" + (kind === "cv" ? " ref-cv" : "") },
      el("span", { class: "ref-value" }, fmt.metric(value)),
      el("span", { class: "ref-track", role: "img", "aria-label": fmt.metric(value) + " on a scale from 0 to 1" }, fill));
  }

  /* -------------------------------------------------------------- sections */
  function summarySection(r) {
    const job = r.job;
    return el("section", { class: "panel" },
      el("div", { class: "panel-head" },
        el("h2", null, "Job #" + job.id + ": " + job.filename),
        el("div", { class: "btn-row" },
          el("a", { class: "btn btn-secondary", href: api.reportUrl(job.id, "csv"), download: "" }, "Download CSV report"),
          el("a", { class: "btn btn-secondary", href: api.reportUrl(job.id, "json"), download: "" }, "Download JSON report"))),
      job.datasetOrigin === "synthetic" ? syntheticFlag() : null,
      job.datasetOrigin === "unknown" ? alertBox("warn", null, "The origin of this dataset was not specified. Treat results as a software demonstration.") : null,
      el("dl", { class: "facts-row" },
        el("div", null, el("dt", null, "Target"), el("dd", null, r.target_column)),
        el("div", null, el("dt", null, "Positive class"), el("dd", null, r.class_mapping.positive_class)),
        el("div", null, el("dt", null, "Rows used"), el("dd", null, fmt.int(r.dataset_summary.n_rows_used))),
        el("div", null, el("dt", null, "Split"), el("dd", null, fmt.split(r.split.train_fraction))),
        el("div", null, el("dt", null, "Train / test rows"), el("dd", null, fmt.int(r.split.train_rows) + " / " + fmt.int(r.split.test_rows))),
        el("div", null, el("dt", null, "Selection metric"), el("dd", null, fmt.metricName(r.scoring_metric))),
        el("div", null, el("dt", null, "Tuning"), el("dd", null, fmt.tuning(r.tuning_level) + ", " + r.cross_validation.n_splits + "-fold CV")),
        el("div", null, el("dt", null, "Seed"), el("dd", null, String(r.random_seed))),
        el("div", null, el("dt", null, "Trained"), el("dd", null, fmt.date(job.finishedAt)))),
      r.warnings.length ? alertBox("warn", "Notes from training", null, r.warnings) : null);
  }

  function selectedSection(r) {
    const best = r.models.find((m) => m.selected) || r.models[0];
    return el("section", { class: "section" },
      el("div", { class: "selected-callout" },
        el("strong", null, "Chosen by cross-validation: " + best.display_name),
        el("span", null, "Mean CV " + fmt.metricName(r.scoring_metric) + " " + fmt.metric(best.cv_score) + " (± " + fmt.metric(best.cv_std) + ") on the training data."),
        el("span", null, r.selection_reason),
        el("span", null, el("a", { href: "predict.html?model=" + best.id }, "Make a prediction with this model"))));
  }

  function labSection(r) {
    const models = r.models;
    const head = el("tr", null, el("th", { scope: "col" }, "Score"),
      models.map((m) => el("th", { scope: "col", class: m.selected ? "is-selected-col" : null }, m.display_name, m.selected ? " " : null, m.selected ? chip("Selected", "teal") : null)));
    const group = (label) => el("tr", { class: "lab-group" }, el("th", { colspan: models.length + 1, scope: "colgroup" }, label));
    const metricRow = (label, getter, kind) => el("tr", null, el("th", { scope: "row" }, label),
      models.map((m) => el("td", { class: m.selected ? "is-selected-col" : null }, refBar(getter(m), kind))));

    return el("section", { class: "section" },
      el("div", { class: "panel" },
        el("div", { class: "panel-head" }, el("h2", null, "Model comparison")),
        el("p", { class: "muted" }, "Teal bars are cross-validation scores on training data, used to choose the model. Blue bars are scores on the held-out test set, measured once after choosing. The faint mark on each bar is 0.5."),
        el("div", { class: "table-wrap" }, el("table", { class: "lab" },
          el("thead", null, head),
          el("tbody", null,
            group("Cross-validation on training data (" + r.cross_validation.n_splits + " folds)"),
            metricRow("Mean " + fmt.metricName(r.scoring_metric), (m) => m.cv_score, "cv"),
            el("tr", null, el("th", { scope: "row" }, "Spread across folds"), models.map((m) => el("td", { class: m.selected ? "is-selected-col" : null }, "± " + fmt.metric(m.cv_std)))),
            group("Held-out test set (" + fmt.int(r.split.test_rows) + " rows)"),
            TEST_METRICS.map((k) => metricRow(fmt.metricName(k), (m) => m.test_metrics[k], "test")),
            group("Tuning"),
            el("tr", null, el("th", { scope: "row" }, "Best settings"), models.map((m) => el("td", { class: "small" + (m.selected ? " is-selected-col" : "") },
              Object.entries(m.best_params).map(([k, v]) => k + " = " + (v === null ? "None" : v)).join(", ")))),
            el("tr", null, el("th", { scope: "row" }, "Settings tried"), models.map((m) => el("td", { class: "small" + (m.selected ? " is-selected-col" : "") },
              m.n_candidates + " (" + m.search + ")"))),
            el("tr", null, el("th", { scope: "row" }, "Training time"), models.map((m) => el("td", { class: "small" + (m.selected ? " is-selected-col" : "") },
              m.fit_seconds.toFixed(1) + " s"))))))),
      r.imbalance.flagged ? alertBox("warn", null, "Classes are imbalanced, so accuracy alone may be misleading.") : null);
  }

  function chartSection(r) {
    const metricCanvas = el("canvas", { role: "img", "aria-label": "Bar chart of held-out test metrics for each model" });
    const rocCanvas = el("canvas", { role: "img", "aria-label": "ROC curves for each model on the test set" });
    const section = el("section", { class: "section grid grid-2" },
      el("div", { class: "panel" }, el("h3", null, "Test-set scores by model"), el("div", { class: "chart-box" }, metricCanvas),
        el("p", { class: "legend-note" }, "Same numbers as the blue bars in the comparison table.")),
      el("div", { class: "panel" }, el("h3", null, "ROC curve (test set)"), el("div", { class: "chart-box" }, rocCanvas),
        el("p", { class: "legend-note" }, "Closer to the top-left corner is better. The dashed diagonal is chance level.")));

    if (!hasChart()) {
      section.prepend(alertBox("warn", null, "Charts could not load, so only tables are shown."));
      return section;
    }
    requestAnimationFrame(() => {
      charts.push(new window.Chart(metricCanvas, {
        type: "bar",
        data: {
          labels: TEST_METRICS.map(fmt.metricName),
          datasets: r.models.map((m) => ({
            label: m.display_name,
            data: TEST_METRICS.map((k) => m.test_metrics[k]),
            backgroundColor: COLORS[m.algorithm] || "#52627a",
            borderRadius: 4,
          })),
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          scales: { y: { min: 0, max: 1, ticks: { stepSize: 0.2 } } },
          plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => c.dataset.label + ": " + fmt.metric(c.raw) } } },
        },
      }));
      const rocSets = r.models.filter((m) => m.roc_curve).map((m) => ({
        label: m.display_name + " (AUC " + fmt.metric(m.test_metrics.roc_auc) + ")",
        data: m.roc_curve.fpr.map((x, i) => ({ x, y: m.roc_curve.tpr[i] })),
        borderColor: COLORS[m.algorithm] || "#52627a",
        backgroundColor: "transparent",
        borderWidth: 2, pointRadius: 0, stepped: false, showLine: true,
      }));
      rocSets.push({ label: "Chance", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: "#b9c6d4", borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0, showLine: true });
      charts.push(new window.Chart(rocCanvas, {
        type: "scatter",
        data: { datasets: rocSets },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          scales: {
            x: { min: 0, max: 1, title: { display: true, text: "False positive rate" } },
            y: { min: 0, max: 1, title: { display: true, text: "True positive rate (recall)" } },
          },
          plugins: { legend: { position: "bottom" } },
        },
      }));
    });
    return section;
  }

  function confusionSection(r) {
    const neg = r.class_mapping.negative_class;
    const pos = r.class_mapping.positive_class;
    const matrices = r.models.map((m) => {
      const [[tn, fp], [fn, tp]] = m.confusion_matrix.matrix;
      const cell = (n, label, correct) => el("td", { class: "cm-cell " + (correct ? "cm-correct" : "cm-wrong") }, fmt.int(n), el("small", null, label));
      return el("div", { class: "panel" },
        el("h3", null, m.display_name),
        el("div", { class: "table-wrap", style: null }, el("table", { class: "cm" },
          el("caption", { class: "visually-hidden" }, "Confusion matrix for " + m.display_name + ": rows are actual classes, columns are predicted classes"),
          el("thead", null,
            el("tr", null, el("td"), el("th", { scope: "colgroup", colspan: 2, class: "cm-axis" }, "Predicted")),
            el("tr", null, el("td"), el("th", { scope: "col" }, neg), el("th", { scope: "col" }, pos))),
          el("tbody", null,
            el("tr", null, el("th", { scope: "row" }, "Actual " + neg), cell(tn, "true negative", true), cell(fp, "false positive", false)),
            el("tr", null, el("th", { scope: "row" }, "Actual " + pos), cell(fn, "false negative", false), cell(tp, "true positive", true))))));
    });
    return el("section", { class: "section" },
      el("h2", null, "Confusion matrices (test set)"),
      el("p", { class: "section-intro" }, "How many test rows each model got right and wrong. A false negative is a real “" + pos + "” case the model missed."),
      el("div", { class: "grid grid-3" }, matrices));
  }

  function importanceSection(r) {
    const usable = r.models.filter((m) => m.feature_importance && m.feature_importance.items.length);
    if (!usable.length) return null;
    const canvasBox = el("div", { class: "chart-box chart-box-tall" });
    const caveat = el("p", { class: "legend-note" });
    const tableBox = el("div");
    const tabs = el("div", { class: "tabs", role: "tablist", "aria-label": "Choose model" });
    let active = null;

    function show(model) {
      active = model;
      [...tabs.children].forEach((t) => t.setAttribute("aria-selected", String(t.dataset.algo === model.algorithm)));
      const fi = model.feature_importance;
      const items = fi.items.slice(0, 15);
      caveat.textContent = fi.caveat;
      render(tableBox, el("details", { class: "more" }, el("summary", null, "Show as a table"),
        el("div", { class: "table-wrap" }, el("table", null,
          el("thead", null, el("tr", null, el("th", { scope: "col" }, "Input (after encoding)"), el("th", { class: "num", scope: "col" }, fi.kind === "coefficient" ? "Coefficient" : "Importance"))),
          el("tbody", null, fi.items.map((it) => el("tr", null, el("td", null, it.feature), el("td", { class: "num" }, it.value.toFixed(4)))))))));
      const old = charts.findIndex((c) => c.canvas && c.canvas.dataset.role === "importance");
      if (old >= 0) charts.splice(old, 1)[0].destroy();
      if (!hasChart()) { render(canvasBox, alertBox("warn", null, "Chart unavailable; see the table below.")); return; }
      const canvas = el("canvas", { role: "img", "aria-label": "Horizontal bar chart of feature " + fi.kind + "s for " + model.display_name, dataset: { role: "importance" } });
      render(canvasBox, canvas);
      charts.push(new window.Chart(canvas, {
        type: "bar",
        data: {
          labels: items.map((i) => i.feature),
          datasets: [{
            label: fi.kind === "coefficient" ? "Coefficient" : "Importance",
            data: items.map((i) => i.value),
            backgroundColor: items.map((i) => (i.value >= 0 ? (COLORS[model.algorithm] || "#1d5faf") : "#b45a4f")),
            borderRadius: 3,
          }],
        },
        options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: false } } },
      }));
    }

    usable.forEach((m) => tabs.append(el("button", { type: "button", class: "tab", role: "tab", dataset: { algo: m.algorithm }, onclick: () => show(m) }, m.display_name)));
    const section = el("section", { class: "section" },
      el("div", { class: "panel" },
        el("div", { class: "panel-head" }, el("h2", null, "What each model relied on")),
        tabs, canvasBox, caveat, tableBox));
    requestAnimationFrame(() => show(usable.find((m) => m.selected) || usable[0]));
    return section;
  }

  function detailsSection(r) {
    const ds = r.dataset_summary;
    const pp = r.preprocessing_summary;
    const split = r.split;
    const labels = Object.keys(split.train_class_counts);
    return el("section", { class: "section" },
      el("h2", null, "Data, split and preprocessing"),
      el("details", { class: "more", open: true }, el("summary", null, "Rows and classes"),
        el("dl", { class: "facts-row" },
          el("div", null, el("dt", null, "Rows in file"), el("dd", null, fmt.int(ds.n_rows_raw))),
          el("div", null, el("dt", null, "Duplicates removed"), el("dd", null, fmt.int(ds.duplicate_rows_removed))),
          el("div", null, el("dt", null, "Rows without a label removed"), el("dd", null, fmt.int(ds.missing_target_rows_removed))),
          el("div", null, el("dt", null, "Rows used"), el("dd", null, fmt.int(ds.n_rows_used))),
          el("div", null, el("dt", null, "Features used"), el("dd", null, fmt.int(ds.n_features_used)))),
        el("div", { class: "table-wrap" }, el("table", null,
          el("thead", null, el("tr", null, el("th", { scope: "col" }, "Class"), el("th", { class: "num", scope: "col" }, "Training rows"), el("th", { class: "num", scope: "col" }, "Test rows"))),
          el("tbody", null, labels.map((l) => el("tr", null,
            el("th", { scope: "row" }, l + (l === r.class_mapping.positive_class ? " (positive class)" : "")),
            el("td", { class: "num" }, fmt.int(split.train_class_counts[l])),
            el("td", { class: "num" }, fmt.int(split.test_class_counts[l] || 0))))))),
        el("p", { class: "small muted" }, r.class_mapping.note + " Stratified split with seed " + split.random_seed + ".")),
      el("details", { class: "more" }, el("summary", null, "Preprocessing pipeline"),
        el("p", null, el("strong", null, "Fitted on: "), pp.fitted_on + "."),
        pp.numeric_features.length ? el("p", null, el("strong", null, "Numeric (" + pp.numeric_features.length + "): "), pp.numeric_features.join(", "), el("br"), el("span", { class: "muted" }, pp.numeric_steps.join(", then ") + ".")) : null,
        pp.categorical_features.length ? el("p", null, el("strong", null, "Categorical (" + pp.categorical_features.length + "): "), pp.categorical_features.join(", "), el("br"), el("span", { class: "muted" }, pp.categorical_steps.join(", then ") + ".")) : null,
        pp.dropped_features.length ? el("p", null, el("strong", null, "Dropped: "), pp.dropped_features.map((d) => d.name + " (" + d.reason + ")").join("; ")) : null),
      el("details", { class: "more" }, el("summary", null, "Classification reports (test set)"),
        r.models.map((m) => {
          const rows = Object.entries(m.classification_report).filter(([, v]) => v && typeof v === "object");
          return el("div", null, el("h3", null, m.display_name), el("div", { class: "table-wrap" }, el("table", null,
            el("thead", null, el("tr", null, el("th", { scope: "col" }, ""), ["Precision", "Recall", "F1", "Support"].map((h) => el("th", { class: "num", scope: "col" }, h)))),
            el("tbody", null, rows.map(([k, v]) => el("tr", null, el("th", { scope: "row" }, k),
              el("td", { class: "num" }, fmt.metric(v.precision)), el("td", { class: "num" }, fmt.metric(v.recall)),
              el("td", { class: "num" }, fmt.metric(v["f1-score"])), el("td", { class: "num" }, fmt.int(v.support)))))))) ;
        })));
  }

  /* -------------------------------------------------------------- loading */
  async function showJob(id) {
    destroyCharts();
    render(root, loading("Loading results for job #" + id));
    try {
      const r = await api.results(id);
      render(root,
        summarySection(r),
        selectedSection(r),
        labSection(r),
        chartSection(r),
        confusionSection(r),
        importanceSection(r),
        detailsSection(r),
        el("section", { class: "section" }, disclaimer()));
    } catch (err) {
      if (err.status === 409) {
        render(root, alertBox("info", "Results are not ready", err.message),
          el("a", { class: "btn btn-primary", href: "train.html?job=" + id }, "Track this job"));
      } else {
        render(root, errorBox(err, "Could not load results"));
      }
    }
  }

  async function init() {
    let jobs;
    try {
      jobs = await api.listJobs(100);
    } catch (err) {
      render(root, errorBox(err, "Could not load training jobs"));
      return;
    }
    const completed = jobs.filter((j) => j.status === "COMPLETED");
    const requested = param("job");
    if (!completed.length && !requested) {
      render(root, empty("No results yet", "Results appear here after a training job finishes.",
        el("a", { class: "btn btn-primary", href: "train.html" }, "Train a model")));
      return;
    }
    if (completed.length) {
      render(picker, completed.map((j) => el("option", { value: j.id }, "#" + j.id + " " + j.filename + " (" + fmt.date(j.finishedAt) + ")")));
      pickerWrap.hidden = false;
      picker.addEventListener("change", () => {
        history.replaceState(null, "", "results.html?job=" + picker.value);
        showJob(picker.value);
      });
    }
    const id = requested && /^\d+$/.test(requested) ? requested : String(completed[0].id);
    if (completed.some((j) => String(j.id) === id)) picker.value = id;
    showJob(id);
  }

  init();
})();
