(function () {
  "use strict";
  const { api, el, render, errorBox, empty, statusChip, chip, fmt, loading } = window.DRP;

  const jobsBox = document.getElementById("jobs-table");
  const predsBox = document.getElementById("predictions-table");
  const refreshBtn = document.getElementById("refresh-history");

  function jobActions(j) {
    if (j.status === "COMPLETED") {
      return el("span", { class: "btn-row" },
        el("a", { href: "results.html?job=" + j.id }, "Results"),
        el("a", { href: api.reportUrl(j.id, "csv"), download: "" }, "CSV"));
    }
    if (j.status === "FAILED") return el("span", { class: "small" }, j.errorMessage || "Failed");
    return el("a", { href: "train.html?job=" + j.id }, "Track progress");
  }

  async function loadJobs() {
    render(jobsBox, loading("Loading jobs"));
    try {
      const jobs = await api.listJobs(200);
      if (!jobs.length) {
        render(jobsBox, empty("No training jobs yet", "Jobs you start from the Train Model page are listed here.",
          el("a", { class: "btn btn-primary", href: "train.html" }, "Train a model")));
        return;
      }
      render(jobsBox, el("div", { class: "table-wrap" }, el("table", null,
        el("thead", null, el("tr", null,
          ["Job", "Created", "File", "Target", "Models", "Metric", "Split", "Status", "Best CV", ""].map((h, i) =>
            el("th", { scope: "col", class: [0, 8].includes(i) ? "num" : null }, h || el("span", { class: "visually-hidden" }, "Details"))))),
        el("tbody", null, jobs.map((j) => el("tr", null,
          el("td", { class: "num" }, "#" + j.id),
          el("td", null, fmt.date(j.createdAt)),
          el("td", null, j.filename, j.datasetOrigin === "synthetic" ? [" ", chip("Synthetic", "warn")] : null),
          el("td", null, j.targetColumn),
          el("td", null, j.algorithms.map(fmt.algo).join(", ")),
          el("td", null, fmt.metricName(j.scoringMetric) + " (" + fmt.tuning(j.tuningLevel) + ")"),
          el("td", null, Math.round(j.trainFraction * 100) + "/" + Math.round((1 - j.trainFraction) * 100)),
          el("td", null, statusChip(j.status)),
          el("td", { class: "num" }, j.bestCvScore != null ? fmt.metric(j.bestCvScore) : "n/a"),
          el("td", null, jobActions(j))))))));
    } catch (err) {
      render(jobsBox, errorBox(err, "Could not load training jobs"));
    }
  }

  async function loadPredictions() {
    render(predsBox, loading("Loading predictions"));
    try {
      const preds = await api.predictions(50);
      if (!preds.length) {
        render(predsBox, empty("No predictions yet", "Predictions you make are logged here (class and probability only).",
          el("a", { class: "btn btn-secondary", href: "predict.html" }, "Make a prediction")));
        return;
      }
      render(predsBox, el("div", { class: "table-wrap" }, el("table", null,
        el("thead", null, el("tr", null,
          el("th", { scope: "col", class: "num" }, "Prediction"), el("th", { scope: "col" }, "Time"), el("th", { scope: "col" }, "Model"),
          el("th", { scope: "col" }, "Model predicted class"), el("th", { scope: "col", class: "num" }, "Model-estimated probability of positive class"))),
        el("tbody", null, preds.map((p) => el("tr", null,
          el("td", { class: "num" }, "#" + p.id),
          el("td", null, fmt.date(p.createdAt)),
          el("td", null, el("a", { href: "predict.html?model=" + p.modelId }, "#" + p.modelId + " " + fmt.algo(p.algorithm))),
          el("td", null, p.predictedClass, " ", p.positiveClassPredicted ? chip("positive", "warn") : chip("negative", "info")),
          el("td", { class: "num" }, fmt.pct(p.probabilityPositive))))))));
    } catch (err) {
      render(predsBox, errorBox(err, "Could not load predictions"));
    }
  }

  async function refresh() {
    refreshBtn.disabled = true;
    await Promise.all([loadJobs(), loadPredictions(), window.DRP.updateServiceStatus()]);
    refreshBtn.disabled = false;
  }

  document.body.dataset.ownStatus = "1";
  refreshBtn.addEventListener("click", refresh);
  refresh();
})();
