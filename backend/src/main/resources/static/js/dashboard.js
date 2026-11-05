(function () {
  "use strict";
  const { api, el, render, errorBox, empty, statusChip, chip, fmt, loading } = window.DRP;

  async function loadStatus() {
    const list = document.getElementById("status-list");
    const button = document.getElementById("refresh-status");
    button.disabled = true;
    render(list, el("li", null, loading("Checking services")));
    const health = await window.DRP.updateServiceStatus();
    button.disabled = false;
    if (!health) {
      render(list,
        el("li", null, el("span", null, "Web server and database"), chip("Offline", "danger")),
        el("li", null, el("span", null, "ML service"), chip("Unknown", null)));
      return;
    }
    render(list,
      el("li", null, el("span", null, "Web server"), chip("Online", "ok")),
      el("li", null, el("span", null, "Database (H2)"), health.database === "UP" ? chip("Online", "ok") : chip("Problem", "danger")),
      el("li", null, el("span", null, "ML service (Python)"), health.mlService === "UP" ? chip("Online", "ok") : chip("Offline", "danger")));
    if (health.mlService !== "UP") {
      list.append(el("li", null, el("span", { class: "small muted" }, health.mlServiceMessage || "Start the ML service to train or predict.")));
    }
  }

  async function loadCounts() {
    const box = document.getElementById("counts");
    try {
      const s = await api.summary();
      const cell = (value, label) => el("div", { class: "count" }, el("span", { class: "count-value" }, fmt.int(value)), el("span", { class: "count-label" }, label));
      render(box,
        cell(s.totalJobs, "Training jobs"),
        cell(s.completedJobs, "Completed jobs"),
        cell(s.availableModels, "Saved models"),
        cell(s.predictions, "Predictions made"));
    } catch (err) {
      render(box, el("div", { class: "count", style: null }, errorBox(err, "Counts unavailable")));
      box.style.display = "block";
    }
  }

  async function loadRecent() {
    const box = document.getElementById("recent-jobs");
    try {
      const jobs = await api.listJobs(5);
      if (!jobs.length) {
        render(box, empty("No training jobs yet",
          "Train your first model with a labelled CSV. You can also load the built-in synthetic demo file in step 1.",
          el("a", { class: "btn btn-primary", href: "train.html" }, "Train a model")));
        return;
      }
      const rows = jobs.map((j) => el("tr", null,
        el("td", { class: "num" }, "#" + j.id),
        el("td", null, j.filename),
        el("td", null, j.targetColumn),
        el("td", null, j.algorithms.map(fmt.algo).join(", ")),
        el("td", null, statusChip(j.status)),
        el("td", { class: "num" }, j.bestCvScore != null ? fmt.metric(j.bestCvScore) + " " : "n/a"),
        el("td", null, j.status === "COMPLETED"
          ? el("a", { href: "results.html?job=" + j.id }, "View results")
          : (j.status === "FAILED" ? el("span", { class: "small muted" }, "See history") : el("a", { href: "train.html?job=" + j.id }, "Track progress")))));
      render(box, el("div", { class: "table-wrap" }, el("table", null,
        el("thead", null, el("tr", null,
          el("th", { class: "num", scope: "col" }, "Job"), el("th", { scope: "col" }, "File"), el("th", { scope: "col" }, "Target"),
          el("th", { scope: "col" }, "Models"), el("th", { scope: "col" }, "Status"), el("th", { class: "num", scope: "col" }, "Best CV score"),
          el("th", { scope: "col" }, el("span", { class: "visually-hidden" }, "Actions")))),
        el("tbody", null, rows))));
    } catch (err) {
      render(box, errorBox(err, "Could not load recent jobs"));
    }
  }

  document.body.dataset.ownStatus = "1";
  document.getElementById("refresh-status").addEventListener("click", loadStatus);
  loadStatus();
  loadCounts();
  loadRecent();
})();
