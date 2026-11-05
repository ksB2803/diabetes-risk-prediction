/*
 * Prediction page. The input form is generated from the selected model's
 * feature schema; no feature names are hard-coded.
 */
(function () {
  "use strict";
  const { api, el, render, errorBox, alertBox, empty, chip, fmt, loading, syntheticFlag, disclaimer, param } = window.DRP;

  const root = document.getElementById("predict-root");
  let models = [];
  let current = null;
  let busy = false;

  function modelLabel(m) {
    return "#" + m.id + " " + m.displayName + " (job #" + m.trainingJobId + (m.datasetFilename ? ", " + m.datasetFilename : "") + ")" +
      (m.selectedByCv ? ", chosen by CV" : "");
  }

  function fieldId(i) { return "f-" + i; }

  function buildForm(model) {
    const fields = model.features.map((f, i) => {
      const id = fieldId(i);
      const errId = id + "-err";
      if (f.type === "numeric") {
        const input = el("input", { type: "number", id, name: f.name, step: "any", required: true, inputmode: "decimal",
          "aria-describedby": id + "-hint " + errId, placeholder: f.median != null ? "e.g. " + f.median : null });
        return el("div", { class: "field" },
          el("label", { for: id }, f.name),
          input,
          el("span", { class: "hint", id: id + "-hint" }, f.min != null
            ? "Training range " + f.min + " to " + f.max + ", median " + f.median + "."
            : "Number."),
          el("span", { class: "field-error", id: errId, "aria-live": "polite" }));
      }
      const select = el("select", { id, name: f.name, required: true, "aria-describedby": id + "-hint " + errId },
        el("option", { value: "" }, "Choose a value"),
        f.categories.map((c) => el("option", { value: c }, c)));
      return el("div", { class: "field" },
        el("label", { for: id }, f.name),
        select,
        el("span", { class: "hint", id: id + "-hint" }, f.categories.length + " values seen in training."),
        el("span", { class: "field-error", id: errId, "aria-live": "polite" }));
    });
    return fields;
  }

  function readForm(form, model) {
    const values = {};
    let firstInvalid = null;
    model.features.forEach((f, i) => {
      const input = form.querySelector("#" + fieldId(i));
      const err = form.querySelector("#" + fieldId(i) + "-err");
      let message = "";
      const raw = input.value.trim();
      if (raw === "") {
        message = "Enter a value.";
      } else if (f.type === "numeric") {
        const n = Number(raw);
        if (!isFinite(n)) message = "Enter a number.";
        else values[f.name] = n;
      } else {
        values[f.name] = raw;
      }
      err.textContent = message;
      input.setAttribute("aria-invalid", message ? "true" : "false");
      if (message && !firstInvalid) firstInvalid = input;
    });
    return { values, firstInvalid };
  }

  function resultEmpty() {
    return [el("h2", null, "Result"), el("p", { class: "muted" }, "Fill in every field and select Predict. The result will appear here."), disclaimer()];
  }

  function resultView(res, model) {
    const p = res.probabilityPositive;
    const fill = el("span", { class: "prob-fill" });
    if (typeof p === "number") fill.style.width = (p * 100).toFixed(1) + "%";
    return [
      el("h2", null, "Result"),
      el("p", { class: "muted small" }, "Model predicted class"),
      el("p", { class: "result-class" }, res.predictedClass),
      el("p", null, res.positiveClassPredicted
        ? chip("This is the positive class (“" + res.positiveClass + "”)", "warn")
        : chip("This is the negative class (“" + res.negativeClass + "”)", "info")),
      typeof p === "number" ? el("div", null,
        el("p", { class: "small", id: "prob-label" }, "Model-estimated probability of “" + res.positiveClass + "”: ", el("strong", null, fmt.pct(p))),
        el("div", { class: "prob-bar", role: "img", "aria-labelledby": "prob-label" }, fill),
        el("p", { class: "small muted" }, "This is the model’s output score from the uploaded training data. It is not a calibrated personal medical risk.")) : null,
      res.warnings.length ? alertBox("warn", "Check these inputs", null, res.warnings) : null,
      model.datasetOrigin === "synthetic" ? syntheticFlag() : null,
      el("p", { class: "small muted" }, "Prediction #" + res.predictionId + " with " + res.algorithm + ", " + fmt.date(res.createdAt) + ". Inputs were not stored."),
      el("p", { class: "disclaimer", role: "note" }, res.disclaimer || window.DRP.DISCLAIMER),
    ];
  }

  function showModel(model) {
    current = model;
    const resultBox = el("aside", { class: "panel sticky", "aria-live": "polite" }, resultEmpty());
    const form = el("form", { novalidate: true }, el("div", { class: "form-grid" }, buildForm(model)));
    const formError = el("div");
    const submitBtn = el("button", { type: "submit", class: "btn btn-primary" }, "Predict");
    const fillBtn = el("button", { type: "button", class: "btn btn-secondary" }, "Fill with example values");
    const clearBtn = el("button", { type: "reset", class: "btn btn-quiet" }, "Clear");

    fillBtn.addEventListener("click", () => {
      model.features.forEach((f, i) => {
        const input = form.querySelector("#" + fieldId(i));
        if (f.type === "numeric" && f.median != null) input.value = f.median;
        if (f.type !== "numeric" && f.categories.length) input.value = f.categories[0];
      });
    });
    form.addEventListener("reset", () => { render(resultBox, resultEmpty()); render(formError); });
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (busy) return;
      render(formError);
      const { values, firstInvalid } = readForm(form, model);
      if (firstInvalid) { firstInvalid.focus(); return; }
      busy = true;
      submitBtn.disabled = true;
      render(submitBtn, el("span", { class: "spinner", "aria-hidden": "true" }), "Predicting");
      try {
        const res = await api.predict(model.id, values);
        render(resultBox, resultView(res, model));
      } catch (err) {
        render(formError, errorBox(err, "Prediction failed"));
        if (err.code === "MODEL_UNAVAILABLE") loadModels();
      } finally {
        busy = false;
        submitBtn.disabled = false;
        render(submitBtn, "Predict");
      }
    });
    form.append(formError, el("div", { class: "wizard-actions" }, el("div", { class: "btn-row" }, fillBtn, clearBtn), submitBtn));

    const tm = model.testMetrics || {};
    const info = el("section", { class: "panel" },
      el("div", { class: "panel-head" }, el("h2", null, model.displayName), el("a", { href: "results.html?job=" + model.trainingJobId }, "See full results")),
      model.datasetOrigin === "synthetic" ? syntheticFlag() : null,
      el("dl", { class: "facts-row" },
        el("div", null, el("dt", null, "Target"), el("dd", null, model.targetColumn)),
        el("div", null, el("dt", null, "Positive class"), el("dd", null, model.positiveClass)),
        el("div", null, el("dt", null, "CV " + fmt.metricName(model.selectionMetric)), el("dd", null, fmt.metric(model.cvScore))),
        el("div", null, el("dt", null, "Test ROC-AUC"), el("dd", null, fmt.metric(tm.roc_auc))),
        el("div", null, el("dt", null, "Test recall"), el("dd", null, fmt.metric(tm.recall))),
        el("div", null, el("dt", null, "Test precision"), el("dd", null, fmt.metric(tm.precision)))),
      el("p", { class: "small muted" }, model.version),
      el("h3", { class: "section" }, "Enter values for one record"),
      el("p", { class: "muted small" }, "All " + model.features.length + " fields are required. Example values use the training median for numbers and the first category for text. Values outside the training range are allowed, but the result will warn you."),
      form);

    render(document.getElementById("predict-main"), el("div", { class: "predict-layout" }, info, resultBox));
  }

  async function loadModels() {
    try {
      models = await api.listModels();
    } catch (err) {
      render(root, errorBox(err, "Could not load saved models"));
      return;
    }
    if (!models.length) {
      render(root, empty("No saved models yet", "Train a model first. Every trained model is saved and appears here.",
        el("a", { class: "btn btn-primary", href: "train.html" }, "Train a model")));
      return;
    }
    const select = el("select", { id: "model-picker" }, models.map((m) => el("option", { value: m.id }, modelLabel(m))));
    const requested = param("model");
    const chosen = models.find((m) => String(m.id) === requested) || models.find((m) => m.selectedByCv) || models[0];
    select.value = String(chosen.id);
    render(root,
      requested && String(chosen.id) !== requested ? alertBox("warn", null, "Model #" + requested + " is not available, so another saved model was selected.") : null,
      el("div", { class: "field section" }, el("label", { for: "model-picker" }, "Saved model"), select,
        el("span", { class: "hint" }, "Models are listed newest first. Each one expects the features it was trained on.")),
      el("div", { id: "predict-main", class: "section" }));
    select.addEventListener("change", () => {
      history.replaceState(null, "", "predict.html?model=" + select.value);
      openModel(select.value);
    });
    openModel(chosen.id);
  }

  async function openModel(id) {
    const main = document.getElementById("predict-main");
    render(main, loading("Loading model details"));
    try {
      showModel(await api.getModel(id));
    } catch (err) {
      render(main, errorBox(err, "Could not load this model"));
    }
  }

  loadModels();
})();
