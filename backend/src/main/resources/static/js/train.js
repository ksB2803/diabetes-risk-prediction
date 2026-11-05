/*
 * Eight-step training wizard. All answers live in `state`, so moving Back and
 * Next never loses a selection. Only real API responses are displayed.
 */
(function () {
  "use strict";
  const { api, el, render, errorBox, alertBox, chip, fmt, loading, syntheticFlag, param } = window.DRP;

  const MAX_UPLOAD_BYTES = 20 * 1024 * 1024; // mirrors the server default; the server is authoritative
  const SAMPLE_URL = "sample/synthetic_diabetes_demo.csv";

  // Mirrors the bounded grids in ml-service/app/training.py (settings tried per model).
  const GRID_SIZE = {
    logistic_regression: { quick: 3, balanced: 8, thorough: 16 },
    random_forest: { quick: 2, balanced: 12, thorough: 20 },
    decision_tree: { quick: 3, balanced: 12, thorough: 160 },
  };
  const FOLDS = { quick: 3, balanced: 5, thorough: 5 };

  const state = {
    step: 0,
    maxVisited: 0,
    file: null,
    origin: "unknown",
    profile: null,
    inspecting: false,
    inspectError: null,
    target: null,
    positiveClass: null,
    features: [],
    featuresTouched: false,
    splitChoice: "0.8",
    customTrainPct: 80,
    seed: 42,
    algorithms: ["logistic_regression", "random_forest"],
    metric: "roc_auc",
    tuning: "balanced",
    submitting: false,
    job: null,
    pollTimer: null,
    stepError: null,
  };

  const panel = document.getElementById("step-panel");
  const rail = document.getElementById("rail");
  const railProgress = document.getElementById("rail-progress");

  /* ================================================================ helpers */
  const columns = () => (state.profile ? state.profile.columns : []);
  const column = (name) => columns().find((c) => c.name === name);
  const trainFraction = () => (state.splitChoice === "custom" ? state.customTrainPct / 100 : parseFloat(state.splitChoice));
  const usableRows = () => (state.profile ? state.profile.n_rows - state.profile.duplicate_rows : 0);

  function targetInfo() {
    const col = column(state.target);
    if (!col || !col.class_counts) return null;
    const total = col.class_counts.reduce((s, c) => s + c.count, 0);
    const minority = Math.min(...col.class_counts.map((c) => c.count));
    return { col, total, minorityShare: total ? minority / total : 0 };
  }

  function defaultFeatures() {
    return columns().filter((c) => c.name !== state.target && c.suggested_feature).map((c) => c.name);
  }

  function featureUnusable(col) {
    if (col.warnings.includes("ALL_EMPTY")) return "Empty in every row";
    if (col.warnings.includes("HIGH_CARDINALITY")) return "Too many distinct text values to encode";
    return null;
  }

  function question(text) { return el("p", { class: "step-question" }, text); }
  function explain(...parts) { return el("div", { class: "explain" }, el("div", null, ...parts)); }

  function radioCard(name, value, checked, title, text, extra) {
    const input = el("input", { type: "radio", name, value, checked: checked || null });
    return el("label", { class: "choice" }, input,
      el("span", { class: "choice-body" }, el("span", { class: "choice-title" }, title, extra || null), text ? el("span", { class: "choice-text" }, text) : null));
  }

  function checkCard(name, value, checked, title, text, extra) {
    const input = el("input", { type: "checkbox", name, value, checked: checked || null });
    return el("label", { class: "choice" }, input,
      el("span", { class: "choice-body" }, el("span", { class: "choice-title" }, title, extra || null), text ? el("span", { class: "choice-text" }, text) : null));
  }

  /* ================================================================ step 1: upload */
  function renderUpload() {
    const fileInput = el("input", { type: "file", id: "csv-file", accept: ".csv,text/csv", class: "visually-hidden" });
    fileInput.addEventListener("change", () => fileInput.files[0] && chooseFile(fileInput.files[0], false));

    const drop = el("label", { class: "dropzone", for: "csv-file" },
      el("strong", null, state.file ? "Choose a different CSV" : "Choose a CSV file"),
      el("span", { class: "muted" }, "or drag and drop it here. CSV only, up to 20 MB."),
      fileInput);
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("is-dragover"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("is-dragover"));
    drop.addEventListener("drop", (e) => {
      e.preventDefault();
      drop.classList.remove("is-dragover");
      const f = e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) chooseFile(f, false);
    });

    const sampleBtn = el("button", { type: "button", class: "btn btn-quiet", disabled: state.inspecting || null }, "Use the built-in synthetic demo file");
    sampleBtn.addEventListener("click", loadSample);

    const origin = el("fieldset", { class: "section" },
      el("legend", null, "Where does this data come from?"),
      el("div", { class: "choice-grid choice-grid-3" },
        radioCard("origin", "synthetic", state.origin === "synthetic", "Synthetic or demo data", "Generated or made-up rows. Results are labelled as a demonstration."),
        radioCard("origin", "real", state.origin === "real", "Real or published dataset", "Collected data, for example a public research dataset."),
        radioCard("origin", "unknown", state.origin === "unknown", "Not sure", "Treat results with extra caution.")));
    origin.addEventListener("change", (e) => { state.origin = e.target.value; });

    const body = [
      el("h2", { id: "step-title" }, "Upload your data"),
      question("Which CSV file should the model learn from?"),
      explain("The file needs one row per person or record, a header row with column names, and one column that holds the known answer (for example 1 for diabetes and 0 for no diabetes)."),
      drop,
      el("div", { class: "btn-row" }, sampleBtn, el("a", { href: SAMPLE_URL, download: "synthetic_diabetes_demo.csv", class: "small" }, "Download the demo CSV")),
      origin,
    ];

    if (state.inspecting) body.push(loading("Reading and checking the file"));
    if (state.inspectError) body.push(errorBox(state.inspectError, "This file could not be used"));
    if (state.profile && !state.inspecting) body.push(renderProfile(state.profile));
    return body;
  }

  function renderProfile(p) {
    const out = [];
    out.push(el("h3", { class: "section" }, "File summary"));
    out.push(el("dl", { class: "facts-row" },
      el("div", null, el("dt", null, "File"), el("dd", null, p.filename)),
      el("div", null, el("dt", null, "Size"), el("dd", null, fmt.bytes(p.size_bytes))),
      el("div", null, el("dt", null, "Rows"), el("dd", null, fmt.int(p.n_rows))),
      el("div", null, el("dt", null, "Columns"), el("dd", null, fmt.int(p.n_columns))),
      el("div", null, el("dt", null, "Duplicate rows"), el("dd", null, fmt.int(p.duplicate_rows)))));
    if (state.origin === "synthetic") out.push(syntheticFlag());
    p.warnings.forEach((w) => out.push(alertBox("warn", null, w)));

    out.push(el("h3", { class: "section" }, "Preview (first " + p.preview.rows.length + " rows)"));
    out.push(el("div", { class: "table-wrap" }, el("table", { class: "preview-table" },
      el("thead", null, el("tr", null, p.preview.columns.map((c) => el("th", { scope: "col" }, c)))),
      el("tbody", null, p.preview.rows.map((r) => el("tr", null, r.map((v) =>
        v === null ? el("td", { class: "empty-cell" }, "missing") : el("td", { class: typeof v === "number" ? "num" : null }, String(v)))))))));

    out.push(el("h3", { class: "section" }, "Columns"));
    out.push(el("div", { class: "table-wrap" }, el("table", null,
      el("thead", null, el("tr", null,
        el("th", { scope: "col" }, "Column"), el("th", { scope: "col" }, "Detected type"),
        el("th", { class: "num", scope: "col" }, "Missing"), el("th", { class: "num", scope: "col" }, "Distinct values"), el("th", { scope: "col" }, "Notes"))),
      el("tbody", null, p.columns.map((c) => el("tr", null,
        el("th", { scope: "row" }, c.name),
        el("td", null, c.inferred_type),
        el("td", { class: "num" }, c.missing_count + " (" + c.missing_pct + "%)"),
        el("td", { class: "num" }, fmt.int(c.unique_count)),
        el("td", null, c.warning_messages.length ? c.warning_messages.join(" ") : el("span", { class: "muted" }, "None"))))))));
    return out;
  }

  async function chooseFile(file, isSample) {
    state.inspectError = null;
    if (!/\.csv$/i.test(file.name)) {
      state.inspectError = new window.DRP.ApiError(400, "NOT_A_CSV", "Only .csv files are accepted.");
      return draw();
    }
    if (file.size === 0) {
      state.inspectError = new window.DRP.ApiError(400, "EMPTY_FILE", "The selected file is empty.");
      return draw();
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      state.inspectError = new window.DRP.ApiError(413, "FILE_TOO_LARGE", "The file is larger than 20 MB.");
      return draw();
    }
    state.file = file;
    state.profile = null;
    state.target = null;
    state.positiveClass = null;
    state.features = [];
    state.featuresTouched = false;
    state.maxVisited = 0;
    if (isSample || /synthetic|demo/i.test(file.name)) state.origin = "synthetic";
    state.inspecting = true;
    draw();
    try {
      const profile = await api.inspect(file);
      state.profile = profile;
      state.target = profile.suggested_target;
      const t = column(state.target);
      state.positiveClass = t ? t.suggested_positive_class : null;
      state.features = defaultFeatures();
    } catch (err) {
      state.file = null;
      state.inspectError = err;
    } finally {
      state.inspecting = false;
      draw();
    }
  }

  async function loadSample() {
    try {
      const res = await fetch(SAMPLE_URL);
      if (!res.ok) throw new Error("The demo file could not be loaded.");
      const blob = await res.blob();
      chooseFile(new File([blob], "synthetic_diabetes_demo.csv", { type: "text/csv" }), true);
    } catch (err) {
      state.inspectError = err;
      draw();
    }
  }

  function validateUpload() {
    if (state.inspecting) return "Wait for the file check to finish.";
    if (!state.profile) return "Choose a CSV file to continue.";
    if (usableRows() < state.profile.limits.min_rows) return "The file needs at least " + state.profile.limits.min_rows + " distinct rows to train a model.";
    return null;
  }

  /* ================================================================ step 2: target */
  const TARGET_REASON = {
    one_class: "Only one value. A classifier needs examples of two outcomes.",
    multiclass: "More than two values. This version supports two classes only.",
    continuous: "Continuous numbers, not a two-class answer.",
    empty: "No values.",
  };

  function renderTarget() {
    const binaryFirst = [...columns()].sort((a, b) => (a.target_status === "binary" ? 0 : 1) - (b.target_status === "binary" ? 0 : 1));
    const list = el("div", { class: "list-choices", role: "radiogroup", "aria-labelledby": "target-q" },
      binaryFirst.map((c) => {
        const ok = c.target_status === "binary";
        const input = el("input", { type: "radio", name: "target", value: c.name, checked: state.target === c.name || null, disabled: ok ? null : true });
        return el("label", { class: "list-choice" + (ok ? "" : " is-disabled") }, input,
          el("span", { class: "lc-main" },
            el("span", { class: "lc-name" }, c.name, state.profile.suggested_target === c.name ? " " : null,
              state.profile.suggested_target === c.name ? chip("Suggested", "teal") : null),
            ok
              ? el("span", { class: "lc-meta" }, c.class_counts.map((k) => chip(k.label + ": " + fmt.int(k.count), null)))
              : el("span", { class: "lc-note" }, TARGET_REASON[c.target_status] || "Not usable as a target.")));
      }));
    list.addEventListener("change", (e) => {
      if (e.target.name !== "target") return;
      state.target = e.target.value;
      const col = column(state.target);
      state.positiveClass = col ? col.suggested_positive_class : null;
      state.features = state.featuresTouched ? state.features.filter((f) => f !== state.target) : defaultFeatures();
      draw();
    });

    const body = [
      el("h2", { id: "step-title" }, "Choose the target"),
      el("p", { class: "step-question", id: "target-q" }, "Which column contains the answer the model should learn to predict?"),
      explain(el("strong", null, "Target (label): "), "the known outcome you want to predict, such as Diabetes = 1 or 0. ",
        el("strong", null, "Features: "), "the other columns the model uses as clues. Only columns with exactly two values can be chosen here."),
      list,
    ];

    const info = targetInfo();
    if (info) {
      const counts = info.col.class_counts;
      body.push(el("h3", { class: "section" }, "Class balance in “" + state.target + "”"));
      const bar = el("div", { class: "class-bar", role: "img", "aria-label": counts.map((c) => c.label + " " + fmt.pct(c.count / info.total)).join(", ") },
        counts.map((c) => {
          const seg = el("span", null, c.label + " " + fmt.pct(c.count / info.total, 0));
          seg.style.width = (100 * c.count / info.total) + "%";
          return seg;
        }));
      body.push(bar);
      if (info.minorityShare < 0.35) {
        body.push(alertBox("warn", "Imbalanced classes", "The smaller class is " + fmt.pct(info.minorityShare) +
          " of rows. Accuracy can look good even when the model misses most of the smaller class, so also check recall, F1 and ROC-AUC."));
      }
      const pos = el("fieldset", { class: "section" },
        el("legend", null, "Which value is the positive outcome?"),
        el("p", { class: "muted small" }, "Precision, recall, F1 and the probability shown on the Predict page all refer to this class. Nothing is guessed: the mapping is shown on the results page."),
        el("div", { class: "choice-grid" }, counts.map((c) => radioCard("positive", c.label, state.positiveClass === c.label, c.label,
          c.label === info.col.suggested_positive_class ? "Suggested: usually means the condition is present." : null))));
      pos.addEventListener("change", (e) => { state.positiveClass = e.target.value; });
      body.push(pos);
    }
    return body;
  }

  function validateTarget() {
    const col = column(state.target);
    if (!col) return "Choose a target column.";
    if (col.target_status !== "binary") return TARGET_REASON[col.target_status] || "This column cannot be a target.";
    if (!state.positiveClass) return "Choose which value is the positive outcome.";
    return null;
  }

  /* ================================================================ step 3: features */
  function renderFeatures() {
    const candidates = columns().filter((c) => c.name !== state.target);
    const selected = new Set(state.features);
    const list = el("div", { class: "list-choices" }, candidates.map((c) => {
      const reason = featureUnusable(c);
      const input = el("input", { type: "checkbox", name: "feature", value: c.name, checked: (!reason && selected.has(c.name)) || null, disabled: reason ? true : null });
      const notes = c.warning_messages.filter((m, i) => !["HIGH_CARDINALITY", "ALL_EMPTY"].includes(c.warnings[i]));
      return el("label", { class: "list-choice" + (reason ? " is-disabled" : "") }, input,
        el("span", { class: "lc-main" },
          el("span", { class: "lc-name" }, c.name),
          el("span", { class: "lc-meta" },
            chip(c.inferred_type === "numeric" ? "Number" : "Category", "info"),
            c.missing_count ? chip(c.missing_pct + "% missing", c.missing_pct > 40 ? "warn" : null) : null,
            c.warnings.includes("ID_LIKE") ? chip("Looks like an ID", "warn") : null,
            c.warnings.includes("CONSTANT") ? chip("Same value everywhere", "warn") : null,
            c.warnings.includes("DUPLICATE_COLUMN") ? chip("Duplicate column", "warn") : null),
          reason ? el("span", { class: "lc-note" }, reason + ".") : (notes.length ? el("span", { class: "lc-note" }, notes.join(" ")) : null)));
    }));
    list.addEventListener("change", (e) => {
      if (e.target.name !== "feature") return;
      state.featuresTouched = true;
      const set = new Set(state.features);
      if (e.target.checked) set.add(e.target.value); else set.delete(e.target.value);
      state.features = candidates.map((c) => c.name).filter((n) => set.has(n));
      counter.textContent = state.features.length + " of " + candidates.length + " columns selected";
    });

    const counter = el("p", { class: "muted small", "aria-live": "polite" }, state.features.length + " of " + candidates.length + " columns selected");
    const btn = (label, fn) => el("button", { type: "button", class: "btn btn-secondary", onclick: () => { state.featuresTouched = true; fn(); draw(); } }, label);

    return [
      el("h2", { id: "step-title" }, "Choose the features"),
      question("Which columns should the model use as clues?"),
      explain("Pick columns that would be known before the outcome. Identifiers (like a patient number) and columns that repeat the answer can make a model look better than it really is. ",
        "Missing numbers are filled with the training median and text categories with the most common training value, inside the training pipeline only."),
      el("div", { class: "btn-row" },
        btn("Use suggested", () => { state.features = defaultFeatures(); }),
        btn("Select all usable", () => { state.features = candidates.filter((c) => !featureUnusable(c)).map((c) => c.name); }),
        btn("Clear all", () => { state.features = []; })),
      counter,
      list,
    ];
  }

  function validateFeatures() {
    if (!state.features.length) return "Select at least one feature.";
    if (state.features.includes(state.target)) return "The target cannot also be a feature.";
    return null;
  }

  /* ================================================================ step 4: split */
  function splitEstimate(fraction) {
    const info = targetInfo();
    const rows = usableRows();
    const test = Math.round(rows * (1 - fraction));
    const perClass = info ? info.col.class_counts.map((c) => ({ label: c.label, test: Math.floor(c.count * (1 - fraction)) })) : [];
    return { rows, train: rows - test, test, perClass };
  }

  function renderSplit() {
    const options = [
      ["0.8", "80% train / 20% test", "Recommended for most datasets.", true],
      ["0.75", "75% train / 25% test", "A slightly bigger test set.", false],
      ["0.7", "70% train / 30% test", "Useful when you have plenty of rows.", false],
      ["custom", "Custom split", "Choose a training share between 50% and 90%.", false],
    ];
    const group = el("fieldset", null,
      el("legend", { class: "visually-hidden" }, "Train/test split"),
      el("div", { class: "choice-grid" }, options.map(([v, title, text, rec]) =>
        radioCard("split", v, state.splitChoice === v, title, text, rec ? chip("Recommended", "teal") : null))));
    group.addEventListener("change", (e) => { if (e.target.name === "split") { state.splitChoice = e.target.value; draw(); } });

    const customInput = el("input", { type: "number", id: "custom-train", min: 50, max: 90, step: 1, value: state.customTrainPct, "aria-describedby": "custom-hint" });
    customInput.addEventListener("input", () => { state.customTrainPct = Number(customInput.value); updateEstimate(); });
    const seedInput = el("input", { type: "number", id: "seed", min: 0, max: 2147483647, step: 1, value: state.seed, "aria-describedby": "seed-hint" });
    seedInput.addEventListener("input", () => { state.seed = seedInput.value === "" ? NaN : Number(seedInput.value); });

    const estimateBox = el("div", { "aria-live": "polite" });
    function updateEstimate() {
      const f = trainFraction();
      if (!(f >= 0.5 && f <= 0.9)) { render(estimateBox, alertBox("error", null, "Enter a training share between 50 and 90.")); return; }
      const e = splitEstimate(f);
      render(estimateBox, el("dl", { class: "facts-row" },
        el("div", null, el("dt", null, "Rows used (duplicates removed)"), el("dd", null, fmt.int(e.rows))),
        el("div", null, el("dt", null, "Training rows (about)"), el("dd", null, fmt.int(e.train))),
        el("div", null, el("dt", null, "Test rows (about)"), el("dd", null, fmt.int(e.test))),
        e.perClass.map((p) => el("div", null, el("dt", null, "Test rows of “" + p.label + "”"), el("dd", null, "≈ " + fmt.int(p.test))))));
    }
    updateEstimate();

    return [
      el("h2", { id: "step-title" }, "Split into training and test data"),
      question("How much data should be kept aside for unseen testing?"),
      explain("The model learns from the ", el("strong", null, "training"), " rows. The ", el("strong", null, "test"),
        " rows are locked away first and used once at the end, so their scores show how the model handles new data. Tuning never looks at the test rows. ",
        "The split is stratified, so both parts keep the same class balance."),
      group,
      el("div", { class: "inline-fields section" },
        state.splitChoice === "custom" ? el("div", { class: "field" }, el("label", { for: "custom-train" }, "Training share (%)"), customInput,
          el("span", { class: "hint", id: "custom-hint" }, "Whole number from 50 to 90.")) : null,
        el("div", { class: "field" }, el("label", { for: "seed" }, "Random seed"), seedInput,
          el("span", { class: "hint", id: "seed-hint" }, "Fixes the shuffle so the same settings give the same split."))),
      estimateBox,
    ];
  }

  function validateSplit() {
    const f = trainFraction();
    if (state.splitChoice === "custom" && (!Number.isInteger(state.customTrainPct) || state.customTrainPct < 50 || state.customTrainPct > 90)) {
      return "Enter a whole-number training share between 50 and 90.";
    }
    if (!Number.isInteger(state.seed) || state.seed < 0 || state.seed > 2147483647) return "The random seed must be a whole number of 0 or more.";
    const e = splitEstimate(f);
    const small = e.perClass.find((p) => p.test < 2);
    if (small) return "With this split the test set would have fewer than 2 rows of “" + small.label + "”. Use a larger test share or more data.";
    return null;
  }

  /* ================================================================ step 5: models */
  function renderModels() {
    const sel = new Set(state.algorithms);
    const group = el("fieldset", null, el("legend", { class: "visually-hidden" }, "Algorithms"),
      el("div", { class: "choice-grid choice-grid-3" },
        checkCard("algo", "logistic_regression", sel.has("logistic_regression"), "Logistic Regression",
          "Despite the name, a classification model. Draws a weighted boundary between the classes. Easy to interpret, a strong baseline; inputs are standardised first."),
        checkCard("algo", "random_forest", sel.has("random_forest"), "Random Forest",
          "Many decision trees that vote. Can capture non-linear patterns and interactions between features."),
        checkCard("algo", "decision_tree", sel.has("decision_tree"), "Decision Tree",
          "A single tree of yes/no questions. Easy to follow, but can overfit; depth is limited by tuning.")));
    group.addEventListener("change", () => {
      state.algorithms = [...group.querySelectorAll("input:checked")].map((i) => i.value);
    });
    return [
      el("h2", { id: "step-title" }, "Choose candidate models"),
      question("Which algorithms should be trained and compared?"),
      explain("Each selected model is tuned and scored the same way. None is assumed to be best: the comparison on the next pages decides."),
      group,
    ];
  }

  function validateModels() {
    return state.algorithms.length ? null : "Select at least one algorithm.";
  }

  /* ================================================================ step 6: metric */
  const METRIC_TEXT = {
    roc_auc: "How well the model ranks positive cases above negative ones, across all thresholds. 0.5 is chance level, 1.0 is perfect.",
    f1: "A balance of precision and recall in one number. Good when both kinds of mistakes matter.",
    recall: "Of the real positive cases, how many the model finds. Choose this if missing a positive case is the bigger worry.",
    accuracy: "The share of all predictions that are correct. Can mislead when one class is much more common.",
    precision: "Of the cases the model flags as positive, how many really are. Choose this if false alarms are the bigger worry.",
  };

  function renderMetric() {
    const order = ["roc_auc", "f1", "recall", "accuracy", "precision"];
    const group = el("fieldset", null, el("legend", { class: "visually-hidden" }, "Selection metric"),
      el("div", { class: "choice-grid" }, order.map((m) =>
        radioCard("metric", m, state.metric === m, fmt.metricName(m), METRIC_TEXT[m], m === "roc_auc" ? chip("Default", "teal") : null))));
    group.addEventListener("change", (e) => { state.metric = e.target.value; draw(); });
    const info = targetInfo();
    return [
      el("h2", { id: "step-title" }, "Choose what matters most"),
      question("Which score should be used to tune and pick the model?"),
      explain("This score is used during cross-validation on the training data. All the other scores are still reported for every model, on both cross-validation and the test set."),
      group,
      info && info.minorityShare < 0.35 && state.metric === "accuracy"
        ? alertBox("warn", "Accuracy on imbalanced data", "Your classes are imbalanced, so a model can score high accuracy while missing most positive cases. Consider ROC-AUC, F1 or recall.")
        : null,
    ];
  }

  /* ================================================================ step 7: tuning */
  function renderTuning() {
    const levels = [
      ["quick", "Quick", "A small search. Fastest; good for a first look.", false],
      ["balanced", "Balanced", "A moderate search with 5-fold cross-validation.", true],
      ["thorough", "Thorough", "A larger but bounded search. Takes longest.", false],
    ];
    const group = el("fieldset", null, el("legend", { class: "visually-hidden" }, "Tuning level"),
      el("div", { class: "choice-grid choice-grid-3" }, levels.map(([v, title, text, rec]) =>
        radioCard("tuning", v, state.tuning === v, title, text, rec ? chip("Recommended", "teal") : null))));
    group.addEventListener("change", (e) => { state.tuning = e.target.value; draw(); });

    const rows = state.algorithms.map((a) => {
      const n = GRID_SIZE[a][state.tuning];
      return el("tr", null, el("th", { scope: "row" }, fmt.algo(a)), el("td", { class: "num" }, fmt.int(n)),
        el("td", { class: "num" }, FOLDS[state.tuning]), el("td", { class: "num" }, fmt.int(n * FOLDS[state.tuning] + 1)));
    });
    return [
      el("h2", { id: "step-title" }, "Tune the settings"),
      question("How hard should the app search for better model settings?"),
      explain("Each model has settings (like tree depth or regularisation strength). The search tries a fixed, limited set of them using stratified cross-validation on the training rows only, then refits the best one on all training rows."),
      group,
      el("h3", { class: "section" }, "What this level means for your models"),
      el("div", { class: "table-wrap" }, el("table", null,
        el("thead", null, el("tr", null, el("th", { scope: "col" }, "Model"), el("th", { class: "num", scope: "col" }, "Settings tried"),
          el("th", { class: "num", scope: "col" }, "CV folds"), el("th", { class: "num", scope: "col" }, "Model fits"))),
        el("tbody", null, rows))),
      el("p", { class: "small muted" }, "Folds may be reduced automatically if one class has very few training rows."),
    ];
  }

  /* ================================================================ step 8: review */
  function collectWarnings() {
    const warnings = [];
    const info = targetInfo();
    if (state.profile.duplicate_rows) warnings.push(state.profile.duplicate_rows + " duplicate row(s) will be removed before splitting.");
    if (info && info.minorityShare < 0.35) warnings.push("Classes are imbalanced (smaller class " + fmt.pct(info.minorityShare) + "). Look beyond accuracy.");
    state.features.forEach((f) => {
      const c = column(f);
      if (!c) return;
      if (c.warnings.includes("ID_LIKE")) warnings.push("“" + f + "” looks like an identifier.");
      if (c.warnings.includes("HIGH_MISSING")) warnings.push("“" + f + "” is missing in " + c.missing_pct + "% of rows; gaps will be filled from training data.");
      if (c.warnings.includes("CONSTANT")) warnings.push("“" + f + "” has the same value everywhere and will be dropped.");
    });
    if (state.origin === "unknown") warnings.push("Dataset origin is not specified.");
    return warnings;
  }

  function renderReview() {
    const p = state.profile;
    const warnings = collectWarnings();
    const row = (k, v, stepIndex) => [el("dt", null, k), el("dd", null, v, " ",
      el("button", { type: "button", class: "btn btn-quiet", onclick: () => go(stepIndex), "aria-label": "Edit " + k }, "Edit"))];
    return [
      el("h2", { id: "step-title" }, "Review and train"),
      question("Does everything look right?"),
      state.origin === "synthetic" ? syntheticFlag() : null,
      el("dl", { class: "review-list" },
        row("Dataset", p.filename + " (" + fmt.int(p.n_rows) + " rows, " + p.n_columns + " columns, " + fmt.bytes(p.size_bytes) + ")", 0),
        row("Data origin", fmt.origin(state.origin), 0),
        row("Target", state.target + " (positive class: " + state.positiveClass + ")", 1),
        row("Features (" + state.features.length + ")", state.features.join(", "), 2),
        row("Split", fmt.split(trainFraction()) + ", stratified", 3),
        row("Random seed", String(state.seed), 3),
        row("Models", state.algorithms.map(fmt.algo).join(", "), 4),
        row("Selection metric", fmt.metricName(state.metric), 5),
        row("Tuning", fmt.tuning(state.tuning) + " (" + FOLDS[state.tuning] + "-fold cross-validation)", 6)),
      warnings.length ? alertBox("warn", "Things to keep in mind", null, warnings) : alertBox("ok", null, "No warnings for these settings."),
    ];
  }

  function buildConfig() {
    return {
      targetColumn: state.target,
      featureColumns: state.features,
      trainFraction: Math.round(trainFraction() * 100) / 100,
      algorithms: state.algorithms,
      scoringMetric: state.metric,
      tuningLevel: state.tuning,
      randomSeed: state.seed,
      positiveClass: state.positiveClass,
      datasetOrigin: state.origin,
    };
  }

  /* ================================================================ job tracking */
  const STAGES = [
    ["QUEUED", "Queued on the server"],
    ["RUNNING", "Validating data, tuning and evaluating models"],
    ["SAVING", "Saving models and results"],
    ["COMPLETED", "Finished"],
  ];

  function stageStates(job) {
    if (job.status === "COMPLETED") return STAGES.map(() => "done");
    const saving = job.stage && /saving/i.test(job.stage);
    const activeIdx = job.status === "QUEUED" ? 0 : (saving ? 2 : 1);
    return STAGES.map((_, i) => {
      if (job.status === "FAILED") return i < activeIdx ? "done" : (i === activeIdx ? "failed" : "todo");
      return i < activeIdx ? "done" : (i === activeIdx ? "active" : "todo");
    });
  }

  function renderJob(pollError) {
    const job = state.job;
    const states = stageStates(job);
    const body = [
      el("h2", { id: "step-title" }, "Training job #" + job.id),
      el("dl", { class: "facts-row" },
        el("div", null, el("dt", null, "Job ID"), el("dd", null, "#" + job.id)),
        el("div", null, el("dt", null, "Status"), el("dd", null, window.DRP.statusChip(job.status))),
        el("div", null, el("dt", null, "Started"), el("dd", null, fmt.date(job.startedAt || job.createdAt)))),
      el("ol", { class: "job-stages", "aria-label": "Progress" }, STAGES.map(([, label], i) =>
        el("li", { "data-state": states[i] }, el("span", { class: "stage-dot", "aria-hidden": "true" }), label,
          states[i] === "active" ? el("span", { class: "visually-hidden" }, " (in progress)") : null))),
      el("p", { class: "small muted" }, "The server reports stages, not percentages, so no progress percentage is shown."),
    ];
    if (pollError) body.push(alertBox("warn", "Lost contact with the server", pollError.message + " Still trying."));
    if (job.status === "COMPLETED") {
      body.push(alertBox("ok", "Training finished", "Models were tuned on the training rows and scored once on the " + (job.testRows || "held-out") + " test rows."));
      body.push(el("div", { class: "btn-row" },
        el("a", { class: "btn btn-primary", href: "results.html?job=" + job.id }, "View results"),
        el("a", { class: "btn btn-secondary", href: "predict.html" + (job.selectedModelId ? "?model=" + job.selectedModelId : "") }, "Make a prediction")));
    } else if (job.status === "FAILED") {
      body.push(alertBox("error", "Training failed", job.errorMessage || "The job failed."));
      const actions = el("div", { class: "btn-row" });
      if (state.file && state.profile) {
        actions.append(el("button", { type: "button", class: "btn btn-primary", onclick: submit }, "Try again"));
        actions.append(el("button", { type: "button", class: "btn btn-secondary", onclick: () => { state.job = null; go(7); } }, "Back and edit"));
      } else {
        actions.append(el("a", { class: "btn btn-primary", href: "train.html" }, "Start a new training job"));
      }
      body.push(actions);
    }
    render(panel, body);
    drawRail();
  }

  function poll(failures = 0) {
    clearTimeout(state.pollTimer);
    state.pollTimer = setTimeout(async () => {
      try {
        state.job = await api.getJob(state.job.id);
        renderJob(null);
        if (state.job.status === "QUEUED" || state.job.status === "RUNNING") poll(0);
      } catch (err) {
        if (err.status === 404) { renderJob(null); render(panel, errorBox(err, "This job no longer exists")); return; }
        renderJob(failures < 20 ? err : null);
        if (failures < 20) poll(failures + 1);
        else panel.append(errorBox(err, "Stopped checking. Refresh the page to try again."));
      }
    }, failures ? Math.min(10000, 1500 * (failures + 1)) : 1500);
  }

  async function submit() {
    if (state.submitting) return;
    state.submitting = true;
    state.stepError = null;
    draw();
    try {
      state.job = await api.createJob(state.file, buildConfig());
      history.replaceState(null, "", "train.html?job=" + state.job.id);
      renderJob(null);
      poll(0);
    } catch (err) {
      state.job = null;
      state.stepError = err;
      draw();
    } finally {
      state.submitting = false;
    }
  }

  /* ================================================================ framework */
  const STEPS = [
    { title: "Upload CSV", render: renderUpload, validate: validateUpload },
    { title: "Choose target", render: renderTarget, validate: validateTarget },
    { title: "Choose features", render: renderFeatures, validate: validateFeatures },
    { title: "Train/test split", render: renderSplit, validate: validateSplit },
    { title: "Models", render: renderModels, validate: validateModels },
    { title: "Metric", render: renderMetric, validate: () => null },
    { title: "Tuning", render: renderTuning, validate: () => null },
    { title: "Review and train", render: renderReview, validate: () => null },
  ];

  function drawRail() {
    const locked = !!state.job || state.submitting;
    render(rail, STEPS.map((s, i) => el("li", null, el("button", {
      type: "button",
      class: "rail-step",
      "data-state": i < state.step || (i <= state.maxVisited && i !== state.step) ? "done" : "todo",
      "aria-current": i === state.step && !state.job ? "step" : null,
      disabled: locked || i > state.maxVisited ? true : null,
      onclick: () => go(i),
    }, el("span", { class: "rail-num", "aria-hidden": "true" }, String(i + 1)), s.title))));
    railProgress.textContent = state.job ? "Training job submitted" : "Step " + (state.step + 1) + " of " + STEPS.length;
  }

  function draw() {
    if (state.job) return renderJob(null);
    const step = STEPS[state.step];
    const content = step.render();
    const isLast = state.step === STEPS.length - 1;

    const back = el("button", { type: "button", class: "btn btn-secondary", disabled: state.step === 0 || state.submitting || null,
      onclick: () => go(state.step - 1) }, isLast ? "Back and edit" : "Back");
    const next = isLast
      ? el("button", { type: "button", class: "btn btn-primary", disabled: state.submitting || null, onclick: confirmAndSubmit },
          state.submitting ? [el("span", { class: "spinner", "aria-hidden": "true" }), "Starting"] : "Start Training")
      : el("button", { type: "button", class: "btn btn-primary", disabled: state.inspecting || null, onclick: nextStep }, "Next");

    render(panel, content,
      state.stepError ? (state.stepError instanceof Error ? errorBox(state.stepError, isLast ? "Training could not start" : null) : alertBox("error", null, state.stepError)) : null,
      el("div", { class: "wizard-actions" }, back, next));
    drawRail();
  }

  function go(index) {
    if (index < 0 || index >= STEPS.length || index > state.maxVisited) return;
    state.step = index;
    state.stepError = null;
    draw();
    const title = document.getElementById("step-title");
    if (title) { title.setAttribute("tabindex", "-1"); title.focus({ preventScroll: false }); }
  }

  function nextStep() {
    const problem = STEPS[state.step].validate();
    if (problem) {
      state.stepError = problem;
      draw();
      const err = panel.querySelector(".alert-error");
      if (err) err.scrollIntoView({ block: "center", behavior: "smooth" });
      return;
    }
    state.maxVisited = Math.max(state.maxVisited, state.step + 1);
    go(state.step + 1);
  }

  function confirmAndSubmit() {
    for (let i = 0; i < STEPS.length - 1; i++) {
      const problem = STEPS[i].validate();
      if (problem) { state.step = i; state.stepError = problem; draw(); return; }
    }
    const dialog = document.getElementById("confirm-dialog");
    const fits = state.algorithms.reduce((s, a) => s + GRID_SIZE[a][state.tuning] * FOLDS[state.tuning] + 1, 0);
    document.getElementById("confirm-text").textContent =
      "The server will run " + fmt.int(fits) + " model fits across " + state.algorithms.length + " algorithm(s) on " +
      fmt.int(usableRows()) + " rows. This can take from a few seconds to a few minutes.";
    if (typeof dialog.showModal === "function") {
      dialog.returnValue = "";
      dialog.showModal();
      dialog.addEventListener("close", function onClose() {
        dialog.removeEventListener("close", onClose);
        if (dialog.returnValue === "confirm") submit();
      });
    } else if (window.confirm("Start training now?")) {
      submit();
    }
  }

  /* ================================================================ start */
  const existingJob = param("job");
  if (existingJob && /^\d+$/.test(existingJob)) {
    render(panel, loading("Loading training job #" + existingJob));
    api.getJob(existingJob).then((job) => {
      state.job = job;
      renderJob(null);
      if (job.status === "QUEUED" || job.status === "RUNNING") poll(0);
    }).catch((err) => {
      render(panel, errorBox(err, "Could not load job #" + existingJob), el("a", { class: "btn btn-primary", href: "train.html" }, "Start a new training job"));
      drawRail();
    });
    drawRail();
  } else {
    draw();
  }
})();
