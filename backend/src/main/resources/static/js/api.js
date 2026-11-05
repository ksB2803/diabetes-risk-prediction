/*
 * Shared API client and small UI helpers.
 * The browser only ever calls the Spring Boot backend (same origin by default).
 * To point the frontend at another backend, define before this script:
 *   <script>window.DRP_CONFIG = { apiBase: "http://localhost:8080" };</script>
 */
(function () {
  "use strict";

  const API_BASE = (window.DRP_CONFIG && window.DRP_CONFIG.apiBase) || "";

  class ApiError extends Error {
    constructor(status, code, message, details) {
      super(message);
      this.status = status;
      this.code = code || "ERROR";
      this.details = Array.isArray(details) ? details : [];
    }
  }

  async function request(path, options = {}) {
    const { method = "GET", json, formData, timeoutMs = 30000 } = options;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const init = { method, signal: controller.signal, headers: { Accept: "application/json" } };
    if (json !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(json);
    } else if (formData) {
      init.body = formData; // the browser sets the multipart boundary itself
    }

    let res;
    try {
      res = await fetch(API_BASE + path, init);
    } catch (err) {
      if (err.name === "AbortError") {
        throw new ApiError(0, "TIMEOUT", "The server took too long to respond. Try again.");
      }
      throw new ApiError(0, "NETWORK", "Cannot reach the web server. Check that the backend is running.");
    } finally {
      clearTimeout(timer);
    }

    const type = res.headers.get("Content-Type") || "";
    let data = null;
    if (type.includes("application/json")) {
      try { data = await res.json(); } catch (e) { data = null; }
    } else {
      data = await res.text();
    }

    if (!res.ok) {
      const body = data && typeof data === "object" ? data : {};
      const fallback = {
        404: "Not found.",
        413: "The file is larger than the upload limit.",
        429: "The server is busy. Try again shortly.",
        502: "The ML service failed while processing the request.",
        503: "The ML service is not available right now.",
      }[res.status] || "The request failed (HTTP " + res.status + ").";
      throw new ApiError(res.status, body.code, body.message || fallback, body.details);
    }
    return data;
  }

  const api = {
    health: () => request("/api/health", { timeoutMs: 10000 }),
    summary: () => request("/api/summary"),
    inspect: (file) => {
      const fd = new FormData();
      fd.append("file", file, file.name);
      return request("/api/datasets/inspect", { method: "POST", formData: fd, timeoutMs: 60000 });
    },
    createJob: (file, config) => {
      const fd = new FormData();
      fd.append("file", file, file.name);
      fd.append("config", new Blob([JSON.stringify(config)], { type: "application/json" }));
      return request("/api/training-jobs", { method: "POST", formData: fd, timeoutMs: 60000 });
    },
    listJobs: (limit = 50) => request("/api/training-jobs?limit=" + encodeURIComponent(limit)),
    getJob: (id) => request("/api/training-jobs/" + encodeURIComponent(id)),
    results: (id) => request("/api/training-jobs/" + encodeURIComponent(id) + "/results"),
    reportUrl: (id, format) => API_BASE + "/api/training-jobs/" + encodeURIComponent(id) + "/report?format=" + format,
    listModels: () => request("/api/models"),
    getModel: (id) => request("/api/models/" + encodeURIComponent(id)),
    predict: (id, values) => request("/api/models/" + encodeURIComponent(id) + "/predict", { method: "POST", json: { values } }),
    predictions: (limit = 20) => request("/api/predictions?limit=" + encodeURIComponent(limit)),
  };

  /* ------------------------------------------------------------------ DOM helpers */

  /** Build an element. Children may be strings (inserted as text, never HTML) or nodes. */
  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [key, value] of Object.entries(attrs)) {
        if (value === null || value === undefined || value === false) continue;
        if (key === "class") node.className = value;
        else if (key === "text") node.textContent = value;
        else if (key === "dataset") Object.assign(node.dataset, value);
        else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
        else if (value === true) node.setAttribute(key, "");
        else node.setAttribute(key, String(value));
      }
    }
    appendAll(node, children);
    return node;
  }

  function appendAll(node, children) {
    for (const child of children.flat(Infinity)) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
    return node;
  }

  function render(node, ...children) {
    clear(node);
    return appendAll(node, children);
  }

  function loading(text) {
    return el("div", { class: "loading", role: "status" }, el("span", { class: "spinner", "aria-hidden": "true" }), text);
  }

  function alertBox(kind, title, message, details) {
    return el("div", { class: "alert alert-" + kind, role: kind === "error" ? "alert" : "status" },
      title ? el("p", { class: "alert-title" }, title) : null,
      message ? el("p", null, message) : null,
      details && details.length ? el("ul", null, details.map((d) => el("li", null, d))) : null);
  }

  function errorBox(err, title) {
    const e = err instanceof ApiError ? err : new ApiError(0, "ERROR", (err && err.message) || "Something went wrong.");
    return alertBox("error", title || "Something went wrong", e.message, e.details);
  }

  function empty(title, text, action) {
    return el("div", { class: "empty" }, el("h3", null, title), el("p", null, text), action || null);
  }

  function chip(text, kind) {
    return el("span", { class: "chip" + (kind ? " chip-" + kind : "") }, text);
  }

  function statusChip(status) {
    const map = { COMPLETED: ["Completed", "ok"], FAILED: ["Failed", "danger"], RUNNING: ["Running", "info"], QUEUED: ["Queued", "warn"] };
    const [label, kind] = map[status] || [status, null];
    return chip(label, kind);
  }

  /* ------------------------------------------------------------------ formatting */
  const METRIC_LABELS = { roc_auc: "ROC-AUC", f1: "F1-score", recall: "Recall", precision: "Precision", accuracy: "Accuracy" };
  const ALGO_LABELS = { logistic_regression: "Logistic Regression", random_forest: "Random Forest", decision_tree: "Decision Tree" };
  const TUNING_LABELS = { quick: "Quick", balanced: "Balanced", thorough: "Thorough" };
  const ORIGIN_LABELS = { synthetic: "Synthetic/demo dataset", real: "Real or published dataset", unknown: "Origin not specified" };

  const fmt = {
    metric: (v, digits = 3) => (typeof v === "number" && isFinite(v) ? v.toFixed(digits) : "n/a"),
    pct: (v, digits = 1) => (typeof v === "number" && isFinite(v) ? (v * 100).toFixed(digits) + "%" : "n/a"),
    int: (v) => (typeof v === "number" ? v.toLocaleString() : "n/a"),
    bytes: (n) => {
      if (typeof n !== "number") return "n/a";
      if (n < 1024) return n + " B";
      if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
      return (n / (1024 * 1024)).toFixed(2) + " MB";
    },
    date: (iso) => {
      if (!iso) return "n/a";
      const d = new Date(iso);
      return isNaN(d) ? "n/a" : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
    },
    metricName: (m) => METRIC_LABELS[m] || m,
    algo: (a) => ALGO_LABELS[a] || a,
    tuning: (t) => TUNING_LABELS[t] || t,
    origin: (o) => ORIGIN_LABELS[o] || ORIGIN_LABELS.unknown,
    split: (f) => Math.round(f * 100) + "% train / " + Math.round((1 - f) * 100) + "% test",
  };

  const SYNTHETIC_TEXT = "Synthetic/demo dataset — results are for software demonstration only.";
  const DISCLAIMER = "Educational project only. This model is not a medical device and cannot diagnose diabetes. " +
    "Results depend on the uploaded dataset and model quality. For health concerns, consult a qualified healthcare professional.";

  function syntheticFlag() {
    return el("p", { class: "synthetic-flag", role: "note" }, SYNTHETIC_TEXT);
  }

  function disclaimer() {
    return el("p", { class: "disclaimer", role: "note" }, DISCLAIMER);
  }

  function param(name) {
    return new URLSearchParams(window.location.search).get(name);
  }

  /* ------------------------------------------------------------------ header status */
  async function updateServiceStatus() {
    const node = document.getElementById("service-status");
    if (!node) return null;
    try {
      const h = await api.health();
      if (h.status === "UP") {
        node.dataset.state = "up";
        node.textContent = "Services online";
      } else {
        node.dataset.state = "degraded";
        node.textContent = h.mlService === "DOWN" ? "ML service offline" : "Database issue";
      }
      return h;
    } catch (err) {
      node.dataset.state = "down";
      node.textContent = "Server offline";
      return null;
    }
  }

  window.DRP = {
    api, ApiError, el, clear, render, loading, alertBox, errorBox, empty, chip, statusChip, fmt,
    syntheticFlag, disclaimer, param, updateServiceStatus, DISCLAIMER,
  };

  document.addEventListener("DOMContentLoaded", () => {
    // Pages that need the health result call updateServiceStatus themselves.
    if (!document.body.dataset.ownStatus) updateServiceStatus();
  });
})();
