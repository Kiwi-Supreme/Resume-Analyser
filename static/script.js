const $ = (id) => document.getElementById(id);

const SAMPLES = {
  ds: "Data Scientist\nWe are hiring a Data Scientist with 3+ years of experience and a master's degree. You will build machine learning models in Python, analyse data with pandas and numpy, write SQL queries, and deploy models using Docker on AWS. Experience with tensorflow, NLP and statistics is a plus.",
  hr: "HR Manager\nLead recruitment, onboarding and employee relations with 5+ years of experience and a bachelor's degree. Responsibilities include payroll oversight, benefits administration, policy compliance, performance management and training programs for staff.",
  it: "Java Developer\nBackend developer with 4+ years of experience and a bachelor's degree. Strong Java, Spring and Hibernate skills, SQL databases, REST APIs, Git, Jenkins and Docker. Agile teams and unit testing required.",
  fin: "Finance Analyst\n3+ years of experience and a bachelor's degree. Required skills: accounting, budgeting, financial reporting, auditing, excel. Preferred: accounts payable.",
};

const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};
const pretty = (s) => s.replace(/-/g, " ");

function chips(container, items, cls) {
  container.innerHTML = "";
  items.forEach((t) => container.appendChild(el("span", "chip " + (cls || ""), t)));
}

function verdictFor(score) {
  if (score >= 70) return ["Strong match", "#16a34a"];
  if (score >= 45) return ["Good match", "#4f46e5"];
  if (score >= 25) return ["Partial match", "#d97706"];
  return ["Weak match", "#dc2626"];
}

function bar(pct) {
  const b = el("div", "bar");
  const i = el("div");
  i.style.width = Math.max(0, Math.min(100, pct)) + "%";
  b.appendChild(i);
  return b;
}

const DIM_LABELS = {
  required_skills: "Required skills",
  semantic_relevance: "Semantic relevance",
  experience: "Experience alignment",
  education: "Education alignment",
};

function renderDims(container, dims) {
  container.innerHTML = "";
  Object.entries(DIM_LABELS).forEach(([key, label]) => {
    const v = dims[key];
    const row = el("div", "dim");
    const l = el("div", "label");
    l.append(el("span", "", label), el("b", "", v === null ? "n/a (not in job)" : v + "%"));
    row.appendChild(l);
    if (v !== null) row.appendChild(bar(v));
    container.appendChild(row);
  });
}

function renderWhy(list, items, positive) {
  list.innerHTML = "";
  items.forEach((it) => {
    const li = el("li");
    const left = el("span", "", it.label + " ");
    left.appendChild(el("span", "val", "(" + it.value + ")"));
    li.append(left, el("span", "imp " + (positive ? "pos" : "neg"), (it.impact > 0 ? "+" : "") + it.impact));
    list.appendChild(li);
  });
}

function fact(parent, value, label) {
  const f = el("div", "fact");
  f.append(el("b", "", value), el("span", "", label));
  parent.appendChild(f);
}

// ---------------------------------------------------------------- single
function render(data) {
  $("results").hidden = false;

  const top = data.predicted_category;
  $("top-cat").innerHTML = "";
  $("top-cat").append(el("span", "", pretty(top.category)), el("small", "", top.confidence + "% confidence"));
  const cats = $("categories");
  cats.innerHTML = "";
  data.top_categories.forEach((c) => {
    const row = el("div", "cat-row");
    const l = el("div", "label");
    l.append(el("span", "", pretty(c.category)), el("span", "", c.confidence + "%"));
    row.append(l, bar(c.confidence));
    cats.appendChild(row);
  });

  const p = data.profile;
  $("facts").innerHTML = "";
  fact($("facts"), p.years_experience ? p.years_experience + " yrs" : "—", "experience");
  fact($("facts"), p.education, "education");
  fact($("facts"), p.skills.length, "skills detected");
  fact($("facts"), p.certifications.length, "certifications");
  chips($("resume-skills"), p.skills);

  const hasJob = data.match_score !== undefined;
  $("match").hidden = !hasJob;
  $("why").hidden = !hasJob;
  $("skill-compare").hidden = !hasJob;
  $("jd-category").textContent = hasJob
    ? "The job description itself looks like: " + pretty(data.job_category.category) + " (" + data.job_category.confidence + "%)"
    : "Add a job description to get a compatibility score.";

  if (hasJob) {
    const [label, color] = verdictFor(data.match_score);
    $("gauge").style.setProperty("--pct", data.match_score);
    $("gauge").style.setProperty("--col", color);
    $("score").textContent = Math.round(data.match_score);
    $("verdict").textContent = label;
    $("verdict").style.color = color;
    renderDims($("dims"), data.dimensions);
    renderWhy($("why-pos"), data.explanation.positive, true);
    renderWhy($("why-neg"), data.explanation.negative, false);

    const jr = data.job_requirements;
    $("job-facts").innerHTML = "";
    fact($("job-facts"), jr.years_experience ? jr.years_experience + "+ yrs" : "—", "experience required");
    fact($("job-facts"), jr.education, "education required");
    fact($("job-facts"), jr.skills.length, "skills required");
    chips($("matched"), data.matched_skills.map((s) => "✓ " + s), "ok");
    chips($("missing"), data.missing_skills.map((s) => "✗ " + s), "miss");
    chips($("keywords"), data.missing_keywords, "kw");
  }
  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------------------------------------------------------------- rank
function renderRank(data) {
  $("rank-results").hidden = false;
  const jr = data.job_requirements;
  $("rank-req").textContent =
    "Job needs " + jr.skills.length + " skills" +
    (jr.years_experience ? ", " + jr.years_experience + "+ years" : "") +
    (jr.education !== "Not specified" ? ", " + jr.education : "") +
    (data.warnings.length ? " · skipped: " + data.warnings.join("; ") : "");

  const list = $("rank-list");
  list.innerHTML = "";
  data.candidates.forEach((c) => {
    const row = el("div", "cand");
    row.appendChild(el("div", "rank-no", c.rank));

    const mid = el("div");
    mid.appendChild(el("div", "cand-name", c.filename));
    mid.appendChild(el("div", "cand-sub",
      "Likely role: " + pretty(c.role.category) + " · " + (c.years_experience || 0) + " yrs experience"));
    mid.appendChild(bar(c.match_score));
    const d = el("div", "mini-dims");
    Object.entries(DIM_LABELS).forEach(([k, label]) => {
      const s = el("span", "", label + " ");
      s.appendChild(el("b", "", c.dimensions[k] === null ? "n/a" : c.dimensions[k] + "%"));
      d.appendChild(s);
    });
    mid.appendChild(d);
    const chipRow = el("div", "chips");
    chipRow.style.marginTop = "8px";
    c.matched_skills.slice(0, 5).forEach((s) => chipRow.appendChild(el("span", "chip ok", "✓ " + s)));
    c.missing_skills.slice(0, 4).forEach((s) => chipRow.appendChild(el("span", "chip miss", "✗ " + s)));
    mid.appendChild(chipRow);
    const why = c.explanation;
    const parts = [];
    if (why.positive[0]) parts.push("+ " + why.positive[0].label);
    if (why.negative[0]) parts.push("− " + why.negative[0].label);
    if (parts.length) mid.appendChild(el("div", "reason", "Biggest drivers: " + parts.join("  ·  ")));
    row.appendChild(mid);

    row.appendChild(el("div", "cand-score", Math.round(c.match_score) + "%"));
    list.appendChild(row);
  });
  $("rank-results").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------------------------------------------------------------- shared form plumbing
function setupDrop(dropId, inputId, textId, onChange, multiple) {
  const drop = $(dropId), input = $(inputId);
  const set = (files) => {
    drop.classList.toggle("filled", files.length > 0);
    const t = $(textId);
    t.innerHTML = "";
    if (files.length) {
      t.append(el("strong", "", multiple ? files.length + " resume" + (files.length > 1 ? "s" : "") + " selected" : files[0].name),
        document.createElement("br"), el("span", "muted", "Click to choose different file" + (multiple ? "s" : "")));
    } else {
      t.innerHTML = "<strong>Click to upload</strong> or drag &amp; drop";
    }
    onChange(files);
  };
  input.addEventListener("change", () => set([...input.files]));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => {
    const files = [...e.dataTransfer.files];
    if (!files.length) return;
    const dt = new DataTransfer();
    (multiple ? files : [files[0]]).forEach((f) => dt.items.add(f));
    input.files = dt.files;
    set([...input.files]);
  });
}

function loading(btnId, textId, on, idle) {
  document.querySelector("#" + btnId + " .spinner").hidden = !on;
  $(textId).textContent = on ? "Working…" : idle;
}

async function post(url, body) {
  const res = await fetch(url, { method: "POST", body });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || "Server error (" + res.status + ")");
  return data;
}

let file = null, rankFiles = [];
setupDrop("drop", "resume", "drop-text", (f) => { file = f[0] || null; $("submit").disabled = !file; }, false);
setupDrop("rank-drop", "rank-files", "rank-drop-text", (f) => { rankFiles = f; $("rank-submit").disabled = f.length < 2; }, true);

document.querySelectorAll("[data-sample]").forEach((b) =>
  b.addEventListener("click", () => { $("jd").value = SAMPLES[b.dataset.sample]; }));

document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
  $("tab-single").hidden = t.dataset.tab !== "single";
  $("tab-rank").hidden = t.dataset.tab !== "rank";
}));

$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!file) return;
  $("error").hidden = true;
  const body = new FormData();
  body.append("resume", file);
  body.append("job_description", $("jd").value);
  $("submit").disabled = true;
  loading("submit", "btn-text", true, "Analyze resume");
  try { render(await post("/api/analyze", body)); }
  catch (err) { $("error").textContent = err.message; $("error").hidden = false; }
  finally { loading("submit", "btn-text", false, "Analyze resume"); $("submit").disabled = !file; }
});

$("rank-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("rank-error").hidden = true;
  const body = new FormData();
  rankFiles.forEach((f) => body.append("resumes", f));
  body.append("job_description", $("rank-jd").value);
  $("rank-submit").disabled = true;
  loading("rank-submit", "rank-btn-text", true, "Rank candidates");
  try { renderRank(await post("/api/rank", body)); }
  catch (err) { $("rank-error").textContent = err.message; $("rank-error").hidden = false; }
  finally { loading("rank-submit", "rank-btn-text", false, "Rank candidates"); $("rank-submit").disabled = rankFiles.length < 2; }
});

// ---------------------------------------------------------------- model report
const f3 = (x) => (x === undefined ? "–" : x.toFixed(3));

function table(headers, rows) {
  const wrap = el("div", "scroll-x");
  const t = el("table");
  const thead = el("thead"), hr = el("tr");
  headers.forEach((h) => hr.appendChild(el("th", "", h)));
  thead.appendChild(hr);
  const tb = el("tbody");
  rows.forEach((r) => {
    const tr = el("tr");
    r.forEach((c) => tr.appendChild(el("td", "", c)));
    tb.appendChild(tr);
  });
  t.append(thead, tb);
  wrap.appendChild(t);
  return wrap;
}

function renderReport(m) {
  const body = $("report-body");
  body.innerHTML = "";
  body.classList.remove("muted");
  const c = m.classifier, r = m.matching;

  if (c) {
    body.appendChild(el("h4", "", "1. Resume → job category classifier"));
    body.appendChild(el("p", "muted", c.best_model + ": " + (c.test_accuracy * 100).toFixed(1) + "% accuracy, " +
      (c.test_f1_macro * 100).toFixed(1) + "% macro-F1 on a held-out test set (" + c.n_resumes + " resumes, " + c.n_categories + " categories)."));
    body.appendChild(table(["Model", "CV macro-F1"], c.comparison.map((x) => [x.model, ((x.tuned_cv_f1_macro ?? x.cv_f1_macro) * 100).toFixed(1) + "%"])));
  }
  if (r) {
    const d = r.data;
    body.appendChild(el("h4", "", "2. Compatibility prediction & candidate ranking"));
    body.appendChild(el("div", "callout",
      "Training data caveat: job descriptions are synthetic and relevance labels (0–3) are weak labels derived from resume categories, " +
      "not recruiter judgments. Split by candidate, job and skill pool. " + d.train_pairs + " train / " + d.test_pairs + " test pairs."));
    body.appendChild(table(
      ["Model", "ROC-AUC", "PR-AUC", "Brier", "NDCG@5", "NDCG@10", "Recall@10"],
      Object.entries(r.model_comparison).map(([n, x]) =>
        [n, f3(x.roc_auc), f3(x.pr_auc), f3(x.brier), f3(x["ndcg@5"]), f3(x["ndcg@10"]), f3(x["recall@10"])])));

    body.appendChild(el("h4", "", "Ablation: remove one feature group"));
    body.appendChild(table(["Removed", "ROC-AUC", "PR-AUC", "NDCG@10"],
      Object.entries(r.ablation).map(([n, x]) => [n, f3(x.roc_auc), f3(x.pr_auc), f3(x["ndcg@10"])])));

    body.appendChild(el("h4", "", "Bias check (identity text added to resumes)"));
    body.appendChild(table(["Variant", "Mean |Δ prob|", "Max |Δ prob|", "Decision flips"],
      Object.entries(r.bias_check.variants).map(([n, x]) => [n, f3(x.mean_abs_change), f3(x.max_abs_change), x.decision_flips])));
    body.appendChild(el("p", "note", "Protected attributes are never model inputs. Full report: models/experiment_report.md"));
  }
}

fetch("/api/metrics").then((r) => r.json()).then(renderReport).catch(() => {
  $("report-body").textContent = "Report unavailable.";
});
