import "@google/model-viewer";

const views = [
  { id: "front", label: "Front view", detail: "Required · face-on", required: true },
  { id: "left", label: "Left profile", detail: "Required · temple visible", required: true },
  { id: "right", label: "Right profile", detail: "Required · temple visible", required: true },
  { id: "back", label: "Back view", detail: "Optional · inner structure", required: false },
  { id: "hero", label: "Hero image", detail: "Optional · thumbnail", required: false },
];

const stages = [
  ["validating", "Validate source images"], ["segmenting", "Isolate frame contours"],
  ["landmarking", "Detect eyewear landmarks"], ["deforming", "Deform symmetric template"],
  ["projecting", "Project multi-view texture"], ["painting", "Generate PBR materials"],
  ["optimizing", "Optimize web asset"], ["uploading", "Publish preview"],
];

const form = document.querySelector("#generation-form");
const grid = document.querySelector("#upload-grid");
const template = document.querySelector("#upload-template");
const message = document.querySelector("#form-message");
const generateButton = document.querySelector("#generate-button");
const stageList = document.querySelector("#stage-list");
const statusTitle = document.querySelector("#status-title");
const progressValue = document.querySelector("#progress-value");
const progressBar = document.querySelector("#progress-bar");
const details = document.querySelector("#job-details");
const reviewPanel = document.querySelector("#review-panel");
const viewer = document.querySelector("#asset-viewer");
const finalizeButton = document.querySelector("#finalize-button");
const finalizeMessage = document.querySelector("#finalize-message");
let activeJob = null;
let pollTimer = null;

for (const view of views) {
  const card = template.content.firstElementChild.cloneNode(true);
  const input = card.querySelector("input");
  input.name = view.id;
  input.required = view.required;
  card.querySelector("strong").textContent = view.label;
  card.querySelector("small").textContent = view.detail;
  input.setAttribute("aria-label", `${view.label} image`);
  input.addEventListener("change", () => showPreview(card, input.files?.[0]));
  for (const event of ["dragenter", "dragover"]) card.addEventListener(event, () => card.classList.add("dragging"));
  for (const event of ["dragleave", "drop"]) card.addEventListener(event, () => card.classList.remove("dragging"));
  grid.append(card);
}

stageList.innerHTML = stages.map(([id, label]) => `<li data-stage="${id}">${label}</li>`).join("");

function showPreview(card, file) {
  if (!file) return;
  const preview = card.querySelector(".upload-preview");
  if (preview.dataset.url) URL.revokeObjectURL(preview.dataset.url);
  const url = URL.createObjectURL(file);
  preview.dataset.url = url;
  preview.style.backgroundImage = `url("${url}")`;
  card.classList.add("has-image");
  card.querySelector(".upload-action").textContent = "Replace image";
}

function apiHeaders() {
  const key = document.querySelector("#api-key").value.trim();
  return key ? { "X-API-Key": key } : {};
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  message.textContent = "";
  if (!form.reportValidity()) return;
  generateButton.disabled = true;
  generateButton.firstChild.textContent = "Uploading… ";
  try {
    const response = await fetch("/v1/jobs", { method: "POST", headers: apiHeaders(), body: new FormData(form) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail?.message || payload.detail || "Upload failed");
    activeJob = payload;
    details.hidden = false;
    document.querySelector("#job-id").textContent = payload.id;
    updateJob(payload);
    schedulePoll();
  } catch (error) {
    message.textContent = error instanceof Error ? error.message : "Could not create the job.";
    generateButton.disabled = false;
    generateButton.firstChild.textContent = "Generate 3D asset ";
  }
});

function schedulePoll() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(pollJob, 1800);
}

async function pollJob() {
  if (!activeJob) return;
  try {
    const response = await fetch(`/v1/jobs/${activeJob.id}`, { headers: apiHeaders() });
    if (!response.ok) throw new Error("Status request failed");
    const job = await response.json();
    activeJob = job;
    updateJob(job);
    if (!["ready_for_review", "failed", "cancelled", "finalized"].includes(job.status)) schedulePoll();
  } catch (error) {
    message.textContent = error instanceof Error ? error.message : "Could not refresh job status.";
    schedulePoll();
  }
}

function updateJob(job) {
  const progress = Math.max(0, Math.min(100, job.progress || 0));
  progressValue.textContent = `${progress}%`;
  progressBar.style.width = `${progress}%`;
  statusTitle.textContent = job.status.replaceAll("_", " ");
  document.querySelector("#job-status").textContent = job.status;
  const currentIndex = ["ready_for_review", "finalized"].includes(job.status)
    ? stages.length
    : stages.findIndex(([id]) => id === job.status);
  for (const [index, item] of [...stageList.children].entries()) {
    item.className = index < currentIndex ? "complete" : index === currentIndex ? "active" : "";
  }
  if (job.status === "failed") {
    message.textContent = job.error?.message || "Generation failed.";
    generateButton.disabled = false;
    generateButton.firstChild.textContent = "Try again ";
  }
  if (["ready_for_review", "finalized"].includes(job.status) && job.outputs?.preview_glb_url) {
    viewer.src = job.outputs.preview_glb_url;
    reviewPanel.hidden = false;
    generateButton.firstChild.textContent = "Asset generated ";
    reviewPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  if (job.status === "finalized") {
    finalizeButton.disabled = true;
    finalizeButton.textContent = "Asset synced";
  }
}

document.querySelector("#lighting").addEventListener("change", (event) => {
  viewer.environmentImage = event.target.value;
});

finalizeButton.addEventListener("click", async () => {
  if (!activeJob) return;
  finalizeButton.disabled = true;
  finalizeMessage.textContent = "Syncing asset…";
  try {
    const response = await fetch(`/v1/jobs/${activeJob.id}/finalize`, { method: "POST", headers: apiHeaders() });
    const job = await response.json();
    if (!response.ok) throw new Error(job.detail || "Finalization failed");
    activeJob = job;
    updateJob(job);
    finalizeMessage.textContent = "Asset saved and synced.";
  } catch (error) {
    finalizeMessage.textContent = error instanceof Error ? error.message : "Could not finalize asset.";
    finalizeButton.disabled = false;
  }
});
