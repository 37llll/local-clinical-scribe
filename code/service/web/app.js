const demoPayload = {
  encounter_id: "demo-001",
  template: "soap",
  segments: [
    {
      id: "seg_0001",
      start: 0.0,
      end: 3.2,
      speaker: "patient",
      text: "我这两天咳嗽，晚上有点发热。"
    },
    {
      id: "seg_0002",
      start: 3.5,
      end: 8.1,
      speaker: "doctor",
      text: "建议先做血常规和胸片检查，注意休息，多喝水。"
    }
  ]
};

const state = {
  encounterId: null,
  draft: null,
  sections: [],
  capabilities: null
};

const el = (id) => document.getElementById(id);

function setStatus(message, tone = "muted") {
  const node = el("draftStatus");
  node.textContent = message;
  node.dataset.tone = tone;
}

function prettyJson(value) {
  return JSON.stringify(value, null, 2);
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options
  });
  const text = await response.text();
  const body = text ? JSON.parse(text) : {};
  if (!response.ok || body.success === false) {
    throw new Error(body.error || response.statusText);
  }
  return body;
}

function showView(name) {
  document.querySelectorAll(".view").forEach((node) => {
    node.classList.toggle("active", node.id === `${name}View`);
  });
  document.querySelectorAll(".nav-item").forEach((node) => {
    node.classList.toggle("active", node.dataset.view === name);
  });
  const titleMap = {
    workspace: ["Workspace", "Draft, review, finalize, export."],
    encounters: ["Encounters", "Saved local records."],
    settings: ["Runtime", "Available local capabilities."]
  };
  const [title, subtitle] = titleMap[name];
  el("viewTitle").textContent = title;
  el("viewSubtitle").textContent = subtitle;
}

async function loadCapabilities() {
  try {
    const data = await requestJson("/capabilities");
    state.capabilities = data;
    el("serviceState").textContent = "Service ready";
    el("coreStatus").textContent = data.core?.clinical_note_drafting ? "Ready" : "Off";
    const audio = data.optional_audio || {};
    const audioReady = Object.values(audio).some(Boolean);
    el("audioStatus").textContent = audioReady ? "Available" : "Core only";
    el("capabilitiesOutput").textContent = prettyJson(data);
  } catch (error) {
    el("serviceState").textContent = "Service unavailable";
    el("coreStatus").textContent = "Offline";
    el("audioStatus").textContent = "Offline";
    el("capabilitiesOutput").textContent = error.message;
  }
}

function loadDemo() {
  el("encounterId").value = demoPayload.encounter_id;
  el("transcriptInput").value = prettyJson(demoPayload.segments);
  setStatus("Demo loaded.");
}

function clearWorkspace() {
  state.encounterId = null;
  state.draft = null;
  state.sections = [];
  el("encounterId").value = "";
  el("transcriptInput").value = "";
  el("exportOutput").value = "";
  el("activeEncounterChip").textContent = "No encounter";
  el("sectionsEditor").className = "sections-editor empty-state";
  el("sectionsEditor").textContent = "No draft loaded.";
  el("finalizeButton").disabled = true;
  el("exportMarkdownButton").disabled = true;
  el("exportJsonButton").disabled = true;
  setStatus("");
}

function parseTranscriptInput() {
  const raw = el("transcriptInput").value.trim();
  if (!raw) {
    throw new Error("Transcript JSON is empty.");
  }
  const parsed = JSON.parse(raw);
  if (Array.isArray(parsed)) {
    return parsed;
  }
  if (Array.isArray(parsed.segments)) {
    return parsed.segments;
  }
  throw new Error("Transcript JSON must be an array or an object with segments.");
}

async function generateDraft() {
  try {
    setStatus("Generating draft...");
    const segments = parseTranscriptInput();
    const encounterId = el("encounterId").value.trim() || undefined;
    const payload = {
      encounter_id: encounterId,
      template: "soap",
      segments
    };
    const data = await requestJson("/api/clinical_note/from_transcript", {
      method: "POST",
      body: JSON.stringify(payload)
    });
    state.encounterId = data.encounter_id;
    state.draft = data.clinical_note;
    state.sections = data.clinical_note.sections || [];
    renderSections();
    await loadEncounters();
    setStatus("Draft saved.");
  } catch (error) {
    setStatus(error.message, "danger");
  }
}

function renderSections() {
  const container = el("sectionsEditor");
  container.className = "sections-editor";
  container.innerHTML = "";
  el("activeEncounterChip").textContent = state.encounterId || "No encounter";

  state.sections.forEach((section, index) => {
    const block = document.createElement("article");
    block.className = "section-editor";
    const evidence = section.evidence || [];
    block.innerHTML = `
      <div class="section-header">
        <strong>${escapeHtml(section.title || section.key)}</strong>
        <span class="status-pill ${escapeHtml(section.status || "")}">
          ${escapeHtml(section.status || "draft")}
        </span>
      </div>
      <textarea class="textarea" data-section-index="${index}">${escapeHtml(section.content || "")}</textarea>
      <ul class="evidence-list">
        ${evidence.map((item) => `
          <li>
            <strong>${escapeHtml(item.segment_id || "")}</strong>
            ${escapeHtml(item.speaker || "unknown")}:
            ${escapeHtml(item.quote || "")}
          </li>
        `).join("")}
      </ul>
    `;
    container.appendChild(block);
  });

  container.querySelectorAll("textarea").forEach((textarea) => {
    textarea.addEventListener("input", (event) => {
      const index = Number(event.target.dataset.sectionIndex);
      state.sections[index].content = event.target.value;
      state.sections[index].status = state.sections[index].status === "missing"
        ? "missing"
        : "reviewed";
    });
  });

  el("finalizeButton").disabled = !state.encounterId;
  el("exportMarkdownButton").disabled = !state.encounterId;
  el("exportJsonButton").disabled = !state.encounterId;
}

async function finalizeEncounter() {
  if (!state.encounterId) return;
  try {
    setStatus("Finalizing...");
    const reviewedSections = state.sections
      .filter((section) => section.status !== "missing")
      .map((section) => ({ ...section, status: "reviewed" }));
    const payload = {
      encounter_id: state.encounterId,
      reviewer: el("reviewerInput").value.trim() || undefined,
      review_notes: el("reviewNotesInput").value.trim() || undefined,
      sections: reviewedSections
    };
    const data = await requestJson("/api/clinical_note/finalize", {
      method: "POST",
      body: JSON.stringify(payload)
    });
    state.draft = data.encounter.draft;
    state.sections = data.encounter.finalized_note?.sections || state.sections;
    renderSections();
    await exportEncounter("markdown");
    await loadEncounters();
    setStatus("Final note saved.");
  } catch (error) {
    setStatus(error.message, "danger");
  }
}

async function exportEncounter(format) {
  if (!state.encounterId) return;
  try {
    const response = await fetch(`/api/clinical_note/encounters/${encodeURIComponent(state.encounterId)}/export?format=${format}`);
    const text = await response.text();
    if (!response.ok) {
      throw new Error(text);
    }
    if (format === "json") {
      el("exportOutput").value = prettyJson(JSON.parse(text));
    } else {
      el("exportOutput").value = text;
    }
  } catch (error) {
    el("exportOutput").value = error.message;
  }
}

async function loadEncounters() {
  const container = el("encounterList");
  try {
    const data = await requestJson("/api/clinical_note/encounters");
    const encounters = data.encounters || [];
    if (!encounters.length) {
      container.className = "empty-state";
      container.textContent = "No encounters saved.";
      return;
    }
    container.className = "encounter-list";
    container.innerHTML = "";
    encounters.forEach((item) => {
      const row = document.createElement("div");
      row.className = "encounter-item";
      row.innerHTML = `
        <div>
          <strong>${escapeHtml(item.encounter_id)}</strong>
          <span>${escapeHtml(item.status)} - ${escapeHtml(item.updated_at)}</span>
        </div>
        <button class="button secondary" type="button">Open</button>
      `;
      row.querySelector("button").addEventListener("click", () => openEncounter(item.encounter_id));
      container.appendChild(row);
    });
  } catch (error) {
    container.className = "empty-state";
    container.textContent = error.message;
  }
}

async function openEncounter(encounterId) {
  try {
    const data = await requestJson(`/api/clinical_note/encounters/${encodeURIComponent(encounterId)}`);
    const record = data.encounter;
    state.encounterId = record.encounter_id;
    state.draft = record.draft;
    state.sections = record.finalized_note?.sections || record.draft?.sections || [];
    el("encounterId").value = record.encounter_id;
    el("transcriptInput").value = prettyJson(record.transcript?.segments || record.transcript || []);
    renderSections();
    await exportEncounter(record.status === "finalized" ? "markdown" : "json");
    showView("workspace");
    setStatus("Encounter loaded.");
  } catch (error) {
    setStatus(error.message, "danger");
  }
}

async function copyExport() {
  const value = el("exportOutput").value;
  if (!value) return;
  await navigator.clipboard.writeText(value);
  setStatus("Export copied.");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function bindEvents() {
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => showView(button.dataset.view));
  });
  el("refreshButton").addEventListener("click", async () => {
    await loadCapabilities();
    await loadEncounters();
  });
  el("loadDemoButton").addEventListener("click", loadDemo);
  el("draftButton").addEventListener("click", generateDraft);
  el("clearButton").addEventListener("click", clearWorkspace);
  el("finalizeButton").addEventListener("click", finalizeEncounter);
  el("exportMarkdownButton").addEventListener("click", () => exportEncounter("markdown"));
  el("exportJsonButton").addEventListener("click", () => exportEncounter("json"));
  el("copyExportButton").addEventListener("click", copyExport);
  el("reloadEncountersButton").addEventListener("click", loadEncounters);
}

async function init() {
  bindEvents();
  loadDemo();
  await loadCapabilities();
  await loadEncounters();
}

init();
