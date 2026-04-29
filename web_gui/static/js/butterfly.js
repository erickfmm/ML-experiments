/* ═══════════════════════════════════════════════════════════════
   Butterfly Segmentation — Enhanced UI
   ═══════════════════════════════════════════════════════════════ */

let pollTimer = null;
let currentRunId = null;
let selectedButterflyFile = null;

// ── Init ────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    checkStatus();
    setupDragDrop();
});

// ── Card Toggle ─────────────────────────────────────────────────
function toggleCard(headerEl) {
    headerEl.classList.toggle("collapsed");
    const body = headerEl.nextElementSibling;
    body.classList.toggle("collapsed");
}

// ── Status ──────────────────────────────────────────────────────
async function checkStatus() {
    try {
        const res = await fetch("/api/butterfly/status");
        const data = await res.json();

        _setDot("dataset-dot", "dataset-status",
            data.dataset_exists, "Dataset: ✅ Downloaded", "Dataset: ❌ Not downloaded", true);

        _setDot("pickle-dot", "pickle-status",
            data.data_exists, "Pickle: ✅ Ready", "Pickle: ⚠️ Generate first");

        _setDot("seg-model-dot", "seg-model-status",
            data.seg_model_exists, "Segmentation: ✅ Trained", "Segmentation: ❌ Not trained");

        _setDot("cls-model-dot", "cls-model-status",
            data.cls_model_exists, "Classifier: ✅ Trained", "Classifier: ❌ Not trained");
    } catch (err) {
        console.error("Status check failed:", err);
    }
}

function _setDot(dotId, statusId, ok, okText, failText, isBlue) {
    const dot = document.getElementById(dotId);
    const status = document.getElementById(statusId);
    if (!dot || !status) return;
    dot.className = "dot " + (ok ? (isBlue ? "blue" : "green") : "red");
    status.textContent = ok ? okText : failText;
}

// ── Logging ─────────────────────────────────────────────────────
function appendLog(text) {
    const el = document.getElementById("log-output");
    el.textContent += text;
    el.scrollTop = el.scrollHeight;
}

function clearLog() {
    document.getElementById("log-output").textContent = "";
}

// ── Polling ─────────────────────────────────────────────────────
function startPolling(runId) {
    currentRunId = runId;
    const stopBtn = document.getElementById("btn-stop");
    if (stopBtn) stopBtn.classList.add("visible");
    pollTimer = setInterval(async () => {
        try {
            const res = await fetch(`/api/output/${runId}`);
            const data = await res.json();
            if (data.lines && data.lines.length) {
                appendLog(data.lines.join(""));
                const last = data.lines[data.lines.length - 1];
                if (last && last.includes("[Process exited")) {
                    clearInterval(pollTimer);
                    currentRunId = null;
                    if (stopBtn) stopBtn.classList.remove("visible");
                    checkStatus();
                    disableButtons(false);
                }
            }
        } catch (err) {
            console.error("Poll error:", err);
        }
    }, 500);
}

// ── Stop ────────────────────────────────────────────────────────
async function stopRun() {
    if (!currentRunId) return;
    await fetch(`/api/kill/${currentRunId}`, { method: "POST" });
    clearInterval(pollTimer);
    appendLog("\n--- Stopped by user ---\n");
    currentRunId = null;
    const stopBtn = document.getElementById("btn-stop");
    if (stopBtn) stopBtn.classList.remove("visible");
    disableButtons(false);
    checkStatus();
}

// ── Buttons ─────────────────────────────────────────────────────
function disableButtons(disabled) {
    document.querySelectorAll(".btn").forEach(b => {
        if (b.id !== "btn-stop") b.disabled = disabled;
    });
}

// ── 1. Download Dataset ─────────────────────────────────────────
async function downloadDataset() {
    clearLog();
    appendLog("⬇️ Downloading butterfly dataset from Kaggle...\n");
    disableButtons(true);
    document.getElementById("download-progress-container").style.display = "block";

    try {
        const res = await fetch("/api/butterfly/download", { method: "POST" });
        const data = await res.json();
        if (data.run_id) {
            startPolling(data.run_id);
        } else {
            appendLog("Error: " + (data.error || "unknown") + "\n");
            disableButtons(false);
        }
    } catch (err) {
        appendLog("Error: " + err.message + "\n");
        disableButtons(false);
    } finally {
        document.getElementById("download-progress-container").style.display = "none";
    }
}

// ── 2. Generate Pickle ──────────────────────────────────────────
async function savePickle() {
    clearLog();
    const imageSize = document.getElementById("hp-image-size").value;
    appendLog(`📦 Generating pickle (image size: ${imageSize}px)...\n`);
    disableButtons(true);

    try {
        const res = await fetch("/api/butterfly/save_pickle", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image_size: parseInt(imageSize) }),
        });
        const data = await res.json();
        if (data.run_id) {
            startPolling(data.run_id);
        } else {
            appendLog("Error: " + (data.error || "unknown") + "\n");
            disableButtons(false);
        }
    } catch (err) {
        appendLog("Error: " + err.message + "\n");
        disableButtons(false);
    }
}

// ── 3. Train Segmentation ───────────────────────────────────────
async function trainSegmentation() {
    clearLog();
    const params = {
        epochs: parseInt(document.getElementById("seg-epochs").value),
        batch_size: parseInt(document.getElementById("seg-batch-size").value),
        learning_rate: parseFloat(document.getElementById("seg-lr").value),
        test_split: parseFloat(document.getElementById("seg-test-split").value),
    };
    appendLog(`🧠 Training segmentation (epochs=${params.epochs}, batch=${params.batch_size}, lr=${params.learning_rate})...\n`);
    disableButtons(true);

    try {
        const res = await fetch("/api/butterfly/train_segmentation", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params),
        });
        const data = await res.json();
        if (data.run_id) {
            startPolling(data.run_id);
        } else {
            appendLog("Error: " + (data.error || "unknown") + "\n");
            disableButtons(false);
        }
    } catch (err) {
        appendLog("Error: " + err.message + "\n");
        disableButtons(false);
    }
}

// ── 4. Train Classifier ─────────────────────────────────────────
async function trainClassifier() {
    clearLog();
    const params = {
        epochs: parseInt(document.getElementById("cls-epochs").value),
        batch_size: parseInt(document.getElementById("cls-batch-size").value),
        learning_rate: parseFloat(document.getElementById("cls-lr").value),
        test_split: parseFloat(document.getElementById("cls-test-split").value),
    };
    appendLog(`🏷️ Training classifier (epochs=${params.epochs}, batch=${params.batch_size}, lr=${params.learning_rate})...\n`);
    disableButtons(true);

    try {
        const res = await fetch("/api/butterfly/train_classifier", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params),
        });
        const data = await res.json();
        if (data.run_id) {
            startPolling(data.run_id);
        } else {
            appendLog("Error: " + (data.error || "unknown") + "\n");
            disableButtons(false);
        }
    } catch (err) {
        appendLog("Error: " + err.message + "\n");
        disableButtons(false);
    }
}

// ── 5. Predict Image ────────────────────────────────────────────
async function predictImage(event) {
    const file = event?.target?.files?.[0] || selectedButterflyFile;
    if (!file) return;
    selectedButterflyFile = file;

    appendLog("\n🔍 Predicting segmentation...\n");

    const formData = new FormData();
    formData.append("image", file);

    try {
        const res = await fetch("/api/butterfly/predict", {
            method: "POST",
            body: formData,
        });
        const data = await res.json();

        if (data.error) {
            appendLog("Error: " + data.error + "\n");
            return;
        }

        document.getElementById("prediction-results").style.display = "grid";
        document.getElementById("img-original").src = "data:image/png;base64," + data.original;
        document.getElementById("img-predicted").src = "data:image/png;base64," + data.predicted;
        document.getElementById("img-cropped").src = "data:image/png;base64," + data.cropped;
        appendLog("✅ Prediction complete!\n");
    } catch (err) {
        appendLog("Error: " + err.message + "\n");
    }
}

async function classifyImage() {
    if (!selectedButterflyFile) {
        appendLog("\n⚠️ Select an image first to classify it.\n");
        return;
    }

    appendLog("\n🏷️ Classifying butterfly image...\n");

    const formData = new FormData();
    formData.append("image", selectedButterflyFile);

    try {
        const res = await fetch("/api/butterfly/classify", {
            method: "POST",
            body: formData,
        });
        const data = await res.json();

        if (data.error) {
            appendLog("Error: " + data.error + "\n");
            return;
        }

        renderClassificationResult(data);
        appendLog(`✅ Classified as ${data.label} (${(data.confidence * 100).toFixed(1)}%)\n`);
    } catch (err) {
        appendLog("Error: " + err.message + "\n");
    }
}

function renderClassificationResult(data) {
    const panel = document.getElementById("classification-panel");
    const labelEl = document.getElementById("classification-label");
    const confidenceEl = document.getElementById("classification-confidence");
    const listEl = document.getElementById("classification-top-list");

    if (!panel || !labelEl || !confidenceEl || !listEl) return;

    panel.style.display = "block";
    labelEl.textContent = data.label || "Unknown";
    confidenceEl.textContent = `Confidence: ${((data.confidence || 0) * 100).toFixed(1)}%`;

    listEl.innerHTML = "";
    (data.top_predictions || []).forEach((item, index) => {
        const row = document.createElement("li");
        row.className = "classification-item";

        const rank = document.createElement("span");
        rank.className = "classification-rank";
        rank.textContent = `#${index + 1}`;

        const label = document.createElement("span");
        label.className = "classification-item-label";
        label.textContent = item.label || `Class ${item.class_index}`;

        const confidence = document.createElement("span");
        confidence.className = "classification-item-confidence";
        confidence.textContent = `${((item.confidence || 0) * 100).toFixed(1)}%`;

        row.appendChild(rank);
        row.appendChild(label);
        row.appendChild(confidence);
        listEl.appendChild(row);
    });
}

// ── Drag & Drop ─────────────────────────────────────────────────
function setupDragDrop() {
    const area = document.getElementById("upload-area");
    if (!area) return;

    area.addEventListener("dragover", e => {
        e.preventDefault();
        area.classList.add("dragover");
    });
    area.addEventListener("dragleave", () => {
        area.classList.remove("dragover");
    });
    area.addEventListener("drop", e => {
        e.preventDefault();
        area.classList.remove("dragover");
        if (e.dataTransfer.files.length) {
            const input = document.getElementById("file-input");
            input.files = e.dataTransfer.files;
            predictImage({ target: input });
        }
    });
}
