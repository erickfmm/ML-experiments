/* ═══════════════════════════════════════════════════════════════
   Butterfly Segmentation
   ═══════════════════════════════════════════════════════════════ */

let pollTimer = null;
let currentRunId = null;

// ── Init ────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    checkStatus();
    setupDragDrop();
});

async function checkStatus() {
    const res = await fetch("/api/butterfly/status");
    const data = await res.json();

    const modelDot = document.getElementById("model-dot");
    const dataDot = document.getElementById("data-dot");
    const modelStatus = document.getElementById("model-status");
    const dataStatus = document.getElementById("data-status");

    if (data.model_exists) {
        modelDot.className = "dot green";
        modelStatus.textContent = "Model: ✅ Trained";
    } else {
        modelDot.className = "dot red";
        modelStatus.textContent = "Model: ❌ Not trained";
    }

    if (data.data_exists) {
        dataDot.className = "dot green";
        dataStatus.textContent = "Data: ✅ Pickle exists";
    } else {
        dataDot.className = "dot yellow";
        dataStatus.textContent = "Data: ⚠️ Need to download & pickle";
    }
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
    pollTimer = setInterval(async () => {
        const res = await fetch(`/api/output/${runId}`);
        const data = await res.json();
        if (data.lines && data.lines.length) {
            appendLog(data.lines.join(""));
            const last = data.lines[data.lines.length - 1];
            if (last && last.includes("[Process exited")) {
                clearInterval(pollTimer);
                currentRunId = null;
                checkStatus();
            }
        }
    }, 500);
}

// ── Actions ─────────────────────────────────────────────────────
async function savePickle() {
    clearLog();
    appendLog("📦 Saving data as pickle (this may take a while)...\n");
    disableButtons(true);

    // Run the butterfly test with BUTTERFLY_TASK=save_pickle
    const res = await fetch("/api/run/IMAGE_segmentation_clasification_butterfly.py", { method: "POST" });
    const data = await res.json();
    if (data.run_id) startPolling(data.run_id);
    else appendLog("Error: " + (data.error || "unknown"));
}

async function trainSegmentation() {
    clearLog();
    appendLog("🧠 Training segmentation model...\n");
    disableButtons(true);

    const res = await fetch("/api/butterfly/train_segmentation", { method: "POST" });
    const data = await res.json();
    if (data.run_id) startPolling(data.run_id);
    else appendLog("Error: " + (data.error || "unknown"));
}

async function trainClassifier() {
    clearLog();
    appendLog("🏷️ Training classifier model...\n");
    disableButtons(true);

    const res = await fetch("/api/butterfly/train_classifier", { method: "POST" });
    const data = await res.json();
    if (data.run_id) startPolling(data.run_id);
    else appendLog("Error: " + (data.error || "unknown"));
}

function disableButtons(disabled) {
    document.querySelectorAll(".btn").forEach(b => b.disabled = disabled);
}

// ── Image Prediction ────────────────────────────────────────────
async function predictImage(event) {
    const file = event.target.files[0];
    if (!file) return;

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

        // Show results
        document.getElementById("prediction-results").style.display = "grid";
        document.getElementById("img-original").src = "data:image/png;base64," + data.original;
        document.getElementById("img-predicted").src = "data:image/png;base64," + data.predicted;
        document.getElementById("img-cropped").src = "data:image/png;base64," + data.cropped;
        appendLog("✅ Prediction complete!\n");
    } catch (err) {
        appendLog("Error: " + err.message + "\n");
    }
}

// ── Drag & Drop ─────────────────────────────────────────────────
function setupDragDrop() {
    const area = document.getElementById("upload-area");

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
