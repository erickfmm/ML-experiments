/* ═══════════════════════════════════════════════════════════════
   Test Files Editor / Runner
   ═══════════════════════════════════════════════════════════════ */

let editor;          // CodeMirror instance
let currentFile = "";
let currentRunId = null;
let pollTimer = null;

// ── Initialise CodeMirror ───────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    editor = CodeMirror.fromTextArea(document.getElementById("code-editor"), {
        mode: "python",
        theme: "monokai",
        lineNumbers: true,
        matchBrackets: true,
        autoCloseBrackets: true,
        indentUnit: 4,
        tabSize: 4,
        indentWithTabs: false,
        lineWrapping: true,
    });
    loadFileList();
});

// ── File list ───────────────────────────────────────────────────
async function loadFileList() {
    const res = await fetch("/api/files");
    const files = await res.json();
    const container = document.getElementById("file-list");
    container.innerHTML = "";
    files.forEach(f => {
        const btn = document.createElement("button");
        btn.className = "file-btn";
        btn.textContent = f;
        btn.onclick = () => selectFile(f, btn);
        container.appendChild(btn);
    });
}

// ── Select a file ───────────────────────────────────────────────
async function selectFile(filename, btnEl) {
    // Highlight active button
    document.querySelectorAll(".file-btn").forEach(b => b.classList.remove("active"));
    if (btnEl) btnEl.classList.add("active");

    const res = await fetch(`/api/file/${encodeURIComponent(filename)}`);
    const data = await res.json();
    if (data.error) { alert(data.error); return; }

    currentFile = filename;
    document.getElementById("editor-title").textContent = `📄 ${filename}`;
    editor.setValue(data.content);
    editor.clearHistory();
}

// ── Save file ───────────────────────────────────────────────────
async function saveFile() {
    if (!currentFile) return;
    const res = await fetch(`/api/file/${encodeURIComponent(currentFile)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: editor.getValue() }),
    });
    const data = await res.json();
    if (data.status === "saved") {
        flash("💾 Saved!", "#28a745");
    }
}

// ── Run / Stop ──────────────────────────────────────────────────
async function toggleRun() {
    const btn = document.getElementById("run-btn");

    if (currentRunId) {
        // Stop
        await fetch(`/api/kill/${currentRunId}`, { method: "POST" });
        setStatus("idle");
        currentRunId = null;
        clearInterval(pollTimer);
        return;
    }

    if (!currentFile) { alert("Select a file first."); return; }

    // Auto-save before running
    await saveFile();

    const res = await fetch(`/api/run/${encodeURIComponent(currentFile)}`, {
        method: "POST",
    });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }

    currentRunId = data.run_id;
    setStatus("running");
    appendOutput(`\n--- Running ${currentFile} ---\n`);

    // Poll for output
    pollTimer = setInterval(pollOutput, 300);
}

async function pollOutput() {
    if (!currentRunId) return;
    const res = await fetch(`/api/output/${currentRunId}`);
    const data = await res.json();
    if (data.lines && data.lines.length) {
        appendOutput(data.lines.join(""));
        // Check if process exited
        const last = data.lines[data.lines.length - 1];
        if (last && last.includes("[Process exited")) {
            const codeMatch = last.match(/code (\d+)/);
            const code = codeMatch ? parseInt(codeMatch[1]) : 0;
            setStatus(code === 0 ? "done" : "error");
            clearInterval(pollTimer);
            currentRunId = null;
        }
    }
}

// ── Output helpers ──────────────────────────────────────────────
function appendOutput(text) {
    const el = document.getElementById("terminal-output");
    el.textContent += text;
    el.scrollTop = el.scrollHeight;
}

function clearOutput() {
    document.getElementById("terminal-output").textContent = "";
}

function setStatus(state) {
    const dot = document.getElementById("status-dot");
    const btn = document.getElementById("run-btn");
    dot.className = "status-dot " + state;
    if (state === "running") {
        btn.textContent = "⏹ Stop";
        btn.classList.add("running");
    } else {
        btn.textContent = "▶ Run";
        btn.classList.remove("running");
    }
}

function flash(msg, color) {
    const el = document.getElementById("editor-title");
    const orig = el.textContent;
    el.textContent = msg;
    el.style.color = color;
    setTimeout(() => { el.textContent = orig; el.style.color = ""; }, 1500);
}

// Keyboard shortcut: Ctrl+S to save, Ctrl+Enter to run
document.addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        saveFile();
    }
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        toggleRun();
    }
});
