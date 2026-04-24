/* ═══════════════════════════════════════════════════════════════
   Psychoacoustic Plots
   ═══════════════════════════════════════════════════════════════ */

let barkChart, erbChart, melChart, soneChart;

document.addEventListener("DOMContentLoaded", () => {
    loadPlots();
});

async function loadPlots() {
    const params = new URLSearchParams({
        min: document.getElementById("min-val").value,
        max: document.getElementById("max-val").value,
        step: document.getElementById("step-val").value,
        max_sone: document.getElementById("max-sone").value,
    });

    const res = await fetch(`/api/psychoacoustic?${params}`);
    const data = await res.json();

    const chartDefaults = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { labels: { color: "#ccc" } },
        },
        scales: {
            x: {
                ticks: { color: "#999" },
                grid: { color: "#444" },
            },
            y: {
                ticks: { color: "#999" },
                grid: { color: "#444" },
            },
        },
    };

    // ── Bark ──────────────────────────────────────────────
    if (barkChart) barkChart.destroy();
    barkChart = new Chart(document.getElementById("bark-chart"), {
        type: "line",
        data: {
            labels: data.x,
            datasets: [
                { label: "7000", data: data.bark["7000"], borderColor: "red", borderWidth: 1.5, pointRadius: 0 },
                { label: "1990 Traunmuller", data: data.bark["1990Traunmuller"], borderColor: "blue", borderWidth: 1.5, pointRadius: 0 },
                { label: "1992 Wang", data: data.bark["1992Wang"], borderColor: "green", borderWidth: 1.5, pointRadius: 0 },
                { label: "7500", data: data.bark["7500"], borderColor: "black", borderWidth: 1.5, pointRadius: 0 },
            ],
        },
        options: { ...chartDefaults, plugins: { ...chartDefaults.plugins, title: { display: false } } },
    });

    // ── ERB ───────────────────────────────────────────────
    if (erbChart) erbChart.destroy();
    erbChart = new Chart(document.getElementById("erb-chart"), {
        type: "line",
        data: {
            labels: data.x,
            datasets: [
                { label: "Linear", data: data.erb.linear, borderColor: "red", borderWidth: 1.5, pointRadius: 0 },
                { label: "Poly 2nd", data: data.erb.poly2nd, borderColor: "black", borderWidth: 1.5, pointRadius: 0 },
                { label: "Matlab", data: data.erb.matlab, borderColor: "orange", borderWidth: 1.5, pointRadius: 0 },
            ],
        },
        options: chartDefaults,
    });

    // ── MEL ───────────────────────────────────────────────
    if (melChart) melChart.destroy();
    melChart = new Chart(document.getElementById("mel-chart"), {
        type: "line",
        data: {
            labels: data.x,
            datasets: [
                { label: "700", data: data.mel["700"], borderColor: "red", borderWidth: 1.5, pointRadius: 0 },
                { label: "1000", data: data.mel["1000"], borderColor: "blue", borderWidth: 1.5, pointRadius: 0 },
                { label: "625", data: data.mel["625"], borderColor: "green", borderWidth: 1.5, pointRadius: 0 },
            ],
        },
        options: chartDefaults,
    });

    // ── Sone ──────────────────────────────────────────────
    if (soneChart) soneChart.destroy();
    soneChart = new Chart(document.getElementById("sone-chart"), {
        type: "line",
        data: {
            labels: data.x_sone,
            datasets: [
                { label: "Sone", data: data.sone.sone, borderColor: "red", borderWidth: 1.5, pointRadius: 0 },
                { label: "Approximation", data: data.sone.approximation, borderColor: "blue", borderWidth: 1.5, pointRadius: 0 },
            ],
        },
        options: chartDefaults,
    });
}
