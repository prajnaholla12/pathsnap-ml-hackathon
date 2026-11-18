// PathSnap evaluator UI (Laptop 2). Talks to the FastAPI server on Laptop 1.
// The UI only sends a trip id; everything else happens on the API.

let API = window.PATHSNAP_API || "http://127.0.0.1:8000";
try { API = localStorage.getItem("pathsnap_api") || API; } catch (e) { /* storage blocked */ }

const $ = (id) => document.getElementById(id);
$("api").value = API;

// ---- map -------------------------------------------------------------
const map = L.map("map", { preferCanvas: true }).setView([20, 0], 2);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

const layers = {
  network: L.layerGroup(), gps: L.layerGroup(), clf: L.layerGroup(), route: L.layerGroup(),
};
layers.network.addTo(map); layers.gps.addTo(map); layers.route.addTo(map);
L.control.layers(null, {
  "Road network (candidates)": layers.network,
  "GPS observations": layers.gps,
  "Classifier only (no HMM)": layers.clf,
  "Predicted route (HMM)": layers.route,
}, { collapsed: false }).addTo(map);

const legend = L.control({ position: "bottomleft" });
legend.onAdd = () => {
  const d = L.DomUtil.create("div", "legend");
  d.innerHTML = '<i style="background:#9aa3ab"></i>road network<br>' +
    '<i class="dot" style="background:#1f4fd1"></i>GPS points<br>' +
    '<i style="background:#e67e22"></i>classifier only<br>' +
    '<i style="background:#d62728"></i>predicted route (HMM)';
  return d;
};
legend.addTo(map);

// ---- helpers ---------------------------------------------------------
function addLog(msg, err) {
  const li = document.createElement("li");
  li.textContent = new Date().toLocaleTimeString() + "  " + msg;
  if (err) li.className = "err";
  $("log").prepend(li);
}
const pct = (x) => (x == null || isNaN(x) ? "–" : (100 * x).toFixed(2) + "%");
const num = (x, d = 3) => (x == null || isNaN(x) ? "–" : Number(x).toFixed(d));

async function call(path, opts) {
  const r = await fetch(API + path, opts);
  if (!r.ok) throw new Error(r.status + " " + (await r.text()));
  return r.json();
}

// ---- connect + trip list --------------------------------------------
async function connect() {
  API = $("api").value.trim().replace(/\/$/, "");
  try { localStorage.setItem("pathsnap_api", API); } catch (e) { /* ignore */ }
  $("status").textContent = "connecting…";
  try {
    const h = await call("/health");
    $("status").textContent = `connected – model ${h.model}`;
    $("status").className = "status ok";
    addLog("GET /health ok");
    const t = await call("/trips");
    addLog(`GET /trips -> ${t.trips.length} trips`);
    const box = $("trips");
    box.innerHTML = "";
    t.trips.forEach((tr) => {
      const b = document.createElement("button");
      b.className = "trip";
      b.innerHTML = `<b>${tr.trip_id}</b><small>${tr.n_points} GPS points</small>`;
      b.onclick = () => runTrip(tr.trip_id, b);
      box.appendChild(b);
    });
  } catch (e) {
    $("status").textContent = "could not reach API: " + e.message;
    $("status").className = "status off";
    addLog("connect failed: " + e.message, true);
  }
}

// ---- prediction -----------------------------------------------------
async function runTrip(tid, btn) {
  document.querySelectorAll(".trip").forEach((x) => x.classList.remove("active"));
  btn.classList.add("active");
  $("busy").hidden = false;
  addLog(`POST /predict trip=${tid}`);
  const t0 = performance.now();
  try {
    const res = await call("/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trip_id: tid }),
    });
    const ms = performance.now() - t0;
    addLog(`<- ${tid}: top-1 ${pct(res.metrics.hmm_top1_accuracy)} in ${(ms / 1000).toFixed(2)} s`);
    draw(res);
    showMetrics(res);
  } catch (e) {
    addLog("predict failed: " + e.message, true);
  } finally {
    $("busy").hidden = true;
  }
}

function draw(res) {
  Object.values(layers).forEach((l) => l.clearLayers());
  res.network.forEach((s) => L.polyline(s, { color: "#9aa3ab", weight: 1, opacity: 0.7 }).addTo(layers.network));
  const step = Math.max(1, Math.floor(res.gps.length / 2500));
  for (let i = 0; i < res.gps.length; i += step) {
    L.circleMarker(res.gps[i], { radius: 1.6, color: "#1f4fd1", weight: 0, fillOpacity: 0.55 })
      .bindTooltip("#" + i).addTo(layers.gps);
  }
  res.classifier_route.forEach((ln) =>
    L.polyline(ln, { color: "#e67e22", weight: 3, opacity: 0.9, dashArray: "4 6" }).addTo(layers.clf));
  res.predicted_route.forEach((ln) =>
    L.polyline(ln, { color: "#d62728", weight: 4, opacity: 0.9 }).addTo(layers.route));
  map.fitBounds(L.latLngBounds(res.gps), { padding: [30, 30] });
}

function showMetrics(res) {
  const m = res.metrics;
  $("resultBox").hidden = false;
  $("resTrip").textContent = res.trip_id;
  $("kAcc").textContent = pct(m.hmm_top1_accuracy);
  $("kClf").textContent = pct(m.classifier_top1_accuracy);
  $("kF1").textContent = num(m.hmm_macro_f1, 4);
  $("kCont").textContent = num(m.route_f1);
  const t = res.timing_s;
  const rows = [
    ["GPS readings", res.n_points],
    ["Classifier-only macro F1", num(m.classifier_macro_f1, 4)],
    ["Route precision / recall (length)", `${num(m.route_precision)} / ${num(m.route_recall)}`],
    ["Direct continuity: classifier → HMM", `${pct(m.classifier_direct_continuity)} → ${pct(m.hmm_direct_continuity)}`],
    ["Candidate miss rate", pct(m.candidate_miss_rate)],
    ["Viterbi breaks", m.viterbi_breaks],
    ["Predicted / reference route arcs", `${m.predicted_route_arcs} / ${m.reference_route_arcs}`],
    ["Candidates + features", t.candidates_features + " s"],
    ["Classifier", t.classifier + " s"],
    ["Viterbi", t.viterbi + " s"],
    ["Total on API", t.total + " s"],
  ];
  $("details").innerHTML = rows.map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join("");
}

$("connect").onclick = connect;
$("api").addEventListener("keydown", (e) => { if (e.key === "Enter") connect(); });
connect();
