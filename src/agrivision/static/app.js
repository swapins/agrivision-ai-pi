const els = {
  backend: document.getElementById('backend'),
  confidence: document.getElementById('confidence'),
  coralStatus: document.getElementById('coralStatus'),
  hum: document.getElementById('hum'),
  last: document.getElementById('last'),
  lastInference: document.getElementById('lastInference'),
  latency: document.getElementById('latency'),
  leaf: document.getElementById('leaf'),
  modeNote: document.getElementById('modeNote'),
  plant: document.getElementById('plant'),
  predictions: document.getElementById('predictions'),
  pump: document.getElementById('pump'),
  pumpOffBtn: document.getElementById('pumpOffBtn'),
  pumpReason: document.getElementById('pumpReason'),
  rawLabel: document.getElementById('rawLabel'),
  rawSoil: document.getElementById('rawSoil'),
  runtimeMode: document.getElementById('runtimeMode'),
  scanBtn: document.getElementById('scanBtn'),
  scanMessage: document.getElementById('scanMessage'),
  soil: document.getElementById('soil'),
  temp: document.getElementById('temp'),
};

const pageSimulation = document.body.dataset.simulation === 'true';

function healthClass(status) {
  return (status || 'neutral').toLowerCase().replaceAll(' ', '-');
}

function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = text ?? '';
  return d.innerHTML;
}

function percent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(1) + '%' : '-';
}

function renderPredictions(rows, simulation) {
  if (!rows || !rows.length) {
    els.predictions.innerHTML = '<span class="muted">No scan yet.</span>';
    return;
  }
  const prefix = simulation ? 'SIM ' : '';
  els.predictions.innerHTML = rows.map(
    (p) => `<div class="pred-row"><b>${prefix}${escapeHtml(p.label)}</b><b>${percent(p.confidence)}</b></div>`
  ).join('');
}

async function refresh() {
  try {
    const r = await fetch('/api/status', { cache: 'no-store' });
    const s = await r.json();
    const simulation = Boolean(s.simulation);

    els.plant.textContent = s.plant;
    els.plant.className = 'plant-status ' + healthClass(s.plant);
    els.runtimeMode.textContent = simulation ? 'SIMULATION ONLY' : 'Raspberry Pi hardware';
    els.backend.textContent = s.backend || '-';
    els.coralStatus.textContent = s.coral_status || '-';
    els.rawLabel.textContent = simulation && s.raw_label !== '-' ? 'SIM ' + s.raw_label : s.raw_label;
    els.confidence.textContent = percent(s.confidence);
    els.soil.textContent = percent(s.soil);
    els.rawSoil.textContent = s.raw_soil === null || s.raw_soil === undefined ? '-' : String(s.raw_soil);
    els.temp.textContent = Number.isFinite(Number(s.temperature)) ? Number(s.temperature).toFixed(1) + ' C' : '-';
    els.hum.textContent = percent(s.humidity);
    els.pump.textContent = s.pump;
    els.pumpReason.textContent = s.pump_reason || '-';
    els.last.textContent = s.last_scan;
    els.lastInference.textContent = s.last_inference_at || '-';
    els.latency.textContent = s.inference_latency_ms === null || s.inference_latency_ms === undefined ? '-' : Number(s.inference_latency_ms).toFixed(1) + ' ms';
    els.scanMessage.textContent = s.message;
    els.modeNote.textContent = simulation
      ? 'All displayed sensor, camera, and inference values are simulated.'
      : 'Hardware mode requires the published Edge TPU model and labels.';
    renderPredictions(s.top_predictions, simulation);
  } catch (e) {
    els.scanMessage.textContent = 'Dashboard connection error: ' + e;
  }
}

async function scanLeaf() {
  els.scanBtn.disabled = true;
  els.scanMessage.textContent = pageSimulation ? 'Capturing simulated image...' : 'Capturing and analysing on Coral Edge TPU...';
  try {
    const r = await fetch('/api/scan', { method: 'POST' });
    const payload = await r.json();
    if (!r.ok) throw new Error(payload.error || 'scan failed');
    els.leaf.style.opacity = 1;
    els.leaf.src = '/latest.jpg?t=' + Date.now();
    await refresh();
  } catch (e) {
    els.scanMessage.textContent = 'Scan failed: ' + e.message;
  } finally {
    els.scanBtn.disabled = false;
  }
}

async function pumpOff() {
  els.pumpOffBtn.disabled = true;
  try {
    const r = await fetch('/api/pump/off', { method: 'POST' });
    const payload = await r.json();
    if (!r.ok) throw new Error(payload.error || 'pump stop failed');
    els.scanMessage.textContent = 'Pump manually stopped.';
    await refresh();
  } catch (e) {
    els.scanMessage.textContent = 'Pump stop failed: ' + e.message;
  } finally {
    els.pumpOffBtn.disabled = false;
  }
}

setInterval(refresh, 1500);
refresh();
