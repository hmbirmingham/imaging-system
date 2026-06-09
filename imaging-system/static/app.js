/* app.js — Plate Imaging System frontend */

'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
let selectedFilename = null;
let allCaptures      = [];
let debounceTimers   = {};

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  connectSSE();
  bindNavigation();
  bindCapture();
  bindQuantify();
  bindLED();
  bindImageControls();
  bindCameraSettings();
  bindLibrary();
  loadRecentThumbs();
});

// ── SSE ───────────────────────────────────────────────────────────────────────
function connectSSE() {
  const es = new EventSource('/api/events');

  es.addEventListener('init', e => {
    const d = JSON.parse(e.data);
    if (!d.camera) setStatus('No camera connected — capture disabled');
    if (!d.led)    console.info('LED hardware not available');
    if (d.led)     syncLED(d.led_brightness);
  });

  es.addEventListener('capture', e => {
    const d = JSON.parse(e.data);
    setStatus(`Captured: ${d.filename}`);
    addThumb(d.filename);
    selectCapture(d.filename);
    refreshLibraryIfOpen();
  });

  es.addEventListener('status', e => {
    setStatus(JSON.parse(e.data).message);
  });

  es.addEventListener('quantify_done', e => {
    const d = JSON.parse(e.data);
    renderResults(d);
    setStatus(`Analysis complete — ${d.count} colonies, ${d.anomaly_count} flagged`);
    refreshLibraryIfOpen();
  });

  es.addEventListener('error_event', e => {
    const d = JSON.parse(e.data);
    setStatus(`Error: ${d.message}`);
    showResultsError(d.message);
  });

  es.onerror = () => {
    console.warn('SSE disconnected — retrying…');
  };
}

// ── Navigation ────────────────────────────────────────────────────────────────
function bindNavigation() {
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
      btn.classList.add('active');
      const view = document.getElementById(`view-${btn.dataset.view}`);
      if (view) {
        view.classList.add('active');
        if (btn.dataset.view === 'library') loadLibrary();
      }
    });
  });
}

// ── Capture ───────────────────────────────────────────────────────────────────
function bindCapture() {
  document.getElementById('capture-btn')?.addEventListener('click', async () => {
    setStatus('Capturing…');
    const r = await fetch('/api/capture', { method: 'POST' });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      setStatus(d.message || 'Capture failed');
    }
    // success handled via SSE
  });
}

// ── Quantify ──────────────────────────────────────────────────────────────────
function bindQuantify() {
  document.getElementById('quantify-btn')?.addEventListener('click', async () => {
    if (!selectedFilename) return;
    showResultsSpinner();
    setStatus('Starting analysis…');
    await fetch('/api/quantify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: selectedFilename }),
    });
    // result handled via SSE quantify_done
  });
}

// ── LED brightness ────────────────────────────────────────────────────────────
function bindLED() {
  const slider = document.getElementById('led-slider');
  const val    = document.getElementById('led-val');
  if (!slider) return;

  slider.addEventListener('input', () => {
    val.textContent = `${slider.value}%`;
  });

  // Send on release only (avoid lag from continuous writes)
  slider.addEventListener('change', async () => {
    val.textContent = `${slider.value}%`;
    await fetch('/api/led', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ brightness: parseInt(slider.value) }),
    });
  });
}

function syncLED(pct) {
  const slider = document.getElementById('led-slider');
  const val    = document.getElementById('led-val');
  if (slider) slider.value = pct;
  if (val)    val.textContent = `${pct}%`;
}

// ── Image controls (brightness / contrast / saturation) ──────────────────────
function bindImageControls() {
  const controls = [
    { id: 'brightness-slider', valId: 'brightness-val',
      fmt: v => String(Math.round(v)),
      key: 'brightness', scale: v => v / 100 },
    { id: 'contrast-slider',   valId: 'contrast-val',
      fmt: v => `${(v/100).toFixed(1)}×`,
      key: 'contrast',   scale: v => v / 100 },
    { id: 'saturation-slider', valId: 'saturation-val',
      fmt: v => `${(v/100).toFixed(1)}×`,
      key: 'saturation', scale: v => v / 100 },
  ];

  controls.forEach(({ id, valId, fmt, key, scale }) => {
    const el = document.getElementById(id);
    const lbl = document.getElementById(valId);
    if (!el) return;

    el.addEventListener('input', () => {
      lbl.textContent = fmt(parseInt(el.value));
    });

    el.addEventListener('change', () => {
      lbl.textContent = fmt(parseInt(el.value));
      debounce(key, 300, () => sendCameraSettings({ [key]: scale(parseInt(el.value)) }));
    });
  });
}

// ── Camera settings ───────────────────────────────────────────────────────────
function bindCameraSettings() {
  const aeToggle       = document.getElementById('ae-toggle');
  const manualGroup    = document.getElementById('manual-exposure');
  const exposureSlider = document.getElementById('exposure-slider');
  const gainSlider     = document.getElementById('gain-slider');
  const exposureVal    = document.getElementById('exposure-val');
  const gainVal        = document.getElementById('gain-val');

  if (!aeToggle) return;

  aeToggle.addEventListener('change', () => {
    const auto = aeToggle.checked;
    manualGroup.classList.toggle('disabled', auto);
    if (exposureSlider) exposureSlider.disabled = auto;
    if (gainSlider)     gainSlider.disabled = auto;
    sendCameraSettings({ auto_exposure: auto });
  });

  exposureSlider?.addEventListener('input', () => {
    exposureVal.textContent = exposureSlider.value;
  });
  exposureSlider?.addEventListener('change', () => {
    debounce('exposure', 300, () =>
      sendCameraSettings({ exposure_time: parseInt(exposureSlider.value) }));
  });

  gainSlider?.addEventListener('input', () => {
    gainVal.textContent = `${(parseInt(gainSlider.value)/10).toFixed(1)}×`;
  });
  gainSlider?.addEventListener('change', () => {
    debounce('gain', 300, () =>
      sendCameraSettings({ analogue_gain: parseInt(gainSlider.value) / 10 }));
  });
}

async function sendCameraSettings(partial) {
  await fetch('/api/camera', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(partial),
  });
}

// ── Collapsible cards ─────────────────────────────────────────────────────────
function toggleCard(id) {
  document.getElementById(id)?.classList.toggle('open');
}

// ── Thumbnail strip ───────────────────────────────────────────────────────────
async function loadRecentThumbs() {
  const r = await fetch('/api/captures').catch(() => null);
  if (!r || !r.ok) return;
  allCaptures = await r.json();
  const strip = document.getElementById('thumb-strip');
  const empty = document.getElementById('strip-empty');
  if (!strip) return;

  allCaptures.slice(0, 12).forEach(c => {
    const el = buildThumbItem(c.filename);
    strip.appendChild(el);
  });

  if (allCaptures.length > 0 && empty) empty.style.display = 'none';
}

function addThumb(filename) {
  const strip = document.getElementById('thumb-strip');
  const empty = document.getElementById('strip-empty');
  if (!strip) return;
  if (empty) empty.style.display = 'none';
  const el = buildThumbItem(filename);
  strip.insertBefore(el, strip.firstChild);
}

function buildThumbItem(filename) {
  const item = document.createElement('div');
  item.className = 'thumb-item';
  item.dataset.filename = filename;

  const img = document.createElement('img');
  img.src = `/api/thumbnail/${filename}`;
  img.alt = filename;
  img.loading = 'lazy';
  item.appendChild(img);

  item.addEventListener('click', () => selectCapture(filename));
  return item;
}

function selectCapture(filename) {
  selectedFilename = filename;

  document.querySelectorAll('.thumb-item').forEach(el => {
    el.classList.toggle('selected', el.dataset.filename === filename);
  });

  const qBtn = document.getElementById('quantify-btn');
  if (qBtn) qBtn.disabled = false;

  // Show camera image in preview if available
  const feed = document.getElementById('camera-feed');
  if (feed) {
    feed.src = `/captures/${filename}`;
    feed.onerror = () => { feed.src = '/stream'; };
  }

  setStatus(`Selected: ${filename} · Click Quantify to analyse`);
}

// ── Results rendering ─────────────────────────────────────────────────────────
function renderResults(data) {
  const empty   = document.getElementById('results-empty');
  const content = document.getElementById('results-content');
  const spinner = document.getElementById('results-spinner');
  const badge   = document.getElementById('results-badge');
  const grid    = document.getElementById('stats-grid');
  const list    = document.getElementById('colony-list');

  if (spinner) spinner.style.display = 'none';
  if (empty)   empty.style.display   = 'none';
  if (content) content.style.display = '';

  const s   = data.summary_stats || {};
  const rep = data.anomaly_report || {};

  // Badge
  if (badge) {
    badge.style.display = '';
    if (data.anomaly_count > 0) {
      badge.className = 'badge badge-red';
      badge.textContent = `${data.anomaly_count} flagged`;
    } else {
      badge.className = 'badge badge-green';
      badge.textContent = 'Clean';
    }
  }

  // Stats grid
  if (grid) {
    grid.innerHTML = '';
    [
      { label: 'Colonies',   value: data.count,               cls: '' },
      { label: 'Anomalies',  value: data.anomaly_count,       cls: data.anomaly_count > 0 ? 'flagged' : 'clean' },
      { label: 'Mean Area',  value: `${(s.mean_area_mm2||0).toFixed(2)} mm²`, cls: '' },
      { label: 'Coverage',   value: `${(s.coverage_pct||0).toFixed(1)}%`,     cls: '' },
      { label: 'Haemolysis', value: s.hemolysis_candidates || 0,              cls: '' },
      { label: 'ML Layer',   value: rep.ml_active ? 'Active' : 'Untrained',   cls: '' },
    ].forEach(({ label, value, cls }) => {
      grid.insertAdjacentHTML('beforeend',
        `<div class="stat-item">
          <div class="stat-label">${label}</div>
          <div class="stat-value ${cls}">${value}</div>
        </div>`
      );
    });
  }

  // Per-colony list
  if (list) {
    list.innerHTML = '';
    const flagged = (data.colonies || []).filter(c =>
      (c.anomaly_flags && c.anomaly_flags.length) || c.ml_anomaly);
    const normal  = data.count - flagged.length;

    flagged.forEach(c => {
      const flags = [...(c.anomaly_flags || [])];
      if (c.ml_anomaly) flags.push('ml_anomaly');
      const chips = flags.map(f => {
        const cls = f === 'hemolysis_candidate' ? 'hemolysis'
                  : f === 'ml_anomaly'          ? 'ml' : '';
        return `<span class="flag-chip ${cls}">${f.replace(/_/g,' ')}</span>`;
      }).join('');
      list.insertAdjacentHTML('beforeend',
        `<div class="colony-row">
          <span class="colony-num anomaly">#${c.id}</span>
          <span class="colony-area">${c.area_mm2.toFixed(3)} mm²</span>
          <span class="colony-flags">${chips}</span>
        </div>`
      );
    });

    if (normal > 0) {
      list.insertAdjacentHTML('beforeend',
        `<div style="font-size:11px;color:var(--tertiary);padding:6px 0;text-align:center">
          ${normal} normal ${normal === 1 ? 'colony' : 'colonies'} — no flags
        </div>`
      );
    }
    if (flagged.length === 0 && data.count > 0) {
      list.insertAdjacentHTML('beforeend',
        `<div style="font-size:11px;color:var(--green);padding:6px 0;text-align:center">
          All ${data.count} colonies within normal range
        </div>`
      );
    }
  }
}

function showResultsSpinner() {
  const empty   = document.getElementById('results-empty');
  const content = document.getElementById('results-content');
  const spinner = document.getElementById('results-spinner');
  if (empty)   empty.style.display   = 'none';
  if (content) content.style.display = 'none';
  if (spinner) spinner.style.display = '';
}

function showResultsError(msg) {
  const spinner = document.getElementById('results-spinner');
  const empty   = document.getElementById('results-empty');
  if (spinner) spinner.style.display = 'none';
  if (empty) {
    empty.style.display = '';
    empty.innerHTML = `<p style="color:var(--red)">${msg}</p>`;
  }
}

// ── Library ───────────────────────────────────────────────────────────────────
function bindLibrary() {
  document.getElementById('lib-search')?.addEventListener('input', filterLibrary);
  document.getElementById('lib-filter')?.addEventListener('change', filterLibrary);
}

async function loadLibrary() {
  const r = await fetch('/api/captures').catch(() => null);
  if (!r || !r.ok) return;
  allCaptures = await r.json();
  renderLibrary(allCaptures);
}

function filterLibrary() {
  const query  = (document.getElementById('lib-search')?.value || '').toLowerCase();
  const filter = document.getElementById('lib-filter')?.value || '';

  let list = allCaptures.filter(c => {
    if (query && !c.filename.toLowerCase().includes(query) &&
        !(c.timestamp || '').toLowerCase().includes(query)) return false;
    if (filter === 'flagged'    && !(c.anomaly_count > 0)) return false;
    if (filter === 'clean'      && !(c.has_result && c.anomaly_count === 0)) return false;
    if (filter === 'unanalysed' && c.has_result) return false;
    return true;
  });

  renderLibrary(list);
}

function renderLibrary(list) {
  const grid  = document.getElementById('photo-grid');
  const empty = document.getElementById('library-empty');
  if (!grid) return;

  // Remove existing cards (preserve empty state node)
  grid.querySelectorAll('.photo-card').forEach(el => el.remove());

  if (list.length === 0) {
    if (empty) empty.style.display = 'flex';
    return;
  }
  if (empty) empty.style.display = 'none';

  list.forEach(c => {
    const ts = c.timestamp ? formatTime(c.timestamp) : 'Unknown time';
    const countText = c.count != null ? `${c.count} colonies` : 'Not analysed';
    const flagText  = c.anomaly_count > 0 ? `${c.anomaly_count} flagged` : '';

    const card = document.createElement('div');
    card.className = 'photo-card';
    card.innerHTML = `
      <img class="photo-card-thumb lazy"
           data-src="/api/thumbnail/${c.filename}"
           src="data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs="
           alt="${c.filename}">
      <div class="photo-card-info">
        <div class="photo-card-time">${ts}</div>
        <div class="photo-card-count">${countText}</div>
        ${flagText ? `<div class="photo-card-flags">${flagText}</div>` : ''}
        <div class="photo-card-name">${c.filename}</div>
      </div>
    `;

    card.addEventListener('click', () => openDetail(c));
    grid.appendChild(card);
  });

  // Lazy load thumbnails via IntersectionObserver
  const observer = new IntersectionObserver((entries, obs) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const img = entry.target;
        img.src = img.dataset.src;
        img.classList.remove('lazy');
        img.classList.add('loaded');
        obs.unobserve(img);
      }
    });
  }, { rootMargin: '100px' });

  grid.querySelectorAll('.photo-card-thumb.lazy').forEach(img => observer.observe(img));
}

function refreshLibraryIfOpen() {
  if (document.getElementById('view-library')?.classList.contains('active')) {
    loadLibrary();
  }
}

// ── Detail panel ──────────────────────────────────────────────────────────────
async function openDetail(capture) {
  const panel   = document.getElementById('detail-panel');
  const overlay = document.getElementById('detail-overlay');
  const title   = document.getElementById('detail-title');
  const body    = document.getElementById('detail-body');

  if (!panel) return;

  title.textContent = capture.filename;
  body.innerHTML = '<div class="spinner-wrap"><div class="spinner"></div></div>';
  panel.classList.add('open');
  overlay.classList.add('visible');
  document.body.style.overflow = 'hidden';

  // Decide which image to show (annotated if exists)
  const stem      = capture.filename.replace(/\.[^.]+$/, '');
  const imgSrc    = capture.has_result
                    ? `/results/annotated_${capture.filename}`
                    : `/captures/${capture.filename}`;

  // Fetch result data
  let result = null;
  if (capture.has_result) {
    const r = await fetch(`/api/result/${stem}`).catch(() => null);
    if (r && r.ok) result = await r.json();
  }

  const ts = capture.timestamp ? formatTime(capture.timestamp) : '—';

  let html = `
    <img class="detail-img" src="${imgSrc}"
         onerror="this.src='/captures/${capture.filename}'" alt="Plate image">

    <div class="detail-section">
      <div class="detail-section-title">Metadata</div>
      <div class="detail-meta-row">
        <span class="detail-meta-key">Captured</span>
        <span class="detail-meta-value">${ts}</span>
      </div>
      <div class="detail-meta-row">
        <span class="detail-meta-key">File</span>
        <span class="detail-meta-value">${capture.filename}</span>
      </div>
    </div>
  `;

  if (result) {
    const s   = result.summary_stats || {};
    const rep = result.anomaly_report || {};
    html += `
      <div class="detail-section">
        <div class="detail-section-title">Quantification</div>
        ${metaRow('Colony count',    result.count)}
        ${metaRow('Anomalies',       result.anomaly_count)}
        ${metaRow('Mean area',       `${(s.mean_area_mm2||0).toFixed(3)} mm²`)}
        ${metaRow('Coverage',        `${(s.coverage_pct||0).toFixed(2)}%`)}
        ${metaRow('Circularity',     (s.mean_circularity||0).toFixed(3))}
        ${metaRow('Haemolysis',      s.hemolysis_candidates || 0)}
        ${metaRow('ML layer',        rep.ml_active ? 'Active' : 'Not trained')}
      </div>
    `;

    const flagged = (result.colonies || []).filter(c =>
      (c.anomaly_flags && c.anomaly_flags.length) || c.ml_anomaly);

    if (flagged.length > 0) {
      html += `<div class="detail-section">
        <div class="detail-section-title">Flagged Colonies (${flagged.length})</div>`;
      flagged.forEach(c => {
        const flags = [...(c.anomaly_flags || [])];
        if (c.ml_anomaly) flags.push('ml_anomaly');
        html += `<div class="detail-colony-row">
          <span class="colony-num anomaly">#${c.id}</span>
          <span class="colony-area">${c.area_mm2.toFixed(3)} mm²</span>
          <span class="colony-flags">${flags.map(f =>
            `<span class="flag-chip">${f.replace(/_/g,' ')}</span>`).join('')}
          </span>
        </div>`;
      });
      html += '</div>';
    }

    html += `<button class="export-btn" onclick="exportCSV('${stem}')">
      Export Results as CSV
    </button>`;
  } else {
    html += `<div style="color:var(--tertiary);font-size:13px;text-align:center;padding:20px">
      Image not yet analysed.<br>Select it in Capture view and click Quantify.
    </div>`;
  }

  body.innerHTML = html;
}

function closeDetail() {
  document.getElementById('detail-panel')?.classList.remove('open');
  document.getElementById('detail-overlay')?.classList.remove('visible');
  document.body.style.overflow = '';
}

async function exportCSV(stem) {
  const r = await fetch(`/api/result/${stem}`).catch(() => null);
  if (!r || !r.ok) return alert('No result data to export.');
  const data = await r.json();
  const cols  = ['id','area_mm2','circularity','aspect_ratio',
                  'hemolysis_delta','anomaly_flags','ml_anomaly','stat_score'];
  const rows  = [cols.join(',')];
  (data.colonies || []).forEach(c => {
    rows.push(cols.map(k =>
      k === 'anomaly_flags'
        ? `"${(c[k]||[]).join('|')}"`
        : (c[k] ?? '')
    ).join(','));
  });
  const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = `${stem}_colonies.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function setStatus(msg) {
  const bar = document.getElementById('status-bar');
  if (bar) bar.textContent = msg;
}

function metaRow(key, value) {
  return `<div class="detail-meta-row">
    <span class="detail-meta-key">${key}</span>
    <span class="detail-meta-value">${value}</span>
  </div>`;
}

function formatTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

function debounce(key, ms, fn) {
  clearTimeout(debounceTimers[key]);
  debounceTimers[key] = setTimeout(fn, ms);
}
