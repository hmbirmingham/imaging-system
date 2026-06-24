/* app.js — Plate Imaging System frontend */

'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
let selectedFilename = null;
let allCaptures      = [];
let debounceTimers   = {};

// Profiles / validation
let profilesData     = { available: false, profiles: [], plate_types: {} };
let selectedProfile  = 'unknown';
let selectedPlate    = 'unknown';
let currentDetail    = null;   // { capture, result, stem }
let valState         = null;   // active validation working state
let appState         = { camera: false, led: false };
let settingsLoaded   = false;
let lastQuantifyData = null;   // for the raw-detection toggle
let showRaw          = false;

// Size / tolerance / reaction vocabularies (mirror profiles.py)
const SIZE_VOCAB      = ['pinpoint', 'small', 'medium', 'large'];
const TOLERANCE_VOCAB = ['tight', 'normal', 'loose'];
const HEMOLYSIS_VOCAB = ['none', 'alpha', 'beta', 'gamma', 'unknown'];
const LACTOSE_VOCAB   = ['fermenter', 'non_fermenter', 'late', 'na', 'unknown'];

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
  loadProfiles();
  bindProfilesView();
  bindSettings();
});

// ── Profiles ──────────────────────────────────────────────────────────────────
async function loadProfiles() {
  const r = await fetch('/api/profiles').catch(() => null);
  if (!r || !r.ok) return;
  profilesData = await r.json();
  if (!profilesData.available) return;

  const orgSel   = document.getElementById('organism-select');
  const plateSel = document.getElementById('plate-select');

  if (orgSel) {
    profilesData.profiles.forEach(p => {
      const o = document.createElement('option');
      o.value = p.profile_id;
      o.textContent = p.display_name + (p.validated ? ' ✓' : '');
      orgSel.appendChild(o);
    });
    orgSel.addEventListener('change', () => {
      selectedProfile = orgSel.value;
      syncPlateOptions();
      updateProfileBadge();
    });
  }
  if (plateSel) {
    fillPlateOptions(plateSel, Object.keys(profilesData.plate_types));
    plateSel.addEventListener('change', () => { selectedPlate = plateSel.value; });
  }

  // Apply saved default organism/plate.
  const cfg = await fetch('/api/settings').then(r => r.ok ? r.json() : null).catch(() => null);
  if (cfg) {
    if (orgSel && [...orgSel.options].some(o => o.value === cfg.default_profile_id)) {
      orgSel.value = selectedProfile = cfg.default_profile_id;
      syncPlateOptions(); updateProfileBadge();
    }
    if (plateSel && [...plateSel.options].some(o => o.value === cfg.default_plate_type)) {
      plateSel.value = selectedPlate = cfg.default_plate_type;
    }
  }
}

// Restrict the plate dropdown to the selected organism's media (falls back to all).
function syncPlateOptions() {
  const plateSel = document.getElementById('plate-select');
  if (!plateSel) return;
  const prof = profilesData.profiles.find(p => p.profile_id === selectedProfile);
  const codes = (prof && prof.plate_types && prof.plate_types.length)
    ? prof.plate_types : Object.keys(profilesData.plate_types);
  fillPlateOptions(plateSel, codes);
  selectedPlate = plateSel.value;
}

function fillPlateOptions(sel, codes) {
  const prev = sel.value;
  sel.innerHTML = '<option value="unknown">— Unknown —</option>';
  codes.forEach(code => {
    const o = document.createElement('option');
    o.value = code;
    o.textContent = profilesData.plate_types[code] || code;
    sel.appendChild(o);
  });
  if ([...sel.options].some(o => o.value === prev)) sel.value = prev;
}

function updateProfileBadge() {
  const badge = document.getElementById('profile-badge');
  if (!badge) return;
  const prof = profilesData.profiles.find(p => p.profile_id === selectedProfile);
  if (!prof) { badge.style.display = 'none'; return; }
  badge.style.display = '';
  badge.className = 'badge ' + (prof.validated ? 'badge-green' : 'badge-gray');
  badge.textContent = prof.validated ? 'validated' : 'unvalidated';
}

// ── SSE ───────────────────────────────────────────────────────────────────────
function connectSSE() {
  const es = new EventSource('/api/events');

  es.addEventListener('init', e => {
    const d = JSON.parse(e.data);
    appState = { camera: d.camera, led: d.led };
    if (!d.camera) setStatus('No camera connected — capture disabled');
    if (!d.led)    console.info('LED hardware not available');
    if (d.led)     syncLED(d.led_brightness);
  });

  es.addEventListener('train_done', e => {
    const d = JSON.parse(e.data);
    setStatus(`Model trained — F1 ${d.cv_f1} on ${d.n} colonies`);
    if (document.getElementById('view-settings')?.classList.contains('active')) loadDiagnostics();
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

  es.addEventListener('validated', e => {
    const d = JSON.parse(e.data);
    setStatus(`Validation saved — ${d.updated} labelled, ${d.added} added`);
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
        if (btn.dataset.view === 'library')  loadLibrary();
        if (btn.dataset.view === 'profiles') loadProfilesView();
        if (btn.dataset.view === 'settings') openSettings();
      }
    });
  });
}

// ── Capture ───────────────────────────────────────────────────────────────────
function bindCapture() {
  document.getElementById('capture-btn')?.addEventListener('click', async () => {
    setStatus('Capturing…');
    const r = await fetch('/api/capture', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ profile_id: selectedProfile, plate_type: selectedPlate }),
    });
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
      body: JSON.stringify({ filename: selectedFilename,
                             profile_id: selectedProfile, plate_type: selectedPlate }),
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
function toggleRaw() {
  showRaw = !showRaw;
  if (lastQuantifyData) renderResults(lastQuantifyData);
}

function renderResults(data) {
  lastQuantifyData = data;
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
    const colFlags = c => showRaw ? (c.raw_anomaly_flags || []) : (c.anomaly_flags || []);
    const hasRaw = (data.colonies || []).some(c =>
      (c.raw_anomaly_flags || []).length !== (c.anomaly_flags || []).length);
    if (hasRaw) {
      list.insertAdjacentHTML('beforeend',
        `<div class="raw-toggle" onclick="toggleRaw()">
          ${showRaw ? '● showing raw detection — click for profile-adjusted'
                    : '○ profile-adjusted — click to show raw detection'}
        </div>`);
    }
    const flagged = (data.colonies || []).filter(c =>
      colFlags(c).length || c.ml_anomaly);
    const normal  = data.count - flagged.length;

    flagged.forEach(c => {
      const flags = [...colFlags(c)];
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
  currentDetail = { capture, result, stem };

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
    if (profilesData.available) {
      html += `<button class="validate-btn" onclick="startValidation()">
        Validate &amp; Sign off
      </button>`;
    }
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

// ── Validation mode ───────────────────────────────────────────────────────────
async function startValidation() {
  if (!currentDetail || !currentDetail.result) return;
  const { result, capture, stem } = currentDetail;

  // Seed working state from the analysis. Each colony starts at the detector's
  // verdict; the human confirms, flips, or rejects it.
  valState = {
    stem,
    filename:   capture.filename,
    profile_id: result.profile_id || capture.profile_id || selectedProfile || 'unknown',
    plate_type: result.plate_type || capture.plate_type || selectedPlate || 'unknown',
    imgW: 0, imgH: 0,
    colonies: (result.colonies || []).map(c => ({
      id:         c.id,
      centroid:   c.centroid || [0, 0],
      area_mm2:   c.area_mm2,
      is_anomaly: !!((c.anomaly_flags && c.anomaly_flags.length) || c.ml_anomaly),
      status:     'confirmed',
    })),
    added: [],
    profile: null,
  };

  // Restore a prior validation if one exists.
  const prior = await fetch(`/api/validation/${stem}`).then(r => r.ok ? r.json() : null).catch(() => null);
  if (prior && prior.colonies) {
    const byId = Object.fromEntries(prior.colonies.map(c => [c.id, c]));
    valState.colonies.forEach(c => {
      const p = byId[c.id];
      if (p) { c.is_anomaly = !!p.is_anomaly; c.status = p.status || 'confirmed'; }
    });
    valState.added = prior.added || [];
    if (prior.profile_id) valState.profile_id = prior.profile_id;
    if (prior.plate_type) valState.plate_type = prior.plate_type;
  }

  if (valState.profile_id !== 'unknown') {
    valState.profile = await fetch(`/api/profile/${valState.profile_id}`)
      .then(r => r.ok ? r.json() : null).catch(() => null);
  }
  renderValidation();
}

function renderValidation() {
  const body = document.getElementById('detail-body');
  const v = valState;
  const imgSrc = currentDetail.capture.has_result
    ? `/results/annotated_${v.filename}` : `/captures/${v.filename}`;

  const orgOpts = ['<option value="unknown">— Unknown —</option>']
    .concat(profilesData.profiles.map(p =>
      `<option value="${p.profile_id}" ${p.profile_id === v.profile_id ? 'selected' : ''}>${p.display_name}</option>`)).join('');
  const plateOpts = ['<option value="unknown">— Unknown —</option>']
    .concat(Object.entries(profilesData.plate_types).map(([code, name]) =>
      `<option value="${code}" ${code === v.plate_type ? 'selected' : ''}>${name}</option>`)).join('');

  body.innerHTML = `
    <div class="val-toolbar">
      <button class="val-back" onclick="cancelValidation()">← Back</button>
      <span class="val-hint">Click a colony: normal → anomaly → false-positive. Click empty agar to add a missed one.</span>
    </div>
    <div class="val-canvas" id="val-canvas">
      <img id="val-img" class="val-img" src="${imgSrc}" alt="Plate">
      <div class="val-overlay" id="val-overlay"></div>
    </div>
    <div class="val-counts" id="val-counts"></div>

    <div class="detail-section">
      <div class="detail-section-title">Sample</div>
      <div class="select-row"><span class="slider-label">Organism</span>
        <select class="filter-select" id="val-organism">${orgOpts}</select></div>
      <div class="select-row"><span class="slider-label">Plate</span>
        <select class="filter-select" id="val-plate">${plateOpts}</select></div>
    </div>

    <div class="detail-section" id="val-biology"></div>

    <div class="detail-section">
      <div class="detail-section-title">Sign-off</div>
      <div class="select-row"><span class="slider-label">Validated by</span>
        <input type="text" id="val-by" class="val-input" placeholder="Name (e.g. Dr Ryberg)"></div>
      <button class="validate-btn" id="val-submit" onclick="submitValidation()">Save validation &amp; sign off</button>
      <div class="val-msg" id="val-msg"></div>
    </div>
  `;

  document.getElementById('val-organism').addEventListener('change', async e => {
    valState.profile_id = e.target.value;
    valState.profile = valState.profile_id !== 'unknown'
      ? await fetch(`/api/profile/${valState.profile_id}`).then(r => r.ok ? r.json() : null).catch(() => null)
      : null;
    renderBiologyEditor();
  });
  document.getElementById('val-plate').addEventListener('change', e => {
    valState.plate_type = e.target.value;
    renderBiologyEditor();
  });

  const img = document.getElementById('val-img');
  if (img.complete) onValImgLoad(); else img.onload = onValImgLoad;
  document.getElementById('val-overlay').addEventListener('click', onCanvasClick);
  renderBiologyEditor();
}

function onValImgLoad() {
  const img = document.getElementById('val-img');
  valState.imgW = img.naturalWidth || img.clientWidth;
  valState.imgH = img.naturalHeight || img.clientHeight;
  placeMarkers();
}

function placeMarkers() {
  const overlay = document.getElementById('val-overlay');
  if (!overlay) return;
  overlay.innerHTML = '';
  const { imgW, imgH } = valState;
  valState.colonies.forEach((c, i) => {
    const m = document.createElement('div');
    m.className = `val-marker ${markerClass(c)}`;
    m.style.left = `${(c.centroid[0] / imgW) * 100}%`;
    m.style.top  = `${(c.centroid[1] / imgH) * 100}%`;
    m.textContent = c.id;
    m.title = `Colony ${c.id} · ${c.area_mm2?.toFixed?.(2) ?? '?'} mm²`;
    m.addEventListener('click', ev => { ev.stopPropagation(); cycleColony(i); });
    overlay.appendChild(m);
  });
  valState.added.forEach((a, i) => {
    const m = document.createElement('div');
    m.className = `val-marker ${a.is_anomaly ? 'm-anomaly' : 'm-normal'} m-added`;
    m.style.left = `${(a.centroid_x / valState.imgW) * 100}%`;
    m.style.top  = `${(a.centroid_y / valState.imgH) * 100}%`;
    m.textContent = '+';
    m.title = 'Added colony — click to toggle / remove';
    m.addEventListener('click', ev => { ev.stopPropagation(); cycleAdded(i); });
    overlay.appendChild(m);
  });
  updateValCount();
}

function markerClass(c) {
  if (c.status === 'false_positive') return 'm-rejected';
  return c.is_anomaly ? 'm-anomaly' : 'm-normal';
}

// normal → anomaly → false-positive → normal
function cycleColony(i) {
  const c = valState.colonies[i];
  if (c.status === 'false_positive')      { c.status = 'confirmed'; c.is_anomaly = false; }
  else if (!c.is_anomaly)                 { c.is_anomaly = true; }
  else                                    { c.status = 'false_positive'; }
  placeMarkers();
}

// added: normal → anomaly → removed
function cycleAdded(i) {
  const a = valState.added[i];
  if (!a.is_anomaly) { a.is_anomaly = true; }
  else               { valState.added.splice(i, 1); }
  placeMarkers();
}

function onCanvasClick(e) {
  const overlay = document.getElementById('val-overlay');
  const rect = overlay.getBoundingClientRect();
  const fx = (e.clientX - rect.left) / rect.width;
  const fy = (e.clientY - rect.top)  / rect.height;
  valState.added.push({
    centroid_x: Math.round(fx * valState.imgW),
    centroid_y: Math.round(fy * valState.imgH),
    is_anomaly: false,
  });
  placeMarkers();
}

function updateValCount() {
  const confirmed = valState.colonies.filter(c => c.status !== 'false_positive').length + valState.added.length;
  const anomalies = valState.colonies.filter(c => c.status !== 'false_positive' && c.is_anomaly).length
                  + valState.added.filter(a => a.is_anomaly).length;
  const rejected  = valState.colonies.filter(c => c.status === 'false_positive').length;
  const el = document.getElementById('val-counts');
  if (el) el.innerHTML =
    `<span class="vc"><b>${confirmed}</b> colonies</span>
     <span class="vc vc-anom"><b>${anomalies}</b> anomalies</span>
     <span class="vc vc-rej"><b>${rejected}</b> false-positive</span>
     <span class="vc vc-add"><b>${valState.added.length}</b> added</span>`;
}

function renderBiologyEditor() {
  const host = document.getElementById('val-biology');
  if (!host) return;
  if (valState.profile_id === 'unknown') {
    host.innerHTML = `<div class="detail-section-title">Biology</div>
      <div class="val-note">Select an organism to edit biology fields.</div>`;
    return;
  }
  const prof = valState.profile || {};
  const bio  = prof.biology || {};
  const plates = (bio.plates || {});
  const pb = Object.assign({}, plates.default || {}, plates[valState.plate_type] || {});

  const sel = (id, vocab, cur) =>
    `<select class="filter-select" id="${id}">` +
    vocab.map(o => `<option value="${o}" ${o === cur ? 'selected' : ''}>${o}</option>`).join('') +
    `</select>`;

  host.innerHTML = `
    <div class="detail-section-title">Biology — ${prof.display_name || valState.profile_id}
      <span class="badge ${prof.validation?.validated ? 'badge-green' : 'badge-gray'}">
        ${prof.validation?.validated ? 'validated' : 'unvalidated'}</span></div>
    <div class="val-subtle">Organism-level</div>
    <div class="toggle-row"><span class="slider-label">Swarming</span>
      <label class="toggle"><input type="checkbox" id="bio-swarming" ${bio.swarming ? 'checked' : ''}>
      <span class="toggle-track"></span></label></div>
    <div class="select-row"><span class="slider-label">Incubation (h)</span>
      <input type="number" class="val-input val-num" id="bio-inc-min" value="${bio.incubation_time_h_min ?? 18}">
      <input type="number" class="val-input val-num" id="bio-inc-max" value="${bio.incubation_time_h_max ?? 24}"></div>
    <div class="val-subtle">Appearance on ${profilesData.plate_types[valState.plate_type] || valState.plate_type}</div>
    <div class="select-row"><span class="slider-label">Colony size</span>${sel('bio-size', SIZE_VOCAB, pb.colony_size || 'medium')}</div>
    <div class="select-row"><span class="slider-label">Tolerance</span>${sel('bio-tol', TOLERANCE_VOCAB, pb.size_tolerance || 'normal')}</div>
    <div class="select-row"><span class="slider-label">Hemolysis</span>${sel('bio-hem', HEMOLYSIS_VOCAB, pb.hemolysis || 'unknown')}</div>
    <div class="select-row"><span class="slider-label">Lactose</span>${sel('bio-lac', LACTOSE_VOCAB, pb.lactose || 'unknown')}</div>
    <div class="select-row"><span class="slider-label">Colour</span>
      <input type="text" class="val-input" id="bio-color" value="${pb.colony_color || ''}" placeholder="e.g. pink"></div>
    <div class="select-row"><span class="slider-label">Notes</span>
      <input type="text" class="val-input" id="bio-notes" value="${pb.notes || ''}" placeholder="optional"></div>
  `;
}

function readBiology() {
  if (valState.profile_id === 'unknown' || !document.getElementById('bio-size')) return {};
  return {
    organism_biology: {
      swarming: document.getElementById('bio-swarming').checked,
      incubation_time_h_min: parseInt(document.getElementById('bio-inc-min').value) || 18,
      incubation_time_h_max: parseInt(document.getElementById('bio-inc-max').value) || 24,
    },
    plate_biology: {
      colony_size:    document.getElementById('bio-size').value,
      size_tolerance: document.getElementById('bio-tol').value,
      hemolysis:      document.getElementById('bio-hem').value,
      lactose:        document.getElementById('bio-lac').value,
      colony_color:   document.getElementById('bio-color').value || null,
      notes:          document.getElementById('bio-notes').value || '',
    },
  };
}

async function submitValidation() {
  const by = document.getElementById('val-by').value.trim();
  const msg = document.getElementById('val-msg');
  if (!by) { msg.textContent = 'Enter a name to sign off.'; msg.className = 'val-msg err'; return; }

  const confirmed = valState.colonies.filter(c => c.status !== 'false_positive').length + valState.added.length;
  const bio = readBiology();
  const payload = {
    filename:     valState.filename,
    profile_id:   valState.profile_id,
    plate_type:   valState.plate_type,
    validated_by: by,
    manual_count: confirmed,
    colonies:     valState.colonies.map(c => ({ id: c.id, is_anomaly: c.is_anomaly, status: c.status })),
    added:        valState.added,
    signoff:      valState.profile_id !== 'unknown' ? { by } : null,
    ...bio,
  };

  const btn = document.getElementById('val-submit');
  btn.disabled = true;
  msg.textContent = 'Saving…'; msg.className = 'val-msg';
  const r = await fetch('/api/validate', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).catch(() => null);

  if (r && r.ok) {
    const d = await r.json();
    msg.textContent = `Saved — ${d.updated} labelled, ${d.added} added. Signed off by ${by}.`;
    msg.className = 'val-msg ok';
    setStatus(`Validated ${valState.filename} — ${confirmed} colonies confirmed`);
    await loadProfiles();   // refresh validated badges
    setTimeout(() => { cancelValidation(); }, 1200);
  } else {
    msg.textContent = 'Save failed.'; msg.className = 'val-msg err';
    btn.disabled = false;
  }
}

function cancelValidation() {
  valState = null;
  if (currentDetail) openDetail(currentDetail.capture);
}

// ── Profiles view ─────────────────────────────────────────────────────────────
function bindProfilesView() {
  document.getElementById('prof-search')?.addEventListener('input', filterProfilesView);
  document.getElementById('prof-filter')?.addEventListener('change', filterProfilesView);
}

async function loadProfilesView() {
  await loadProfilesData();
  filterProfilesView();
}

async function loadProfilesData() {
  const r = await fetch('/api/profiles').catch(() => null);
  if (r && r.ok) profilesData = await r.json();
}

function filterProfilesView() {
  const q = (document.getElementById('prof-search')?.value || '').toLowerCase();
  const f = document.getElementById('prof-filter')?.value || '';
  const list = (profilesData.profiles || []).filter(p => {
    if (q && !p.display_name.toLowerCase().includes(q) && !p.profile_id.includes(q)) return false;
    if (f === 'validated'   && !p.validated) return false;
    if (f === 'unvalidated' &&  p.validated) return false;
    if (f === 'swarming'    && !p.swarming)  return false;
    return true;
  });
  renderProfilesGrid(list);
}

function renderProfilesGrid(list) {
  const grid = document.getElementById('profiles-grid');
  if (!grid) return;
  grid.innerHTML = '';
  if (!profilesData.available) {
    grid.innerHTML = `<div class="library-empty" style="display:flex">
      <h3 class="empty-title">Profiles unavailable</h3>
      <p class="empty-sub">PyYAML is not installed on this device.</p></div>`;
    return;
  }
  list.forEach(p => {
    const chips = (p.plate_types || []).map(c =>
      `<span class="plate-chip">${(profilesData.plate_types && profilesData.plate_types[c]) ? c : c}</span>`).join('');
    const card = document.createElement('div');
    card.className = 'profile-card';
    card.innerHTML = `
      <div class="profile-card-top">
        <span class="profile-name">${p.display_name}</span>
        <span class="badge ${p.validated ? 'badge-green' : 'badge-gray'}">${p.validated ? '✓ validated' : 'unvalidated'}</span>
      </div>
      <div class="profile-meta">
        <span class="gram-tag gram-${p.gram || 'unknown'}">Gram ${p.gram === 'positive' ? '+' : p.gram === 'negative' ? '−' : '?'}</span>
        ${p.swarming ? '<span class="swarm-tag">swarming</span>' : ''}
      </div>
      <div class="plate-chips">${chips}</div>`;
    card.addEventListener('click', () => openProfileEditor(p.profile_id));
    grid.appendChild(card);
  });
}

async function openProfileEditor(profileId) {
  const panel = document.getElementById('detail-panel');
  const overlay = document.getElementById('detail-overlay');
  const title = document.getElementById('detail-title');
  const body = document.getElementById('detail-body');
  if (!panel) return;

  title.textContent = 'Profile';
  body.innerHTML = '<div class="spinner-wrap"><div class="spinner"></div></div>';
  panel.classList.add('open'); overlay.classList.add('visible');
  document.body.style.overflow = 'hidden';

  const prof = await fetch(`/api/profile/${profileId}`).then(r => r.ok ? r.json() : null).catch(() => null);
  if (!prof) { body.innerHTML = '<p style="color:var(--red)">Could not load profile.</p>'; return; }
  title.textContent = prof.display_name || profileId;

  const bio = prof.biology || {};
  const codes = (prof.plate_types && prof.plate_types.length)
    ? prof.plate_types : Object.keys(profilesData.plate_types || {});
  const plateOpts = codes.map(c =>
    `<option value="${c}">${(profilesData.plate_types && profilesData.plate_types[c]) || c}</option>`).join('');

  body.innerHTML = `
    <div class="detail-section">
      <div class="detail-section-title">Organism
        <span class="badge ${prof.validation?.validated ? 'badge-green' : 'badge-gray'}">
          ${prof.validation?.validated ? 'validated' : 'unvalidated'}</span></div>
      ${metaRow('Gram', bio.gram || '—')}
      ${metaRow('Morphology', bio.cell_morphology || '—')}
      <div class="toggle-row"><span class="slider-label">Swarming</span>
        <label class="toggle"><input type="checkbox" id="pe-swarming" ${bio.swarming ? 'checked' : ''}>
        <span class="toggle-track"></span></label></div>
      <div class="select-row"><span class="slider-label">Incubation (h)</span>
        <input type="number" class="val-input val-num" id="pe-inc-min" value="${bio.incubation_time_h_min ?? 18}">
        <input type="number" class="val-input val-num" id="pe-inc-max" value="${bio.incubation_time_h_max ?? 24}"></div>
    </div>
    <div class="detail-section">
      <div class="detail-section-title">Appearance per plate</div>
      <div class="select-row"><span class="slider-label">Plate</span>
        <select class="filter-select" id="pe-plate">${plateOpts}</select></div>
      <div id="pe-plate-fields"></div>
    </div>
    <div class="detail-section">
      <div class="select-row"><span class="slider-label">Validated by</span>
        <input type="text" id="pe-by" class="val-input" placeholder="Name (optional for draft)"></div>
      <button class="validate-btn" onclick="saveProfile('${profileId}', true)">Save &amp; sign off</button>
      <button class="export-btn" onclick="saveProfile('${profileId}', false)">Save draft</button>
      <div class="val-msg" id="pe-msg"></div>
    </div>`;

  const sel = document.getElementById('pe-plate');
  sel.addEventListener('change', () => renderProfilePlateFields(prof, sel.value));
  renderProfilePlateFields(prof, sel.value);
}

function renderProfilePlateFields(prof, plateType) {
  const host = document.getElementById('pe-plate-fields');
  if (!host) return;
  const plates = (prof.biology || {}).plates || {};
  const pb = Object.assign({}, plates.default || {}, plates[plateType] || {});
  const sel = (id, vocab, cur) => `<select class="filter-select" id="${id}">` +
    vocab.map(o => `<option value="${o}" ${o === cur ? 'selected' : ''}>${o}</option>`).join('') + `</select>`;
  host.innerHTML = `
    <div class="select-row"><span class="slider-label">Colony size</span>${sel('pe-size', SIZE_VOCAB, pb.colony_size || 'medium')}</div>
    <div class="select-row"><span class="slider-label">Tolerance</span>${sel('pe-tol', TOLERANCE_VOCAB, pb.size_tolerance || 'normal')}</div>
    <div class="select-row"><span class="slider-label">Hemolysis</span>${sel('pe-hem', HEMOLYSIS_VOCAB, pb.hemolysis || 'unknown')}</div>
    <div class="select-row"><span class="slider-label">Lactose</span>${sel('pe-lac', LACTOSE_VOCAB, pb.lactose || 'unknown')}</div>
    <div class="select-row"><span class="slider-label">Colour</span>
      <input type="text" class="val-input" id="pe-color" value="${pb.colony_color || ''}" placeholder="e.g. pink"></div>
    <div class="select-row"><span class="slider-label">Notes</span>
      <input type="text" class="val-input" id="pe-notes" value="${pb.notes || ''}" placeholder="optional"></div>`;
}

async function saveProfile(profileId, signoff) {
  const by = document.getElementById('pe-by').value.trim();
  const msg = document.getElementById('pe-msg');
  if (signoff && !by) { msg.textContent = 'Enter a name to sign off.'; msg.className = 'val-msg err'; return; }
  const payload = {
    organism_biology: {
      swarming: document.getElementById('pe-swarming').checked,
      incubation_time_h_min: parseInt(document.getElementById('pe-inc-min').value) || 18,
      incubation_time_h_max: parseInt(document.getElementById('pe-inc-max').value) || 24,
    },
    plate_type: document.getElementById('pe-plate').value,
    plate_biology: {
      colony_size:    document.getElementById('pe-size').value,
      size_tolerance: document.getElementById('pe-tol').value,
      hemolysis:      document.getElementById('pe-hem').value,
      lactose:        document.getElementById('pe-lac').value,
      colony_color:   document.getElementById('pe-color').value || null,
      notes:          document.getElementById('pe-notes').value || '',
    },
    signoff: signoff ? { by } : null,
  };
  const r = await fetch(`/api/profile/${profileId}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).catch(() => null);
  if (r && r.ok) {
    msg.textContent = signoff ? `Signed off by ${by}.` : 'Saved.';
    msg.className = 'val-msg ok';
    await loadProfilesData(); await loadProfiles();
    if (document.getElementById('view-profiles')?.classList.contains('active')) filterProfilesView();
  } else { msg.textContent = 'Save failed.'; msg.className = 'val-msg err'; }
}

// ── Settings ──────────────────────────────────────────────────────────────────
function bindSettings() {
  document.querySelectorAll('#settings-tabs .seg-btn').forEach(b => {
    b.addEventListener('click', () => {
      document.querySelectorAll('#settings-tabs .seg-btn').forEach(x => x.classList.remove('active'));
      document.querySelectorAll('.settings-panel').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      document.getElementById(`tab-${b.dataset.tab}`).classList.add('active');
      if (b.dataset.tab === 'diagnostics') loadDiagnostics();
      if (b.dataset.tab === 'about')       loadAbout();
    });
  });

  document.getElementById('set-diameter')?.addEventListener('change', e =>
    saveSetting({ plate_diameter_mm: parseInt(e.target.value) }));
  document.getElementById('set-autoq')?.addEventListener('change', e =>
    saveSetting({ auto_quantify: e.target.checked }));
  document.getElementById('set-organism')?.addEventListener('change', e =>
    saveSetting({ default_profile_id: e.target.value }));
  document.getElementById('set-plate')?.addEventListener('change', e =>
    saveSetting({ default_plate_type: e.target.value }));

  const z = document.getElementById('set-z');
  z?.addEventListener('input', () => {
    const v = (parseInt(z.value) / 10);
    document.getElementById('set-z-val').textContent = v.toFixed(1);
    document.getElementById('set-z-word').textContent = zWord(v);
  });
  z?.addEventListener('change', () => saveSetting({ anomaly_z_thresh: parseInt(z.value) / 10 }));
}

function zWord(z) {
  if (z < 2.0) return 'Very sensitive';
  if (z < 2.5) return 'Sensitive';
  if (z === 2.5) return 'Balanced';
  if (z <= 3.0) return 'Conservative';
  return 'Very conservative';
}

async function openSettings() {
  // activate General tab
  document.querySelectorAll('#settings-tabs .seg-btn').forEach(x => x.classList.remove('active'));
  document.querySelectorAll('.settings-panel').forEach(x => x.classList.remove('active'));
  document.querySelector('#settings-tabs .seg-btn[data-tab="general"]')?.classList.add('active');
  document.getElementById('tab-general')?.classList.add('active');
  await loadSettings();
}

async function loadSettings() {
  if (!profilesData.profiles.length) await loadProfilesData();
  const cfg = await fetch('/api/settings').then(r => r.ok ? r.json() : null).catch(() => null);
  if (!cfg) return;
  const diam = document.getElementById('set-diameter'); if (diam) diam.value = String(cfg.plate_diameter_mm);
  const autoq = document.getElementById('set-autoq'); if (autoq) autoq.checked = !!cfg.auto_quantify;
  const z = document.getElementById('set-z');
  if (z) { z.value = Math.round(cfg.anomaly_z_thresh * 10);
    document.getElementById('set-z-val').textContent = cfg.anomaly_z_thresh.toFixed(1);
    document.getElementById('set-z-word').textContent = zWord(cfg.anomaly_z_thresh); }

  const org = document.getElementById('set-organism');
  if (org && org.options.length <= 1) {
    profilesData.profiles.forEach(p => org.insertAdjacentHTML('beforeend',
      `<option value="${p.profile_id}">${p.display_name}</option>`));
  }
  if (org) org.value = cfg.default_profile_id;
  const plate = document.getElementById('set-plate');
  if (plate && plate.options.length <= 1) {
    Object.entries(profilesData.plate_types || {}).forEach(([c, n]) =>
      plate.insertAdjacentHTML('beforeend', `<option value="${c}">${n}</option>`));
  }
  if (plate) plate.value = cfg.default_plate_type;
  settingsLoaded = true;
}

async function saveSetting(partial) {
  const el = document.getElementById('settings-saved');
  await fetch('/api/settings', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(partial),
  }).catch(() => null);
  if (el) { el.textContent = 'Saved'; el.classList.add('show'); setTimeout(() => el.classList.remove('show'), 1200); }
}

// ── ML Diagnostics ────────────────────────────────────────────────────────────
async function loadDiagnostics() {
  const host = document.getElementById('diag-content');
  if (!host) return;
  host.innerHTML = '<div class="spinner-wrap"><div class="spinner"></div></div>';
  const d = await fetch('/api/diagnostics').then(r => r.ok ? r.json() : null).catch(() => null);
  if (!d) { host.innerHTML = '<p style="color:var(--red)">Diagnostics unavailable.</p>'; return; }
  renderDiagnostics(d);
}

function renderDiagnostics(d) {
  const c = d.counting || {}, a = d.anomaly || {};
  const host = document.getElementById('diag-content');
  const pct = v => (v == null ? '—' : `${Math.round(v * 100)}%`);
  const num = v => (v == null ? '—' : v);

  const countCards = c.validated_plates
    ? `<div class="diag-cards">
        <div class="mc"><div class="mc-l">Avg count error</div><div class="mc-v">±${num(c.avg_count_error)}</div><div class="mc-s">${num(c.within_1_pct)}% within ±1</div></div>
        <div class="mc"><div class="mc-l">Detection precision</div><div class="mc-v">${pct(c.detection_precision)}</div><div class="mc-s">${c.false_positives} false positive${c.false_positives === 1 ? '' : 's'}</div></div>
        <div class="mc"><div class="mc-l">Detection recall</div><div class="mc-v">${pct(c.detection_recall)}</div><div class="mc-s mc-warn">${c.missed} missed</div></div>
        <div class="mc"><div class="mc-l">Validated plates</div><div class="mc-v">${c.validated_plates}</div><div class="mc-s">${c.colonies_validated} colonies</div></div>
      </div>
      <div class="card"><div class="card-title" style="margin-bottom:10px">Auto vs validated count</div>${svgCountChart(c.plates || [])}
        <div class="chart-legend"><span><i class="dot dot-green"></i>validated</span><span><i class="dot dot-blue"></i>auto</span></div></div>`
    : `<div class="diag-empty">No validated plates yet. Validate captures in the Library to measure counting accuracy.</div>`;

  const anomCards = a.trained
    ? `<div class="diag-pe">
        <div class="peb"><div class="peb-v">${pct(a.recall)}</div><div class="peb-l">of real anomalies caught <span class="hint">(recall)</span></div></div>
        <div class="peb"><div class="peb-v">${pct(a.precision)}</div><div class="peb-l">of flags are correct <span class="hint">(precision)</span></div></div>
        <div class="peb"><div class="peb-v">${a.cv_f1 != null ? a.cv_f1.toFixed(2) : '—'}</div><div class="peb-l">overall score <span class="hint">(F1)</span></div></div>
      </div>` + featureBars(a.feature_importances)
    : `<div class="diag-empty">Model not trained yet. ${a.n_labelled}/${a.min_labels} labelled colonies — ${a.ready_to_train ? 'ready to train.' : `${a.min_labels - a.n_labelled} more needed.`}</div>`;

  const progress = Math.min(100, Math.round(100 * a.n_labelled / a.min_labels));
  host.innerHTML = `
    <div class="diag-sec">Counting &amp; placement accuracy</div>
    ${countCards}
    <div class="diag-sec">Anomaly classifier</div>
    <div class="card">
      <div class="prog-row"><span>Labelling progress</span><span>${a.n_labelled} / ${a.min_labels}</span></div>
      <div class="prog-bar"><div style="width:${progress}%"></div></div>
    </div>
    ${anomCards}
    <div class="diag-actions">
      <button class="validate-btn" id="retrain-btn" ${a.ready_to_train ? '' : 'disabled'} onclick="retrain()">${a.trained ? 'Retrain now' : 'Train model'}</button>
      ${a.model_card ? `<a class="export-btn" href="/${a.model_card}" target="_blank">View model card ↗</a>` : ''}
    </div>`;
}

function featureBars(fi) {
  if (!fi || !fi.length) return '';
  const max = Math.max(...fi.map(f => f.importance)) || 1;
  return `<div class="card"><div class="card-title" style="margin-bottom:10px">What the model weighs most</div>` +
    fi.slice(0, 6).map(f =>
      `<div class="fbar"><span class="fbar-n">${f.feature}</span>
        <span class="fbar-t"><span style="width:${Math.round(100 * f.importance / max)}%"></span></span>
        <span class="fbar-v">${f.importance.toFixed(2)}</span></div>`).join('') + `</div>`;
}

// Inline SVG dual-line chart — no external library (Pi runs offline).
function svgCountChart(plates) {
  if (!plates.length) return '<div class="diag-empty">No data.</div>';
  const W = 600, H = 170, pad = 26;
  const maxY = Math.max(1, ...plates.map(p => Math.max(p.auto, p.manual)));
  const x = i => pad + (plates.length === 1 ? (W - 2 * pad) / 2 : i * (W - 2 * pad) / (plates.length - 1));
  const y = v => H - pad - (v / maxY) * (H - 2 * pad);
  const line = (key, color, dash) => {
    const pts = plates.map((p, i) => `${x(i).toFixed(1)},${y(p[key]).toFixed(1)}`).join(' ');
    const dots = plates.map((p, i) => `<circle cx="${x(i).toFixed(1)}" cy="${y(p[key]).toFixed(1)}" r="2.5" fill="${color}"/>`).join('');
    return `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" ${dash ? 'stroke-dasharray="5 4"' : ''}/>${dots}`;
  };
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Auto versus validated colony count per plate">
    <line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" stroke="rgba(0,0,0,0.12)"/>
    <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${H - pad}" stroke="rgba(0,0,0,0.12)"/>
    <text x="4" y="${pad + 4}" font-size="10" fill="#a1a1a6">${maxY}</text>
    <text x="8" y="${H - pad + 4}" font-size="10" fill="#a1a1a6">0</text>
    ${line('manual', '#34c759', false)}
    ${line('auto', '#0071e3', true)}
  </svg>`;
}

async function retrain() {
  const btn = document.getElementById('retrain-btn');
  if (btn) { btn.disabled = true; btn.textContent = 'Training…'; }
  setStatus('Training anomaly model…');
  const r = await fetch('/api/train', { method: 'POST' }).catch(() => null);
  if (!r || !r.ok) {
    const d = r ? await r.json().catch(() => ({})) : {};
    setStatus(d.message || 'Training failed');
    if (btn) { btn.disabled = false; btn.textContent = 'Train model'; }
  }
  // success handled via SSE train_done → loadDiagnostics()
}

// ── About ─────────────────────────────────────────────────────────────────────
async function loadAbout() {
  const host = document.getElementById('about-content');
  if (!host) return;
  const nValidated = (profilesData.profiles || []).filter(p => p.validated).length;
  const nProf = (profilesData.profiles || []).length;
  const nPlates = Object.keys(profilesData.plate_types || {}).length;
  host.innerHTML = `
    <div class="card">
      <div class="card-header-row"><span class="card-title">System</span></div>
      ${aboutRow('Camera', appState.camera ? 'Connected' : 'Demo mode', appState.camera ? 'badge-green' : 'badge-gray')}
      ${aboutRow('LED backlight', appState.led ? 'Connected' : 'Not connected', appState.led ? 'badge-green' : 'badge-gray')}
      ${metaRow('Organisms', nProf)}
      ${metaRow('Plate types', nPlates)}
      ${metaRow('Validated profiles', `${nValidated} signed off`)}
    </div>`;
}

function aboutRow(key, value, badgeCls) {
  return `<div class="detail-meta-row"><span class="detail-meta-key">${key}</span>
    <span class="badge ${badgeCls}">${value}</span></div>`;
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
