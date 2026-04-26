// ================================================================
//  COMMON UTILITIES  –  shared across every page
// ================================================================

// ---------- Formatting helpers ----------

function formatCurrency(val) {
  if (val == null || isNaN(val)) return '—';
  return '$' + parseFloat(val).toLocaleString('en-US', {
    minimumFractionDigits: 2, maximumFractionDigits: 2
  });
}

// Score or ratio with N decimals
function fmt(val, decimals = 2) {
  if (val == null || isNaN(val)) return '—';
  return parseFloat(val).toFixed(decimals);
}

// Format a percentage that is stored as a fraction (0.12 → "12.00%")
function fmtPct(val, decimals = 2) {
  if (val == null || isNaN(val)) return '—';
  return (parseFloat(val) * 100).toFixed(decimals) + '%';
}

// Format a percentage that is already a percent value (12.5 → "12.50%")
function fmtPctDirect(val, decimals = 2) {
  if (val == null || isNaN(val)) return '—';
  return parseFloat(val).toFixed(decimals) + '%';
}

// ISO timestamp → "Apr 26, 2026 14:30"
function formatDate(iso) {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
         + ' ' + d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  } catch (_) { return iso; }
}

// ISO timestamp → "Apr 26"
function formatDateShort(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  } catch (_) { return iso; }
}

// ---------- CSS class helpers ----------

function pctClass(val) {
  if (val == null) return 'neu';
  return val > 0 ? 'pos' : val < 0 ? 'neg' : 'neu';
}

// ---------- HTML component helpers ----------

function decisionBadge(decision) {
  const d = (decision || '').toLowerCase();
  if (d === 'buy')   return `<span class="badge badge-buy">Buy</span>`;
  if (d === 'watch') return `<span class="badge badge-watch">Watch</span>`;
  return `<span class="badge badge-skip">Skip</span>`;
}

function statusBadge(status) {
  const s = (status || '').toLowerCase();
  if (s === 'ok')    return `<span class="badge badge-ok">OK</span>`;
  if (s === 'error') return `<span class="badge badge-error">Error</span>`;
  return `<span class="badge badge-info">${status}</span>`;
}

// Visual score bar  (score is 0.0–1.0)
function scoreBar(score) {
  if (score == null) return '—';
  const pct  = Math.min(100, Math.max(0, score * 100));
  const cls  = pct >= 60 ? 'high' : pct >= 30 ? 'med' : 'low';
  return `
    <div class="score-bar">
      <div class="score-track">
        <div class="score-fill ${cls}" style="width:${pct.toFixed(1)}%"></div>
      </div>
      <span class="score-num">${score.toFixed(2)}</span>
    </div>`;
}

// Loading spinner HTML
function loadingHTML() {
  return `<div class="state-box"><div class="spinner"></div><span>Loading live data…</span></div>`;
}

// Empty state HTML
function emptyHTML(msg = 'No data available.') {
  return `<div class="state-box"><span style="font-size:28px">📭</span><span>${msg}</span></div>`;
}

// ---------- Nav helpers ----------

// Highlight the current page link in the nav
function setActiveNav() {
  const page = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('nav a.nav-link[data-page]').forEach(a => {
    a.classList.toggle('active', a.dataset.page === page);
  });
}

// Update the "last updated" text in the nav
function setNavInfo(isoStr, runStatus) {
  const el  = document.getElementById('nav-updated');
  const dot = document.getElementById('nav-dot');
  if (el)  el.textContent  = isoStr ? 'Updated ' + formatDate(isoStr) : 'Waiting for data…';
  if (dot) dot.className   = 'status-dot ' + (runStatus === 'ok' ? 'ok' : runStatus === 'error' ? 'error' : '');
}

// ---------- Regime banner ----------

function renderRegimeBanner(containerId, regime) {
  const el = document.getElementById(containerId);
  if (!el) return;
  if (!regime) { el.style.display = 'none'; return; }

  const label  = (regime.label  || 'inactive').toLowerCase().replace(' ', '_');
  const symbol = regime.benchmark_symbol || 'SPY';
  const ret21  = regime.ret21  != null ? (regime.ret21 * 100).toFixed(1) + '%' : '—';
  const sma50  = regime.sma50  != null ? formatCurrency(regime.sma50)          : '—';
  const price  = regime.benchmark_price != null ? formatCurrency(regime.benchmark_price) : '—';

  el.className = 'regime-banner ' + label;
  el.innerHTML = `
    <span class="regime-pill">${label.replace('_', ' ')}</span>
    <span class="regime-summary">${regime.summary || '—'}</span>
    <span class="regime-meta">
      <strong>${symbol}</strong> ${price} &nbsp;·&nbsp; SMA50 ${sma50} &nbsp;·&nbsp; 21d&nbsp;ret ${ret21}
    </span>`;
}

// ---------- Auto-run on every page ----------
document.addEventListener('DOMContentLoaded', setActiveNav);
