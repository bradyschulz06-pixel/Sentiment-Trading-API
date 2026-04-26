// ================================================================
//  STATUS PAGE — js/status.js
//  Reads latest state + run history from /runs collection
// ================================================================

// ── Current state ─────────────────────────────────────────────
db.collection('state').doc('latest')
  .onSnapshot(doc => {
    if (!doc.exists) return;
    const data   = doc.data();
    const run    = data.run    || {};
    const regime = data.regime || {};
    const config = data.config || {};

    setNavInfo(data.updated_at, run.status);
    renderHealthCards(run, regime, config);
    renderWarnings(data.warnings || []);
  });


// ── Run history (last 20 runs) ─────────────────────────────────
db.collection('runs')
  .orderBy('completed_at', 'desc')
  .limit(20)
  .onSnapshot(snap => {
    const runs = snap.docs.map(d => d.data());
    renderRunHistory(runs);
  },
  err => {
    document.getElementById('run-history').innerHTML =
      `<div class="alert alert-warn" style="margin:16px;">
         Run history not available. Make sure Firestore indexes are created.
       </div>`;
  });


// ── Health cards ──────────────────────────────────────────────

function renderHealthCards(run, regime, config) {
  // Last run time
  document.getElementById('sys-last-run').textContent = run.completed_at
    ? formatDate(run.completed_at) : '—';
  document.getElementById('sys-trigger').textContent  = 'trigger: ' + (run.trigger || '—');

  // Regime
  const label = (regime.label || 'unknown').replace('_', ' ').toUpperCase();
  const regEl = document.getElementById('sys-regime');
  regEl.textContent  = label;
  regEl.className    = 'stat-value ' + (regime.label === 'risk_on'
    ? 'pos' : regime.label === 'risk_off' ? 'neg' : '');
  document.getElementById('sys-regime-sub').textContent = regime.benchmark_symbol
    ? regime.benchmark_symbol + ' vs SMA50'
    : '—';

  // Auto-trade
  const atEl = document.getElementById('sys-autotrade');
  if (config.auto_trade_enabled != null) {
    atEl.textContent  = config.auto_trade_enabled ? 'ENABLED' : 'DISABLED';
    atEl.className    = 'stat-value ' + (config.auto_trade_enabled ? 'pos' : 'neg');
  } else {
    atEl.textContent = '—';
  }
}


// ── Warnings list ─────────────────────────────────────────────

function renderWarnings(warnings) {
  const el = document.getElementById('warnings-list');
  // Filter out the regime warning prefix (it's structural, not a user warning)
  const userWarnings = warnings.filter(w => !w.startsWith('MARKET_REGIME::'));

  if (!userWarnings.length) {
    el.innerHTML = `
      <div class="state-box" style="padding:24px;">
        <span style="font-size:20px">✅</span>
        <span>No warnings in the last engine run.</span>
      </div>`;
    return;
  }

  el.innerHTML = userWarnings.map(w =>
    `<div class="alert alert-warn">${w}</div>`
  ).join('');
}


// ── Run history ───────────────────────────────────────────────

function renderRunHistory(runs) {
  const el = document.getElementById('run-history');
  if (!runs.length) {
    el.innerHTML = emptyHTML('No run history yet.');
    return;
  }

  el.innerHTML = runs.map(r => {
    const signalInfo = r.signal_count != null
      ? `${r.buy_count || 0} buy / ${r.signal_count || 0} signals`
      : '';
    const posInfo = r.position_count != null
      ? ` · ${r.position_count} positions` : '';

    return `
    <div class="run-row">
      <div>
        ${statusBadge(r.status)}
        <span class="run-trigger">${r.trigger || 'manual'}</span>
      </div>
      <div class="run-body">
        <div>${r.summary || '—'}</div>
        ${signalInfo
          ? `<div style="font-size:11px;color:var(--muted);margin-top:3px;">
               ${signalInfo}${posInfo}
               ${r.regime_label ? ' · regime: <strong>' + r.regime_label.replace('_',' ') + '</strong>' : ''}
             </div>`
          : ''}
        ${r.error
          ? `<div class="alert alert-error" style="margin-top:6px;font-size:12px;">${r.error}</div>`
          : ''}
      </div>
      <div class="run-time">${formatDate(r.completed_at)}</div>
    </div>`;
  }).join('');
}
