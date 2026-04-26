// ================================================================
//  DASHBOARD PAGE — js/dashboard.js
//  Listens to a single Firestore document: /state/latest
//  The Python sync script writes this doc after every engine run.
// ================================================================

// ── Single real-time listener ──────────────────────────────────
db.collection('state').doc('latest')
  .onSnapshot(doc => {
    if (!doc.exists) {
      document.getElementById('last-run').innerHTML =
        '<div class="alert alert-info" style="margin:16px;">No engine runs recorded yet. ' +
        'Run the sync script after your first engine run.</div>';
      return;
    }

    const data = doc.data();

    // Nav bar
    setNavInfo(data.updated_at, data.run && data.run.status);

    // Stat cards
    renderStats(data);

    // Regime banner
    renderRegimeBanner('regime-banner', data.regime);

    // Signals preview (top 5 buy decisions)
    const buys = (data.signals || [])
      .filter(s => s.decision === 'buy')
      .slice(0, 5);
    renderSignalsPreview(buys);

    // Positions preview
    renderPositionsPreview(data.positions || []);

    // Last run card
    renderLastRun(data.run || {});
  },
  err => {
    console.error('Firestore error:', err);
    document.getElementById('last-run').innerHTML =
      `<div class="alert alert-error">Could not connect to Firestore: ${err.message}</div>`;
  });

// ── Listener for performance summary (equity + account) ────────
db.collection('performance').doc('summary')
  .onSnapshot(doc => {
    if (!doc.exists) return;
    const p = doc.data();

    if (p.equity != null) {
      document.getElementById('stat-equity').textContent = formatCurrency(p.equity);
    }
    if (p.buying_power != null) {
      document.getElementById('stat-buying-power').textContent =
        'Buying power: ' + formatCurrency(p.buying_power);
    }
    if (p.win_rate_pct != null) {
      const el = document.getElementById('stat-win-rate');
      el.textContent = fmtPctDirect(p.win_rate_pct);
      el.className   = 'stat-value ' + pctClass(p.win_rate_pct - 50);
    }
    if (p.total_trades != null) {
      document.getElementById('stat-total-trades').textContent =
        p.total_trades + ' closed trades recorded';
    }
  });


// ── Render helpers ────────────────────────────────────────────

function renderStats(data) {
  const run = data.run || {};

  // Signal threshold from run meta
  if (run.signal_threshold != null) {
    document.getElementById('stat-threshold').textContent = fmt(run.signal_threshold);
  }

  // Open P&L from positions
  const positions = data.positions || [];
  if (positions.length) {
    // unrealized_plpc is a fraction (e.g. 0.05 = 5%)
    const totalPnl = positions.reduce((sum, p) => {
      return sum + (p.unrealized_plpc * p.avg_entry_price * p.qty);
    }, 0);
    const pnlEl = document.getElementById('stat-open-pnl');
    pnlEl.textContent  = formatCurrency(totalPnl);
    pnlEl.className    = 'stat-value ' + pctClass(totalPnl);
    document.getElementById('stat-positions-count').textContent =
      positions.length + ' open position' + (positions.length !== 1 ? 's' : '');
  } else {
    document.getElementById('stat-open-pnl').textContent = '$0.00';
    document.getElementById('stat-positions-count').textContent = 'No open positions';
  }
}

function renderLastRun(run) {
  if (!run.status) {
    document.getElementById('last-run').innerHTML = emptyHTML('No run data.');
    return;
  }
  document.getElementById('last-run').innerHTML = `
    <div class="run-row">
      <div class="run-body">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
          ${statusBadge(run.status)}
          <span class="run-trigger">${run.trigger || 'manual'}</span>
        </div>
        <div>${run.summary || '—'}</div>
        ${run.error
          ? `<div class="alert alert-error" style="margin-top:8px;">${run.error}</div>`
          : ''}
      </div>
      <div class="run-time">${formatDate(run.completed_at)}</div>
    </div>`;
}

function renderSignalsPreview(signals) {
  const el = document.getElementById('signals-preview');
  if (!signals.length) {
    el.innerHTML = emptyHTML('No buy signals in current run.');
    return;
  }

  const rows = signals.map(s => `
    <tr>
      <td><strong>${s.symbol}</strong></td>
      <td>${formatCurrency(s.price)}</td>
      <td>${scoreBar(s.earnings_score)}</td>
      <td>${decisionBadge(s.decision)}</td>
    </tr>`).join('');

  el.innerHTML = `
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Symbol</th><th>Price</th><th>Earnings Score</th><th>Signal</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <a href="signals.html" class="more-link">View all signals →</a>`;
}

function renderPositionsPreview(positions) {
  const el = document.getElementById('positions-preview');
  if (!positions.length) {
    el.innerHTML = emptyHTML('No open positions.');
    return;
  }

  const rows = positions.map(p => {
    const pnlPct = p.unrealized_plpc * 100;
    return `
    <tr>
      <td><strong>${p.symbol}</strong></td>
      <td>${p.qty} sh</td>
      <td>${formatCurrency(p.market_value)}</td>
      <td class="${pctClass(pnlPct)}">${pnlPct.toFixed(2)}%</td>
    </tr>`;
  }).join('');

  el.innerHTML = `
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Symbol</th><th>Qty</th><th>Mkt Value</th><th>P&amp;L %</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <a href="trades.html" class="more-link">View all trades →</a>`;
}
