// ================================================================
//  SIGNALS PAGE — js/signals.js
//  Reads signals array from /state/latest
// ================================================================

db.collection('state').doc('latest')
  .onSnapshot(doc => {
    if (!doc.exists) {
      document.getElementById('signals-table').innerHTML =
        '<div class="alert alert-info" style="margin:16px;">No data yet. Run the engine and sync script first.</div>';
      return;
    }

    const data     = doc.data();
    const run      = data.run      || {};
    const signals  = data.signals  || [];
    const regime   = data.regime;

    setNavInfo(data.updated_at, run.status);
    renderRegimeBanner('regime-banner', regime);
    renderRunMeta(run, signals);
    renderSignalsTable(signals);
  });


// ── Run metadata bar ──────────────────────────────────────────

function renderRunMeta(run, signals) {
  document.getElementById('meta-trigger').textContent   = run.trigger   || 'manual';
  document.getElementById('meta-count').textContent     = signals.length;
  document.getElementById('meta-buys').textContent      = signals.filter(s => s.decision === 'buy').length;
  document.getElementById('meta-threshold').textContent = run.signal_threshold != null
    ? run.signal_threshold.toFixed(2) : '—';
  document.getElementById('meta-time').textContent      = formatDate(run.completed_at);
}


// ── Full signals table ────────────────────────────────────────

function renderSignalsTable(signals) {
  const el = document.getElementById('signals-table');
  if (!signals.length) {
    el.innerHTML = emptyHTML('No signals in this run.');
    return;
  }

  // Sort: buys first, then by composite score descending
  const sorted = [...signals].sort((a, b) => {
    if (a.decision === 'buy' && b.decision !== 'buy') return -1;
    if (b.decision === 'buy' && a.decision !== 'buy') return  1;
    return (b.composite_score || 0) - (a.composite_score || 0);
  });

  const rows = sorted.map(s => {
    const stopStr   = s.stop_price   ? formatCurrency(s.stop_price)   : '—';
    const targetStr = s.target_price ? formatCurrency(s.target_price) : '—';
    const earningsStr = s.next_earnings_date
      ? `<span class="badge badge-info">${s.next_earnings_date}</span>` : '—';

    return `
    <tr>
      <td><strong>${s.symbol}</strong></td>
      <td>${formatCurrency(s.price)}</td>
      <td>${scoreBar(s.earnings_score)}</td>
      <td>${scoreBar(s.composite_score)}</td>
      <td>${decisionBadge(s.decision)}</td>
      <td>${stopStr}</td>
      <td>${targetStr}</td>
      <td>${earningsStr}</td>
      <td class="truncate hide-sm">${s.rationale || '—'}</td>
    </tr>`;
  }).join('');

  el.innerHTML = `
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Symbol</th>
          <th>Price</th>
          <th>Earnings Score</th>
          <th>Composite</th>
          <th>Decision</th>
          <th>Stop</th>
          <th>Target</th>
          <th>Next Earnings</th>
          <th class="hide-sm">Rationale</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}
