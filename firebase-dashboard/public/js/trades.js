// ================================================================
//  TRADES PAGE — js/trades.js
//  Positions + trade intents from /state/latest
//  Closed trade history from /trades collection
// ================================================================

// ── Current positions + intents from /state/latest ────────────
db.collection('state').doc('latest')
  .onSnapshot(doc => {
    if (!doc.exists) {
      setEmptyAll();
      return;
    }
    const data = doc.data();
    setNavInfo(data.updated_at, data.run && data.run.status);
    renderPositionsTable(data.positions    || []);
    renderIntentsTable  (data.trade_intents || []);
  });


// ── Closed trade history from /trades collection ───────────────
// Listen for the 50 most recently closed trades
db.collection('trades')
  .orderBy('exit_at', 'desc')
  .limit(50)
  .onSnapshot(snap => {
    const trades = snap.docs.map(d => d.data());
    renderHistoryTable(trades);
  },
  err => {
    document.getElementById('history-table').innerHTML =
      `<div class="alert alert-warn" style="margin:16px;">
         Trade history not available yet. Run the journal sync to populate this.
       </div>`;
  });


// ── Render helpers ────────────────────────────────────────────

function setEmptyAll() {
  document.getElementById('positions-table').innerHTML = emptyHTML('No positions data.');
  document.getElementById('intents-table').innerHTML   = emptyHTML('No intent data.');
}

function renderPositionsTable(positions) {
  const el = document.getElementById('positions-table');
  if (!positions.length) {
    el.innerHTML = emptyHTML('No open positions right now.');
    return;
  }

  const rows = positions.map(p => {
    const pnlPct  = p.unrealized_plpc * 100;
    const pnlAmt  = p.unrealized_plpc * p.avg_entry_price * p.qty;
    return `
    <tr>
      <td><strong>${p.symbol}</strong></td>
      <td><span class="badge badge-info">${p.side || 'long'}</span></td>
      <td>${p.qty}</td>
      <td>${formatCurrency(p.avg_entry_price)}</td>
      <td>${formatCurrency(p.market_value)}</td>
      <td class="${pctClass(pnlPct)}">${pnlPct.toFixed(2)}%</td>
      <td class="${pctClass(pnlAmt)}">${formatCurrency(pnlAmt)}</td>
    </tr>`;
  }).join('');

  el.innerHTML = `
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Symbol</th><th>Side</th><th>Qty</th>
          <th>Avg Entry</th><th>Mkt Value</th>
          <th>P&amp;L %</th><th>P&amp;L $</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

function renderIntentsTable(intents) {
  const el = document.getElementById('intents-table');
  if (!intents.length) {
    el.innerHTML = emptyHTML('No trade intents in the last engine run.');
    return;
  }

  const rows = intents.map(t => `
    <tr>
      <td><strong>${t.symbol}</strong></td>
      <td>${t.side === 'buy'
        ? '<span class="badge badge-buy">Buy</span>'
        : '<span class="badge badge-error">Sell</span>'}</td>
      <td>${t.qty}</td>
      <td>${formatCurrency(t.notional)}</td>
      <td>${t.status || '—'}</td>
      <td class="truncate hide-sm">${t.reason || '—'}</td>
    </tr>`).join('');

  el.innerHTML = `
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Symbol</th><th>Side</th><th>Qty</th>
          <th>Notional</th><th>Status</th>
          <th class="hide-sm">Reason</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

function renderHistoryTable(trades) {
  const el = document.getElementById('history-table');
  if (!trades.length) {
    el.innerHTML = `
      <div class="alert alert-info" style="margin:16px;">
        No closed trades yet. The journal sync writes trades here after paper fills.
      </div>`;
    return;
  }

  const rows = trades.map(t => {
    const retPct = (t.return_pct || 0) * 100;
    return `
    <tr>
      <td><strong>${t.symbol}</strong></td>
      <td>${formatDateShort(t.entry_at)}</td>
      <td>${formatDateShort(t.exit_at)}</td>
      <td>${formatCurrency(t.entry_price)}</td>
      <td>${formatCurrency(t.exit_price)}</td>
      <td>${t.qty}</td>
      <td class="${pctClass(t.pnl)}">${formatCurrency(t.pnl)}</td>
      <td class="${pctClass(retPct)}">${retPct.toFixed(2)}%</td>
      <td>${t.hold_days != null ? t.hold_days + 'd' : '—'}</td>
      <td class="hide-sm">${t.dominant_factor || '—'}</td>
      <td class="truncate hide-sm">${t.entry_rationale || '—'}</td>
    </tr>`;
  }).join('');

  el.innerHTML = `
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Symbol</th><th>Entry</th><th>Exit</th>
          <th>Entry $</th><th>Exit $</th><th>Qty</th>
          <th>P&amp;L</th><th>Return</th><th>Days</th>
          <th class="hide-sm">Factor</th>
          <th class="hide-sm">Rationale</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}
