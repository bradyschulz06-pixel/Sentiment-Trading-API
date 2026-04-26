// ================================================================
//  PERFORMANCE PAGE — js/performance.js
//  Reads from /performance/summary (written by sync script)
// ================================================================

// Chart instances — keep references to destroy on re-render
let equityChart   = null;
let winlossChart  = null;
let drawdownChart = null;

// ── Real-time listener ─────────────────────────────────────────
db.collection('performance').doc('summary')
  .onSnapshot(doc => {
    if (!doc.exists) {
      document.getElementById('trade-table').innerHTML = `
        <div class="alert alert-info" style="margin:16px;">
          No performance data yet. Run the sync script with --journal or after a backtest.
        </div>`;
      setNavInfo(null, null);
      return;
    }

    const p = doc.data();
    setNavInfo(p.updated_at, 'ok');
    renderStats(p);
    renderCharts(p);
    renderTradeTable(p.closed_trades || []);
  });


// ── Stat cards ────────────────────────────────────────────────

function renderStats(p) {
  // Total return
  const retEl = document.getElementById('stat-return');
  if (p.total_return_pct != null) {
    retEl.textContent = fmtPctDirect(p.total_return_pct);
    retEl.className   = 'stat-value ' + pctClass(p.total_return_pct);
  }

  // vs benchmark
  if (p.benchmark_return_pct != null) {
    const out = p.total_return_pct - p.benchmark_return_pct;
    document.getElementById('stat-vs-bench').textContent =
      `vs ${p.benchmark_symbol || 'benchmark'}: `
      + fmtPctDirect(p.benchmark_return_pct)
      + ` (${out >= 0 ? '+' : ''}${out.toFixed(2)}%)`;
  }

  // Win rate
  if (p.win_rate_pct != null) {
    const wrEl = document.getElementById('stat-winrate');
    wrEl.textContent = fmtPctDirect(p.win_rate_pct);
    wrEl.className   = 'stat-value ' + pctClass(p.win_rate_pct - 50);
    document.getElementById('stat-trades').textContent =
      (p.total_trades || 0) + ' total trades';
  }

  // Max drawdown (already negative)
  if (p.max_drawdown_pct != null) {
    document.getElementById('stat-dd').textContent = fmtPctDirect(p.max_drawdown_pct);
  }

  // Sharpe
  if (p.sharpe_ratio != null) {
    const sharpeEl = document.getElementById('stat-sharpe');
    sharpeEl.textContent = fmt(p.sharpe_ratio);
    sharpeEl.className   = 'stat-value ' + pctClass(p.sharpe_ratio);
  }

  // Extra metrics
  if (p.average_trade_return_pct != null) {
    const avgEl = document.getElementById('stat-avg-trade');
    avgEl.textContent = fmtPctDirect(p.average_trade_return_pct);
    avgEl.className   = 'stat-value ' + pctClass(p.average_trade_return_pct);
  }
  if (p.best_trade_pnl  != null) document.getElementById('stat-best').textContent  = formatCurrency(p.best_trade_pnl);
  if (p.worst_trade_pnl != null) document.getElementById('stat-worst').textContent = formatCurrency(p.worst_trade_pnl);
}


// ── Charts ────────────────────────────────────────────────────

// Shared Chart.js default colours
const C = {
  blue:    '#58a6ff',
  yellow:  '#e3b341',
  green:   '#3fb950',
  red:     '#f85149',
  grid:    'rgba(48,54,61,0.7)',
  text:    '#8b949e',
};

function chartDefaults() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { mode: 'index', intersect: false }
    },
    scales: {
      x: {
        ticks: { color: C.text, maxTicksLimit: 8, font: { size: 10 } },
        grid:  { color: C.grid }
      },
      y: {
        ticks: { color: C.text, font: { size: 10 } },
        grid:  { color: C.grid }
      }
    }
  };
}

function renderCharts(p) {
  const curve = p.equity_curve || [];

  // ── Equity curve ──
  if (curve.length) {
    const labels     = curve.map(pt => formatDateShort(pt.date));
    const strategy   = curve.map(pt => pt.equity);
    const benchmark  = curve.map(pt => pt.benchmark_equity);

    if (equityChart) equityChart.destroy();
    equityChart = new Chart(
      document.getElementById('equity-chart').getContext('2d'),
      {
        type: 'line',
        data: {
          labels,
          datasets: [
            {
              label: 'Strategy',
              data:  strategy,
              borderColor:     C.blue,
              backgroundColor: 'rgba(88,166,255,0.08)',
              borderWidth: 2,
              pointRadius: 0,
              fill: true,
              tension: 0.2
            },
            {
              label: 'Benchmark',
              data:  benchmark,
              borderColor:     C.yellow,
              borderWidth:     1.5,
              borderDash:      [4, 3],
              pointRadius:     0,
              fill:            false,
              tension:         0.2
            }
          ]
        },
        options: {
          ...chartDefaults(),
          scales: {
            ...chartDefaults().scales,
            y: {
              ...chartDefaults().scales.y,
              ticks: {
                ...chartDefaults().scales.y.ticks,
                callback: v => '$' + (v / 1000).toFixed(0) + 'k'
              }
            }
          }
        }
      }
    );
  } else {
    document.getElementById('equity-chart').closest('.chart-container').innerHTML =
      '<div class="state-box">No equity curve data yet.</div>';
  }

  // ── Win / Loss donut ──
  const wins   = p.wins   || 0;
  const losses = p.losses || 0;
  if (wins + losses > 0) {
    if (winlossChart) winlossChart.destroy();
    winlossChart = new Chart(
      document.getElementById('winloss-chart').getContext('2d'),
      {
        type: 'doughnut',
        data: {
          labels: ['Wins', 'Losses'],
          datasets: [{
            data:            [wins, losses],
            backgroundColor: [C.green, C.red],
            borderColor:     ['#161b22', '#161b22'],
            borderWidth:     3
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: true,
              position: 'bottom',
              labels: { color: C.text, font: { size: 11 } }
            },
            tooltip: {
              callbacks: {
                label: ctx => ` ${ctx.label}: ${ctx.parsed} trades`
              }
            }
          },
          cutout: '60%'
        }
      }
    );
  }

  // ── Drawdown chart ──
  if (curve.length) {
    // Calculate drawdown from equity curve
    let peak = -Infinity;
    const drawdown = curve.map(pt => {
      peak = Math.max(peak, pt.equity);
      return peak > 0 ? ((pt.equity - peak) / peak) * 100 : 0;
    });

    if (drawdownChart) drawdownChart.destroy();
    drawdownChart = new Chart(
      document.getElementById('drawdown-chart').getContext('2d'),
      {
        type: 'line',
        data: {
          labels: curve.map(pt => formatDateShort(pt.date)),
          datasets: [{
            label: 'Drawdown',
            data:  drawdown,
            borderColor:     C.red,
            backgroundColor: 'rgba(248,81,73,0.12)',
            borderWidth:     1.5,
            pointRadius:     0,
            fill:            true,
            tension:         0.2
          }]
        },
        options: {
          ...chartDefaults(),
          scales: {
            ...chartDefaults().scales,
            y: {
              ...chartDefaults().scales.y,
              ticks: {
                ...chartDefaults().scales.y.ticks,
                callback: v => v.toFixed(1) + '%'
              }
            }
          }
        }
      }
    );
  }
}


// ── Closed trade table ────────────────────────────────────────

function renderTradeTable(trades) {
  const el = document.getElementById('trade-table');
  if (!trades.length) {
    el.innerHTML = emptyHTML('No closed trade data. Run the journal sync to populate this.');
    return;
  }

  const rows = [...trades]
    .sort((a, b) => (b.exit_at || '').localeCompare(a.exit_at || ''))
    .slice(0, 50)
    .map(t => {
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
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}
