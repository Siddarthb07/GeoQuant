const state = {
  longCandidates: [],
  shortCandidates: [],
  candidateMap: {},
  selected: null,
  selectedKey: null,
  chartTimeframe: "5m",
  autoChart: true,
};
let lastTrainState = null;
let chartRefreshTimer = null;

const $ = (id) => document.getElementById(id);

function candidateKey(item) {
  return `${item.symbol}__${item.direction}`;
}

async function api(path, options = {}) {
  const resp = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || `API error ${resp.status}`);
  }
  return resp.json();
}

function formatPct(v) {
  return `${(v * 100).toFixed(2)}%`;
}

function formatNum(v, d = 2) {
  return Number(v).toFixed(d);
}

function pnlClass(v) {
  return Number(v) >= 0 ? "long" : "short";
}

function renderNews(items) {
  const el = $("newsList");
  if (!items.length) {
    el.innerHTML = "<p class='status'>No news feed returned. Try Refresh Signals.</p>";
    return;
  }
  el.innerHTML = "";
  items.forEach((item) => {
    const sentimentClass = item.sentiment >= 0 ? "long" : "short";
    const div = document.createElement("div");
    div.className = "news-item";
    div.innerHTML = `
      <div class="news-title">
        <a href="${item.link}" target="_blank" rel="noopener">${item.title}</a>
      </div>
      <div class="news-meta">
        ${new Date(item.published_at).toLocaleString()} | ${item.source}
      </div>
      <div class="news-meta ${sentimentClass}">
        Sentiment: ${item.sentiment.toFixed(3)} | Relevance: ${item.relevance.toFixed(2)}
      </div>
      <div class="news-tags">${(item.tags || []).join(", ")}</div>
    `;
    el.appendChild(div);
  });
}

function renderNewsImpact(items) {
  const el = $("newsImpact");
  if (!items.length) {
    el.innerHTML = "No strong stock-news impact detected in latest headlines.";
    return;
  }
  el.innerHTML =
    `<div><strong>Most Affected By Current News</strong></div>` +
    items
      .slice(0, 10)
      .map(
        (item) => `
      <div class="impact-row">
        <div><strong>${item.symbol}</strong> <span class="${item.expected_move === "UP" ? "long" : "short"}">${item.expected_move}</span> (${formatNum(item.impact_score, 3)})</div>
        <div class="news-meta">${(item.themes || []).join(", ")}</div>
      </div>
    `
      )
      .join("");
}

function candidateCard(c) {
  const directionClass = c.direction === "LONG" ? "long" : "short";
  const key = candidateKey(c);
  return `
    <div class="candidate-row ${state.selectedKey === key ? "active" : ""}" data-key="${key}">
      <div class="candidate-symbol">
        <span>${c.symbol}</span>
        <span class="badge ${directionClass}">${c.direction}</span>
      </div>
      <div class="mini-grid">
        <div>Exp Profit: ${formatNum(c.expected_profit_pct)}%</div>
        <div>Risk: ${formatNum(c.risk_score)}</div>
        <div>R:R: ${formatNum(c.risk_reward)}</div>
        <div>Exp Acc: ${formatPct(c.expected_accuracy)}</div>
        <div>Next Day Up: ${formatPct(c.next_day_up_prob)}</div>
        <div>Intraday Up: ${formatPct(c.intraday_up_prob)}</div>
      </div>
    </div>
  `;
}

function bindCandidateClicks(containerId) {
  const el = $(containerId);
  el.querySelectorAll(".candidate-row").forEach((row) => {
    row.addEventListener("click", () => {
      const key = row.dataset.key;
      const selected = state.candidateMap[key];
      if (selected) {
        selectCandidate(selected);
        renderCandidates();
      }
    });
  });
}

function renderCandidateGroup(containerId, list) {
  const el = $(containerId);
  if (!list.length) {
    el.innerHTML = "<p class='status'>No data yet.</p>";
    return;
  }
  el.innerHTML = list.map(candidateCard).join("");
  bindCandidateClicks(containerId);
}

function renderCandidates() {
  renderCandidateGroup("candidateLongTable", state.longCandidates);
  renderCandidateGroup("candidateShortTable", state.shortCandidates);
}

function setChartMeta(payload) {
  const delayedText = payload.delayed ? "Delayed exchange feed" : "End-of-day feed";
  $("chartMeta").textContent = `Source: ${payload.source} | Timeframe: ${payload.timeframe} | ${delayedText}`;
}

async function renderChart(symbol) {
  const bars = state.chartTimeframe === "1d" ? 250 : 320;
  const payload = await api(
    `/api/chart/${encodeURIComponent(symbol)}?bars=${bars}&timeframe=${encodeURIComponent(state.chartTimeframe)}`
  );
  const points = payload.points || [];
  if (!points.length) {
    $("chart").innerHTML = "<p class='status'>No chart data available.</p>";
    return;
  }
  setChartMeta(payload);

  const x = points.map((p) => p.timestamp);
  const trace = {
    x,
    open: points.map((p) => p.open),
    high: points.map((p) => p.high),
    low: points.map((p) => p.low),
    close: points.map((p) => p.close),
    type: "candlestick",
    name: symbol,
    increasing: { line: { color: "#37d399", width: 1.1 } },
    decreasing: { line: { color: "#ff6b6b", width: 1.1 } },
  };
  const sma20 = {
    x,
    y: points.map((p) => p.sma20),
    mode: "lines",
    line: { color: "#f5a524", width: 1.6 },
    name: "SMA20",
  };
  const sma50 = {
    x,
    y: points.map((p) => p.sma50),
    mode: "lines",
    line: { color: "#8ad8ff", width: 1.4 },
    name: "SMA50",
  };
  const vwap = {
    x,
    y: points.map((p) => p.vwap),
    mode: "lines",
    line: { color: "#c9ff6b", width: 1.2, dash: "dot" },
    name: "VWAP",
  };
  const volume = {
    x,
    y: points.map((p) => p.volume),
    type: "bar",
    yaxis: "y2",
    marker: { color: "rgba(142, 203, 255, 0.28)" },
    name: "Volume",
  };

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { l: 48, r: 38, t: 20, b: 30 },
    xaxis: {
      color: "#90a8b8",
      gridcolor: "rgba(255,255,255,0.05)",
      rangeslider: { visible: false },
      showspikes: true,
      spikemode: "across",
      spikecolor: "#7ea0b6",
      spikethickness: 1,
    },
    yaxis: {
      color: "#90a8b8",
      gridcolor: "rgba(255,255,255,0.06)",
      title: "Price",
    },
    yaxis2: {
      overlaying: "y",
      side: "right",
      showgrid: false,
      color: "#6f8798",
      title: "Vol",
      rangemode: "tozero",
    },
    hovermode: "x unified",
    dragmode: "zoom",
    showlegend: true,
    legend: { orientation: "h", y: 1.08, x: 0 },
  };

  Plotly.newPlot("chart", [trace, sma20, sma50, vwap, volume], layout, {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  });
}

function setChartAutoRefresh() {
  if (chartRefreshTimer) {
    clearInterval(chartRefreshTimer);
    chartRefreshTimer = null;
  }
  if (!state.autoChart || !state.selected) {
    return;
  }
  chartRefreshTimer = setInterval(() => {
    renderChart(state.selected.symbol).catch(() => {});
  }, 15000);
}

function selectCandidate(c) {
  state.selected = c;
  state.selectedKey = candidateKey(c);
  $("selectedSymbol").textContent = `${c.symbol} (${c.market})`;
  $("ticketSymbol").value = c.symbol;
  $("ticketMarket").value = c.market;
  $("ticketSide").value = c.direction === "LONG" ? "buy" : "sell";
  $("tradeDetails").innerHTML = `
    <p><strong>Setup:</strong> ${c.direction} bias with ${formatPct(c.combined_confidence)} confidence.</p>
    <p><strong>Entry:</strong> ${formatNum(c.entry_price)} | <strong>Stop:</strong> ${formatNum(c.stop_loss)} | <strong>Target:</strong> ${formatNum(c.target_price)}</p>
    <p><strong>Expected Profit:</strong> ${formatNum(c.expected_profit_pct)}% | <strong>R:R:</strong> ${formatNum(c.risk_reward)} | <strong>Max Loss:</strong> ${formatNum(c.max_loss_pct)}% | <strong>Take Profit:</strong> ${formatNum(c.take_profit_pct)}%</p>
    <p><strong>Expected Accuracy:</strong> ${formatPct(c.expected_accuracy)}</p>
    <p><strong>Accuracy Reasoning:</strong> ${c.accuracy_reasoning}</p>
    <p><strong>Model Rationale:</strong> ${c.rationale}</p>
  `;
  renderChart(c.symbol).catch((err) => {
    $("tradeDetails").innerHTML = `<p class="status short">${err.message}</p>`;
  });
  setChartAutoRefresh();
}

async function loadCandidates() {
  try {
    const previousKey = state.selectedKey;
    const payload = await api("/api/candidates/split?per_side=15");
    state.longCandidates = Array.isArray(payload.longs) ? payload.longs : [];
    state.shortCandidates = Array.isArray(payload.shorts) ? payload.shorts : [];
    state.candidateMap = {};
    [...state.longCandidates, ...state.shortCandidates].forEach((item) => {
      state.candidateMap[candidateKey(item)] = item;
    });

    state.selected = null;
    state.selectedKey = null;
    if (previousKey && state.candidateMap[previousKey]) {
      selectCandidate(state.candidateMap[previousKey]);
    } else if (state.longCandidates.length) {
      selectCandidate(state.longCandidates[0]);
    } else if (state.shortCandidates.length) {
      selectCandidate(state.shortCandidates[0]);
    } else {
      $("selectedSymbol").textContent = "No candidates available";
      $("chart").innerHTML = "<p class='status'>Chart will appear when candidates are available.</p>";
      $("tradeDetails").innerHTML =
        "<p class='status'>No candidate data received yet. Try Refresh Signals.</p>";
    }
    renderCandidates();
  } catch (err) {
    state.longCandidates = [];
    state.shortCandidates = [];
    state.candidateMap = {};
    state.selected = null;
    state.selectedKey = null;
    $("selectedSymbol").textContent = "Candidate load failed";
    $("candidateLongTable").innerHTML = `<p class='status short'>Failed to load: ${String(err.message || err)}</p>`;
    $("candidateShortTable").innerHTML = `<p class='status short'>Failed to load: ${String(err.message || err)}</p>`;
    $("chart").innerHTML = "<p class='status'>Chart unavailable while candidate API is failing.</p>";
    $("tradeDetails").innerHTML =
      "<p class='status short'>Resolve API error and click Refresh Signals.</p>";
  }
}

async function loadModelMetrics() {
  try {
    const m = await api("/api/model-metrics");
    $("modelAccuracy").innerHTML = `
      <div><strong>Model Quality</strong></div>
      <div>Daily Acc: ${formatPct(m.daily_accuracy)} | Daily AUC: ${formatNum(m.daily_auc, 3)}</div>
      <div>Intraday Acc: ${formatPct(m.intraday_accuracy)} | Intraday AUC: ${formatNum(m.intraday_auc, 3)}</div>
      <div>Expected Accuracy Band: <strong>${m.expected_accuracy_band}</strong></div>
      <div>${m.reasoning}</div>
    `;
  } catch (err) {
    $("modelAccuracy").innerHTML = `<span class='short'>Model metrics unavailable: ${String(err.message || err)}</span>`;
  }
}

async function loadSelfLearningStatus() {
  try {
    const s = await api("/api/self-learning/status");
    $("selfLearningStatus").innerHTML = `
      <div><strong>Self Learning</strong></div>
      <div>State: ${s.state.toUpperCase()} | Resolved Signals: ${s.resolved_signals}</div>
      <div>Rolling Hit-Rate: ${formatPct(s.rolling_hit_rate)} (${s.sample_size} samples)</div>
      <div>Cycle: ${s.cycle_minutes} min | Retrain: ${s.retrain_hours} hr</div>
      <div class="news-meta">${s.message}</div>
    `;
  } catch (err) {
    $("selfLearningStatus").innerHTML = `<span class='short'>Self-learning status unavailable: ${String(err.message || err)}</span>`;
  }
}

async function loadNews() {
  try {
    const data = await api("/api/news?limit=60");
    renderNews(data);
  } catch (err) {
    $("newsList").innerHTML = `<p class='status short'>News load failed: ${String(err.message || err)}</p>`;
  }
}

async function loadNewsImpact() {
  try {
    const data = await api("/api/news-impact?limit=15");
    renderNewsImpact(data);
  } catch (err) {
    $("newsImpact").innerHTML = `Impact map unavailable: ${String(err.message || err)}`;
  }
}

async function loadOrders() {
  try {
    const data = await api("/api/orders?limit=40");
    const el = $("ordersList");
    const items = data.items || [];
    if (!items.length) {
      el.innerHTML = "<p class='status'>No orders yet.</p>";
      return;
    }
    el.innerHTML = items
      .map(
        (o) => `
        <div class="order-row">
          <div><strong>${o.symbol}</strong> ${o.side.toUpperCase()} ${o.qty}</div>
          <div class="news-meta">${o.mode.toUpperCase()} | ${o.status}</div>
          <div class="news-meta">${new Date(o.created_at).toLocaleString()}</div>
        </div>
      `
      )
      .join("");
  } catch (err) {
    $("ordersList").innerHTML = `<p class='status short'>Orders unavailable: ${String(err.message || err)}</p>`;
  }
}

async function loadPortfolio() {
  try {
    const data = await api("/api/portfolio?limit=3000");
    const summary = data.summary || {};
    const items = data.items || [];

    $("portfolioSummary").innerHTML = `
      <div><strong>Portfolio P&L</strong></div>
      <div>Symbols Traded: ${summary.symbols_traded || 0} | Open Positions: ${summary.open_positions || 0}</div>
      <div>Realized: <span class="${pnlClass(summary.total_realized_pnl || 0)}">${formatNum(summary.total_realized_pnl || 0)}</span></div>
      <div>Unrealized: <span class="${pnlClass(summary.total_unrealized_pnl || 0)}">${formatNum(summary.total_unrealized_pnl || 0)}</span></div>
      <div>Total: <strong class="${pnlClass(summary.total_pnl || 0)}">${formatNum(summary.total_pnl || 0)}</strong></div>
    `;

    const listEl = $("portfolioList");
    if (!items.length) {
      listEl.innerHTML = "<p class='status'>No executed trades yet.</p>";
      return;
    }

    listEl.innerHTML = items
      .map(
        (p) => `
        <div class="order-row">
          <div><strong>${p.symbol}</strong> (${p.market}) | ${p.open_side} | Net Qty: ${formatNum(p.net_qty, 3)}</div>
          <div class="news-meta">Bought: ${formatNum(p.total_bought_qty, 3)} | Sold: ${formatNum(p.total_sold_qty, 3)} | Orders: ${p.executed_orders}</div>
          <div class="news-meta">Avg Entry: ${p.avg_entry_price == null ? "-" : formatNum(p.avg_entry_price, 4)} | Current: ${p.current_price == null ? "-" : formatNum(p.current_price, 4)}</div>
          <div class="news-meta">
            Realized: <span class="${pnlClass(p.realized_pnl)}">${formatNum(p.realized_pnl)}</span> |
            Unrealized: <span class="${pnlClass(p.unrealized_pnl)}">${formatNum(p.unrealized_pnl)}</span> |
            Total: <span class="${pnlClass(p.total_pnl)}">${formatNum(p.total_pnl)}</span>
          </div>
        </div>
      `
      )
      .join("");
  } catch (err) {
    $("portfolioSummary").innerHTML = `<span class='short'>P&L summary unavailable: ${String(err.message || err)}</span>`;
    $("portfolioList").innerHTML = `<p class='status short'>P&L list unavailable: ${String(err.message || err)}</p>`;
  }
}

async function updateTrainStatus() {
  try {
    const status = await api("/api/train/status");
    $("trainStatus").textContent = `${status.state.toUpperCase()}: ${status.message}`;
    if (status.state === "completed" && lastTrainState === "running") {
      await api("/api/reload-models", { method: "POST" });
      await Promise.all([loadCandidates(), loadModelMetrics(), loadSelfLearningStatus()]);
    }
    lastTrainState = status.state;
    loadSelfLearningStatus();
  } catch (err) {
    $("trainStatus").textContent = `ERROR: ${String(err.message || err)}`;
  }
}

async function refreshAll() {
  const results = await Promise.allSettled([
    loadNews(),
    loadNewsImpact(),
    loadCandidates(),
    loadOrders(),
    loadPortfolio(),
    updateTrainStatus(),
    loadModelMetrics(),
    loadSelfLearningStatus(),
  ]);
  const failedCalls = results.filter((r) => r.status === "rejected").length;
  if (state.longCandidates.length || state.shortCandidates.length) {
    $("trainStatus").textContent = `Updated: ${state.longCandidates.length} long, ${state.shortCandidates.length} short candidates loaded.`;
  } else if (failedCalls) {
    $("trainStatus").textContent = `Refresh completed with ${failedCalls} failed calls. Check panels for details.`;
  } else {
    $("trainStatus").textContent = "Refresh completed, but no candidates were returned.";
  }
}

$("refreshBtn").addEventListener("click", async () => {
  $("trainStatus").textContent = "Refreshing market and news context...";
  await refreshAll();
});

$("trainBtn").addEventListener("click", async () => {
  $("trainStatus").textContent = "Starting full retraining...";
  try {
    await api("/api/train", { method: "POST", body: JSON.stringify({ force: true }) });
    setTimeout(refreshAll, 1800);
  } catch (err) {
    $("trainStatus").textContent = `Retrain request failed: ${String(err.message || err)}`;
  }
});

$("placeOrderBtn").addEventListener("click", async () => {
  const symbol = $("ticketSymbol").value.trim();
  const market = $("ticketMarket").value.trim();
  const side = $("ticketSide").value;
  const qty = Number($("ticketQty").value || "0");
  const live = $("liveToggle").checked;
  if (!symbol || qty <= 0) {
    $("orderStatus").textContent = "Set symbol and qty before placing the order.";
    return;
  }
  $("orderStatus").textContent = "Placing order...";
  try {
    const data = await api("/api/order", {
      method: "POST",
      body: JSON.stringify({ symbol, side, qty, market, live }),
    });
    $("orderStatus").textContent = `${data.mode.toUpperCase()} ${data.status}: ${data.note}`;
    loadOrders();
    loadPortfolio();
  } catch (err) {
    $("orderStatus").textContent = String(err.message || err);
  }
});

$("timeframeSelect").addEventListener("change", async (event) => {
  state.chartTimeframe = event.target.value;
  if (state.selected) {
    await renderChart(state.selected.symbol);
  }
  setChartAutoRefresh();
});

$("autoChartToggle").addEventListener("change", (event) => {
  state.autoChart = event.target.checked;
  setChartAutoRefresh();
});

refreshAll();
setInterval(updateTrainStatus, 7000);
