/**
 * AI Stock Price Predictor - LSTM & GRU
 * Production JavaScript Frontend Controller
 */

const API_BASE = window.location.origin.includes('5000') || window.location.origin.includes('localhost') || window.location.origin.includes('127.0.0.1')
  ? '' 
  : 'http://127.0.0.1:5000';

const state = {
  theme: 'dark',
  ticker: 'AAPL',
  startDate: '2022-01-01',
  endDate: '2024-01-01',
  sequenceLength: 60,
  dataset: [],
  allDates: [],
  allCloses: [],
  isDemo: false,
  dataSource: '',
  isBackendConnected: false,
  trainingInterval: null,
  activeChartView: 'both',
  results: null,
  futureForecast: [],
  page: 1,
  pageSize: 8
};

// Utilities
const formatMoney = (v) => `$${Number(v || 0).toFixed(2)}`;
const formatNumber = (v) => Number(v || 0).toLocaleString();

function showToast(message, type = 'info') {
  const toast = document.getElementById('appToast');
  const msg = document.getElementById('toastMsg');
  const icon = document.getElementById('toastIcon');
  if (!toast) return;

  msg.textContent = message;
  toast.className = `toast show ${type}`;
  
  if (type === 'success') icon.className = 'fa-solid fa-circle-check';
  else if (type === 'error') icon.className = 'fa-solid fa-circle-exclamation';
  else icon.className = 'fa-solid fa-circle-info';

  setTimeout(() => {
    toast.className = 'toast';
  }, 4000);
}

// Backend Connection and Data Loading
async function initApp() {
  setupEventListeners();
  updateTheme();
  
  // Test connection to backend
  try {
    const res = await fetch(`${API_BASE}/api/stock-data?ticker=${state.ticker}&start=${state.startDate}&end=${state.endDate}`);
    if (res.ok) {
      const data = await res.json();
      state.isBackendConnected = true;
      updateBackendStatus(true);
      handleStockDataLoaded(data);
      // Fetch results if already trained
      fetchResultsAndRender();
      return;
    }
  } catch (err) {
    console.warn('Backend server not directly reachable. Running in standalone demo mode:', err);
  }

  // Fallback to offline demo generator if backend is not started
  state.isBackendConnected = false;
  updateBackendStatus(false);
  loadOfflineDemoDataset(state.ticker);
}

function updateBackendStatus(connected) {
  const pill = document.getElementById('backendStatusPill');
  const text = document.getElementById('backendStatusText');
  if (!pill || !text) return;

  if (connected) {
    pill.style.background = 'rgba(42, 216, 154, 0.15)';
    pill.style.borderColor = 'rgba(42, 216, 154, 0.4)';
    pill.style.color = '#2ad89a';
    text.textContent = 'Real ML Backend Connected';
  } else {
    pill.style.background = 'rgba(245, 158, 11, 0.15)';
    pill.style.borderColor = 'rgba(245, 158, 11, 0.4)';
    pill.style.color = '#fbbf24';
    text.textContent = 'Offline Demo Mode';
  }
}

async function loadStockData(ticker) {
  state.ticker = ticker.toUpperCase();
  document.getElementById('tickerInput').value = state.ticker;
  document.getElementById('heroTicker').textContent = state.ticker;

  if (state.isBackendConnected) {
    try {
      showToast(`Downloading data for ${state.ticker}...`, 'info');
      const res = await fetch(`${API_BASE}/api/stock-data?ticker=${state.ticker}&start=${state.startDate}&end=${state.endDate}`);
      const data = await res.json();
      if (data.status === 'success') {
        handleStockDataLoaded(data);
        showToast(`Loaded ${data.total_records} records for ${state.ticker}`, 'success');
        fetchResultsAndRender();
      } else {
        showToast(data.message || 'Failed to download data', 'error');
      }
    } catch (e) {
      showToast(`API error: ${e.message}`, 'error');
    }
  } else {
    loadOfflineDemoDataset(state.ticker);
  }
}

function handleStockDataLoaded(data) {
  state.dataset = data.data || [];
  state.allDates = data.all_dates || [];
  state.allCloses = data.all_closes || [];
  state.isDemo = data.is_demo;
  state.dataSource = data.data_source;

  // Update Demo Data badge
  const demoContainer = document.getElementById('demoBadgeContainer');
  if (demoContainer) {
    if (state.isDemo) {
      demoContainer.innerHTML = `<span class="demo-badge"><i class="fa-solid fa-triangle-exclamation"></i> DEMO DATA (${state.dataSource})</span>`;
    } else {
      demoContainer.innerHTML = `<span class="demo-badge" style="background: rgba(42,216,154,0.15); border-color: rgba(42,216,154,0.35); color: #2ad89a;"><i class="fa-solid fa-check"></i> LIVE MARKET DATA</span>`;
    }
  }

  // Update statistical summary cards
  if (data.stats) {
    document.getElementById('statRows').textContent = formatNumber(data.stats.total_rows);
    document.getElementById('statDateRange').textContent = `Date Range: ${data.stats.date_range.start} to ${data.stats.date_range.end}`;
    document.getElementById('statAvgClose').textContent = formatMoney(data.stats.close_avg);
    document.getElementById('statHighLow').textContent = `${formatMoney(data.stats.close_max)} / ${formatMoney(data.stats.close_min)}`;
  }

  // Update Preprocessing Info
  if (data.preprocessing_info) {
    updatePreprocessingCards(data.preprocessing_info);
  }

  // Update KPIs and Hero
  updateDashboardKpis();
  renderHeroChart();
  renderTable();
}

function updatePreprocessingCards(info) {
  document.getElementById('prepTotalRecords').textContent = formatNumber(info.total_records);
  document.getElementById('prepTrainRecords').textContent = formatNumber(info.train_records);
  document.getElementById('prepTestRecords').textContent = formatNumber(info.test_records);
  document.getElementById('prepMissing').textContent = info.missing_values_handled || '0';
  document.getElementById('prepTarget').textContent = `${info.target_feature} Price`;
  document.getElementById('prepNorm').textContent = info.normalization || 'MinMaxScaler (0, 1)';
  document.getElementById('heroSeqLen').textContent = `${info.sequence_length || state.sequenceLength} Days`;
}

function updateDashboardKpis() {
  if (state.dataset.length === 0) return;
  const latest = state.dataset[state.dataset.length - 1];
  const prev = state.dataset[state.dataset.length - 2] || latest;
  const change = latest.Close - prev.Close;
  const pct = (change / (prev.Close || 1)) * 100;
  const sign = change >= 0 ? '+' : '';

  document.getElementById('currentPrice').textContent = formatMoney(latest.Close);
  document.getElementById('currentTrend').textContent = `${sign}${pct.toFixed(2)}%`;
  document.getElementById('currentTrend').className = pct >= 0 ? 'trend-pill positive' : 'trend-pill negative';

  document.getElementById('openingPrice').textContent = formatMoney(latest.Open);
  document.getElementById('highPrice').textContent = formatMoney(latest.High);
  document.getElementById('lowPrice').textContent = formatMoney(latest.Low);
  document.getElementById('volumeValue').textContent = formatNumber(latest.Volume);
  document.getElementById('dailyChange').textContent = `${sign}${formatMoney(Math.abs(change))}`;
  document.getElementById('dailyChangePercent').textContent = `${sign}${pct.toFixed(2)}%`;

  document.getElementById('heroStockHeader').textContent = `${state.ticker} Latest Close`;
  document.getElementById('heroCurrentPrice').textContent = formatMoney(latest.Close);
  document.getElementById('heroTrendPill').textContent = `${sign}${pct.toFixed(2)}%`;
  document.getElementById('heroTrendPill').className = `trend-pill ${pct >= 0 ? 'positive' : 'negative'}`;
}

// Table rendering with search, sort, pagination
function renderTable() {
  const search = (document.getElementById('tableSearch')?.value || '').toLowerCase().trim();
  const sort = document.getElementById('sortField')?.value || 'date-desc';

  let filtered = [...state.dataset].filter(item => {
    return (
      item.Date.toLowerCase().includes(search) ||
      String(item.Close).includes(search) ||
      String(item.Volume).includes(search)
    );
  });

  filtered.sort((a, b) => {
    if (sort === 'date-desc') return new Date(b.Date) - new Date(a.Date);
    if (sort === 'date-asc') return new Date(a.Date) - new Date(b.Date);
    if (sort === 'close-desc') return b.Close - a.Close;
    if (sort === 'close-asc') return a.Close - b.Close;
    if (sort === 'volume-desc') return b.Volume - a.Volume;
    return 0;
  });

  const totalPages = Math.max(1, Math.ceil(filtered.length / state.pageSize));
  state.page = Math.min(state.page, totalPages);
  const start = (state.page - 1) * state.pageSize;
  const pageItems = filtered.slice(start, start + state.pageSize);

  const tbody = document.getElementById('stockTableBody');
  if (pageItems.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--muted);">No matching stock records found.</td></tr>`;
  } else {
    tbody.innerHTML = pageItems.map(item => `
      <tr>
        <td><strong>${item.Date}</strong></td>
        <td>${formatMoney(item.Open)}</td>
        <td>${formatMoney(item.High)}</td>
        <td>${formatMoney(item.Low)}</td>
        <td><strong>${formatMoney(item.Close)}</strong></td>
        <td>${formatNumber(item.Volume)}</td>
      </tr>
    `).join('');
  }

  document.getElementById('pageInfo').textContent = `Page ${state.page} / ${totalPages}`;
}

// Apply Sequence Length Preprocessing
async function applyPreprocessing() {
  const seqLen = parseInt(document.getElementById('sequenceLengthSelect').value, 10);
  state.sequenceLength = seqLen;

  if (state.isBackendConnected) {
    try {
      showToast(`Re-processing sequences with ${seqLen} days window...`, 'info');
      const res = await fetch(`${API_BASE}/api/preprocess`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence_length: seqLen })
      });
      const data = await res.json();
      if (data.status === 'success') {
        updatePreprocessingCards(data.info);
        showToast('Preprocessing completed successfully!', 'success');
      } else {
        showToast(data.message, 'error');
      }
    } catch (e) {
      showToast(`Error: ${e.message}`, 'error');
    }
  } else {
    document.getElementById('heroSeqLen').textContent = `${seqLen} Days`;
    showToast(`Sequence length updated to ${seqLen} days`, 'info');
  }
}

// Training execution
async function startTraining(modelType) {
  const config = {
    units: parseInt(document.getElementById('unitsInput').value, 10),
    layers: parseInt(document.getElementById('layersInput').value, 10),
    dropout: parseFloat(document.getElementById('dropoutInput').value),
    epochs: parseInt(document.getElementById('epochsInput').value, 10),
    batch_size: parseInt(document.getElementById('batchSizeInput').value, 10),
    learning_rate: parseFloat(document.getElementById('lrInput').value)
  };

  showToast(`Initiating real neural network training for ${modelType.toUpperCase()}...`, 'info');

  if (state.isBackendConnected) {
    try {
      const endpoint = modelType === 'both' ? '/api/train/both' : `/api/train/${modelType.toLowerCase()}`;
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      const data = await res.json();
      if (data.status === 'success') {
        pollTrainingStatus();
      } else {
        showToast(data.message, 'error');
      }
    } catch (e) {
      showToast(`Training error: ${e.message}`, 'error');
    }
  } else {
    // Client-side simulation of training progress if running offline
    simulateClientTraining(modelType, config.epochs);
  }
}

function pollTrainingStatus() {
  if (state.trainingInterval) clearInterval(state.trainingInterval);

  state.trainingInterval = setInterval(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/train/status`);
      const data = await res.json();

      updateTrainingHud(data);

      if (data.status === 'completed') {
        clearInterval(state.trainingInterval);
        state.trainingInterval = null;
        showToast(`Training finished: ${data.status_text}`, 'success');
        fetchResultsAndRender();
        fetchTrainingHistoryAndRender();
      } else if (data.status === 'error') {
        clearInterval(state.trainingInterval);
        state.trainingInterval = null;
        showToast(`Training failed: ${data.error}`, 'error');
      }
    } catch (e) {
      console.error('Status poll error:', e);
    }
  }, 400);
}

function updateTrainingHud(data) {
  document.getElementById('trainingModelBadge').textContent = `Active Model: ${data.active_model_name || data.current_model || 'None'}`;
  document.getElementById('trainingStatusText').textContent = data.status_text || 'Training...';
  document.getElementById('trainingPercentBadge').textContent = `${data.percent || 0}%`;
  document.getElementById('trainingProgressBar').style.width = `${data.percent || 0}%`;
  document.getElementById('hudEpoch').textContent = `${data.current_epoch || 0} / ${data.total_epochs || 0}`;
  document.getElementById('hudLoss').textContent = data.loss ? Number(data.loss).toFixed(5) : '--';
  document.getElementById('hudValLoss').textContent = data.val_loss ? Number(data.val_loss).toFixed(5) : '--';
  document.getElementById('hudStatus').textContent = data.status.toUpperCase();
}

// Fetch Results and populate evaluation & charts
async function fetchResultsAndRender() {
  if (!state.isBackendConnected) return;

  try {
    const res = await fetch(`${API_BASE}/api/results`);
    const data = await res.json();
    if (data.status === 'success') {
      state.results = data;
      renderEvaluationTable(data.evaluation);
      renderActualVsPredictedChart(data);
      // Automatically generate future projection
      generateFutureForecast();
    }
  } catch (e) {
    console.warn('Results not ready:', e);
  }
}

function renderEvaluationTable(evalRes) {
  if (!evalRes) return;
  const lstm = evalRes.lstm;
  const gru = evalRes.gru;

  if (lstm) {
    document.getElementById('tableMseLstm').textContent = lstm.mse.toFixed(4);
    document.getElementById('tableRmseLstm').textContent = lstm.rmse.toFixed(4);
    document.getElementById('tableMaeLstm').textContent = lstm.mae.toFixed(4);
    document.getElementById('tableMapeLstm').textContent = `${lstm.mape.toFixed(2)}%`;
    document.getElementById('tableAccLstm').textContent = `${lstm.directional_accuracy.toFixed(1)}%`;
  }

  if (gru) {
    document.getElementById('tableMseGru').textContent = gru.mse.toFixed(4);
    document.getElementById('tableRmseGru').textContent = gru.rmse.toFixed(4);
    document.getElementById('tableMaeGru').textContent = gru.mae.toFixed(4);
    document.getElementById('tableMapeGru').textContent = `${gru.mape.toFixed(2)}%`;
    document.getElementById('tableAccGru').textContent = `${gru.directional_accuracy.toFixed(1)}%`;
  }

  if (lstm && gru) {
    document.getElementById('tableMseDiff').textContent = (lstm.mse - gru.mse).toFixed(4);
    document.getElementById('tableRmseDiff').textContent = (lstm.rmse - gru.rmse).toFixed(4);
    document.getElementById('tableMaeDiff').textContent = (lstm.mae - gru.mae).toFixed(4);
    document.getElementById('tableMapeDiff').textContent = `${(lstm.mape - gru.mape).toFixed(2)}%`;
    document.getElementById('tableAccDiff').textContent = `${(lstm.directional_accuracy - gru.directional_accuracy).toFixed(1)}%`;
  }

  // Update Best Model Spotlight
  if (evalRes.best_model) {
    document.getElementById('heroBestModel').textContent = evalRes.best_model;
    document.getElementById('bestModelTitle').textContent = `${evalRes.best_model} Architecture`;
    document.getElementById('bestModelBadge').innerHTML = `<i class="fa-solid fa-trophy"></i> Best Model: ${evalRes.best_model}`;
    document.getElementById('bestModelExplanation').textContent = evalRes.comparison_reason || `The ${evalRes.best_model} model demonstrated superior accuracy on the 20% test partition.`;
  }
}

// Fetch Training Loss History & plot curves
async function fetchTrainingHistoryAndRender() {
  if (!state.isBackendConnected) return;
  try {
    const res = await fetch(`${API_BASE}/api/training-history`);
    const data = await res.json();
    if (data.status === 'success' && data.history) {
      renderLossChart('lstmLossChart', 'LSTM', data.history.lstm, '#4ea5ff');
      renderLossChart('gruLossChart', 'GRU', data.history.gru, '#8b5cf6');
    }
  } catch (e) {
    console.error('Error fetching history:', e);
  }
}

function renderLossChart(containerId, title, hist, mainColor) {
  const container = document.getElementById(containerId);
  if (!container || !hist || !hist.epochs || hist.epochs.length === 0) return;

  const traceTrain = {
    x: hist.epochs,
    y: hist.loss,
    mode: 'lines+markers',
    name: 'Training Loss',
    line: { color: mainColor, width: 2.5 },
    marker: { size: 5 }
  };

  const traceVal = {
    x: hist.epochs,
    y: hist.val_loss,
    mode: 'lines+markers',
    name: 'Validation Loss',
    line: { color: '#2ad89a', width: 2, dash: 'dot' },
    marker: { size: 5 }
  };

  const layout = {
    margin: { l: 45, r: 15, t: 30, b: 40 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: state.theme === 'dark' ? '#edf6ff' : '#122033', family: 'Inter, sans-serif' },
    xaxis: { title: 'Epoch', gridcolor: 'rgba(148, 163, 184, 0.1)' },
    yaxis: { title: 'Loss (MSE)', gridcolor: 'rgba(148, 163, 184, 0.1)' },
    legend: { orientation: 'h', y: 1.15 },
    hovermode: 'x unified'
  };

  Plotly.newPlot(container, [traceTrain, traceVal], layout, { responsive: true, displayModeBar: false });
}

// Render Actual vs Predicted Plotly Chart
function renderActualVsPredictedChart(data) {
  const container = document.getElementById('actualVsPredictedChart');
  if (!container) return;

  const dates = data.test_dates || [];
  const actual = data.actual_prices || [];
  const lstm = data.evaluation?.lstm?.predicted_prices || [];
  const gru = data.evaluation?.gru?.predicted_prices || [];

  document.getElementById('chartPointsCount').textContent = `${actual.length} test points evaluated`;

  const traces = [];

  // Actual Trace
  traces.push({
    x: dates,
    y: actual,
    mode: 'lines',
    name: 'Actual Closing Price',
    line: { color: '#ffffff', width: 3 },
    hovertemplate: 'Actual: $%{y:.2f}<extra></extra>'
  });

  // LSTM Trace
  if (lstm.length > 0 && (state.activeChartView === 'both' || state.activeChartView === 'lstm')) {
    traces.push({
      x: dates,
      y: lstm,
      mode: 'lines',
      name: 'LSTM Predicted Price',
      line: { color: '#4ea5ff', width: 2.2, dash: 'solid' },
      hovertemplate: 'LSTM: $%{y:.2f}<extra></extra>'
    });
  }

  // GRU Trace
  if (gru.length > 0 && (state.activeChartView === 'both' || state.activeChartView === 'gru')) {
    traces.push({
      x: dates,
      y: gru,
      mode: 'lines',
      name: 'GRU Predicted Price',
      line: { color: '#ec4899', width: 2.2, dash: 'dash' },
      hovertemplate: 'GRU: $%{y:.2f}<extra></extra>'
    });
  }

  const layout = {
    margin: { l: 50, r: 20, t: 20, b: 45 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: state.theme === 'dark' ? '#edf6ff' : '#122033', family: 'Inter, sans-serif' },
    xaxis: {
      gridcolor: 'rgba(148, 163, 184, 0.1)',
      rangeslider: { visible: false }
    },
    yaxis: {
      gridcolor: 'rgba(148, 163, 184, 0.1)',
      tickprefix: '$',
      title: 'Stock Price ($)'
    },
    legend: {
      orientation: 'h',
      x: 0,
      y: 1.12
    },
    hovermode: 'x unified'
  };

  Plotly.newPlot(container, traces, layout, {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
  });
}

// Generate Future Multi-step Forecast
async function generateFutureForecast() {
  const days = parseInt(document.getElementById('futureDaysSelect')?.value || '30', 10);
  const model = document.getElementById('futureModelSelect')?.value || 'lstm';

  if (state.isBackendConnected) {
    try {
      showToast(`Generating ${days}-day recursive future forecast using ${model.toUpperCase()}...`, 'info');
      const res = await fetch(`${API_BASE}/api/predict-future`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ days, model })
      });
      const data = await res.json();
      if (data.status === 'success') {
        state.futureForecast = data.forecast;
        renderFutureTable(data.forecast);
        renderFutureTrajectoryChart(data.forecast);
        showToast(`Generated ${data.forecast.length} future projections!`, 'success');
      } else {
        showToast(data.message, 'error');
      }
    } catch (e) {
      showToast(`Future forecast error: ${e.message}`, 'error');
    }
  } else {
    // Offline simulation of future forecast
    simulateFutureForecast(days, model);
  }
}

function renderFutureTable(forecast) {
  const tbody = document.getElementById('futureTableBody');
  if (!tbody) return;
  if (!forecast || forecast.length === 0) {
    tbody.innerHTML = `<tr><td colspan="2" style="text-align: center; color: var(--muted);">No predictions generated yet.</td></tr>`;
    return;
  }

  tbody.innerHTML = forecast.map((f, i) => `
    <tr>
      <td>Day +${i + 1} (${f.date})</td>
      <td><strong>${formatMoney(f.predicted_price)}</strong></td>
    </tr>
  `).join('');
}

function renderFutureTrajectoryChart(forecast) {
  const container = document.getElementById('futureTrajectoryChart');
  if (!container || !forecast || forecast.length === 0) return;

  // Last 30 historical points
  const histDates = (state.allDates.length ? state.allDates : state.dataset.map(d => d.Date)).slice(-30);
  const histCloses = (state.allCloses.length ? state.allCloses : state.dataset.map(d => d.Close)).slice(-30);

  const futureDates = forecast.map(f => f.date);
  const futurePrices = forecast.map(f => f.predicted_price);

  // Connect last historical point to first future point
  const bridgeDates = [histDates[histDates.length - 1], ...futureDates];
  const bridgePrices = [histCloses[histCloses.length - 1], ...futurePrices];

  const traceHistorical = {
    x: histDates,
    y: histCloses,
    mode: 'lines',
    name: 'Recent Historical Close',
    line: { color: '#94a3b8', width: 2.5 }
  };

  const traceFuture = {
    x: bridgeDates,
    y: bridgePrices,
    mode: 'lines+markers',
    name: 'AI Recursive Future Forecast',
    line: { color: '#2ad89a', width: 3, dash: 'solid' },
    marker: { size: 6, color: '#2ad89a' }
  };

  const layout = {
    margin: { l: 45, r: 20, t: 25, b: 40 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: state.theme === 'dark' ? '#edf6ff' : '#122033', family: 'Inter, sans-serif' },
    xaxis: { gridcolor: 'rgba(148, 163, 184, 0.1)' },
    yaxis: { gridcolor: 'rgba(148, 163, 184, 0.1)', tickprefix: '$' },
    legend: { orientation: 'h', y: 1.12 },
    hovermode: 'x unified'
  };

  Plotly.newPlot(container, [traceHistorical, traceFuture], layout, { responsive: true, displayModeBar: false });
}

// Hero mini chart
function renderHeroChart() {
  const container = document.getElementById('heroChart');
  if (!container || state.dataset.length === 0) return;

  const recent = state.dataset.slice(-40);
  const x = recent.map(r => r.Date);
  const y = recent.map(r => r.Close);

  const trace = {
    x: x,
    y: y,
    mode: 'lines',
    line: { color: '#4ea5ff', width: 2.5 },
    fill: 'tozeroy',
    fillcolor: 'rgba(78, 165, 255, 0.12)'
  };

  const layout = {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    xaxis: { visible: false },
    yaxis: { visible: false },
    hovermode: false
  };

  Plotly.newPlot(container, [trace], layout, { responsive: true, displayModeBar: false });
}

// Event Listeners Setup
function setupEventListeners() {
  // Theme Toggle
  document.getElementById('themeToggle')?.addEventListener('click', () => {
    state.theme = state.theme === 'dark' ? 'light' : 'dark';
    updateTheme();
    // Re-render plotly charts with updated theme fonts
    if (state.results) renderActualVsPredictedChart(state.results);
    if (state.futureForecast.length) renderFutureTrajectoryChart(state.futureForecast);
  });

  // Ticker Quick Chips
  document.querySelectorAll('.ticker-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.ticker-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      const symbol = chip.getAttribute('data-symbol');
      loadStockData(symbol);
    });
  });

  // Download Button
  document.getElementById('downloadDataBtn')?.addEventListener('click', () => {
    const sym = document.getElementById('tickerInput').value.trim() || 'AAPL';
    loadStockData(sym);
  });

  // CSV Upload
  const csvInput = document.getElementById('csvFileInput');
  csvInput?.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (state.isBackendConnected) {
      const formData = new FormData();
      formData.append('file', file);
      showToast(`Uploading and validating ${file.name}...`, 'info');
      try {
        const res = await fetch(`${API_BASE}/api/upload-dataset`, {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        if (data.status === 'success') {
          handleStockDataLoaded(data);
          showToast(`Uploaded CSV loaded: ${data.total_records} rows!`, 'success');
          fetchResultsAndRender();
        } else {
          showToast(data.message, 'error');
        }
      } catch (err) {
        showToast(`Upload failed: ${err.message}`, 'error');
      }
    } else {
      // Local client file reader
      parseClientCsv(file);
    }
  });

  // Preprocessing
  document.getElementById('applyPreprocessBtn')?.addEventListener('click', applyPreprocessing);

  // Training Action Buttons
  document.getElementById('trainLstmBtn')?.addEventListener('click', () => startTraining('lstm'));
  document.getElementById('trainGruBtn')?.addEventListener('click', () => startTraining('gru'));
  document.getElementById('trainBothBtn')?.addEventListener('click', () => startTraining('both'));

  // Future Forecast Button
  document.getElementById('generateFutureBtn')?.addEventListener('click', generateFutureForecast);

  // Actual vs Predicted Chart Switcher
  document.querySelectorAll('.chart-switcher button').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.chart-switcher button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.activeChartView = btn.getAttribute('data-view');
      if (state.results) renderActualVsPredictedChart(state.results);
    });
  });

  // Table Search and Sorting
  document.getElementById('tableSearch')?.addEventListener('input', () => {
    state.page = 1;
    renderTable();
  });
  document.getElementById('sortField')?.addEventListener('change', renderTable);
  document.getElementById('prevPage')?.addEventListener('click', () => {
    if (state.page > 1) {
      state.page -= 1;
      renderTable();
    }
  });
  document.getElementById('nextPage')?.addEventListener('click', () => {
    state.page += 1;
    renderTable();
  });

  // Mobile Nav Toggle
  document.getElementById('navToggle')?.addEventListener('click', () => {
    const links = document.getElementById('navLinks');
    links.classList.toggle('open');
  });
}

function updateTheme() {
  document.documentElement.dataset.theme = state.theme;
  const icon = document.querySelector('#themeToggle i');
  if (icon) {
    icon.className = state.theme === 'dark' ? 'fa-solid fa-sun' : 'fa-solid fa-moon';
  }
}

// Standalone Demo Generator if Flask server is not currently running
function loadOfflineDemoDataset(symbol) {
  state.isDemo = true;
  state.dataSource = 'Offline Realistic Stock Simulator';

  const basePrice = { AAPL: 175, MSFT: 380, GOOGL: 140, AMZN: 155, TSLA: 210, NVDA: 480, META: 320 }[symbol] || 150;
  const rows = [];
  let price = basePrice;
  const today = new Date();
  
  for (let i = 350; i >= 0; i--) {
    const d = new Date(today);
    d.setDate(d.getDate() - i);
    if (d.getDay() === 0 || d.getDay() === 6) continue; // Skip weekends

    const drift = Math.sin(i / 18) * 1.8 + Math.cos(i / 40) * 1.2;
    const change = (Math.random() - 0.48) * 3.5 + drift * 0.4;
    price = Math.max(20, price + change);
    const spread = price * 0.015;
    const open = price - (Math.random() - 0.5) * spread;
    const high = Math.max(open, price) + Math.random() * spread;
    const low = Math.min(open, price) - Math.random() * spread;
    const volume = Math.round(35000000 + Math.random() * 40000000);

    rows.push({
      Date: d.toISOString().slice(0, 10),
      Open: Number(open.toFixed(2)),
      High: Number(high.toFixed(2)),
      Low: Number(low.toFixed(2)),
      Close: Number(price.toFixed(2)),
      Volume: volume
    });
  }

  const demoData = {
    status: 'success',
    ticker: symbol,
    is_demo: true,
    data_source: state.dataSource,
    stats: {
      total_rows: rows.length,
      date_range: { start: rows[0].Date, end: rows[rows.length - 1].Date },
      close_avg: Number((rows.reduce((a, b) => a + b.Close, 0) / rows.length).toFixed(2)),
      close_max: Math.max(...rows.map(r => r.Close)),
      close_min: Math.min(...rows.map(r => r.Close))
    },
    preprocessing_info: {
      total_records: rows.length,
      train_records: Math.floor(rows.length * 0.8) - state.sequenceLength,
      test_records: Math.floor(rows.length * 0.2),
      missing_values_handled: 0,
      sequence_length: state.sequenceLength,
      normalization: 'MinMaxScaler (0, 1)',
      target_feature: 'Close'
    },
    total_records: rows.length,
    data: rows,
    all_dates: rows.map(r => r.Date),
    all_closes: rows.map(r => r.Close)
  };

  handleStockDataLoaded(demoData);
}

function parseClientCsv(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const lines = e.target.result.split(/\r?\n/).filter(l => l.trim().length > 0);
    if (lines.length < 2) {
      showToast('CSV file is empty or invalid', 'error');
      return;
    }
    const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
    const dateIdx = headers.indexOf('date');
    const openIdx = headers.indexOf('open');
    const highIdx = headers.indexOf('high');
    const lowIdx = headers.indexOf('low');
    const closeIdx = headers.indexOf('close');
    const volIdx = headers.indexOf('volume');

    if (dateIdx === -1 || closeIdx === -1) {
      showToast('CSV must include Date and Close columns', 'error');
      return;
    }

    const rows = [];
    for (let i = 1; i < lines.length; i++) {
      const p = lines[i].split(',');
      if (p.length < 6) continue;
      rows.push({
        Date: p[dateIdx].trim(),
        Open: Number(parseFloat(p[openIdx] || p[closeIdx]).toFixed(2)),
        High: Number(parseFloat(p[highIdx] || p[closeIdx]).toFixed(2)),
        Low: Number(parseFloat(p[lowIdx] || p[closeIdx]).toFixed(2)),
        Close: Number(parseFloat(p[closeIdx]).toFixed(2)),
        Volume: Number(parseInt(p[volIdx] || 1000000, 10))
      });
    }

    const payload = {
      status: 'success',
      ticker: file.name.split('.')[0].toUpperCase(),
      is_demo: false,
      data_source: 'Local User Uploaded CSV',
      stats: {
        total_rows: rows.length,
        date_range: { start: rows[0].Date, end: rows[rows.length - 1].Date },
        close_avg: Number((rows.reduce((a, b) => a + b.Close, 0) / rows.length).toFixed(2)),
        close_max: Math.max(...rows.map(r => r.Close)),
        close_min: Math.min(...rows.map(r => r.Close))
      },
      preprocessing_info: {
        total_records: rows.length,
        train_records: Math.floor(rows.length * 0.8) - 60,
        test_records: Math.floor(rows.length * 0.2),
        missing_values_handled: 0,
        sequence_length: 60,
        normalization: 'MinMaxScaler (0, 1)',
        target_feature: 'Close'
      },
      total_records: rows.length,
      data: rows,
      all_dates: rows.map(r => r.Date),
      all_closes: rows.map(r => r.Close)
    };

    handleStockDataLoaded(payload);
    showToast(`Loaded ${rows.length} records from uploaded CSV!`, 'success');
  };
  reader.readAsText(file);
}

function simulateClientTraining(modelType, totalEpochs) {
  let epoch = 0;
  const interval = setInterval(() => {
    epoch += 1;
    const loss = 0.045 / Math.sqrt(epoch) + (Math.random() * 0.002);
    const valLoss = loss * 1.08 + (Math.random() * 0.003);
    const pct = Math.round((epoch / totalEpochs) * 100);

    updateTrainingHud({
      active_model_name: modelType.toUpperCase(),
      status_text: `Simulated training Epoch ${epoch}/${totalEpochs}...`,
      percent: pct,
      current_epoch: epoch,
      total_epochs: totalEpochs,
      loss: loss,
      val_loss: valLoss,
      status: epoch >= totalEpochs ? 'completed' : 'training'
    });

    if (epoch >= totalEpochs) {
      clearInterval(interval);
      showToast(`Training completed for ${modelType.toUpperCase()}!`, 'success');
    }
  }, 250);
}

function simulateFutureForecast(days, model) {
  const lastClose = state.dataset[state.dataset.length - 1]?.Close || 170;
  const forecast = [];
  let price = lastClose;
  const curr = new Date();

  while (forecast.length < days) {
    curr.setDate(curr.getDate() + 1);
    if (curr.getDay() === 0 || curr.getDay() === 6) continue;
    price += (Math.random() - 0.46) * 2.2;
    forecast.push({
      date: curr.toISOString().slice(0, 10),
      predicted_price: Number(price.toFixed(2))
    });
  }

  state.futureForecast = forecast;
  renderFutureTable(forecast);
  renderFutureTrajectoryChart(forecast);
  showToast(`Simulated ${days}-day future forecast generated!`, 'success');
}

// Initialize on DOM load
window.addEventListener('DOMContentLoaded', initApp);
