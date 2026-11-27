/* ================================================================
   m1.js — 模块1：二维互相关
   ================================================================ */
(function () {
  'use strict';

  const rndInt = function (a, b) { return Math.floor(Math.random() * (b - a + 1)) + a; };
  const round3 = function (v) { return Math.round(v * 1000) / 1000; };
  const randMat = function (n, a, b) {
    return Array.from({ length: n }, function () {
      return Array.from({ length: n }, function () { return rndInt(a, b); });
    });
  };

  const SPEEDS = [250, 400, 600, 900];

  const S = {
    n: 5, k: 3, pad: 1, stride: 1,
    input: randMat(5, 0, 9),
    kernel: [[1, 0, -1], [0, 1, 0], [-1, 0, 1]],
    pos: 0, playing: false, speed: 1,
    detailOpen: false
  };

  let el = {};
  let autoTimer = null;
  let accumToken = 0;

  let curInputCell = null;
  let curKCell = null;
  let curXCell = null;
  let curPCell = null;

  let chartBars = [];
  let chartSigma = null;
  let chartMaxAbs = 1;

  function outSize() {
    const v = Math.floor((S.n + 2 * S.pad - S.k) / S.stride) + 1;
    return v > 0 ? v : 0;
  }

  function compute(oi, oj) {
    const n = S.n, k = S.k, pad = S.pad, s = S.stride;
    const r0 = oi * s, c0 = oj * s;
    const items = [];
    let sum = 0;
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) {
        const rr = r0 + i - pad, cc = c0 + j - pad;
        const inb = (rr >= 0 && rr < n && cc >= 0 && cc < n);
        const x = inb ? S.input[rr][cc] : 0;
        const kv = S.kernel[i][j];
        const p = round3(x * kv);
        sum = round3(sum + p);
        items.push({ i: i, j: j, rr: rr, cc: cc, x: x, kv: kv, p: p, inb: inb });
      }
    }
    return { items: items, sum: sum };
  }

  function renderInput() {
    const n = S.n, k = S.k, pad = S.pad, s = S.stride;
    const total = n + 2 * pad;
    const os = outSize();
    el.input.style.gridTemplateColumns = 'repeat(' + total + ', 36px)';
    el.input.innerHTML = '';
    let win = null;
    if (os > 0) {
      const oi = Math.floor(S.pos / os), oj = S.pos % os;
      win = { r0: oi * s, c0: oj * s };
    }
    for (let r = 0; r < total; r++) {
      for (let c = 0; c < total; c++) {
        const rr = r - pad, cc = c - pad;
        const isPad = (rr < 0 || rr >= n || cc < 0 || cc >= n);
        const d = document.createElement('div');
        d.className = 'cell' + (isPad ? ' pad' : '');
        d.textContent = isPad ? '0' : S.input[rr][cc];
        d.dataset.r = r; d.dataset.c = c;
        if (win && r >= win.r0 && r < win.r0 + k && c >= win.c0 && c < win.c0 + k) {
          d.classList.add('win');
        }
        if (!isPad) makeEditable(d, rr, cc);
        el.input.appendChild(d);
      }
    }
  }

  function makeEditable(cell, r, c) {
    cell.classList.add('editable');
    cell.addEventListener('click', function () {
      if (cell.isContentEditable) return;
      cell.contentEditable = 'true';
      cell.focus();
      const range = document.createRange();
      range.selectNodeContents(cell);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    });
    cell.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') { e.preventDefault(); cell.blur(); }
    });
    cell.addEventListener('blur', function () {
      cell.contentEditable = 'false';
      let v = parseFloat(String(cell.textContent).replace(/[^\d.\-]/g, ''));
      if (isNaN(v)) v = 0;
      v = Math.max(-99, Math.min(99, Math.round(v * 10) / 10));
      if (v !== S.input[r][c]) { S.input[r][c] = v; refresh(); }
      else { cell.textContent = v; }
    });
  }

  function renderKernel() {
    const k = S.k;
    el.kernel.style.gridTemplateColumns = 'repeat(' + k + ', 38px)';
    el.kernel.innerHTML = '';
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) {
        const d = document.createElement('div');
        d.className = 'cell kernel';
        d.textContent = S.kernel[i][j];
        d.dataset.i = i; d.dataset.j = j;
        el.kernel.appendChild(d);
      }
    }
  }

  function renderOutput() {
    const os = outSize();
    if (os <= 0) {
      el.output.style.gridTemplateColumns = '1fr';
      el.output.innerHTML = '<div class="empty-tip">配置无效：K &gt; n + 2P</div>';
      return;
    }
    el.output.style.gridTemplateColumns = 'repeat(' + os + ', 38px)';
    el.output.innerHTML = '';
    for (let i = 0; i < os; i++) {
      for (let j = 0; j < os; j++) {
        const idx = i * os + j;
        const d = document.createElement('div');
        d.className = 'cell out';
        d.dataset.idx = idx;
        if (idx < S.pos) { d.classList.add('done'); d.textContent = compute(i, j).sum; }
        else if (idx === S.pos) d.classList.add('cur');
        el.output.appendChild(d);
      }
    }
  }

  function renderPair() {
    const k = S.k;
    const os = outSize();
    el.pairK.style.gridTemplateColumns = 'repeat(' + k + ', 34px)';
    el.pairX.style.gridTemplateColumns = 'repeat(' + k + ', 34px)';
    el.pairP.style.gridTemplateColumns = 'repeat(' + k + ', 34px)';
    el.pairK.innerHTML = '';
    el.pairX.innerHTML = '';
    el.pairP.innerHTML = '';
    if (os <= 0) return;
    const oi = Math.floor(S.pos / os), oj = S.pos % os;
    const res = compute(oi, oj);
    res.items.forEach(function (it, idx) {
      const kc = document.createElement('div');
      kc.className = 'cell kernel';
      kc.textContent = it.kv;
      kc.dataset.pidx = idx;
      el.pairK.appendChild(kc);

      const xc = document.createElement('div');
      xc.className = 'cell' + (it.inb ? '' : ' pad');
      xc.textContent = it.x;
      xc.dataset.pidx = idx;
      el.pairX.appendChild(xc);

      const pc = document.createElement('div');
      pc.className = 'cell pcell';
      pc.dataset.pidx = idx;
      el.pairP.appendChild(pc);
    });
    curInputCell = curKCell = curXCell = curPCell = null;
  }

  function buildChart(items, sum) {
    el.accumChart.innerHTML = '';
    chartBars = [];
    chartSigma = null;
    let maxAbs = Math.abs(sum);
    items.forEach(function (it) { maxAbs = Math.max(maxAbs, Math.abs(it.p)); });
    if (maxAbs < 1) maxAbs = 1;
    chartMaxAbs = maxAbs;
    items.forEach(function (it, idx) {
      const slot = document.createElement('div');
      slot.className = 'bar-slot';
      const neg = document.createElement('div'); neg.className = 'bar-neg';
      const pos = document.createElement('div'); pos.className = 'bar-pos';
      slot.appendChild(neg); slot.appendChild(pos);
      slot.title = '第 ' + (idx + 1) + ' 项：' + it.kv + ' × ' + it.x + ' = ' + it.p;
      el.accumChart.appendChild(slot);
      chartBars.push({ slot: slot, pos: pos, neg: neg });
    });
    const sSlot = document.createElement('div');
    sSlot.className = 'bar-slot sigma-slot';
    const sNeg = document.createElement('div'); sNeg.className = 'bar-neg sigma';
    const sPos = document.createElement('div'); sPos.className = 'bar-pos sigma';
    sSlot.appendChild(sNeg); sSlot.appendChild(sPos);
    sSlot.title = '累加和 Σ';
    el.accumChart.appendChild(sSlot);
    chartSigma = { slot: sSlot, pos: sPos, neg: sNeg };
  }

  function setBar(bar, value) {
    const half = 60;
    const h = Math.max(2, Math.abs(value) / chartMaxAbs * half);
    if (value > 0) { bar.pos.style.height = h + 'px'; bar.neg.style.height = '0px'; }
    else if (value < 0) { bar.neg.style.height = h + 'px'; bar.pos.style.height = '0px'; }
    else { bar.pos.style.height = '0px'; bar.neg.style.height = '0px'; }
  }

  function clearActiveCells() {
    if (curInputCell) { curInputCell.classList.remove('hit'); curInputCell = null; }
    if (curKCell) { curKCell.classList.remove('hit'); curKCell.classList.add('done-lit'); curKCell = null; }
    if (curXCell) { curXCell.classList.remove('hit'); curXCell.classList.add('done-lit'); curXCell = null; }
    if (curPCell) { curPCell.classList.remove('active'); curPCell = null; }
  }

  function setActiveCells(idx, it) {
    const kc = el.pairK.querySelector('[data-pidx="' + idx + '"]');
    if (kc) { kc.classList.remove('done-lit'); kc.classList.add('hit'); curKCell = kc; }
    const xc = el.pairX.querySelector('[data-pidx="' + idx + '"]');
    if (xc) { xc.classList.remove('done-lit'); xc.classList.add('hit'); curXCell = xc; }
    const pc = el.pairP.querySelector('[data-pidx="' + idx + '"]');
    if (pc) {
      pc.textContent = it.p;
      pc.classList.add('has-val');
      pc.classList.add(it.p > 0 ? 'pos' : (it.p < 0 ? 'neg' : 'zero'));
      pc.classList.add('active');
      curPCell = pc;
    }
    const xr = it.rr + S.pad, xc2 = it.cc + S.pad;
    const ic = el.input.querySelector('.cell[data-r="' + xr + '"][data-c="' + xc2 + '"]');
    if (ic) { ic.classList.add('hit'); curInputCell = ic; }
    const xLabel = it.inb ? 'X[' + it.rr + ',' + it.cc + ']' : 'X[' + it.rr + ',' + it.cc + ']补零';
    const cls = it.p > 0 ? 'plus' : (it.p < 0 ? 'minus' : '');
    el.accumStep.innerHTML =
      'K[' + it.i + ',' + it.j + '] &times; ' + xLabel +
      ' = ' + it.kv + ' &times; ' + it.x +
      ' = <b class="' + cls + '">' + it.p + '</b>';
    const bar = chartBars[idx];
    if (bar) { bar.slot.classList.add('active'); setBar(bar, it.p); }
  }

  function updateSigma(acc) {
    const b = el.accumSum;
    b.textContent = acc;
    b.classList.remove('pos', 'neg');
    if (acc > 0) b.classList.add('pos');
    else if (acc < 0) b.classList.add('neg');
    if (chartSigma) setBar(chartSigma, acc);
  }

  function updateCompact(oi, oj, idx, total, it, acc) {
    if (!el.compactPos) return;
    el.compactPos.textContent = 'Y[' + oi + ',' + oj + ']';
    el.compactStep.innerHTML = '步骤 <b>' + (idx + 1) + '</b> / ' + total;
    if (el.compactBar) {
      el.compactBar.style.width = Math.round((idx + 1) / total * 100) + '%';
    }
    if (it) {
      const cls = it.p > 0 ? 'plus' : (it.p < 0 ? 'minus' : '');
      el.compactTerm.innerHTML =
        'K[' + it.i + ',' + it.j + '] &times; X[' + it.rr + ',' + it.cc + '] = <b class="' + cls + '">' + it.p + '</b>';
    }
    el.compactSum.textContent = acc;
  }

  function animateAccum() {
    const token = ++accumToken;
    const os = outSize();
    clearActiveCells();
    if (os <= 0) {
      el.accumStep.textContent = '等待有效参数…';
      el.accumChart.innerHTML = '';
      updateSigma(0);
      if (el.compactPos) el.compactPos.textContent = '—';
      if (el.compactStep) el.compactStep.textContent = '等待有效参数';
      if (el.compactTerm) el.compactTerm.textContent = '—';
      if (el.compactSum) el.compactSum.textContent = '0';
      return;
    }
    const oi = Math.floor(S.pos / os), oj = S.pos % os;
    const res = compute(oi, oj);
    const items = res.items;
    const sum = res.sum;
    buildChart(items, sum);
    Array.prototype.forEach.call(el.pairP.children, function (c) { c.textContent = ''; c.className = 'cell pcell'; });
    Array.prototype.forEach.call(el.pairK.children, function (c) { c.classList.remove('done-lit'); });
    Array.prototype.forEach.call(el.pairX.children, function (c) { c.classList.remove('done-lit'); });
    el.accumStep.textContent = '准备计算 Y[' + oi + ',' + oj + '] …';
    updateSigma(0);
    updateCompact(oi, oj, 0, items.length, null, 0);

    const delay = SPEEDS[S.speed];
    let acc = 0;
    let prevIdx = -1;

    items.forEach(function (it, idx) {
      window.setTimeout(function () {
        if (token !== accumToken) return;

        if (prevIdx >= 0) {
          const prevBar = chartBars[prevIdx];
          if (prevBar) prevBar.slot.classList.remove('active');
        }
        if (curKCell) { curKCell.classList.remove('hit'); curKCell.classList.add('done-lit'); curKCell = null; }
        if (curXCell) { curXCell.classList.remove('hit'); curXCell.classList.add('done-lit'); curXCell = null; }
        if (curPCell) { curPCell.classList.remove('active'); curPCell = null; }
        if (curInputCell) { curInputCell.classList.remove('hit'); curInputCell = null; }

        setActiveCells(idx, it);

        acc = (idx === items.length - 1) ? sum : round3(acc + it.p);
        updateSigma(acc);
        updateCompact(oi, oj, idx, items.length, it, acc);

        const outCell = el.output.querySelector('.cell[data-idx="' + S.pos + '"]');
        if (outCell) outCell.textContent = acc;

        prevIdx = idx;

        if (idx === items.length - 1) {
          el.accumStep.innerHTML = '完成 &nbsp;⇒&nbsp; Y[' + oi + ',' + oj + '] = <b>' + sum + '</b>';
        }
      }, delay * (idx + 1));
    });
  }

  function renderFormula() {
    const n = S.n, k = S.k, pad = S.pad, s = S.stride;
    const os = outSize();
    const num = n + 2 * pad - k;
    el.formula.innerHTML =
      '<span>O = &lfloor;(n + 2P &minus; K) / S&rfloor; + 1 = ' +
        '&lfloor;(' + n + ' + 2&times;' + pad + ' &minus; ' + k + ') / ' + s + '&rfloor; + 1 = ' +
        '&lfloor;' + num + ' / ' + s + '&rfloor; + 1 = <b>' + (os > 0 ? os : '—') + '</b></span>' +
      '<span class="dim">输出特征图：' + (os > 0 ? os + ' × ' + os : '无效配置') + '</span>' +
      '<span class="dim">参数量（单个核）：' + (k * k) + '</span>';
  }

  function refresh() {
    renderInput(); renderKernel(); renderOutput(); renderFormula(); renderPair(); animateAccum();
  }
  function renderWindow() {
    renderInput(); renderOutput(); renderPair(); animateAccum();
  }

  function syncUI() {
    el.stN.querySelector('.step-val').textContent = S.n;
    el.stK.querySelector('.step-val').textContent = S.k;
    el.stP.querySelector('.step-val').textContent = S.pad;
    el.stS.querySelector('.step-val').textContent = S.stride;
    el.stSpeed.querySelector('.step-val').textContent = SPEEDS[S.speed] + 'ms';
    refresh();
  }

  function resetPos() { stopPlay(); S.pos = 0; }

  function setN(v) {
    const old = S.input;
    const arr = [];
    for (let r = 0; r < v; r++) {
      const row = [];
      for (let c = 0; c < v; c++) {
        row.push((old && old[r] && old[r][c] !== undefined) ? old[r][c] : rndInt(0, 9));
      }
      arr.push(row);
    }
    S.input = arr; S.n = v; resetPos(); syncUI();
  }
  function setK(v) {
    const old = S.kernel;
    const arr = [];
    for (let i = 0; i < v; i++) {
      const row = [];
      for (let j = 0; j < v; j++) {
        row.push((old && old[i] && old[i][j] !== undefined) ? old[i][j] : rndInt(-1, 1));
      }
      arr.push(row);
    }
    S.kernel = arr; S.k = v; resetPos(); syncUI();
  }
  function setPad(v) { S.pad = v; resetPos(); syncUI(); }
  function setStride(v) { S.stride = v; resetPos(); syncUI(); }

  function stopPlay() {
    S.playing = false;
    if (autoTimer) { window.clearTimeout(autoTimer); autoTimer = null; }
    if (el.btnPlay) el.btnPlay.textContent = '自动播放';
  }
  function startPlay() {
    if (S.playing) return;
    if (outSize() <= 0) return;
    S.playing = true;
    el.btnPlay.textContent = '暂停';
    const step = function () {
      if (!S.playing) return;
      const total = outSize() * outSize();
      S.pos = (S.pos + 1) % total;
      renderWindow();
      const dur = S.k * S.k * SPEEDS[S.speed] + 800;
      autoTimer = window.setTimeout(step, dur);
    };
    autoTimer = window.setTimeout(step, 400);
  }

  function toggleDetail() {
    S.detailOpen = !S.detailOpen;
    if (el.detailed) el.detailed.style.display = S.detailOpen ? '' : 'none';
    if (el.btnDetail) el.btnDetail.textContent = S.detailOpen ? '收起详细视图 ↑' : '展开详细视图 ↓';
  }

  function bindEvents() {
    el.stN.querySelector('[data-act="dec"]').addEventListener('click', function () {
      if (S.n > 3 && S.k <= (S.n - 1) + 2 * S.pad) setN(S.n - 1);
    });
    el.stN.querySelector('[data-act="inc"]').addEventListener('click', function () {
      if (S.n < 8) setN(S.n + 1);
    });
    el.stK.querySelector('[data-act="dec"]').addEventListener('click', function () {
      if (S.k > 1) setK(S.k - 1);
    });
    el.stK.querySelector('[data-act="inc"]').addEventListener('click', function () {
      if (S.k < 5 && (S.k + 1) <= S.n + 2 * S.pad) setK(S.k + 1);
    });
    el.stP.querySelector('[data-act="dec"]').addEventListener('click', function () {
      if (S.pad > 0 && S.k <= S.n + 2 * (S.pad - 1)) setPad(S.pad - 1);
    });
    el.stP.querySelector('[data-act="inc"]').addEventListener('click', function () {
      if (S.pad < 3) setPad(S.pad + 1);
    });
    el.stS.querySelector('[data-act="dec"]').addEventListener('click', function () {
      if (S.stride > 1) setStride(S.stride - 1);
    });
    el.stS.querySelector('[data-act="inc"]').addEventListener('click', function () {
      if (S.stride < 3) setStride(S.stride + 1);
    });
    el.stSpeed.querySelector('[data-act="dec"]').addEventListener('click', function () {
      if (S.speed > 0) {
        S.speed--;
        el.stSpeed.querySelector('.step-val').textContent = SPEEDS[S.speed] + 'ms';
        animateAccum();
      }
    });
    el.stSpeed.querySelector('[data-act="inc"]').addEventListener('click', function () {
      if (S.speed < SPEEDS.length - 1) {
        S.speed++;
        el.stSpeed.querySelector('.step-val').textContent = SPEEDS[S.speed] + 'ms';
        animateAccum();
      }
    });
    el.btnRand.addEventListener('click', function () { S.input = randMat(S.n, 0, 9); resetPos(); refresh(); });
    el.btnRandK.addEventListener('click', function () { S.kernel = randMat(S.k, -1, 1); resetPos(); refresh(); });
    el.btnReset.addEventListener('click', function () { resetPos(); refresh(); });
    el.btnPrev.addEventListener('click', function () {
      const total = outSize() * outSize();
      if (total <= 0) return;
      stopPlay(); S.pos = (S.pos - 1 + total) % total; renderWindow();
    });
    el.btnNext.addEventListener('click', function () {
      const total = outSize() * outSize();
      if (total <= 0) return;
      stopPlay(); S.pos = (S.pos + 1) % total; renderWindow();
    });
    el.btnPlay.addEventListener('click', function () { if (S.playing) stopPlay(); else startPlay(); });
    if (el.btnDetail) el.btnDetail.addEventListener('click', toggleDetail);
  }

  Lab.register({
    id: 'ch4-m1',
    index: '01',
    title: '二维互相关',
    desc: '卷积核在输入上按步长滑动，每个位置做「对应元素相乘再求和」。滑动窗口扫过的所有结果，拼成输出特征图。',
    template: function () {
      return [
        '<div class="panel">',
        '  <h3>参数控制台</h3>',
        '  <div class="ctrl-row">',
        '    <div class="ctrl"><label>输入尺寸 n</label>',
        '      <div class="stepper" data-ref="st-n">',
        '        <button class="step-btn" data-act="dec">−</button>',
        '        <span class="step-val">5</span>',
        '        <button class="step-btn" data-act="inc">+</button>',
        '      </div></div>',
        '    <div class="ctrl"><label>卷积核 K</label>',
        '      <div class="stepper" data-ref="st-k">',
        '        <button class="step-btn" data-act="dec">−</button>',
        '        <span class="step-val">3</span>',
        '        <button class="step-btn" data-act="inc">+</button>',
        '      </div></div>',
        '    <div class="ctrl"><label>Padding P</label>',
        '      <div class="stepper" data-ref="st-p">',
        '        <button class="step-btn" data-act="dec">−</button>',
        '        <span class="step-val">1</span>',
        '        <button class="step-btn" data-act="inc">+</button>',
        '      </div></div>',
        '    <div class="ctrl"><label>Stride S</label>',
        '      <div class="stepper" data-ref="st-s">',
        '        <button class="step-btn" data-act="dec">−</button>',
        '        <span class="step-val">1</span>',
        '        <button class="step-btn" data-act="inc">+</button>',
        '      </div></div>',
        '  </div>',
        '  <div class="ctrl-row">',
        '    <button class="btn" data-ref="btn-rand">随机化输入</button>',
        '    <button class="btn" data-ref="btn-randk">随机化卷积核</button>',
        '    <button class="btn" data-ref="btn-reset">重置动画</button>',
        '    <button class="btn" data-ref="btn-prev">← 上一步</button>',
        '    <button class="btn primary" data-ref="btn-next">下一步 →</button>',
        '    <button class="btn" data-ref="btn-play">自动播放</button>',
        '  </div>',
        '  <div class="formula-bar" data-ref="formula"></div>',
        '</div>',
        '<div class="stage">',
        '  <div class="panel"><h3>输入 X（含补零）</h3><div class="matrix" data-ref="input"></div></div>',
        '  <div class="panel"><h3>卷积核 K</h3><div class="matrix" data-ref="kernel"></div></div>',
        '  <div class="panel"><h3>输出 Y</h3><div class="matrix" data-ref="output"></div></div>',
        '</div>',
        '<div class="panel" style="margin-top:18px;">',
        '  <div class="panel-head-row">',
        '    <h3>计算明细</h3>',
        '    <button class="btn" data-ref="btn-detail">展开详细视图 ↓</button>',
        '  </div>',
        '  <div class="compact-line">',
        '    <span data-ref="compact-pos">—</span>',
        '    <span class="sep">·</span>',
        '    <span data-ref="compact-step">—</span>',
        '    <span class="sep">·</span>',
        '    <span data-ref="compact-term">—</span>',
        '    <span class="sep">·</span>',
        '    <span>Σ = <b data-ref="compact-sum">0</b></span>',
        '    <span class="progress"><span data-ref="compact-bar" style="width:0;"></span></span>',
        '  </div>',
        '  <div data-ref="detailed" style="display:none; margin-top:16px;">',
        '    <div class="pair-stage">',
        '      <div class="pair-col"><div class="pair-label">卷积核 K</div><div class="matrix small" data-ref="pair-k"></div></div>',
        '      <div class="pair-op">×</div>',
        '      <div class="pair-col"><div class="pair-label">窗口 X</div><div class="matrix small" data-ref="pair-x"></div></div>',
        '      <div class="pair-op">=</div>',
        '      <div class="pair-col"><div class="pair-label">乘积 P</div><div class="matrix small" data-ref="pair-p"></div></div>',
        '    </div>',
        '    <div class="accum-head">',
        '      <span class="accum-step" data-ref="accum-step">—</span>',
        '      <div class="accum-right">',
        '        <div class="ctrl"><label>间隔</label>',
        '          <div class="stepper" data-ref="st-speed">',
        '            <button class="step-btn" data-act="dec">−</button>',
        '            <span class="step-val">400ms</span>',
        '            <button class="step-btn" data-act="inc">+</button>',
        '          </div></div>',
        '        <span class="accum-live">Σ = <b data-ref="accum-sum">0</b></span>',
        '      </div>',
        '    </div>',
        '    <div class="accum-chart" data-ref="accum-chart"></div>',
        '    <div class="accum-note">每根柱 = 一个乘积项；绿上红下；最右 Σ 柱 = 累加和。</div>',
        '    <div class="legend">',
        '      <span><i class="sw sw-win"></i>滑动窗口</span>',
        '      <span><i class="sw sw-hit"></i>当前项</span>',
        '      <span><i class="sw sw-pad"></i>补零</span>',
        '      <span><i class="sw sw-cur"></i>当前输出格</span>',
        '      <span><i class="sw sw-pos"></i>正贡献</span>',
        '      <span><i class="sw sw-neg"></i>负贡献</span>',
        '    </div>',
        '  </div>',
        '</div>'
      ].join('\n');
    },
    init: function (root) {
      el = {
        input: root.querySelector('[data-ref="input"]'),
        kernel: root.querySelector('[data-ref="kernel"]'),
        output: root.querySelector('[data-ref="output"]'),
        formula: root.querySelector('[data-ref="formula"]'),
        stN: root.querySelector('[data-ref="st-n"]'),
        stK: root.querySelector('[data-ref="st-k"]'),
        stP: root.querySelector('[data-ref="st-p"]'),
        stS: root.querySelector('[data-ref="st-s"]'),
        stSpeed: root.querySelector('[data-ref="st-speed"]'),
        btnRand: root.querySelector('[data-ref="btn-rand"]'),
        btnRandK: root.querySelector('[data-ref="btn-randk"]'),
        btnReset: root.querySelector('[data-ref="btn-reset"]'),
        btnPrev: root.querySelector('[data-ref="btn-prev"]'),
        btnNext: root.querySelector('[data-ref="btn-next"]'),
        btnPlay: root.querySelector('[data-ref="btn-play"]'),
        btnDetail: root.querySelector('[data-ref="btn-detail"]'),
        detailed: root.querySelector('[data-ref="detailed"]'),
        compactPos: root.querySelector('[data-ref="compact-pos"]'),
        compactStep: root.querySelector('[data-ref="compact-step"]'),
        compactTerm: root.querySelector('[data-ref="compact-term"]'),
        compactSum: root.querySelector('[data-ref="compact-sum"]'),
        compactBar: root.querySelector('[data-ref="compact-bar"]'),
        pairK: root.querySelector('[data-ref="pair-k"]'),
        pairX: root.querySelector('[data-ref="pair-x"]'),
        pairP: root.querySelector('[data-ref="pair-p"]'),
        accumChart: root.querySelector('[data-ref="accum-chart"]'),
        accumSum: root.querySelector('[data-ref="accum-sum"]'),
        accumStep: root.querySelector('[data-ref="accum-step"]')
      };
      bindEvents(); syncUI();
    },
    onHide: function () { stopPlay(); accumToken++; }
  });

})();