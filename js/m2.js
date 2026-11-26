/* ================================================================
   m2.js — 模块2：多通道与 1×1 卷积（大白话版）
   ================================================================ */
(function () {
  'use strict';

  const rndInt = function (a, b) { return Math.floor(Math.random() * (b - a + 1)) + a; };
  const round3 = function (v) { return Math.round(v * 1000) / 1000; };
  const SPEEDS = [300, 500, 700, 1000];
  const PAD = 1;

  const S = {
    Cin: 3,
    Cout: 2,
    K: 3,
    n: 4,
    input: null,
    kernels: null,
    bias: null,
    W1x1: null,
    b1x1: null,
    selOut: 0,
    selPos: 0,
    step: 0,       // 当前推导到了第几个通道（0~Cin）
    speed: 1,
    playing: false
  };

  let el = {};
  let autoTimer = null;

  // 生成数据
  function genInput(n, cin) {
    return Array.from({ length: cin }, function () {
      return Array.from({ length: n }, function () {
        return Array.from({ length: n }, function () { return rndInt(0, 9); });
      });
    });
  }
  function genKernels(cout, cin, k) {
    return Array.from({ length: cout }, function () {
      return Array.from({ length: cin }, function () {
        return Array.from({ length: k }, function () {
          return Array.from({ length: k }, function () { return rndInt(-1, 1); });
        });
      });
    });
  }
  function genBias(cout) {
    return Array.from({ length: cout }, function () { return rndInt(-3, 3); });
  }
  function genW1(cout, cin) {
    return Array.from({ length: cout }, function () {
      return Array.from({ length: cin }, function () {
        return round3(rndInt(-15, 15) / 10);
      });
    });
  }
  function genB1(cout) {
    return Array.from({ length: cout }, function () { return round3(rndInt(-10, 10) / 10); });
  }

  function initData() {
    S.input = genInput(S.n, S.Cin);
    S.kernels = genKernels(S.Cout, S.Cin, S.K);
    S.bias = genBias(S.Cout);
    S.W1x1 = genW1(S.Cout, S.Cin);
    S.b1x1 = genB1(S.Cout);
    S.selOut = Math.min(S.selOut, S.Cout - 1);
    clampPos();
    S.step = 0;
  }

  function outSize() {
    const v = Math.floor((S.n + 2 * PAD - S.K) / 1) + 1;
    return v > 0 ? v : 0;
  }
  function clampPos() {
    const os = outSize();
    const total = os * os;
    if (total <= 0) { S.selPos = 0; return; }
    S.selPos = Math.max(0, Math.min(S.selPos, total - 1));
  }

  function computeChannelSum(o, ci, r, c) {
    let p = 0;
    for (let i = 0; i < S.K; i++) {
      for (let j = 0; j < S.K; j++) {
        const rr = r + i - PAD;
        const cc = c + j - PAD;
        const x = (rr >= 0 && rr < S.n && cc >= 0 && cc < S.n) ? S.input[ci][rr][cc] : 0;
        p += x * S.kernels[o][ci][i][j];
      }
    }
    return round3(p);
  }

  function computeOut(o, r, c) {
    let s = S.bias[o];
    for (let ci = 0; ci < S.Cin; ci++) {
      s += computeChannelSum(o, ci, r, c);
    }
    return round3(s);
  }

  // ============================================================
  // 渲染全景图（带通道颜色标记）
  // ============================================================
  function renderOverview() {
    const os = outSize();
    const r = Math.floor(S.selPos / os) || 0;
    const c = S.selPos % os || 0;

    // 1. 输入通道
    el.ovIn.innerHTML = '';
    for (let ci = 0; ci < S.Cin; ci++) {
      const item = document.createElement('div');
      item.className = 'ov-ch-item';
      item.innerHTML = '<div class="ov-ch-label"><span class="ch-tag bg-ch' + (ci % 3) + '">输入通道 ' + ci + '</span></div>';
      const grid = document.createElement('div');
      grid.className = 'matrix-grid';
      grid.style.gridTemplateColumns = 'repeat(' + (S.n + 2 * PAD) + ', 24px)';
      
      for (let rr = -PAD; rr < S.n + PAD; rr++) {
        for (let cc = -PAD; cc < S.n + PAD; cc++) {
          const isPad = (rr < 0 || rr >= S.n || cc < 0 || cc >= S.n);
          const cell = document.createElement('div');
          // 给格子加通道边框色
          cell.className = 'ov-cell edge-ch' + (ci % 3) + (isPad ? ' pad' : '');
          cell.textContent = isPad ? '0' : S.input[ci][rr][cc];
          
          if (!isPad && S.step === ci) {
            if (rr >= r && rr < r + S.K && cc >= c && cc < c + S.K) {
              cell.classList.add('win');
            }
          }
          grid.appendChild(cell);
        }
      }
      item.appendChild(grid);
      el.ovIn.appendChild(item);
    }

    // 2. 卷积核
    el.ovK.innerHTML = '';
    const o = S.selOut;
    for (let ci = 0; ci < S.Cin; ci++) {
      const item = document.createElement('div');
      item.className = 'ov-ch-item';
      item.innerHTML = '<div class="ov-ch-label"><span class="ch-tag bg-ch' + (ci % 3) + '">核切片 ' + ci + '</span> (属于输出 ' + o + ')</div>';
      const grid = document.createElement('div');
      grid.className = 'matrix-grid';
      grid.style.gridTemplateColumns = 'repeat(' + S.K + ', 24px)';
      
      for (let i = 0; i < S.K; i++) {
        for (let j = 0; j < S.K; j++) {
          const cell = document.createElement('div');
          cell.className = 'ov-cell kernel edge-ch' + (ci % 3);
          cell.textContent = S.kernels[o][ci][i][j];
          if (S.step === ci) cell.style.background = '#eaf3f9';
          grid.appendChild(cell);
        }
      }
      item.appendChild(grid);
      el.ovK.appendChild(item);
    }

    // 3. 输出特征图
    el.ovOut.innerHTML = '';
    if (os > 0) {
      const item = document.createElement('div');
      item.className = 'ov-ch-item';
      item.innerHTML = '<div class="ov-ch-label"><span class="ch-tag" style="background:#c0392b;">输出通道 ' + o + '</span></div>';
      const grid = document.createElement('div');
      grid.className = 'matrix-grid';
      grid.style.gridTemplateColumns = 'repeat(' + os + ', 24px)';
      
      for (let i = 0; i < os; i++) {
        for (let j = 0; j < os; j++) {
          const idx = i * os + j;
          const cell = document.createElement('div');
          cell.className = 'ov-cell out done';
          cell.textContent = computeOut(o, i, j);
          if (idx === S.selPos) cell.classList.add('cur');
          grid.appendChild(cell);
        }
      }
      item.appendChild(grid);
      el.ovOut.appendChild(item);
    }
  }

  // ============================================================
  // 渲染分步明细
  // ============================================================
  function createMatrixBox(title, rows, cols, getter) {
    const box = document.createElement('div');
    box.className = 'mat-box';
    const t = document.createElement('div');
    t.className = 'mat-title';
    t.innerHTML = title;
    box.appendChild(t);

    const grid = document.createElement('div');
    grid.className = 'matrix-grid';
    grid.style.gridTemplateColumns = 'repeat(' + cols + ', 22px)';

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const g = getter(i, j);
        const cell = document.createElement('div');
        cell.className = 'sm-cell ' + (g.cls || '');
        cell.textContent = g.text;
        grid.appendChild(cell);
      }
    }
    box.appendChild(grid);
    return box;
  }

  function renderBreakdown() {
    el.breakdown.innerHTML = '';
    const os = outSize();
    if (os <= 0) return;

    const r = Math.floor(S.selPos / os);
    const c = S.selPos % os;
    const o = S.selOut;

    for (let ci = 0; ci < S.Cin; ci++) {
      const row = document.createElement('div');
      row.className = 'step-row';
      if (S.step === ci) row.classList.add('active');
      else if (S.step > ci) row.classList.add('done');

      const xBox = createMatrixBox('输入窗口 <span class="ch-tag bg-ch' + (ci % 3) + '">Ch ' + ci + '</span>', S.K, S.K, function (i, j) {
        const rr = r + i - PAD;
        const cc = c + j - PAD;
        const inb = (rr >= 0 && rr < S.n && cc >= 0 && cc < S.n);
        return { text: inb ? S.input[ci][rr][cc] : '0', cls: inb ? 'win' : 'pad' };
      });
      
      const op1 = document.createElement('div'); op1.className = 'op'; op1.textContent = '×';
      
      const kBox = createMatrixBox('核切片 <span class="ch-tag bg-ch' + (ci % 3) + '">Ch ' + ci + '</span>', S.K, S.K, function (i, j) {
        return { text: S.kernels[o][ci][i][j], cls: 'kernel' };
      });

      const op2 = document.createElement('div'); op2.className = 'op'; op2.textContent = '=';

      const p = computeChannelSum(o, ci, r, c);
      const valBox = document.createElement('div');
      valBox.className = 'res-box';
      if (S.step > ci) {
        valBox.textContent = p;
        valBox.classList.add('has-val');
        if (p > 0) valBox.classList.add('pos');
        else if (p < 0) valBox.classList.add('neg');
      } else {
        valBox.textContent = '?';
      }

      row.appendChild(xBox); row.appendChild(op1); row.appendChild(kBox); row.appendChild(op2); row.appendChild(valBox);
      el.breakdown.appendChild(row);
    }

    const finalRow = document.createElement('div');
    finalRow.className = 'res-row';
    if (S.step === S.Cin) finalRow.classList.add('active');
    const totalSum = computeOut(o, r, c);

    finalRow.innerHTML = 
      '<span style="font-family:monospace; font-size:13px; color:#666;">把 3 个通道的乘积加起来，再加上偏置 b[' + o + '] (' + S.bias[o] + ') = </span>' +
      '<span class="res-box' + (S.step === S.Cin ? ' has-val pos' : '') + '" style="margin-top:0;">' + 
      (S.step === S.Cin ? totalSum : '?') + 
      '</span>';
    el.breakdown.appendChild(finalRow);
  }

  // ============================================================
  // 1x1 卷积
  // ============================================================
  function renderFC() {
    const cin = S.Cin, cout = S.Cout;
    const rowH = 44, padY = 24, w = 380;
    const used = Math.max(cin, cout) * rowH;
    const h = padY * 2 + used;
    const xIn = 60, xOut = 320;
    const yInStart = padY + (used - cin * rowH) / 2;
    const yOutStart = padY + (used - cout * rowH) / 2;

    const svg = el.fcSvg;
    svg.setAttribute('width', w); svg.setAttribute('height', h); svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h);

    const o = S.selOut;
    let html = '';

    for (let ci = 0; ci < cin; ci++) {
      const y1 = yInStart + ci * rowH + rowH / 2;
      for (let co = 0; co < cout; co++) {
        const y2 = yOutStart + co * rowH + rowH / 2;
        const wgt = S.W1x1[co][ci];
        const color = wgt >= 0 ? '#2980b9' : '#c0392b';
        const stroke = Math.max(0.6, Math.min(3, Math.abs(wgt) * 1.6));
        const opacity = (co === o) ? 0.8 : 0.1;
        html += '<line x1="' + xIn + '" y1="' + y1 + '" x2="' + xOut + '" y2="' + y2 + '" stroke="' + color + '" stroke-width="' + stroke + '" stroke-opacity="' + opacity + '" />';
      }
    }

    const cy = Math.min(S.n - 1, Math.floor(S.n / 2));
    for (let ci = 0; ci < cin; ci++) {
      const y = yInStart + ci * rowH + rowH / 2;
      const v = S.input[ci][cy][cy];
      html += '<circle cx="' + xIn + '" cy="' + y + '" r="14" fill="#eaf3f9" stroke="#2980b9" />';
      html += '<text x="' + xIn + '" y="' + (y + 4) + '" text-anchor="middle" font-size="11" font-family="Consolas, monospace" fill="#1b4f72">' + v + '</text>';
    }

    for (let co = 0; co < cout; co++) {
      const y = yOutStart + co * rowH + rowH / 2;
      let val = S.b1x1[co];
      for (let ci = 0; ci < cin; ci++) val += S.W1x1[co][ci] * S.input[ci][cy][cy];
      val = round3(val);
      
      const isActive = (co === o);
      html += '<g data-fc="' + co + '" style="cursor:pointer">';
      html += '<circle cx="' + xOut + '" cy="' + y + '" r="16" fill="' + (isActive ? '#fdf1ef' : '#ffffff') + '" stroke="' + (isActive ? '#c0392b' : '#dcdcdc') + '" stroke-width="' + (isActive ? 2 : 1) + '" />';
      html += '<text x="' + xOut + '" y="' + (y + 4) + '" text-anchor="middle" font-size="11" font-family="Consolas, monospace" fill="' + (isActive ? '#c0392b' : '#555555') + '">' + val + '</text>';
      html += '</g>';
    }

    svg.innerHTML = html;
    svg.onclick = function (e) {
      let node = e.target;
      while (node && node !== svg) {
        if (node.dataset && node.dataset.fc !== undefined) {
          S.selOut = parseInt(node.dataset.fc, 10);
          refreshAll();
          return;
        }
        node = node.parentNode;
      }
    };
  }

  function refreshAll() {
    renderOverview();
    renderBreakdown();
    renderFC();
    renderFormula();
    syncUI();
  }

  function syncUI() {
    el.stCin.querySelector('.step-val').textContent = S.Cin;
    el.stCout.querySelector('.step-val').textContent = S.Cout;
    el.stK.querySelector('.step-val').textContent = S.K;
    el.stN.querySelector('.step-val').textContent = S.n;
    el.stSpeed.querySelector('.step-val').textContent = SPEEDS[S.speed] + 'ms';
  }

  function renderFormula() {
    const nOut = outSize();
    const params = S.Cout * (S.Cin * S.K * S.K + 1);
    el.formula.innerHTML =
      '<span>输入通道 Cin = <b>' + S.Cin + '</b> &nbsp; 输出通道 Cout = <b>' + S.Cout + '</b></span>' +
      '<span>参数量 = Cout × (Cin×K² + 1) = <b>' + params + '</b></span>' +
      '<span class="dim">输出特征图：' + (nOut > 0 ? nOut + ' × ' + nOut + ' × ' + S.Cout : '无效配置') + '</span>';
  }

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

    const stepFn = function () {
      if (!S.playing) return;
      if (S.step < S.Cin) {
        S.step++; refreshAll();
        autoTimer = window.setTimeout(stepFn, SPEEDS[S.speed] + 400);
      } else {
        const total = outSize() * outSize();
        S.selPos = (S.selPos + 1) % total;
        S.step = 0; refreshAll();
        autoTimer = window.setTimeout(stepFn, 1000);
      }
    };
    autoTimer = window.setTimeout(stepFn, 400);
  }

  function resetPos() { stopPlay(); S.selPos = 0; S.step = 0; }

  // ============================================================
  // 注册模块
  // ============================================================
  Lab.register({
    id: 'm2',
    index: '02',
    title: '多通道与 1×1 卷积',
    desc: '通道就像“视角”或“图层”。输入有几个图层，你的卷积核就必须有几个图层（三明治）。放几个三明治，就得到几个输出图层。',

    template: function () {
      return [
        '<div class="concept-panel">',
        '  <h4>💡 什么是通道（Channel）？</h4>',
        '  <p>想象一张彩色图片，它其实是由 <b>红、绿、蓝（RGB）三个图层</b>叠在一起组成的。这三个图层就是 <b>输入通道（Cin=3）</b>。</p>',
        '  <p>现在你要提取特征，你的“过滤器（卷积核）”也必须像三明治一样叠三层（一层滤红色，一层滤绿色，一层滤蓝色）。<b>所以卷积核的层数必须等于输入通道数！</b></p>',
        '  <p>你如果想找 2 种不同的特征（比如找横线和找竖线），就得准备 2 个不同的三明治（2 个卷积核）。最后就会输出 2 张图，这就是 <b>输出通道（Cout=2）</b>。</p>',
        '</div>',

        '<div class="shape-bar">',
        '  <div class="shape-box"><span class="sh-name">输入 X</span><span class="sh-val" data-ref="sh-in">—</span></div>',
        '  <span class="shape-arrow">→</span>',
        '  <div class="shape-box kernel"><span class="sh-name">卷积核 W</span><span class="sh-val" data-ref="sh-kernel">—</span></div>',
        '  <span class="shape-arrow">→</span>',
        '  <div class="shape-box out"><span class="sh-name">输出 Y</span><span class="sh-val" data-ref="sh-out">—</span></div>',
        '</div>',

        '<div class="panel">',
        '  <h3>参数控制台</h3>',
        '  <div class="ctrl-row">',
        '    <div class="ctrl"><label>输入通道 Cin (图层数)</label>',
        '      <div class="stepper" data-ref="st-cin"><button class="step-btn" data-act="dec">−</button><span class="step-val">3</span><button class="step-btn" data-act="inc">+</button></div></div>',
        '    <div class="ctrl"><label>输出通道 Cout (三明治个数)</label>',
        '      <div class="stepper" data-ref="st-cout"><button class="step-btn" data-act="dec">−</button><span class="step-val">2</span><button class="step-btn" data-act="inc">+</button></div></div>',
        '    <div class="ctrl"><label>卷积核 K</label>',
        '      <div class="stepper" data-ref="st-k"><button class="step-btn" data-act="dec">−</button><span class="step-val">3</span><button class="step-btn" data-act="inc">+</button></div></div>',
        '    <div class="ctrl"><label>输入尺寸 n</label>',
        '      <div class="stepper" data-ref="st-n"><button class="step-btn" data-act="dec">−</button><span class="step-val">4</span><button class="step-btn" data-act="inc">+</button></div></div>',
        '  </div>',
        '  <div class="ctrl-row">',
        '    <button class="btn" data-ref="btn-rand">随机化全部</button>',
        '    <button class="btn" data-ref="btn-reset">重置动画</button>',
        '    <button class="btn" data-ref="btn-prev">← 上一步</button>',
        '    <button class="btn primary" data-ref="btn-next">下一步 →</button>',
        '    <button class="btn" data-ref="btn-play">自动播放</button>',
        '    <div class="ctrl"><label>间隔</label>',
        '      <div class="stepper" data-ref="st-speed"><button class="step-btn" data-act="dec">−</button><span class="step-val">500ms</span><button class="step-btn" data-act="inc">+</button></div></div>',
        '  </div>',
        '  <div class="formula-bar" data-ref="formula"></div>',
        '</div>',

        '<div class="panel">',
        '  <h3>全景视图（注意颜色，相同的颜色互相乘）</h3>',
        '  <div class="overview-stage">',
        '    <div class="ov-group">',
        '      <div class="ov-title">输入 X（3个图层）</div>',
        '      <div class="ov-channels" data-ref="ov-in"></div>',
        '    </div>',
        '    <div class="ov-group">',
        '      <div class="ov-title">卷积核 K（3层三明治）</div>',
        '      <div class="ov-channels" data-ref="ov-k"></div>',
        '    </div>',
        '    <div class="ov-group">',
        '      <div class="ov-title">输出 Y（1张新图）</div>',
        '      <div class="ov-channels" data-ref="ov-out"></div>',
        '    </div>',
        '  </div>',
        '</div>',

        '<div class="panel">',
        '  <h3>计算过程：像三明治一样切开算</h3>',
        '  <p class="tip" style="margin-bottom:12px;">当前推导：计算输出 Y 的某一个点。它是由 3 个输入通道分别与对应的核切片做内积，然后加在一起得到的。</p>',
        '  <div data-ref="breakdown"></div>',
        '</div>',

        '<div class="panel">',
        '  <h3>1×1 卷积 · 逐像素全连接</h3>',
        '  <p class="tip" style="margin-bottom:14px;">取中心像素：把该像素的 3 个通道值，通过一组权重映射成 2 个输出值，等价于对这个像素做一次全连接。</p>',
        '  <div class="fc-stage"><svg class="fc-svg" data-ref="fc-svg"></svg></div>',
        '  <div class="fc-legend">线宽表示权重绝对值大小；蓝线 = 正权重，红线 = 负权重。</div>',
        '</div>'
      ].join('\n');
    },

    init: function (root) {
      el = {
        shIn: root.querySelector('[data-ref="sh-in"]'),
        shKernel: root.querySelector('[data-ref="sh-kernel"]'),
        shOut: root.querySelector('[data-ref="sh-out"]'),
        stCin: root.querySelector('[data-ref="st-cin"]'),
        stCout: root.querySelector('[data-ref="st-cout"]'),
        stK: root.querySelector('[data-ref="st-k"]'),
        stN: root.querySelector('[data-ref="st-n"]'),
        stSpeed: root.querySelector('[data-ref="st-speed"]'),
        btnRand: root.querySelector('[data-ref="btn-rand"]'),
        btnReset: root.querySelector('[data-ref="btn-reset"]'),
        btnPrev: root.querySelector('[data-ref="btn-prev"]'),
        btnNext: root.querySelector('[data-ref="btn-next"]'),
        btnPlay: root.querySelector('[data-ref="btn-play"]'),
        formula: root.querySelector('[data-ref="formula"]'),
        ovIn: root.querySelector('[data-ref="ov-in"]'),
        ovK: root.querySelector('[data-ref="ov-k"]'),
        ovOut: root.querySelector('[data-ref="ov-out"]'),
        breakdown: root.querySelector('[data-ref="breakdown"]'),
        fcSvg: root.querySelector('[data-ref="fc-svg"]')
      };

      initData();

      el.stCin.querySelector('[data-act="dec"]').addEventListener('click', function () { if (S.Cin > 1) { S.Cin--; initData(); resetPos(); refreshAll(); } });
      el.stCin.querySelector('[data-act="inc"]').addEventListener('click', function () { if (S.Cin < 4) { S.Cin++; initData(); resetPos(); refreshAll(); } });
      el.stCout.querySelector('[data-act="dec"]').addEventListener('click', function () { if (S.Cout > 1) { S.Cout--; initData(); resetPos(); refreshAll(); } });
      el.stCout.querySelector('[data-act="inc"]').addEventListener('click', function () { if (S.Cout < 4) { S.Cout++; initData(); resetPos(); refreshAll(); } });
      el.stK.querySelector('[data-act="dec"]').addEventListener('click', function () { if (S.K > 1) { S.K--; initData(); resetPos(); refreshAll(); } });
      el.stK.querySelector('[data-act="inc"]').addEventListener('click', function () { if (S.K < 3 && S.K + 1 <= S.n + 2 * PAD) { S.K++; initData(); resetPos(); refreshAll(); } });
      el.stN.querySelector('[data-act="dec"]').addEventListener('click', function () { if (S.n > 3) { S.n--; initData(); resetPos(); refreshAll(); } });
      el.stN.querySelector('[data-act="inc"]').addEventListener('click', function () { if (S.n < 6) { S.n++; initData(); resetPos(); refreshAll(); } });
      el.stSpeed.querySelector('[data-act="dec"]').addEventListener('click', function () { if (S.speed > 0) { S.speed--; syncUI(); } });
      el.stSpeed.querySelector('[data-act="inc"]').addEventListener('click', function () { if (S.speed < SPEEDS.length - 1) { S.speed++; syncUI(); } });

      el.btnRand.addEventListener('click', function () { initData(); resetPos(); refreshAll(); });
      el.btnReset.addEventListener('click', function () { resetPos(); refreshAll(); });
      
      el.btnPrev.addEventListener('click', function () {
        stopPlay();
        if (S.step > 0) {
          S.step--;
        } else {
          const total = outSize() * outSize();
          S.selPos = (S.selPos - 1 + total) % total;
          S.step = S.Cin;
        }
        refreshAll();
      });
      
      el.btnNext.addEventListener('click', function () {
        stopPlay();
        if (S.step < S.Cin) {
          S.step++;
        } else {
          const total = outSize() * outSize();
          S.selPos = (S.selPos + 1) % total;
          S.step = 0;
        }
        refreshAll();
      });

      el.btnPlay.addEventListener('click', function () {
        if (S.playing) stopPlay(); else startPlay();
      });

      refreshAll();
    },

    onHide: function () { stopPlay(); }
  });

})();