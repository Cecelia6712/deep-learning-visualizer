/* ================================================================
   m3.js — 模块3：池化层（含滑动对比）
   ================================================================ */
(function () {
  'use strict';

  const rndInt = function (a, b) { return Math.floor(Math.random() * (b - a + 1)) + a; };
  const round1 = function (v) { return Math.round(v * 10) / 10; };

  const S = {
    n: 5,
    k: 2,
    stride: 2,
    type: 'max',
    input: [],
    pos: 0,
    playing: false,
    speed: 1
  };
  const SPEEDS = [500, 800, 1200, 1800];

  let el = {};
  let autoTimer = null;

  function genInput(n) {
    const arr = [];
    for (let i = 0; i < n; i++) {
      const row = [];
      for (let j = 0; j < n; j++) row.push(rndInt(0, 9));
      arr.push(row);
    }
    return arr;
  }

  function outSize() {
    const v = Math.floor((S.n - S.k) / S.stride) + 1;
    return v > 0 ? v : 0;
  }

  function computePool(oi, oj) {
    const r = oi * S.stride;
    const c = oj * S.stride;
    const vals = [];
    for (let i = 0; i < S.k; i++) {
      for (let j = 0; j < S.k; j++) {
        vals.push(S.input[r + i][c + j]);
      }
    }
    if (S.type === 'max') {
      const maxVal = Math.max(...vals);
      return { vals: vals, res: maxVal, maxVal: maxVal };
    } else {
      const sum = vals.reduce((a, b) => a + b, 0);
      return { vals: vals, res: round1(sum / vals.length), sum: sum };
    }
  }

  // ============================================================
  // 渲染全景图
  // ============================================================
  function renderOverview() {
    const os = outSize();
    const oi = Math.floor(S.pos / os) || 0;
    const oj = S.pos % os || 0;

    // 1. 输入矩阵
    el.inMatrix.innerHTML = '';
    el.inMatrix.style.gridTemplateColumns = 'repeat(' + S.n + ', 40px)';

    let poolRes = null;
    if (os > 0) poolRes = computePool(oi, oj);

    for (let r = 0; r < S.n; r++) {
      for (let c = 0; c < S.n; c++) {
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.textContent = S.input[r][c];

        if (os > 0) {
          const winR0 = oi * S.stride;
          const winC0 = oj * S.stride;
          if (r >= winR0 && r < winR0 + S.k && c >= winC0 && c < winC0 + S.k) {
            cell.classList.add('win');
            if (S.type === 'max' && S.input[r][c] === poolRes.maxVal) {
              cell.classList.add('max-hit');
            }
          }
        }
        el.inMatrix.appendChild(cell);
      }
    }

    // 2. 输出矩阵
    el.outMatrix.innerHTML = '';
    if (os <= 0) {
      el.outMatrix.innerHTML = '<div class="tip">核太大，无法池化</div>';
      return;
    }
    el.outMatrix.style.gridTemplateColumns = 'repeat(' + os + ', 40px)';

    for (let i = 0; i < os; i++) {
      for (let j = 0; j < os; j++) {
        const idx = i * os + j;
        const cell = document.createElement('div');
        cell.className = 'cell out';
        if (idx < S.pos) {
          cell.classList.add('done');
          cell.textContent = computePool(i, j).res;
        } else if (idx === S.pos) {
          cell.classList.add('cur');
          cell.textContent = computePool(i, j).res;
        }
        el.outMatrix.appendChild(cell);
      }
    }

    // 3. 算子图标及大小提示
    el.opIcon.textContent = S.type === 'max' ? '取最大 Max' : '算平均 Avg';
    el.curPosText.textContent = '当前处理输出位置: [' + oi + ',' + oj + ']';
  }

  // ============================================================
  // 渲染计算明细
  // ============================================================
  function renderDetail() {
    const os = outSize();
    if (os <= 0) {
      el.detailBox.innerHTML = '<span class="tip">无有效窗口</span>';
      return;
    }
    const oi = Math.floor(S.pos / os);
    const oj = S.pos % os;
    const res = computePool(oi, oj);

    let html = '<div class="extracted-nums">';
    res.vals.forEach(v => {
      let isMax = (S.type === 'max' && v === res.maxVal);
      html += `<div class="num-card${isMax ? ' max' : ''}">${v}</div>`;
    });
    html += '</div>';

    html += '<div class="math-text">';
    if (S.type === 'max') {
      html += `挑出最大值 <span class="highlight">${res.maxVal}</span>`;
      html += `<br><span class="dim">因为窗口是 ${S.k}x${S.k}，里面共有 ${res.vals.length} 个数字，最大的那个胜出。</span>`;
    } else {
      html += `${res.vals.join(' + ')} = ${res.sum}`;
      html += `<br><span class="highlight">${res.sum} ÷ ${res.vals.length} = ${res.res}</span>`;
      html += `<br><span class="dim">全部加起来，除以 ${res.vals.length}。</span>`;
    }
    html += '</div>';
    el.detailBox.innerHTML = html;
  }

  function refreshAll() {
    renderOverview();
    renderDetail();
    syncUI();
  }

  function syncUI() {
    el.stN.querySelector('.step-val').textContent = S.n;
    el.stK.querySelector('.step-val').textContent = S.k;
    el.stS.querySelector('.step-val').textContent = S.stride;
    el.stSpeed.querySelector('.step-val').textContent = SPEEDS[S.speed] + 'ms';
    el.btnMax.classList.toggle('active', S.type === 'max');
    el.btnAvg.classList.toggle('active', S.type === 'avg');
  }

  function resetPos() { stopPlay(); S.pos = 0; }
  
  function stopPlay() {
    S.playing = false;
    if (autoTimer) { clearTimeout(autoTimer); autoTimer = null; }
    el.btnPlay.textContent = '自动播放';
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
      refreshAll();
      autoTimer = setTimeout(step, SPEEDS[S.speed]);
    };
    autoTimer = setTimeout(step, 400);
  }

  // ============================================================
  // 注册模块
  // ============================================================
  Lab.register({
    id: 'm3',
    index: '03',
    title: '池化层',
    desc: '池化层没有参数，通道数不变，只变分辨率。',
    template: function () {
      return [
        '<div class="concept-panel">',
        '  <h4>💡 为什么池化层看起来和卷积不一样？</h4>',
        '  <p><b>卷积（前一个模块）：</b> 步长通常是 <b>1</b>，窗口是一格一格爬过去的，<b>大面积重叠</b>，为了提取每一寸像素的特征。</p>',
        '  <p><b>池化（本模块）：</b> 步长通常等于核大小（比如 <b>2</b>），窗口是<b>跳跃</b>的，<b>不重叠</b>。为了把图缩小（降维），保留最强特征。</p>',
        '  <p>👉 <b>动手试试：</b> 你在下面把步长（Stride）调成 1，池化窗口就会像卷积一样重叠了！</p>',
        '</div>',

        '<div class="compare-panel">',
        '  <div class="compare-item">',
        '    <div class="compare-title">卷积的滑动 <span>(Stride=1，重叠)</span></div>',
        '    <div class="mini-grid">',
        '      <div class="mini-cell win-overlap"></div><div class="mini-cell win-overlap"></div><div class="mini-cell win-overlap"></div><div class="mini-cell"></div>',
        '      <div class="mini-cell win-overlap"></div><div class="mini-cell win-overlap"></div><div class="mini-cell win-overlap"></div><div class="mini-cell"></div>',
        '      <div class="mini-cell win-overlap"></div><div class="mini-cell win-overlap"></div><div class="mini-cell win-overlap"></div><div class="mini-cell"></div>',
        '      <div class="mini-cell"></div><div class="mini-cell"></div><div class="mini-cell"></div><div class="mini-cell"></div>',
        '    </div>',
        '  </div>',
        '  <div class="compare-item">',
        '    <div class="compare-title">池化的滑动 <span>(Stride=2，不重叠)</span></div>',
        '    <div class="mini-grid">',
        '      <div class="mini-cell win-step"></div><div class="mini-cell win-step"></div><div class="mini-cell win-step-2"></div><div class="mini-cell win-step-2"></div>',
        '      <div class="mini-cell win-step"></div><div class="mini-cell win-step"></div><div class="mini-cell win-step-2"></div><div class="mini-cell win-step-2"></div>',
        '      <div class="mini-cell"></div><div class="mini-cell"></div><div class="mini-cell"></div><div class="mini-cell"></div>',
        '      <div class="mini-cell"></div><div class="mini-cell"></div><div class="mini-cell"></div><div class="mini-cell"></div>',
        '    </div>',
        '  </div>',
        '</div>',

        '<div class="toolbar">',
        '  <div class="segmented">',
        '    <button class="seg-btn active" data-ref="btn-max">Max Pooling (取最大)</button>',
        '    <button class="seg-btn" data-ref="btn-avg">Avg Pooling (取平均)</button>',
        '  </div>',
        '  <div class="ctrl"><label>输入尺寸 n</label>',
        '    <div class="stepper" data-ref="st-n"><button class="step-btn" data-act="dec">−</button><span class="step-val">5</span><button class="step-btn" data-act="inc">+</button></div></div>',
        '  <div class="ctrl"><label>池化核 K</label>',
        '    <div class="stepper" data-ref="st-k"><button class="step-btn" data-act="dec">−</button><span class="step-val">2</span><button class="step-btn" data-act="inc">+</button></div></div>',
        '  <div class="ctrl"><label>步长 S</label>',
        '    <div class="stepper" data-ref="st-s"><button class="step-btn" data-act="dec">−</button><span class="step-val">2</span><button class="step-btn" data-act="inc">+</button></div></div>',
        '</div>',

        '<div class="panel">',
        '  <div class="ctrl-row" style="margin-bottom: 20px;">',
        '    <button class="btn" data-ref="btn-rand">随机化数据</button>',
        '    <button class="btn" data-ref="btn-reset">重置动画</button>',
        '    <button class="btn" data-ref="btn-prev">← 上一步</button>',
        '    <button class="btn primary" data-ref="btn-next">下一步 →</button>',
        '    <button class="btn" data-ref="btn-play">自动播放</button>',
        '    <div class="ctrl"><label>间隔</label>',
        '      <div class="stepper" data-ref="st-speed"><button class="step-btn" data-act="dec">−</button><span class="step-val">800ms</span><button class="step-btn" data-act="inc">+</button></div></div>',
        '  </div>',
        
        '  <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #eee; padding-bottom:8px; margin-bottom:12px;">',
        '    <h3 style="margin:0; padding:0; border:none;">全景视图</h3>',
        '    <span style="font-family:Consolas; font-size:13px; color:#c0392b;" data-ref="cur-pos-text"></span>',
        '  </div>',
        '  <div class="stage">',
        '    <div class="stage-col">',
        '      <div class="col-label">输入矩阵 (Input)</div>',
        '      <div class="matrix" data-ref="in-matrix"></div>',
        '    </div>',
        '    <div class="stage-op">',
        '      <div class="op-icon" data-ref="op-icon">Max</div>',
        '      <div class="op-arrow">→</div>',
        '    </div>',
        '    <div class="stage-col">',
        '      <div class="col-label">输出矩阵 (Output)</div>',
        '      <div class="matrix" data-ref="out-matrix"></div>',
        '    </div>',
        '  </div>',
        '</div>',

        '<div class="panel">',
        '  <h3>当前窗口计算明细</h3>',
        '  <div class="detail-box" data-ref="detail-box"></div>',
        '</div>'
      ].join('\n');
    },

    init: function (root) {
      el = {
        btnMax: root.querySelector('[data-ref="btn-max"]'),
        btnAvg: root.querySelector('[data-ref="btn-avg"]'),
        stN: root.querySelector('[data-ref="st-n"]'),
        stK: root.querySelector('[data-ref="st-k"]'),
        stS: root.querySelector('[data-ref="st-s"]'),
        stSpeed: root.querySelector('[data-ref="st-speed"]'),
        btnRand: root.querySelector('[data-ref="btn-rand"]'),
        btnReset: root.querySelector('[data-ref="btn-reset"]'),
        btnPrev: root.querySelector('[data-ref="btn-prev"]'),
        btnNext: root.querySelector('[data-ref="btn-next"]'),
        btnPlay: root.querySelector('[data-ref="btn-play"]'),
        inMatrix: root.querySelector('[data-ref="in-matrix"]'),
        outMatrix: root.querySelector('[data-ref="out-matrix"]'),
        opIcon: root.querySelector('[data-ref="op-icon"]'),
        detailBox: root.querySelector('[data-ref="detail-box"]'),
        curPosText: root.querySelector('[data-ref="cur-pos-text"]')
      };

      S.input = genInput(S.n);

      el.btnMax.addEventListener('click', function () {
        if (S.type !== 'max') { S.type = 'max'; resetPos(); refreshAll(); }
      });
      el.btnAvg.addEventListener('click', function () {
        if (S.type !== 'avg') { S.type = 'avg'; resetPos(); refreshAll(); }
      });

      el.stN.querySelector('[data-act="dec"]').addEventListener('click', function () {
        if (S.n > 4) { S.n--; S.input = genInput(S.n); resetPos(); refreshAll(); }
      });
      el.stN.querySelector('[data-act="inc"]').addEventListener('click', function () {
        if (S.n < 6) { S.n++; S.input = genInput(S.n); resetPos(); refreshAll(); }
      });
      el.stK.querySelector('[data-act="dec"]').addEventListener('click', function () {
        if (S.k > 2) { S.k--; resetPos(); refreshAll(); }
      });
      el.stK.querySelector('[data-act="inc"]').addEventListener('click', function () {
        if (S.k < 3 && S.k + 1 <= S.n) { S.k++; resetPos(); refreshAll(); }
      });
      el.stS.querySelector('[data-act="dec"]').addEventListener('click', function () {
        if (S.stride > 1) { S.stride--; resetPos(); refreshAll(); }
      });
      el.stS.querySelector('[data-act="inc"]').addEventListener('click', function () {
        if (S.stride < 3) { S.stride++; resetPos(); refreshAll(); }
      });
      el.stSpeed.querySelector('[data-act="dec"]').addEventListener('click', function () {
        if (S.speed > 0) { S.speed--; syncUI(); }
      });
      el.stSpeed.querySelector('[data-act="inc"]').addEventListener('click', function () {
        if (S.speed < SPEEDS.length - 1) { S.speed++; syncUI(); }
      });

      el.btnRand.addEventListener('click', function () { S.input = genInput(S.n); resetPos(); refreshAll(); });
      el.btnReset.addEventListener('click', function () { resetPos(); refreshAll(); });
      
      el.btnPrev.addEventListener('click', function () {
        stopPlay();
        const total = outSize() * outSize();
        if (total <= 0) return;
        S.pos = (S.pos - 1 + total) % total;
        refreshAll();
      });
      
      el.btnNext.addEventListener('click', function () {
        stopPlay();
        const total = outSize() * outSize();
        if (total <= 0) return;
        S.pos = (S.pos + 1) % total;
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