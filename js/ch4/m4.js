/* ================================================================
   m4.js — 模块4：Padding 与 Stride 实验（带动态自定义行）
   ================================================================ */
(function () {
  'use strict';

  const rndInt = function (a, b) { return Math.floor(Math.random() * (b - a + 1)) + a; };

  const S = {
    n: 5,
    k: 3,
    pad: 0,
    stride: 1,
    input: [],
    pos: 0,
    playing: false
  };

  let el = {};
  let autoTimer = null;

  const PRESETS = [
    { n: 5, pad: 0, k: 3, stride: 1, name: '原始卷积', note: '无补零，步长1' },
    { n: 5, pad: 1, k: 3, stride: 1, name: 'Same 卷积', note: '补一圈0，尺寸不变' },
    { n: 5, pad: 0, k: 3, stride: 2, name: '步长加倍', note: '无补零，步子跨大' },
    { n: 5, pad: 1, k: 3, stride: 2, name: '经典降维', note: '补零并跨步，尺寸减半' }
  ];

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
    const v = Math.floor((S.n + 2 * S.pad - S.k) / S.stride) + 1;
    return v > 0 ? v : 0;
  }

  // ============================================================
  // 渲染动态表格
  // ============================================================
  function renderTable() {
    let html = '';
    let anyMatch = false;

    // 预设行
    PRESETS.forEach((p, idx) => {
      const isMatch = p.n === S.n && p.pad === S.pad && p.k === S.k && p.stride === S.stride;
      if (isMatch) anyMatch = true;
      html += `
        <tr data-idx="${idx}" class="${isMatch ? 'active' : ''}">
          <td>${p.name}</td>
          <td>${p.n}</td>
          <td>${p.pad}</td>
          <td>${p.k}</td>
          <td>${p.stride}</td>
        </tr>
      `;
    });

    // 自定义行
    html += `
      <tr data-idx="-1" class="custom-row ${!anyMatch ? 'active' : ''}" style="${!anyMatch ? '' : 'cursor:default;'}">
        <td>${anyMatch ? '手动调节' : '当前自定义'}</td>
        <td>${S.n}</td>
        <td>${S.pad}</td>
        <td>${S.k}</td>
        <td>${S.stride}</td>
      </tr>
    `;

    el.tableBody.innerHTML = html;

    // 绑定点击
    Array.from(el.tableBody.children).forEach((tr) => {
      tr.addEventListener('click', function () {
        const idx = parseInt(tr.dataset.idx);
        if (idx >= 0) {
          const p = PRESETS[idx];
          S.n = p.n; S.pad = p.pad; S.k = p.k; S.stride = p.stride;
          S.input = genInput(S.n);
          refresh();
        }
      });
    });
  }

  // ============================================================
  // 渲染公式
  // ============================================================
  function renderFormula() {
    const os = outSize();
    const inner = S.n + 2 * S.pad - S.k;
    
    el.formulaBox.innerHTML = 
      `<div>O = ⌊ (n + 2P − K) / S ⌋ + 1</div>` +
      `<div class="dim" style="margin-top:8px;">= ⌊ (${S.n} + 2×${S.pad} − ${S.k}) / ${S.stride} ⌋ + 1</div>` +
      `<div class="dim">= ⌊ ${inner} / ${S.stride} ⌋ + 1</div>` +
      `<div style="margin-top:12px;">输出尺寸 <span class="highlight">${os > 0 ? os + ' × ' + os : '无效'}</span></div>`;
  }

  // ============================================================
  // 渲染全景矩阵
  // ============================================================
  function renderMatrix() {
    const os = outSize();
    const total = S.n + 2 * S.pad;
    
    const oi = os > 0 ? Math.floor(S.pos / os) : 0;
    const oj = os > 0 ? S.pos % os : 0;
    
    const winR0 = oi * S.stride;
    const winC0 = oj * S.stride;

    // 输入矩阵
    el.inGrid.style.gridTemplateColumns = `repeat(${total}, 32px)`;
    el.inGrid.innerHTML = '';
    for (let r = 0; r < total; r++) {
      for (let c = 0; c < total; c++) {
        const rr = r - S.pad;
        const cc = c - S.pad;
        const isPad = (rr < 0 || rr >= S.n || cc < 0 || cc >= S.n);
        
        const cell = document.createElement('div');
        cell.className = 'cell' + (isPad ? ' pad' : '');
        cell.textContent = isPad ? '0' : S.input[rr][cc];

        if (os > 0 && r >= winR0 && r < winR0 + S.k && c >= winC0 && c < winC0 + S.k) {
          cell.classList.add('win');
        }
        el.inGrid.appendChild(cell);
      }
    }

    // 输出矩阵
    el.outGrid.innerHTML = '';
    if (os <= 0) {
      el.outGrid.innerHTML = '<div class="tip" style="color:#c0392b;">尺寸无效，卷积核太大</div>';
      return;
    }
    el.outGrid.style.gridTemplateColumns = `repeat(${os}, 32px)`;
    
    for (let i = 0; i < os; i++) {
      for (let j = 0; j < os; j++) {
        const idx = i * os + j;
        const cell = document.createElement('div');
        cell.className = 'cell out';
        
        if (idx < S.pos) {
          cell.classList.add('done');
          cell.textContent = '✓';
        } else if (idx === S.pos) {
          cell.classList.add('cur');
          cell.textContent = '?';
        }
        el.outGrid.appendChild(cell);
      }
    }
  }

  // ============================================================
  // 同步 UI 和数据
  // ============================================================
  function syncUI() {
    el.stN.querySelector('.step-val').textContent = S.n;
    el.stK.querySelector('.step-val').textContent = S.k;
    el.stP.querySelector('.step-val').textContent = S.pad;
    el.stS.querySelector('.step-val').textContent = S.stride;
    
    renderTable();
    renderFormula();
    renderMatrix();
  }

  function refresh() {
    S.pos = 0;
    stopPlay();
    syncUI();
  }

  function stopPlay() {
    S.playing = false;
    if (autoTimer) { clearTimeout(autoTimer); autoTimer = null; }
    el.btnPlay.textContent = '自动播放';
  }

  function startPlay() {
    if (S.playing) return;
    const total = outSize() * outSize();
    if (total <= 0) return;
    
    S.playing = true;
    el.btnPlay.textContent = '暂停';

    const step = function () {
      if (!S.playing) return;
      S.pos = (S.pos + 1) % total;
      renderMatrix();
      autoTimer = setTimeout(step, 800);
    };
    autoTimer = setTimeout(step, 400);
  }

  // ============================================================
  // 注册模块
  // ============================================================
  Lab.register({
    id: 'ch4-m4',
    index: '04',
    title: 'Padding 与 Stride',
    desc: 'Padding 是补零加边框，Stride 是滑动步子迈多大。这俩决定了输出特征图的尺寸。',
    template: function () {
      return [
        '<div class="concept-panel">',
        '  <h4>💡 两个参数改变一切</h4>',
        '  <p><b>Padding (P)：</b> 给图的外面补一圈 0（虚线框）。为什么补？因为不补的话，图越卷越小，而且边缘的像素只被算过一次，信息容易丢失。补一圈，尺寸就能保住。</p>',
        '  <p><b>Stride (S)：</b> 卷积核滑动的步伐。步子迈 1，是挨着走；步子迈 2，是隔一个走，输出尺寸直接砍半。</p>',
        '  <p>👉 <b>动手试试：</b> 点击左边的表格，或者自己调步进器。看右边图上的虚线框（补零）和红框（滑动窗口）是怎么跟着变的。</p>',
        '</div>',

        '<div class="layout-split">',
        '  <div class="left-panel">',
        '    <div class="panel" style="padding:0; overflow:hidden;">',
        '      <table class="exp-table">',
        '        <thead><tr><th>实验方案</th><th>n</th><th>P</th><th>K</th><th>S</th></tr></thead>',
        '        <tbody data-ref="table-body"></tbody>',
        '      </table>',
        '    </div>',
        '    <div class="formula-box" data-ref="formula-box"></div>',
        '    <div class="control-box">',
        '      <div class="ctrl"><label>n</label><div class="stepper" data-ref="st-n"><button class="step-btn" data-act="dec">−</button><span class="step-val">5</span><button class="step-btn" data-act="inc">+</button></div></div>',
        '      <div class="ctrl"><label>P</label><div class="stepper" data-ref="st-p"><button class="step-btn" data-act="dec">−</button><span class="step-val">0</span><button class="step-btn" data-act="inc">+</button></div></div>',
        '      <div class="ctrl"><label>K</label><div class="stepper" data-ref="st-k"><button class="step-btn" data-act="dec">−</button><span class="step-val">3</span><button class="step-btn" data-act="inc">+</button></div></div>',
        '      <div class="ctrl"><label>S</label><div class="stepper" data-ref="st-s"><button class="step-btn" data-act="dec">−</button><span class="step-val">1</span><button class="step-btn" data-act="inc">+</button></div></div>',
        '    </div>',
        '  </div>',

        '  <div class="right-panel">',
        '    <div class="panel" style="padding:16px;">',
        '      <div class="ctrl-row" style="justify-content: center; margin-bottom: 10px;">',
        '        <button class="btn" data-ref="btn-rand">随机化输入</button>',
        '        <button class="btn" data-ref="btn-prev">← 上一步</button>',
        '        <button class="btn primary" data-ref="btn-next">下一步 →</button>',
        '        <button class="btn" data-ref="btn-play">自动播放</button>',
        '      </div>',
        '      <div class="matrix-stage">',
        '        <div class="matrix-group">',
        '          <div class="matrix-label">输入矩阵 (Input)</div>',
        '          <div class="grid-matrix" data-ref="in-grid"></div>',
        '        </div>',
        '        <div class="stage-op">→</div>',
        '        <div class="matrix-group">',
        '          <div class="matrix-label">输出矩阵 (Output)</div>',
        '          <div class="grid-matrix" data-ref="out-grid"></div>',
        '        </div>',
        '      </div>',
        '    </div>',
        '  </div>',
        '</div>'
      ].join('\n');
    },

    init: function (root) {
      el = {
        tableBody: root.querySelector('[data-ref="table-body"]'),
        formulaBox: root.querySelector('[data-ref="formula-box"]'),
        stN: root.querySelector('[data-ref="st-n"]'),
        stP: root.querySelector('[data-ref="st-p"]'),
        stK: root.querySelector('[data-ref="st-k"]'),
        stS: root.querySelector('[data-ref="st-s"]'),
        btnRand: root.querySelector('[data-ref="btn-rand"]'),
        btnPrev: root.querySelector('[data-ref="btn-prev"]'),
        btnNext: root.querySelector('[data-ref="btn-next"]'),
        btnPlay: root.querySelector('[data-ref="btn-play"]'),
        inGrid: root.querySelector('[data-ref="in-grid"]'),
        outGrid: root.querySelector('[data-ref="out-grid"]')
      };

      S.input = genInput(S.n);

      // 步进器
      el.stN.querySelector('[data-act="dec"]').addEventListener('click', () => { if (S.n > 4) { S.n--; S.input = genInput(S.n); refresh(); } });
      el.stN.querySelector('[data-act="inc"]').addEventListener('click', () => { if (S.n < 7) { S.n++; S.input = genInput(S.n); refresh(); } });
      
      el.stP.querySelector('[data-act="dec"]').addEventListener('click', () => { if (S.pad > 0) { S.pad--; refresh(); } });
      el.stP.querySelector('[data-act="inc"]').addEventListener('click', () => { if (S.pad < 2) { S.pad++; refresh(); } });
      
      el.stK.querySelector('[data-act="dec"]').addEventListener('click', () => { if (S.k > 1) { S.k--; refresh(); } });
      el.stK.querySelector('[data-act="inc"]').addEventListener('click', () => { if (S.k < 5) { S.k++; refresh(); } });
      
      el.stS.querySelector('[data-act="dec"]').addEventListener('click', () => { if (S.stride > 1) { S.stride--; refresh(); } });
      el.stS.querySelector('[data-act="inc"]').addEventListener('click', () => { if (S.stride < 3) { S.stride++; refresh(); } });

      // 按钮
      el.btnRand.addEventListener('click', () => { S.input = genInput(S.n); refresh(); });
      el.btnPrev.addEventListener('click', () => {
        stopPlay();
        const total = outSize() * outSize();
        if (total <= 0) return;
        S.pos = (S.pos - 1 + total) % total;
        renderMatrix();
      });
      el.btnNext.addEventListener('click', () => {
        stopPlay();
        const total = outSize() * outSize();
        if (total <= 0) return;
        S.pos = (S.pos + 1) % total;
        renderMatrix();
      });
      el.btnPlay.addEventListener('click', () => { if (S.playing) stopPlay(); else startPlay(); });

      syncUI();
    },

    onHide: function () { stopPlay(); }
  });

})();