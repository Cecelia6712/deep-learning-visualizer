/* ================================================================
   m5.js — 模块5：LeNet 张量流水线（真实全连接版）
   ================================================================ */
(function () {
  'use strict';

  const LAYERS = [
    { id: 'in', name: '输入', shape: '1 × 28 × 28', desc: '原始图像' },
    { id: 'c1', name: 'Conv1', shape: '6 × 28 × 28', desc: '6个 5×5 卷积 (补零)' },
    { id: 'p1', name: 'Pool1', shape: '6 × 14 × 14', desc: '2×2 最大池化' },
    { id: 'c2', name: 'Conv2', shape: '16 × 10 × 10', desc: '16个 5×5 卷积' },
    { id: 'p2', name: 'Pool2', shape: '16 × 5 × 5', desc: '2×2 最大池化' },
    { id: 'f0', name: 'Flatten', shape: '400', desc: '拉平展直' },
    { id: 'f1', name: 'FC1', shape: '120', desc: '全连接 400→120' },
    { id: 'f2', name: 'FC2', shape: '84', desc: '全连接 120→84' },
    { id: 'out', name: '输出', shape: '10', desc: '分类得分' }
  ];

  const S = {
    activeIdx: 0,
    tensors: {}
  };

  let el = {};

  // 生成张量数据
  function initData() {
    const inSize = 28;
    const img = Array.from({length: inSize}, () => Array(inSize).fill(0));
    for(let r=4; r<24; r++) img[r][13] = 1;
    for(let c=11; c<16; c++) img[20][c] = 1;
    for(let r=4; r<24; r++) { img[r][12] = 0.5; img[r][14] = 0.5; }

    const c1 = [];
    for(let ch=0; ch<6; ch++) {
      const layer = Array.from({length: inSize}, () => Array(inSize).fill(0));
      for(let r=0; r<inSize; r++) {
        for(let c=0; c<inSize; c++) {
          if (img[r][c] === 0) continue;
          if (ch === 0 && c >= 11 && c <= 15) layer[r][c] = 1;
          if (ch === 1 && r >= 18 && r <= 22) layer[r][c] = 1;
          if (ch === 2 && Math.abs(r - c) < 2) layer[r][c] = 1;
          if (ch === 3 && Math.abs(r + c - 28) < 2) layer[r][c] = 1;
          if (ch === 4 && Math.random() > 0.8) layer[r][c] = 1;
          if (ch === 5 && r % 5 === 0) layer[r][c] = 1;
        }
      }
      c1.push(layer);
    }

    const p1 = [];
    for(let ch=0; ch<6; ch++) {
      const layer = Array.from({length: 14}, () => Array(14).fill(0));
      for(let r=0; r<14; r++) {
        for(let c=0; c<14; c++) {
          layer[r][c] = Math.max(c1[ch][r*2][c*2], c1[ch][r*2][c*2+1], c1[ch][r*2+1][c*2], c1[ch][r*2+1][c*2+1]);
        }
      }
      p1.push(layer);
    }

    const c2 = [];
    for(let ch=0; ch<16; ch++) {
      const layer = Array.from({length: 10}, () => Array(10).fill(0));
      for(let r=0; r<10; r++) {
        for(let c=0; c<10; c++) {
          if (Math.random() > 0.75) layer[r][c] = 1;
        }
      }
      c2.push(layer);
    }

    const p2 = [];
    for(let ch=0; ch<16; ch++) {
      const layer = Array.from({length: 5}, () => Array(5).fill(0));
      for(let r=0; r<5; r++) {
        for(let c=0; c<5; c++) {
          layer[r][c] = Math.max(c2[ch][r*2][c*2], c2[ch][r*2][c*2+1], c2[ch][r*2+1][c*2], c2[ch][r*2+1][c*2+1]);
        }
      }
      p2.push(layer);
    }

    S.tensors = { in: [img], c1: c1, p1: p1, c2: c2, p2: p2 };
  }

  function renderPipeline() {
    el.pipeline.innerHTML = '';
    LAYERS.forEach((layer, idx) => {
      const node = document.createElement('div');
      node.className = 'pipe-node' + (idx === S.activeIdx ? ' active' : '');
      node.innerHTML = `<div class="node-name">${layer.name}</div><div class="node-shape">${layer.shape}</div>`;
      node.addEventListener('click', () => {
        S.activeIdx = idx; renderPipeline(); renderDetail();
      });
      el.pipeline.appendChild(node);
      if (idx < LAYERS.length - 1) {
        el.pipeline.appendChild(Object.assign(document.createElement('div'), { className: 'pipe-arrow', innerHTML: '→' }));
      }
    });
  }

  function createTensorView(channels, data, rows, cols) {
    const container = document.createElement('div');
    container.className = 'stack-container';
    const showCh = Math.min(channels, 6);
    const cellSize = 3; 
    const width = cols * cellSize;
    const height = rows * cellSize;

    for (let c = showCh - 1; c >= 0; c--) {
      const card = document.createElement('div');
      card.className = 'layer-card';
      card.style.width = width + 'px'; card.style.height = height + 'px';
      const offset = showCh > 4 ? 4 : 6;
      card.style.transform = `translate(${c * offset}px, ${-c * offset}px)`;
      card.style.zIndex = 100 - c;
      if (c > 0) card.style.opacity = 0.85;

      const grid = document.createElement('div');
      grid.className = 'grid-box';
      grid.style.gridTemplateColumns = `repeat(${cols}, ${cellSize}px)`;
      const layerData = data[c] || [];

      for (let r = 0; r < rows; r++) {
        for (let col = 0; col < cols; col++) {
          const cell = document.createElement('div');
          cell.className = 'grid-cell';
          cell.style.width = cellSize + 'px'; cell.style.height = cellSize + 'px';
          if (layerData[r] && layerData[r][col] > 0) {
            cell.classList.add('on');
            if (layerData[r][col] < 1) cell.style.opacity = layerData[r][col];
          }
          grid.appendChild(cell);
        }
      }
      card.appendChild(grid);
      container.appendChild(card);
    }
    container.style.width = (width + showCh * 6 + 16) + 'px';
    container.style.height = (height + showCh * 6 + 16) + 'px';
    return container;
  }

  // 展平层：真·长面条 (40x10=400)
  function createFlattenView(length) {
    const container = document.createElement('div');
    container.className = 'flatten-container';
    
    const cols = 40, rows = 10, cellSize = 3; 
    const grid = document.createElement('div');
    grid.className = 'grid-box';
    grid.style.gridTemplateColumns = `repeat(${cols}, ${cellSize}px)`;
    
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cell = document.createElement('div');
        cell.className = 'grid-cell';
        cell.style.width = cellSize + 'px'; cell.style.height = cellSize + 'px';
        if (Math.random() > 0.4) cell.style.background = '#2980b9'; 
        grid.appendChild(cell);
      }
    }
    container.appendChild(grid);
    
    const label = document.createElement('div');
    label.style.fontSize = '11px'; label.style.color = '#666';
    label.style.marginTop = '8px'; label.style.fontFamily = 'Consolas, monospace';
    label.textContent = `拉平后的一维向量 (长度: ${length})`;
    container.appendChild(label);

    return container;
  }

  // 全连接网格：扁长型
  function createNeuronGrid(count, isOut) {
    const container = document.createElement('div');
    container.className = 'fc-grid';
    
    if (isOut) {
      container.style.gridTemplateColumns = 'repeat(5, 1fr)';
      container.style.gap = '8px';
      for (let i = 0; i < count; i++) {
        const dot = document.createElement('div');
        dot.className = 'fc-neuron out-node' + (i === 1 ? ' active' : '');
        dot.textContent = i;
        container.appendChild(dot);
      }
    } else {
      let cols = 20; 
      let rows = Math.ceil(count / cols);
      container.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
      
      for (let i = 0; i < rows * cols; i++) {
        const dot = document.createElement('div');
        dot.className = 'fc-neuron';
        if (i >= count) {
          dot.style.opacity = '0';
        } else if (Math.random() > 0.6) {
          dot.style.background = '#2980b9';
        }
        container.appendChild(dot);
      }
    }
    return container;
  }

  // ============================================================
  // 渲染详情
  // ============================================================
  function renderDetail() {
    const idx = S.activeIdx;
    el.detailStage.innerHTML = '';

    if (idx === 0) {
      const v = document.createElement('div');
      v.className = 'tensor-view';
      v.innerHTML = `<div class="tensor-title">输入数据<span>1 × 28 × 28</span></div>`;
      v.appendChild(createTensorView(1, S.tensors.in, 28, 28));
      const op = document.createElement('div');
      op.className = 'op-box';
      op.innerHTML = `<div class="op-icon">🖼️</div><div class="op-desc">这是我们要输入给神经网络的原始图像。网格上的黑色像素点勾勒出了数字的轮廓。</div>`;
      el.detailStage.appendChild(v); el.detailStage.appendChild(op);
      return;
    }

    const prev = LAYERS[idx - 1];
    const cur = LAYERS[idx];

    // --- 左侧 ---
    const vLeft = document.createElement('div');
    vLeft.className = 'tensor-view';
    vLeft.innerHTML = `<div class="tensor-title">上一层的输出<span>${prev.name} · ${prev.shape}</span></div>`;
    if (prev.id === 'f0') {
      vLeft.appendChild(createFlattenView(400));
    } else if (prev.id === 'f1' || prev.id === 'f2') {
      vLeft.appendChild(createNeuronGrid(parseInt(prev.shape), false));
    } else if (prev.id === 'out') {
      vLeft.appendChild(createNeuronGrid(10, true));
    } else {
      const parts = prev.shape.split(' × ');
      vLeft.appendChild(createTensorView(parseInt(parts[0]), S.tensors[prev.id], parseInt(parts[1]), parseInt(parts[2])));
    }

    // --- 右侧 ---
    const vRight = document.createElement('div');
    vRight.className = 'tensor-view';
    vRight.innerHTML = `<div class="tensor-title">当前层输出<span>${cur.name} · ${cur.shape}</span></div>`;
    if (cur.id === 'f0') {
      vRight.appendChild(createFlattenView(400));
    } else if (cur.id === 'f1' || cur.id === 'f2') {
      vRight.appendChild(createNeuronGrid(parseInt(cur.shape), false));
    } else if (cur.id === 'out') {
      vRight.appendChild(createNeuronGrid(10, true));
    } else {
      const parts = cur.shape.split(' × ');
      vRight.appendChild(createTensorView(parseInt(parts[0]), S.tensors[cur.id], parseInt(parts[1]), parseInt(parts[2])));
    }

    // --- 中间：说明文字与图标 ---
    const opBox = document.createElement('div');
    opBox.className = 'op-box';
    let desc = '', formula = '', iconHTML = '↓';

    if (cur.id === 'c1' || cur.id === 'c2') {
      desc = `【卷积变厚】用 <b>${cur.id === 'c1' ? 6 : 16} 个</b>不同的卷积核去提取特征，有的找竖线，有的找横线，最终叠加成 <b>${cur.id === 'c1' ? 6 : 16} 层</b>。`;
      formula = cur.id === 'c1' ? 'P=2, K=5, S=1' : 'P=0, K=5, S=1';
    } else if (cur.id === 'p1' || cur.id === 'p2') {
      desc = `【池化浓缩】进行 2×2 的最大池化，层数不变，但长和宽直接<b>砍半</b>！仔细看，右边的图小了一圈。`;
      formula = 'K=2, S=2';
    } else if (cur.id === 'f0') {
      desc = `【拉平展直】把 16 层 5×5 的小方块，全部拆开揉碎，<b>拉成一根 400 长度的面条</b>。`;
    } else if (cur.id === 'f1' || cur.id === 'f2') {
      desc = `【全连接投票】把上一层的特征，通过<b>权重网络交叉连接</b>起来，压缩成 <b>${cur.shape} 个</b>神经元。`;
      
      // 真正的大号全连接交叉网络
      iconHTML = `
        <svg class="fc-network-svg" viewBox="0 0 200 160">
          <!-- 画线：左边6个点，右边6个点，全交叉 -->
          ${[20, 50, 80, 110, 140].map(y1 =>
            [20, 50, 80, 110, 140].map(y2 =>
              `<line x1="40" y1="${y1}" x2="160" y2="${y2}" stroke="#bce0fd" stroke-width="1.2" opacity="0.7"/>`
            ).join('')
          ).join('')}
          
          <!-- 左边节点 (上一层) -->
          ${[20, 50, 80, 110, 140].map(y =>
            `<circle cx="40" cy="${y}" r="5" fill="#2980b9" />`
          ).join('')}
          
          <!-- 右边节点 (当前层) -->
          ${[20, 50, 80, 110, 140].map((y, i) =>
            `<circle cx="160" cy="${y}" r="5" fill="${i%2===0?'#c0392b':'#fff'}" stroke="#c0392b" stroke-width="2" />`
          ).join('')}
        </svg>
      `;
    } else if (cur.id === 'out') {
      desc = `【最终分类】吐出 <b>10 个分数</b>，代表 0~9 的概率。数字 1 的得分最高（红色）。`;
    }

    // 将 SVG 或箭头放入中间
    if (cur.id === 'f1' || cur.id === 'f2') {
      opBox.innerHTML = `
        ${iconHTML}
        <div class="op-desc">${desc}</div>
      `;
    } else {
      opBox.innerHTML = `
        <div class="op-icon">${iconHTML}</div>
        <div class="op-desc">${desc}</div>
        ${formula ? `<div class="op-formula">${formula}</div>` : ''}
      `;
    }

    el.detailStage.appendChild(vLeft);
    el.detailStage.appendChild(opBox);
    el.detailStage.appendChild(vRight);
  }

  Lab.register({
    id: 'ch4-m5',
    index: '05',
    title: 'LeNet 张量流水线',
    desc: 'LeNet 是由 Yann LeCun 提出的早期卷积神经网络。点击流水线，看看数据到底变了什么魔术。',
    template: function () {
      return [
        '<div class="concept-panel">',
        '  <h4>💡 一起来走一遍 LeNet 的高速公路</h4>',
        '  <p>点击下面流水线上的任意一站，看看数据在每一层究竟变成了什么鬼样子。</p>',
        '</div>',
        '<div class="pipeline-bar" data-ref="pipeline"></div>',
        '<div class="panel" style="padding: 0; overflow: hidden;">',
        '  <div class="detail-stage" data-ref="detail-stage"></div>',
        '</div>'
      ].join('\n');
    },
    init: function (root) {
      el = {
        pipeline: root.querySelector('[data-ref="pipeline"]'),
        detailStage: root.querySelector('[data-ref="detail-stage"]')
      };
      initData();
      renderPipeline();
      renderDetail();
    },
    onHide: function () {}
  });
})();