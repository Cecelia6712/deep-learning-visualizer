/* ================================================================
   m6.js — 模块6：过拟合实验台（修复白屏版）
   ================================================================ */
(function () {
  'use strict';

  const S = {
    epoch: 0,
    maxEpoch: 100,
    isPlaying: false,
    earlyStop: false,
    dropout: false,
    dataAug: false,
    timer: null
  };

  let el = {};
  let ctx = null;
  let chartW = 0, chartH = 0;
  let resizeObserver = null;

  // 核心数学模型：模拟损失曲线
  function getLoss(epoch) {
    const trainLoss = 0.05 + 0.95 * Math.exp(-epoch / 15);

    let testLoss = 0.1 + 0.9 * Math.exp(-epoch / 15) + 0.0003 * epoch * epoch;

    if (S.dropout || S.dataAug) {
      testLoss = 0.1 + 0.9 * Math.exp(-epoch / 15) + 0.00002 * epoch * epoch;
      if (S.dropout && S.dataAug) testLoss -= 0.02;
    }

    if (S.earlyStop && epoch > 28) {
      epoch = 28;
      testLoss = 0.1 + 0.9 * Math.exp(-28 / 15) + 0.0003 * 28 * 28;
      if (S.dropout || S.dataAug) testLoss = 0.1 + 0.9 * Math.exp(-28 / 15) + 0.00002 * 28 * 28;
    }

    return { train: trainLoss, test: testLoss };
  }

  // 修复 Canvas 尺寸计算，防止死锁
  function resizeCanvas() {
    const canvas = el.canvas;
    if (!canvas) return;
    
    const wrap = canvas.parentElement;
    const rect = wrap.getBoundingClientRect();
    
    // 如果父元素还没渲染出尺寸，暂不处理
    if (rect.width === 0 || rect.height === 0) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    chartW = rect.width;
    chartH = rect.height;

    ctx = canvas.getContext('2d');
    // 关键：用 setTransform 替代 scale，防止重绘时倍数累加
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); 

    drawChart();
  }

  // 坐标映射
  function getX(epoch) { return (epoch / S.maxEpoch) * (chartW - 60) + 50; }
  function getY(loss) { return chartH - 40 - (loss / 1.5) * (chartH - 80); }

  // 绘制主图表
  function drawChart() {
    if (!ctx || chartW === 0) return;
    ctx.clearRect(0, 0, chartW, chartH);

    const padL = 50, padB = 40;
    const plotW = chartW - padL - 20;
    const plotH = chartH - padB - 20;

    // 1. 画背景分区
    const xUnder = getX(15);
    ctx.fillStyle = '#fef9e7';
    ctx.fillRect(padL, 20, xUnder - padL, plotH);

    const xBest = getX(35);
    ctx.fillStyle = '#eafaf1';
    ctx.fillRect(xUnder, 20, xBest - xUnder, plotH);

    ctx.fillStyle = '#fdf1ef';
    ctx.fillRect(xBest, 20, chartW - 20 - xBest, plotH);

    // 2. 画网格和坐标轴
    ctx.strokeStyle = '#dcdcdc';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= 1.5; i += 0.5) {
      const y = getY(i);
      ctx.moveTo(padL, y);
      ctx.lineTo(chartW - 20, y);
      ctx.fillStyle = '#888';
      ctx.font = '12px Consolas';
      ctx.textAlign = 'right';
      ctx.fillText(i.toFixed(1), padL - 10, y + 4);
    }
    for (let i = 0; i <= S.maxEpoch; i += 20) {
      const x = getX(i);
      ctx.moveTo(x, 20);
      ctx.lineTo(x, chartH - padB);
      ctx.fillStyle = '#888';
      ctx.textAlign = 'center';
      ctx.fillText(i, x, chartH - 15);
    }
    ctx.stroke();

    // 坐标轴标签
    ctx.fillStyle = '#555';
    ctx.font = '13px Georgia';
    ctx.textAlign = 'center';
    ctx.fillText('训练轮数 (Epoch)', chartW / 2, chartH - 5);
    ctx.save();
    ctx.translate(15, chartH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('损失 (Loss)', 0, 0);
    ctx.restore();

    // 3. 画分区文字
    ctx.font = 'bold 12px sans-serif';
    ctx.fillStyle = '#d4ac0d';
    ctx.fillText('欠拟合区', (padL + xUnder) / 2, 40);
    ctx.fillStyle = '#27ae60';
    ctx.fillText('最佳区', (xUnder + xBest) / 2, 40);
    ctx.fillStyle = '#c0392b';
    ctx.fillText('过拟合区', (xBest + chartW - 20) / 2, 40);

    // 4. 画曲线
    const drawLine = (isTrain, color) => {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.lineJoin = 'round';

      for (let i = 0; i <= S.epoch; i++) {
        const loss = isTrain ? getLoss(i).train : getLoss(i).test;
        const x = getX(i), y = getY(loss);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      if (S.epoch > 0) {
        const curLoss = isTrain ? getLoss(S.epoch).train : getLoss(S.epoch).test;
        const cx = getX(S.epoch), cy = getY(curLoss);
        ctx.beginPath();
        ctx.arc(cx, cy, 5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    };

    drawLine(true, '#c0392b');
    drawLine(false, '#2980b9');
  }

  // 更新状态面板
  function updateStatus() {
    const loss = getLoss(S.epoch);
    el.valEpoch.textContent = S.epoch;
    el.valTrain.textContent = loss.train.toFixed(3);
    el.valTest.textContent = loss.test.toFixed(3);

    let tagText = '未开始训练';
    let tagClass = 'diag-normal';

    if (S.epoch > 0) {
      if (S.epoch <= 15) {
        tagText = '欠拟合：训练/测试都挺高，还没学明白';
        tagClass = 'diag-under';
      } else if (S.epoch <= 35) {
        tagText = '最佳状态：测试集损失最低';
        tagClass = 'diag-best';
      } else {
        if (S.earlyStop) {
          tagText = '已早停：在最佳点切断，防止过拟合';
          tagClass = 'diag-best';
        } else if (S.dropout || S.dataAug) {
          tagText = '缓解过拟合：裂口被压制住了';
          tagClass = 'diag-best';
        } else {
          tagText = '过拟合：训练极低，测试反弹飙升';
          tagClass = 'diag-over';
        }
      }
    }

    el.diagTag.textContent = tagText;
    el.diagTag.className = 'diagnosis-tag ' + tagClass;
  }

  function render() {
    drawChart();
    updateStatus();
    el.stEpoch.querySelector('.step-val').textContent = S.epoch;
  }

  function startPlay() {
    if (S.isPlaying) return;
    if (S.epoch >= S.maxEpoch || (S.earlyStop && S.epoch >= 28)) {
      S.epoch = 0;
    }
    S.isPlaying = true;
    el.btnPlay.textContent = '暂停训练';

    const step = () => {
      if (!S.isPlaying) return;
      const maxLimit = S.earlyStop ? 28 : S.maxEpoch;
      if (S.epoch < maxLimit) {
        S.epoch++;
        render();
        S.timer = setTimeout(step, 50);
      } else {
        S.isPlaying = false;
        el.btnPlay.textContent = '重新训练';
      }
    };
    S.timer = setTimeout(step, 50);
  }

  function stopPlay() {
    S.isPlaying = false;
    if (S.timer) clearTimeout(S.timer);
    el.btnPlay.textContent = '继续训练';
  }

  Lab.register({
    id: 'm6',
    index: '06',
    title: '过拟合实验台',
    desc: '观察红线（训练集）和蓝线（测试集）的动态变化。尝试用“解药”去缝合它们之间的裂口。',
    template: function () {
      return [
        '<div class="concept-panel">',
        '  <h4>💡 到底什么是“过拟合”？</h4>',
        '  <p>红线是<b>训练集损失</b>，它会一直下降，因为网络在“死记硬背”做过的题。</p>',
        '  <p>蓝线是<b>测试集损失</b>，它一开始也会下降，但背得太多以后，遇到没见过的题就不会做了，导致损失<b>反弹飙升</b>。</p>',
        '  <p>红蓝两线之间越来越大的裂口，就是过拟合！下面有三颗“解药”，点一下试试看，裂口是怎么被压下去的。</p>',
        '</div>',

        '<div class="lab-layout">',
        '  <div class="chart-panel">',
        '    <div class="chart-header">',
        '      <div class="chart-title">损失曲线 (Loss Curve)</div>',
        '      <div class="legend-box">',
        '        <div class="legend-item"><div class="legend-line train"></div>训练集</div>',
        '        <div class="legend-item"><div class="legend-line test"></div>测试集</div>',
        '      </div>',
        '    </div>',
        '    <div class="canvas-wrap"><canvas data-ref="canvas"></canvas></div>',
        '  </div>',

        '  <div class="control-panel">',
        '    <div class="status-box">',
        '      <div class="status-title">实时数据监控</div>',
        '      <div class="status-data"><span>当前轮数 (Epoch):</span> <span class="val" data-ref="val-epoch">0</span></div>',
        '      <div class="status-data"><span>训练损失 (Train):</span> <span class="val train" data-ref="val-train">-</span></div>',
        '      <div class="status-data"><span>测试损失 (Test):</span> <span class="val test" data-ref="val-test">-</span></div>',
        '      <div class="diagnosis-tag diag-normal" data-ref="diag-tag">未开始训练</div>',
        '    </div>',

        '    <div class="ctrl-row" style="margin-bottom: 10px;">',
        '      <div class="ctrl"><label>跳转轮数</label>',
        '        <div class="stepper" data-ref="st-epoch"><button class="step-btn" data-act="dec">−</button><span class="step-val">0</span><button class="step-btn" data-act="inc">+</button></div>',
        '      </div>',
        '    </div>',
        '    <div class="ctrl-row" style="margin-bottom: 20px;">',
        '      <button class="btn primary" data-ref="btn-play" style="flex:1;">开始训练</button>',
        '      <button class="btn" data-ref="btn-reset">重置</button>',
        '    </div>',

        '    <div class="antidote-box">',
        '      <div class="antidote-title">💊 解药配方</div>',
        '      <button class="antidote-btn" data-ref="btn-early">',
        '        早停 (Early Stopping)',
        '        <span class="desc">发现测试损失开始反弹时，立刻停止训练</span>',
        '      </button>',
        '      <button class="antidote-btn" data-ref="btn-dropout">',
        '        Dropout',
        '        <span class="desc">训练时随机“掐断”一部分神经元，防止死记硬背</span>',
        '      </button>',
        '      <button class="antidote-btn" data-ref="btn-aug">',
        '        数据增强 (Data Augmentation)',
        '        <span class="desc">把原图旋转、裁剪、变形，变出更多题来练</span>',
        '      </button>',
        '    </div>',
        '  </div>',
        '</div>'
      ].join('\n');
    },

    init: function (root) {
      el = {
        canvas: root.querySelector('[data-ref="canvas"]'),
        valEpoch: root.querySelector('[data-ref="val-epoch"]'),
        valTrain: root.querySelector('[data-ref="val-train"]'),
        valTest: root.querySelector('[data-ref="val-test"]'),
        diagTag: root.querySelector('[data-ref="diag-tag"]'),
        stEpoch: root.querySelector('[data-ref="st-epoch"]'),
        btnPlay: root.querySelector('[data-ref="btn-play"]'),
        btnReset: root.querySelector('[data-ref="btn-reset"]'),
        btnEarly: root.querySelector('[data-ref="btn-early"]'),
        btnDropout: root.querySelector('[data-ref="btn-dropout"]'),
        btnAug: root.querySelector('[data-ref="btn-aug"]')
      };

      // 使用 ResizeObserver 监听容器尺寸，解决初始化时不可见的问题
      resizeObserver = new ResizeObserver(() => {
        resizeCanvas();
      });
      resizeObserver.observe(el.canvas.parentElement);

      // 防抖处理大小变化
      let resizeTimer = null;
      window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(resizeCanvas, 100);
      });

      // 步进器
      el.stEpoch.querySelector('[data-act="dec"]').addEventListener('click', () => {
        stopPlay();
        if (S.epoch > 0) { S.epoch--; render(); }
      });
      el.stEpoch.querySelector('[data-act="inc"]').addEventListener('click', () => {
        stopPlay();
        const limit = S.earlyStop ? 28 : S.maxEpoch;
        if (S.epoch < limit) { S.epoch++; render(); }
      });

      // 播放控制
      el.btnPlay.addEventListener('click', () => {
        if (S.isPlaying) stopPlay(); else startPlay();
      });
      el.btnReset.addEventListener('click', () => {
        stopPlay();
        S.epoch = 0;
        render();
        el.btnPlay.textContent = '开始训练';
      });

      // 解药
      el.btnEarly.addEventListener('click', () => {
        S.earlyStop = !S.earlyStop;
        el.btnEarly.classList.toggle('active', S.earlyStop);
        if (S.earlyStop && S.epoch > 28) S.epoch = 28;
        render();
      });
      el.btnDropout.addEventListener('click', () => {
        S.dropout = !S.dropout;
        el.btnDropout.classList.toggle('active', S.dropout);
        render();
      });
      el.btnAug.addEventListener('click', () => {
        S.dataAug = !S.dataAug;
        el.btnAug.classList.toggle('active', S.dataAug);
        render();
      });

      // 初次渲染，给浏览器留一帧时间计算布局
      requestAnimationFrame(() => {
        resizeCanvas();
        render();
      });
    },

    onHide: function () {
      stopPlay();
      if (resizeObserver && el.canvas) {
        resizeObserver.unobserve(el.canvas.parentElement);
        resizeObserver.disconnect();
      }
    }
  });

})();