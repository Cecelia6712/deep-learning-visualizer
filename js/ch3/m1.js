(function () {
  'use strict';

  /* =========================================================
   *  状态与常量
   * ========================================================= */
  const S = {
    fn: 'sigmoid', // 'sigmoid' | 'tanh' | 'relu'
    x: 1.5,
    showTaylor: false // 场景2：是否显示一阶泰勒展开
  };

  let el = {};
  let rafId = null;

  const VB_W = 900;
  const VB_H = 400;
  const PAD_L = 60, PAD_R = 40, PAD_T = 40, PAD_B = 60;
  const PLOT_W = VB_W - PAD_L - PAD_R;
  const PLOT_H = VB_H - PAD_T - PAD_B;

  const X_MIN = -10;
  const X_MAX = 10;

  const FUNCS = {
    sigmoid: {
      name: 'Sigmoid',
      f: x => 1 / (1 + Math.exp(-x)),
      df: x => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); },
      yMin: -0.1, yMax: 1.1,
      color: '#2980b9'
    },
    tanh: {
      name: 'Tanh',
      f: x => Math.tanh(x),
      df: x => { const t = Math.tanh(x); return 1 - t * t; },
      yMin: -1.1, yMax: 1.1,
      color: '#2980b9'
    },
    relu: {
      name: 'ReLU',
      f: x => Math.max(0, x),
      df: x => (x > 0 ? 1 : (x < 0 ? 0 : NaN)),
      yMin: -1, yMax: 11,
      color: '#2980b9'
    }
  };

  // 坐标映射
  const mapX = x => PAD_L + ((x - X_MIN) / (X_MAX - X_MIN)) * PLOT_W;
  const mapY = (y, f) => PAD_T + ((f.yMax - y) / (f.yMax - f.yMin)) * PLOT_H;

  /* =========================================================
   *  渲染核心
   * ========================================================= */
  function render() {
    const f = FUNCS[S.fn];
    const x = S.x;
    const y = f.f(x);
    const k = f.df(x);

    // 泰勒展开计算（如果是 ReLU 则无意义）
    const isTaylorValid = S.fn !== 'relu';
    // 泰勒直线 g(x) = f(0) + f'(0) * x
    const taylorY = isTaylorValid ? f.f(0) + f.df(0) * x : 0;
    const errorVal = isTaylorValid ? Math.abs(y - taylorY) : 0;

    // ================= 1. 更新公式面板 =================
    let formulaHtml = '';
    let explainHtml = '';

    // ---- Sigmoid 公式与解剖 ----
    if (S.fn === 'sigmoid') {
      formulaHtml = `
        <div class="m1-math-line">
          <span class="m1-fn-name">σ(x)</span> = 
          <div class="m1-frac">
            <span class="m1-num">1</span>
            <span class="m1-den">1 + e<sup class="m1-sup">−x</sup></span>
          </div>
        </div>
      `;
      explainHtml = `
        <div class="m1-explain-item">
          <div class="m1-ex-title">① 为什么是平滑的 S 型？</div>
          <div class="m1-ex-text">x=0 时 e<sup>0</sup>=1，分母 1+1=2，输出恰为 0.5。这是曲线的物理中心。</div>
        </div>
        <div class="m1-explain-item">
          <div class="m1-ex-title">② 为什么左边贴地趋近 0？</div>
          <div class="m1-ex-text">x 是负数时，−x 是正数，e<sup>10</sup> 爆炸成 22026。分母极大，1 除以极大数，结果被狠狠砸向 0。</div>
        </div>
        <div class="m1-explain-item">
          <div class="m1-ex-title">③ 为什么右边封顶趋近 1？</div>
          <div class="m1-ex-text">x 是正数时，−x 是负数，e<sup>-10</sup> 衰减到极小的 0.00004。分母趋近 1，输出触顶在 1。</div>
        </div>
      `;
    } 
    // ---- Tanh 公式与解剖 ----
    else if (S.fn === 'tanh') {
      formulaHtml = `
        <div class="m1-math-line">
          <span class="m1-fn-name">tanh(x)</span> = 
          <div class="m1-frac">
            <span class="m1-num">e<sup class="m1-sup">x</sup> − e<sup class="m1-sup">−x</sup></span>
            <span class="m1-den">e<sup class="m1-sup">x</sup> + e<sup class="m1-sup">−x</sup></span>
          </div>
        </div>
      `;
      explainHtml = `
        <div class="m1-explain-item">
          <div class="m1-ex-title">① 与 Sigmoid 的关系</div>
          <div class="m1-ex-text">等价于 2σ(2x) - 1。就是把 S 曲线撑开到了 <b>(-1, 1)</b> 区间，且以 0 为中心。</div>
        </div>
        <div class="m1-explain-item">
          <div class="m1-ex-title">② 为什么能缓解震荡？</div>
          <div class="m1-ex-text">输出有正有负。下一层权重更新时，不会因为全是正数而导致方向同号（Zig-Zag 震荡）。</div>
        </div>
      `;
    } 
    // ---- ReLU 公式与解剖 ----
    else if (S.fn === 'relu') {
      formulaHtml = `
        <div class="m1-math-line">
          <span class="m1-fn-name">ReLU(x)</span> = 
          <span style="font-size: 20px; font-family: Consolas, monospace;">max(0, x)</span>
        </div>
      `;
      explainHtml = `
        <div class="m1-explain-item">
          <div class="m1-ex-title">① 为什么是两段直线？</div>
          <div class="m1-ex-text">在 0 处一刀切：左边负数直接变成 0（死寂）；右边正数原样放行，变成斜率为 1 的直线。</div>
        </div>
        <div class="m1-explain-item">
          <div class="m1-ex-title">② 为什么能解决梯度消失？</div>
          <div class="m1-ex-text">只要 x > 0，它的斜率永远是 1。梯度无损传回底层，彻底终结 Sigmoid 越传越小的问题。</div>
        </div>
      `;
    }

    // 拼接左侧面板 HTML
    const isWeak = isNaN(k) || Math.abs(k) < 0.05;
    const gradColor = isWeak ? '#c0392b' : '#2980b9';

    let taylorHtml = '';
    if (S.showTaylor && isTaylorValid) {
      const isLocal = Math.abs(x) < 1.5;
      const statusColor = isLocal ? '#27ae60' : '#c0392b';
      const statusText = isLocal ? '局部完美重合' : '误差迅速崩坏';
      
      taylorHtml = `
        <div class="m1-f-divider"></div>
        <div class="m1-taylor-box">
          <div class="m1-taylor-formula">
            g(x) = f(0) + f'(0)x = ${f.f(0).toFixed(2)} + ${f.df(0).toFixed(2)}x
          </div>
          <div class="m1-taylor-vals">
            <div>真实值 f(${x.toFixed(1)}) = <b>${y.toFixed(3)}</b></div>
            <div>近似值 g(${x.toFixed(1)}) = <b>${taylorY.toFixed(3)}</b></div>
            <div style="color: ${statusColor}; margin-top: 4px;">
              误差 = <b>${errorVal.toFixed(3)}</b> (${statusText})
            </div>
          </div>
        </div>
      `;
    }

    el.formulaBox.innerHTML = `
      <div class="m1-math-block">${formulaHtml}</div>
      <div class="m1-f-divider"></div>
      <div class="m1-explain-block">${explainHtml}</div>
      <div class="m1-f-divider"></div>
      <div class="m1-f-val">
        <div>输入 <span class="m1-var-x">x</span> = <b>${x.toFixed(1)}</b></div>
        <div>输出 <span class="m1-var-x">f(x)</span> = <b>${isNaN(y) ? 'NaN' : y.toFixed(3)}</b></div>
        <div>导数 <span class="m1-var-x">f'(x)</span> = <b style="color:${gradColor}">${isNaN(k) ? '不可导' : k.toFixed(4)}</b></div>
      </div>
      ${taylorHtml}
    `;

    el.xVal.textContent = x.toFixed(1);
    el.slider.value = x;

    // ================= 2. 绘制曲线 =================
    if (S.fn === 'relu') {
      el.curveSingle.style.display = 'none';
      el.curveLeft.style.display = 'block';
      el.curveRight.style.display = 'block';
      
      const y0 = mapY(0, f);
      el.curveLeft.setAttribute('d', `M ${mapX(-10)} ${y0} L ${mapX(0)} ${y0}`);
      el.curveRight.setAttribute('d', `M ${mapX(0)} ${y0} L ${mapX(10)} ${mapY(10, f)}`);
      el.reluDivider.style.display = 'block';
    } else {
      el.curveLeft.style.display = 'none';
      el.curveRight.style.display = 'none';
      el.reluDivider.style.display = 'none';
      
      let d = `M ${mapX(-10)} ${mapY(f.f(-10), f)}`;
      for (let i = -9.9; i <= 10; i += 0.1) {
        d += ` L ${mapX(i)} ${mapY(f.f(i), f)}`;
      }
      el.curveSingle.setAttribute('d', d);
      el.curveSingle.style.display = 'block';
    }

    // ================= 3. 绘制泰勒展开线（绿线） =================
    if (S.showTaylor && isTaylorValid) {
      el.taylorLine.style.display = 'block';
      // 计算 x=-10 和 x=10 处的绿线 y 坐标（这里故意不限制在 yMin~yMax 之间，让它直接飞出屏幕）
      const taylorYLeft = f.f(0) + f.df(0) * (-10);
      const taylorYRight = f.f(0) + f.df(0) * (10);
      
      el.taylorLine.setAttribute('x1', mapX(-10));
      el.taylorLine.setAttribute('y1', mapY(taylorYLeft, f));
      el.taylorLine.setAttribute('x2', mapX(10));
      el.taylorLine.setAttribute('y2', mapY(taylorYRight, f));
    } else {
      el.taylorLine.style.display = 'none';
    }

    // ================= 4. 几何切线 =================
    const cx = mapX(x);
    const cy = mapY(y, f);
    
    const scaleX = PLOT_W / (X_MAX - X_MIN);
    const scaleY = PLOT_H / (f.yMax - f.yMin);
    const kPx = -k * (scaleY / scaleX);

    const lineLen = 240;
    let dx, dy;
    if (isNaN(kPx)) {
      dx = 0; dy = lineLen / 2;
      el.tangentLine.setAttribute('stroke-dasharray', '5 5');
    } else {
      dx = (lineLen / 2) / Math.sqrt(1 + kPx * kPx);
      dy = kPx * dx;
      el.tangentLine.removeAttribute('stroke-dasharray');
    }
    
    el.tangentLine.setAttribute('x1', cx - dx);
    el.tangentLine.setAttribute('y1', cy - dy);
    el.tangentLine.setAttribute('x2', cx + dx);
    el.tangentLine.setAttribute('y2', cy + dy);

    // ================= 5. 误差测量线（红点至绿线的垂线） =================
    if (S.showTaylor && isTaylorValid) {
      el.errorLine.style.display = 'block';
      const tcy = mapY(taylorY, f);
      el.errorLine.setAttribute('x1', cx);
      el.errorLine.setAttribute('y1', cy);
      el.errorLine.setAttribute('x2', cx);
      el.errorLine.setAttribute('y2', tcy);
    } else {
      el.errorLine.style.display = 'none';
    }

    // ================= 6. 更新动点与辅助线 =================
    el.point.setAttribute('cx', cx);
    el.point.setAttribute('cy', cy);
    
    el.guideX.setAttribute('x1', cx);
    el.guideX.setAttribute('y1', cy);
    el.guideX.setAttribute('x2', cx);
    el.guideX.setAttribute('y2', PAD_T + PLOT_H);
    
    el.guideY.setAttribute('x1', cx);
    el.guideY.setAttribute('y1', cy);
    el.guideY.setAttribute('x2', PAD_L);
    el.guideY.setAttribute('y2', cy);

    // ================= 7. 梯度残血条 =================
    const absK = Math.abs(k);
    let ratio = isNaN(absK) ? 0 : Math.min(1, absK);
    el.gradFill.style.width = (ratio * 100) + '%';
    el.gradFill.classList.toggle('is-weak', isWeak);
    el.gradNum.classList.toggle('is-weak', isWeak);
    
    if (isNaN(absK)) {
      el.gradNum.textContent = '不可导';
      el.gradWarn.textContent = '折点处无定义';
      el.gradWarn.classList.add('is-on');
    } else if (isWeak) {
      el.gradNum.textContent = absK.toFixed(4);
      el.gradWarn.textContent = '梯度消失区：切线已水平';
      el.gradWarn.classList.add('is-on');
    } else {
      el.gradNum.textContent = absK.toFixed(4);
      el.gradWarn.classList.remove('is-on');
    }
  }

  function drawGrid() {
    const f = FUNCS[S.fn];
    let html = '';
    for (let i = -10; i <= 10; i += 5) {
      const px = mapX(i);
      html += `<line class="m1-grid-line" x1="${px}" y1="${PAD_T}" x2="${px}" y2="${PAD_T + PLOT_H}" ${i===0?'stroke="#dcdcdc"':''} />`;
      html += `<text class="m1-grid-label" x="${px}" y="${VB_H - 20}" text-anchor="middle">${i}</text>`;
    }
    const yStep = (f.yMax - f.yMin) / 4;
    for (let i = 0; i <= 4; i++) {
      const yVal = f.yMin + i * yStep;
      const py = mapY(yVal, f);
      html += `<line class="m1-grid-line" x1="${PAD_L}" y1="${py}" x2="${PAD_L + PLOT_W}" y2="${py}" ${Math.abs(yVal)<0.01?'stroke="#dcdcdc"':''} />`;
      let label = yVal.toFixed(1);
      if (S.fn === 'relu') label = yVal.toFixed(0);
      html += `<text class="m1-grid-label" x="${PAD_L - 12}" y="${py + 4}" text-anchor="end">${label}</text>`;
    }
    el.grid.innerHTML = html;
  }

  /* =========================================================
   *  Lab 模块注册
   * ========================================================= */
  Lab.register({
    id: 'ch3-m1',
    index: '01',
    title: '激活函数的几何映射',
    desc: '解剖激活函数的导数与几何切线，直观感受梯度消失与分段函数的物理本质。',

    template: function () {
      return `
        <div class="m1-wrap">
          <div class="m1-toolbar">
            <div class="m1-btn-group">
              <button class="m1-btn is-active" data-fn="sigmoid">Sigmoid</button>
              <button class="m1-btn" data-fn="tanh">Tanh</button>
              <button class="m1-btn" data-fn="relu">ReLU</button>
            </div>
            <button class="m1-toggle-btn" data-role="btn-taylor">展开 1 阶泰勒近似</button>
          </div>

          <div class="m1-main">
            <div class="m1-sidebar">
              <div class="m1-formula-box" data-role="formula"></div>
              
              <div class="m1-control-box">
                <div class="m1-slider-label">
                  <span>调整输入 x</span>
                  <div class="m1-stepper">
                    <button data-step="-1">−</button>
                    <span data-role="xval">1.5</span>
                    <button data-step="1">+</button>
                  </div>
                </div>
                <input type="range" class="m1-range" min="-10" max="10" step="0.1" value="1.5" data-role="slider">
              </div>

              <div class="m1-grad-box">
                <div class="m1-grad-head">
                  <span>梯度残血条 (|f'(x)|)</span>
                  <span class="m1-grad-num" data-role="gradnum">0.0000</span>
                </div>
                <div class="m1-grad-track">
                  <div class="m1-grad-fill" data-role="gradfill"></div>
                </div>
                <div class="m1-grad-warn" data-role="gradwarn">梯度消失区：切线已水平</div>
              </div>
            </div>

            <div class="m1-canvas-box">
              <svg class="m1-svg" viewBox="0 0 900 400" preserveAspectRatio="xMidYMid meet">
                <!-- 这一步定义裁剪区域，防止泰勒线飞出画布时覆盖到外面去，但没有它反而更有冲击力 -->
                <g data-role="grid"></g>
                
                <!-- 泰勒展开绿线 -->
                <line data-role="taylor-line" stroke="#27ae60" stroke-width="2" stroke-dasharray="6 4" style="display:none;" />
                <!-- 误差垂线 -->
                <line data-role="error-line" stroke="#c0392b" stroke-width="1.5" stroke-dasharray="3 3" style="display:none;" />
                
                <path data-role="curve-single" fill="none" stroke="#2980b9" stroke-width="3" />
                <path data-role="curve-left" fill="none" stroke="#95a5a6" stroke-width="3" />
                <path data-role="curve-right" fill="none" stroke="#2980b9" stroke-width="3" />
                <line data-role="relu-divider" stroke="#e74c3c" stroke-dasharray="4 4" stroke-width="1.5" />
                
                <line data-role="guide-x" stroke="#dcdcdc" stroke-dasharray="3 3" />
                <line data-role="guide-y" stroke="#dcdcdc" stroke-dasharray="3 3" />
                <line data-role="tangent" stroke="#c0392b" stroke-width="2.5" />
                
                <circle data-role="point" r="6" fill="#c0392b" stroke="#fff" stroke-width="2" />
              </svg>
            </div>
          </div>
        </div>
      `;
    },

    init: function (root) {
      el = {
        btnGroup: Array.from(root.querySelectorAll('.m1-btn')),
        btnTaylor: root.querySelector('[data-role="btn-taylor"]'),
        formulaBox: root.querySelector('[data-role="formula"]'),
        slider: root.querySelector('[data-role="slider"]'),
        stepper: root.querySelector('.m1-stepper'),
        xVal: root.querySelector('[data-role="xval"]'),
        
        grid: root.querySelector('[data-role="grid"]'),
        curveSingle: root.querySelector('[data-role="curve-single"]'),
        curveLeft: root.querySelector('[data-role="curve-left"]'),
        curveRight: root.querySelector('[data-role="curve-right"]'),
        reluDivider: root.querySelector('[data-role="relu-divider"]'),
        taylorLine: root.querySelector('[data-role="taylor-line"]'),
        errorLine: root.querySelector('[data-role="error-line"]'),
        
        guideX: root.querySelector('[data-role="guide-x"]'),
        guideY: root.querySelector('[data-role="guide-y"]'),
        tangentLine: root.querySelector('[data-role="tangent"]'),
        point: root.querySelector('[data-role="point"]'),
        
        gradFill: root.querySelector('[data-role="gradfill"]'),
        gradNum: root.querySelector('[data-role="gradnum"]'),
        gradWarn: root.querySelector('[data-role="gradwarn"]')
      };

      drawGrid();
      render();

      el.btnGroup.forEach(btn => {
        btn.addEventListener('click', () => {
          if (btn.classList.contains('is-active')) return;
          el.btnGroup.forEach(b => b.classList.remove('is-active'));
          btn.classList.add('is-active');
          S.fn = btn.dataset.fn;
          
          // 如果切到 ReLU，强制关闭泰勒展开并禁用按钮
          if (S.fn === 'relu') {
            S.showTaylor = false;
            el.btnTaylor.disabled = true;
            el.btnTaylor.style.opacity = '0.5';
            el.btnTaylor.classList.remove('is-active');
          } else {
            el.btnTaylor.disabled = false;
            el.btnTaylor.style.opacity = '1';
            if (S.showTaylor) el.btnTaylor.classList.add('is-active');
          }
          
          drawGrid();
          render();
        });
      });

      // 切换泰勒展开
      el.btnTaylor.addEventListener('click', () => {
        if (el.btnTaylor.disabled) return;
        S.showTaylor = !S.showTaylor;
        el.btnTaylor.classList.toggle('is-active', S.showTaylor);
        render();
      });

      el.slider.addEventListener('input', (e) => {
        S.x = parseFloat(e.target.value);
        render();
      });

      el.stepper.addEventListener('click', e => {
        const btn = e.target.closest('[data-step]');
        if (!btn) return;
        let val = S.x + Number(btn.dataset.step) * 0.5;
        S.x = Math.min(10, Math.max(-10, val));
        render();
      });
    },

    onHide: function () {
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
    }
  });
})();