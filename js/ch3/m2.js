// ============================================================
//  js/ch3-m2.js  —— 四场景连贯故事版（绝对安全与极客推导版）
// ============================================================
(function () {
  'use strict';

  const ACT_FNS = {
    sigmoid: {
      name: 'Sigmoid', sym: 'σ', f: v => 1 / (1 + Math.exp(-v)), df: v => { const s = 1 / (1 + Math.exp(-v)); return s * (1 - s); },
      range: '[0, 1]', threshold: 0.5, grad: '最大 0.25（z=0 处）',
      view: { yMin: 0, yMax: 1, xHalf: 8 }, satBand: true, satThreshold: 4,
      pros: '输出 0~1，天然像概率', cons: '两端梯度≈0，深层易梯度消失',
      useCase: '二分类的<b>输出层</b>'
    },
    tanh: {
      name: 'Tanh', sym: 'tanh', f: v => Math.tanh(v), df: v => 1 - Math.tanh(v)**2,
      range: '[-1, 1]', threshold: 0, grad: '最大 1.0（z=0 处）',
      view: { yMin: -1, yMax: 1, xHalf: 8 }, satBand: true, satThreshold: 3,
      pros: '零中心，收敛比 sigmoid 快', cons: '仍有饱和区，深层仍会消失',
      useCase: '小规模网络的<b>隐藏层</b>'
    },
    relu: {
      name: 'ReLU', sym: 'ReLU', f: v => Math.max(0, v), df: v => v > 0 ? 1 : 0,
      range: '[0, +∞)', threshold: 0, grad: '0 或 1（非平滑）',
      view: { yMin: 0, yMax: 6, xHalf: 8 }, satBand: false, satThreshold: 0,
      pros: '正区间梯度恒为 1，训练极快', cons: '负数区直接死掉（神经元死亡）',
      useCase: '现代深度网络的<b>隐藏层首选</b>'
    },
    linear: {
      name: 'Linear', sym: 'id', f: v => v, df: v => 1,
      range: '(-∞, +∞)', threshold: 0, grad: '恒为 1',
      view: { yMin: -8, yMax: 8, xHalf: 8 }, satBand: false, satThreshold: 0,
      pros: '无扭曲，输出可任意大小', cons: '无非线性，多层等价于一层',
      useCase: '回归问题的<b>输出层</b>'
    }
  };

  const S = {
    tab: 'part1',
    p1_x: [3.0, 1.0], p1_w: [1.5, 1.0], p1_b: -5.0, p1_act: 'sigmoid',
    p2_W: [1.0, 0.5, 0.5, 1.0], p2_b: [0.0, 0.0], p2_x: [2.0, 1.0],
    p3_x: [1.0, -0.5, 1.0], p3_act: 'sigmoid',
    p2_map_row: 0,
    p4_eta: 0.1
  };

  let el = {};
  // 梯度下降状态
  let geoAnimId = null;
  let geoW = 0.9; // 初始 P3_W2[0][0]
  let geoIter = 0;
  let geoCurve = null;

  const P1_CX = 300, P1_CY = 150, P1_SCALE = 30;
  const P1_AXIS_CX = 300, P1_AXIS_Y = 320, P1_AXIS_SCALE = 15;
  const P1_ACT_CX = 680, P1_ACT_CY = 360, P1_ACT_H = 100;
  const P1_GRID_RANGE = 4;

  const P2_CX = 400, P2_CY = 250, P2_SCALE = 20;
  const P2_GRID_RANGE = 8;

  const P3_W1 = [
    [ 0.8,  0.5, -0.3],
    [-0.6,  1.2,  0.4],
    [ 0.5, -0.8,  0.9],
    [ 0.3,  0.7,  0.2]
  ];
  const P3_b1 = [ 0.10, -0.20,  0.00,  0.15];
  const P3_W2 = [
    [ 0.9, -0.6,  0.7,  0.4],
    [-0.5,  1.1, -0.3,  0.8]
  ];
  const P3_b2 = [ 0.05, -0.10];

  const P3 = { tickIdx: -1, playing: false, speed: 1, interval: 400, timer: null };
  const P4 = { 
    tickIdx: -1, playing: false, speed: 1, interval: 500, timer: null,
    updatedW1: null, updatedW2: null
  };
  
  const P4_TARGET = [1.0, 0.0];

  const P4_X_X = 150, P4_H_X = 400, P4_Y_X = 660;
  const P4_X_YS = [140, 240, 340];
  const P4_H_YS = [100, 200, 300, 400];
  const P4_Y_YS = [180, 320];
  const P4_R = 24;

  const ACT_SYM = { sigmoid: 'σ', tanh: 'tanh', relu: 'ReLU', linear: 'id' };

  // ---------------------------------------------------------
  //  计算真实的 Loss 曲线 (Loss vs W2[0][0])
  // ---------------------------------------------------------
  function computeForwardWithW(W1, W2, b1, b2, x, actKey) {
    const fn = ACT_FNS[actKey].f;
    const hRaw = W1.map((row, i) => row[0]*x[0] + row[1]*x[1] + row[2]*x[2] + b1[i]);
    const h = hRaw.map(fn);
    const yRaw = W2.map((row, i) => row[0]*h[0] + row[1]*h[1] + row[2]*h[2] + row[3]*h[3] + b2[i]);
    const y = yRaw.map(fn);
    return { hRaw, h, yRaw, y };
  }

  function computeLoss(y, target) {
    return 0.5 * (Math.pow(y[0] - target[0], 2) + Math.pow(y[1] - target[1], 2));
  }

  function getLossAtW(w) {
    const W1 = P3_W1.map(r => [...r]);
    const W2 = P3_W2.map(r => [...r]);
    W2[0][0] = w;
    const fwd = computeForwardWithW(W1, W2, P3_b1, P3_b2, S.p3_x, S.p3_act);
    return computeLoss(fwd.y, P4_TARGET);
  }

  function buildGeoCurve() {
    const minW = 0.0, maxW = 2.0;
    const steps = 60;
    let points = [];
    let maxL = 0;
    for (let i = 0; i <= steps; i++) {
      const w = minW + (maxW - minW) * (i / steps);
      const L = getLossAtW(w);
      points.push({ w, L });
      if (L > maxL) maxL = L;
    }
    maxL *= 1.1;
    const mapX = w => 20 + ((w - minW) / (maxW - minW)) * 200;
    const mapY = l => 90 - (l / maxL) * 70;
    let pathD = "";
    points.forEach((p, i) => {
      const x = mapX(p.w);
      const y = mapY(p.L);
      if (i === 0) pathD += `M ${x} ${y}`;
      else pathD += ` L ${x} ${y}`;
    });
    return { points, maxL, mapX, mapY, pathD, minW, maxW };
  }

  // ---------------------------------------------------------
  //  基础工具
  // ---------------------------------------------------------
  function buildP3Ticks() {
    const t = [{ type: 'x_ready' }];
    for (let r = 0; r < 4; r++) {
      t.push({ type: 'h_start', row: r });
      for (let i = 0; i < 3; i++) t.push({ type: 'h_term', row: r, term: i });
      t.push({ type: 'h_bias', row: r });
      t.push({ type: 'h_activate', row: r });
    }
    t.push({ type: 'h_done' });
    t.push({ type: 'h_act_each' });
    for (let r = 0; r < 2; r++) {
      t.push({ type: 'y_start', row: r });
      for (let i = 0; i < 4; i++) t.push({ type: 'y_term', row: r, term: i });
      t.push({ type: 'y_bias', row: r });
      t.push({ type: 'y_activate', row: r });
    }
    t.push({ type: 'loss' });
    t.push({ type: 'done' });
    return t;
  }
  const P3_TICKS = buildP3Ticks();

  const P4_TICKS = [
    { type: 'init' },            // 展示初始状态
    { type: 'loss_calc' },       // 算出现有 Loss
    { type: 'grad_y' },          // 算对 y 的偏导 δy = y - y*
    { type: 'y1_flow', term: 0 },{ type: 'y1_flow', term: 1 },{ type: 'y1_flow', term: 2 },{ type: 'y1_flow', term: 3 },
    { type: 'y2_flow', term: 0 },{ type: 'y2_flow', term: 1 },{ type: 'y2_flow', term: 2 },{ type: 'y2_flow', term: 3 },
    { type: 'grad_z2' },         // ★ 乘激活导数：δz2 = δy * σ'(z2)
    { type: 'grad_W2' },         // ★ 算 W2 梯度：∂L/∂W2 = δz2 · h
    { type: 'h_accum' },         // δh = δz2 · W2
    { type: 'h1_flow', term: 0 },{ type: 'h1_flow', term: 1 },{ type: 'h1_flow', term: 2 },
    { type: 'h2_flow', term: 0 },{ type: 'h2_flow', term: 1 },{ type: 'h2_flow', term: 2 },
    { type: 'h3_flow', term: 0 },{ type: 'h3_flow', term: 1 },{ type: 'h3_flow', term: 2 },
    { type: 'h4_flow', term: 0 },{ type: 'h4_flow', term: 1 },{ type: 'h4_flow', term: 2 },
    { type: 'grad_z1' },         // ★ 乘激活导数：δz1 = δh * σ'(z1)
    { type: 'grad_W1' },         // ★ 算 W1 梯度：∂L/∂W1 = δz1 · x
    { type: 'update_formula' },
    { type: 'update_demo' },
    { type: 'update_done' },
    { type: 'loop_forward' },
    { type: 'loop_loss' },
    { type: 'loop_summary' },
    { type: 'done' }
  ];

  const toSvg1_2D = (x, y) => ({ px: P1_CX + x * P1_SCALE, py: P1_CY - y * P1_SCALE });
  const toSvg1_1D = v => ({ px: P1_AXIS_CX + v * P1_AXIS_SCALE, py: P1_AXIS_Y });
  const toSvg2 = (x, y) => ({ px: P2_CX + x * P2_SCALE, py: P2_CY - y * P2_SCALE });
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
  const len = a => Math.sqrt(a[0] * a[0] + a[1] * a[1]);
  const fmt = (v, n) => (Object.is(v, -0) ? 0 : v).toFixed(n);
  const num2 = v => fmt(v, 2);
  const num3 = v => fmt(v, 3);

  const updateArrowhead = (p, s, e) => {
    const dx = e.px - s.px, dy = e.py - s.py, angle = Math.atan2(dy, dx), size = 8;
    const p1x = e.px - size * Math.cos(angle - Math.PI / 6);
    const p1y = e.py - size * Math.sin(angle - Math.PI / 6);
    const p2x = e.px - size * Math.cos(angle + Math.PI / 6);
    const p2y = e.py - size * Math.sin(angle + Math.PI / 6);
    p.setAttribute('points', `${e.px},${e.py} ${p1x},${p1y} ${p2x},${p2y}`);
  };

  const mapActY = (raw, view) => {
    let v = raw;
    if (v > view.yMax) v = view.yMax;
    if (v < view.yMin) v = view.yMin;
    return P1_ACT_CY - ((v - view.yMin) / (view.yMax - view.yMin)) * P1_ACT_H;
  };

  function updateStoryline(tab) {
    const idx = { part1: 1, part2: 2, part3: 3, part4: 4 }[tab];
    el.storyNodes.forEach(n => {
      const i = Number(n.dataset.story);
      n.classList.toggle('is-active', i === idx);
      n.classList.toggle('is-past', i < idx);
    });
  }

  // ---------------------------------------------------------
  //  梯度下降界面渲染
  // ---------------------------------------------------------
  function drawGeoFrame() {
    if (!el.geoBall || !geoCurve) return;
    const { mapX, mapY } = geoCurve;
    const currentL = getLossAtW(geoW);
    const svgX = mapX(geoW);
    const svgY = mapY(currentL);
    
    el.geoBall.setAttribute('cx', svgX);
    el.geoBall.setAttribute('cy', svgY);
    
    if (el.geoEtaLabel) {
      el.geoEtaLabel.innerHTML = `W₂[0][0]=${geoW.toFixed(3)} | Loss=${currentL.toFixed(4)} | η=${S.p4_eta.toFixed(2)}`;
    }
  }

  function resetGeo() {
    cancelAnimationFrame(geoAnimId);
    geoW = 0.9; // 初始 P3_W2[0][0] 的值
    geoIter = 0;
    drawGeoFrame();
  }

  function startGeoAnim() {
    cancelAnimationFrame(geoAnimId);
    let lastTime = 0;
    function step(time) {
      if (time - lastTime >= 700) {
        lastTime = time;
        geoIter++;
        const currentL = getLossAtW(geoW);
        const eps = 0.001;
        const grad = (getLossAtW(geoW + eps) - currentL) / eps;
        geoW = geoW - S.p4_eta * grad;
        drawGeoFrame();
        if (geoIter > 20 || Math.abs(grad) < 0.01) return;
      }
      geoAnimId = requestAnimationFrame(step);
    }
    geoAnimId = requestAnimationFrame(step);
  }

  /* =========================================================
   *  场景 1 & 2 & 3
   * ========================================================= */
  function renderScene1Grid() {
    if (!el.p1Grid) return;
    const R = P1_GRID_RANGE;
    let html = '';
    for (let i = -R; i <= R; i++) {
      const pV = toSvg1_2D(i, -R), pV2 = toSvg1_2D(i, R);
      const pH = toSvg1_2D(-R, i), pH2 = toSvg1_2D(R, i);
      const c = i === 0 ? '#dcdcdc' : '#f2f2f2';
      html += `<line x1="${pV.px}" y1="${pV.py}" x2="${pV2.px}" y2="${pV2.py}" stroke="${c}" stroke-width="1"/>`;
      html += `<line x1="${pH.px}" y1="${pH.py}" x2="${pH2.px}" y2="${pH2.py}" stroke="${c}" stroke-width="1"/>`;
    }
    el.p1Grid.innerHTML = html;
    el.p1AxisLine.setAttribute('x1', P1_AXIS_CX - 14 * P1_AXIS_SCALE);
    el.p1AxisLine.setAttribute('x2', P1_AXIS_CX + 14 * P1_AXIS_SCALE);
    let tk = '';
    for (let i = -14; i <= 14; i += 2) {
      const px = P1_AXIS_CX + i * P1_AXIS_SCALE;
      tk += `<line x1="${px}" y1="${P1_AXIS_Y - 4}" x2="${px}" y2="${P1_AXIS_Y + 4}" stroke="#bbb"/>`;
      if (i % 4 === 0) tk += `<text x="${px}" y="${P1_AXIS_Y + 18}" fill="#aaa" font-size="10" text-anchor="middle" font-family="Consolas, monospace">${i}</text>`;
    }
    el.p1AxisTicks.innerHTML = tk;
  }

  function renderScene1() {
    const x = S.p1_x, w = S.p1_w, b = S.p1_b;
    const act = ACT_FNS[S.p1_act];
    const z = dot(w, x), wLen = len(w);
    const projLen = wLen === 0 ? 0 : z / wLen;
    const zb = z + b, a = act.f(zb);
    const isActive = a > act.threshold;

    el.p1MathProj.textContent = `(${x[0].toFixed(1)}×${w[0].toFixed(1)}) + (${x[1].toFixed(1)}×${w[1].toFixed(1)})`;
    el.p1MathProjLen.textContent = projLen.toFixed(2);
    el.p1MathLenW.textContent = wLen.toFixed(2);
    el.p1MathZ.textContent = `= ${projLen.toFixed(2)} × ${wLen.toFixed(2)} = ${z.toFixed(2)}`;
    el.p1MathZb.textContent = `${z.toFixed(2)} + (${b.toFixed(1)}) = ${zb.toFixed(2)}`;
    el.p1MathA.textContent = `${act.sym}(${zb.toFixed(2)}) = ${a.toFixed(3)}`;
    el.p1MathA.style.color = isActive ? '#27ae60' : '#95a5a6';
    el.p1ActTitle.style.color = isActive ? '#27ae60' : '#555';
    el.p1ActTitle.textContent = isActive ? `③ 激活：放行 (${act.name})` : `③ 激活：抑制 (${act.name})`;

    el.p1ActName.textContent = act.name;
    el.p1ActRange.textContent = act.range;
    el.p1ActGrad.textContent = act.grad;
    el.p1ActPros.textContent = act.pros;
    el.p1ActCons.textContent = act.cons;
    el.p1ActUse.textContent = act.useCase;
    el.p1ActName.style.color = isActive ? '#27ae60' : '#95a5a6';
    el.p1ActBtns.forEach(btn => btn.classList.toggle('is-active', btn.dataset.act === S.p1_act));

    const pO = toSvg1_2D(0, 0), pX = toSvg1_2D(x[0], x[1]), pW = toSvg1_2D(w[0], w[1]);
    const pct = wLen === 0 ? 0 : projLen / wLen;
    const pProj = toSvg1_2D(w[0] * pct, w[1] * pct);

    el.p1VecX.setAttribute('x1', pO.px); el.p1VecX.setAttribute('y1', pO.py);
    el.p1VecX.setAttribute('x2', pX.px); el.p1VecX.setAttribute('y2', pX.py);
    updateArrowhead(el.p1ArrowX, pO, pX);
    el.p1LabelX.setAttribute('x', pX.px - 20); el.p1LabelX.setAttribute('y', pX.py + 20);
    el.p1LabelX.textContent = `x(${x[0].toFixed(1)}, ${x[1].toFixed(1)})`;

    el.p1VecW.setAttribute('x1', pO.px); el.p1VecW.setAttribute('y1', pO.py);
    el.p1VecW.setAttribute('x2', pW.px); el.p1VecW.setAttribute('y2', pW.py);
    updateArrowhead(el.p1ArrowW, pO, pW);
    el.p1LabelW.setAttribute('x', pW.px + 10); el.p1LabelW.setAttribute('y', pW.py - 10);
    el.p1LabelW.textContent = `w(${w[0].toFixed(1)}, ${w[1].toFixed(1)})`;

    el.p1ProjLine.setAttribute('x1', pX.px); el.p1ProjLine.setAttribute('y1', pX.py);
    el.p1ProjLine.setAttribute('x2', pProj.px); el.p1ProjLine.setAttribute('y2', pProj.py);
    el.p1ProjHighlight.setAttribute('x1', pO.px); el.p1ProjHighlight.setAttribute('y1', pO.py);
    el.p1ProjHighlight.setAttribute('x2', pProj.px); el.p1ProjHighlight.setAttribute('y2', pProj.py);
    el.p1LabelProj.setAttribute('x', pProj.px); el.p1LabelProj.setAttribute('y', pProj.py - 10);

    const pz = toSvg1_1D(z), pzb = toSvg1_1D(zb);
    el.p1DataFlowLine.setAttribute('x1', pProj.px); el.p1DataFlowLine.setAttribute('y1', pProj.py);
    el.p1DataFlowLine.setAttribute('x2', pz.px); el.p1DataFlowLine.setAttribute('y2', pz.py);
    el.p1DotZ.setAttribute('cx', pz.px); el.p1DotZ.setAttribute('cy', pz.py);
    el.p1LabelZ.setAttribute('x', pz.px); el.p1LabelZ.setAttribute('y', pz.py - 15);
    el.p1LabelZ.textContent = `z = ${z.toFixed(2)}`;

    if (b !== 0) {
      el.p1BiasGroup.style.display = '';
      el.p1BiasArrow.setAttribute('x1', pz.px); el.p1BiasArrow.setAttribute('y1', pz.py);
      el.p1BiasArrow.setAttribute('x2', pzb.px); el.p1BiasArrow.setAttribute('y2', pzb.py);
      updateArrowhead(el.p1BiasArrowHead, pz, pzb);
      el.p1DotZb.setAttribute('cx', pzb.px); el.p1DotZb.setAttribute('cy', pzb.py);
      el.p1LabelZb.setAttribute('x', pzb.px); el.p1LabelZb.setAttribute('y', pzb.py - 15);
      el.p1LabelZb.textContent = `z+b = ${zb.toFixed(2)}`;
    } else el.p1BiasGroup.style.display = 'none';

    el.p1ActInputLine.setAttribute('x1', pzb.px); el.p1ActInputLine.setAttribute('y1', pzb.py);
    el.p1ActInputLine.setAttribute('x2', P1_ACT_CX);
    el.p1ActInputLine.setAttribute('y2', mapActY(act.f(zb), act.view));

    let d = '';
    for (let i = -8; i <= 8; i += 0.2) {
      d += (i === -8 ? 'M ' : ' L ') + `${P1_ACT_CX + i * 10} ${mapActY(act.f(i), act.view)}`;
    }
    el.p1ActCurve.setAttribute('d', d);
    el.p1ActCurve.setAttribute('stroke', isActive ? '#27ae60' : '#95a5a6');

    const showSat = act.satBand;
    el.p1SatLeft.style.display = showSat ? '' : 'none';
    el.p1SatRight.style.display = showSat ? '' : 'none';
    el.p1SatLeftLabel.style.display = showSat ? '' : 'none';
    el.p1SatRightLabel.style.display = showSat ? '' : 'none';
    if (showSat) {
      const th = act.satThreshold;
      el.p1SatLeft.setAttribute('x', P1_ACT_CX - 80);
      el.p1SatLeft.setAttribute('width', (8 - th) * 10);
      el.p1SatRight.setAttribute('x', P1_ACT_CX + th * 10);
      el.p1SatRight.setAttribute('width', (8 - th) * 10);
      el.p1SatLeftLabel.setAttribute('x', P1_ACT_CX - 80 + (8 - th) * 5);
      el.p1SatRightLabel.setAttribute('x', P1_ACT_CX + th * 10 + (8 - th) * 5);
    }

    let zbC = zb;
    if (zbC > act.view.xHalf) zbC = act.view.xHalf;
    if (zbC < -act.view.xHalf) zbC = -act.view.xHalf;
    const actX = P1_ACT_CX + zbC * 10, actY = mapActY(a, act.view);
    el.p1ActInVLine.setAttribute('x1', actX); el.p1ActInVLine.setAttribute('y1', P1_ACT_CY);
    el.p1ActInVLine.setAttribute('x2', actX); el.p1ActInVLine.setAttribute('y2', actY);
    el.p1ActOutHLine.setAttribute('x1', P1_ACT_CX - 15); el.p1ActOutHLine.setAttribute('y1', actY);
    el.p1ActOutHLine.setAttribute('x2', actX); el.p1ActOutHLine.setAttribute('y2', actY);
    el.p1ActDot.setAttribute('cx', actX); el.p1ActDot.setAttribute('cy', actY);
    el.p1ActDot.setAttribute('fill', isActive ? '#27ae60' : '#95a5a6');
    el.p1LabelA.setAttribute('x', P1_ACT_CX - 20); el.p1LabelA.setAttribute('y', actY + 5);
    el.p1LabelA.setAttribute('fill', isActive ? '#27ae60' : '#95a5a6');
    el.p1LabelA.textContent = `a = ${a.toFixed(2)}`;
    el.p1ActRangeLabel.textContent = `输出范围 ${act.range}`;
  }

  function renderScene2() {
    const w11 = S.p2_W[0], w12 = S.p2_W[1], w21 = S.p2_W[2], w22 = S.p2_W[3];
    const b1 = S.p2_b[0], b2 = S.p2_b[1];
    const x1 = S.p2_x[0], x2 = S.p2_x[1];
    const z1 = w11 * x1 + w12 * x2 + b1, z2 = w21 * x1 + w22 * x2 + b2;

    const det = w11 * w22 - w12 * w21, absDet = Math.abs(det);
    let lvl, msg, col;
    if (absDet < 0.1) { lvl = '⚠ 空间塌缩'; msg = '网格几乎被压成一条线。信息严重丢失。'; col = '#c0392b'; }
    else if (absDet < 0.5) { lvl = '压缩'; msg = '空间被较强压缩，少数神经元主导。'; col = '#e67e22'; }
    else if (absDet < 2) { lvl = '✓ 健康'; msg = '伸缩适度，空间信息保留完好。'; col = '#27ae60'; }
    else { lvl = '强拉伸'; msg = '空间过度放大，训练易数值爆炸。'; col = '#8e44ad'; }
    el.p2DetVal.textContent = det.toFixed(2);
    el.p2DetVal.style.color = col;
    el.p2DetLevel.textContent = lvl;
    el.p2DetLevel.style.color = col;
    el.p2DetMsg.textContent = msg;

    el.p2MatW.innerHTML = `<div>${w11.toFixed(1)}</div><div>${w12.toFixed(1)}</div><div>${w21.toFixed(1)}</div><div>${w22.toFixed(1)}</div>`;
    el.p2MathCalc.innerHTML = `
      <div style="margin-bottom:8px;">z₁ = ${w11.toFixed(1)}×${x1.toFixed(1)} + ${w12.toFixed(1)}×${x2.toFixed(1)} + ${b1.toFixed(1)} = <b style="color:#c0392b;">${z1.toFixed(2)}</b></div>
      <div>z₂ = ${w21.toFixed(1)}×${x1.toFixed(1)} + ${w22.toFixed(1)}×${x2.toFixed(1)} + ${b2.toFixed(1)} = <b style="color:#c0392b;">${z2.toFixed(2)}</b></div>`;

    const R = P2_GRID_RANGE;
    let oH = '', tH = '';
    for (let i = -R; i <= R; i++) {
      const pa = toSvg2(-R, i), pb = toSvg2(R, i), pc = toSvg2(i, -R), pd = toSvg2(i, R);
      const sc = i === 0 ? '#e6e6e6' : '#f5f5f5';
      const sw = i === 0 ? 1.5 : 1;
      oH += `<line x1="${pa.px}" y1="${pa.py}" x2="${pb.px}" y2="${pb.py}" stroke="${sc}" stroke-width="${sw}"/>`;
      oH += `<line x1="${pc.px}" y1="${pc.py}" x2="${pd.px}" y2="${pd.py}" stroke="${sc}" stroke-width="${sw}"/>`;
      const va1 = w11 * i + w12 * (-R) + b1, vb1 = w21 * i + w22 * (-R) + b2;
      const va2 = w11 * i + w12 * (R) + b1, vb2 = w21 * i + w22 * (R) + b2;
      const ha1 = w11 * (-R) + w12 * i + b1, hb1 = w21 * (-R) + w22 * i + b2;
      const ha2 = w11 * (R) + w12 * i + b1, hb2 = w21 * (R) + w22 * i + b2;
      const p1 = toSvg2(va1, vb1), p2 = toSvg2(va2, vb2), p3 = toSvg2(ha1, hb1), p4 = toSvg2(ha2, hb2);
      const tc = i === 0 ? '#2980b9' : '#bcd5e8', tw = i === 0 ? 2 : 1;
      tH += `<line x1="${p1.px}" y1="${p1.py}" x2="${p2.px}" y2="${p2.py}" stroke="${tc}" stroke-width="${tw}"/>`;
      tH += `<line x1="${p3.px}" y1="${p3.py}" x2="${p4.px}" y2="${p4.py}" stroke="${tc}" stroke-width="${tw}"/>`;
    }
    el.p2OrigGrid.innerHTML = oH;
    el.p2TransGrid.innerHTML = tH;

    const pO = toSvg2(0, 0), pB = toSvg2(b1, b2), pX = toSvg2(x1, x2), pZ = toSvg2(z1, z2);
    el.p2VecX.setAttribute('x1', pO.px); el.p2VecX.setAttribute('y1', pO.py);
    el.p2VecX.setAttribute('x2', pX.px); el.p2VecX.setAttribute('y2', pX.py);
    el.p2LabelX.setAttribute('x', pX.px + 10); el.p2LabelX.setAttribute('y', pX.py + 5);
    el.p2LabelX.textContent = `x(${x1.toFixed(1)}, ${x2.toFixed(1)})`;
    el.p2VecZ.setAttribute('x1', pB.px); el.p2VecZ.setAttribute('y1', pB.py);
    el.p2VecZ.setAttribute('x2', pZ.px); el.p2VecZ.setAttribute('y2', pZ.py);
    el.p2LabelZ.setAttribute('x', pZ.px + 10); el.p2LabelZ.setAttribute('y', pZ.py + 5);
    el.p2LabelZ.textContent = `z(${z1.toFixed(1)}, ${z2.toFixed(1)})`;
    el.p2BiasPoint.setAttribute('cx', pB.px); el.p2BiasPoint.setAttribute('cy', pB.py);

    renderP2Map();
  }

  function buildP2MapHtml() {
    const W = S.p2_W, b = S.p2_b;
    let html = '';
    for (let i = 0; i < 2; i++) {
      const w0 = W[i * 2], w1 = W[i * 2 + 1];
      html += `
        <div class="m2-map-row" data-role="p2-map-row-${i}">
          <div class="m2-map-neuron">
            <div class="m2-map-neuron-title">神经元 ${i + 1}</div>
            <div class="m2-map-wvec">w<sup>(${i + 1})</sup> = [${num2(w0)}, ${num2(w1)}]</div>
            <div class="m2-map-bias">b<sub>${i + 1}</sub> = ${num2(b[i])}</div>
          </div>
          <div class="m2-map-arrow">⟶</div>
          <div class="m2-map-matrow">
            <span class="m2-map-bracket">[</span>
            <span class="m2-map-wcell">${num2(w0)}</span>
            <span class="m2-map-wcell">${num2(w1)}</span>
            <span class="m2-map-bracket">]</span>
            <span class="m2-map-rowtag">= W 的第 ${i + 1} 行</span>
          </div>
        </div>`;
    }
    return html;
  }

  function renderP2Map() {
    if (!el.p2MapBox) return;
    el.p2MapBox.innerHTML = buildP2MapHtml();

    const r = S.p2_map_row;
    for (let i = 0; i < 2; i++) {
      const node = el.p2MapBox.querySelector(`[data-role="p2-map-row-${i}"]`);
      if (node) node.classList.toggle('is-active', i === r);
    }
    if (el.p2MapRowVal) el.p2MapRowVal.textContent = String(r + 1);

    const W = S.p2_W, b = S.p2_b, x = S.p2_x;
    const w0 = W[r * 2], w1 = W[r * 2 + 1];
    const z = w0 * x[0] + w1 * x[1] + b[r];

    if (el.p2MapCalc) {
      el.p2MapCalc.innerHTML =
        `第 ${r + 1} 行 = 神经元 ${r + 1} 的全部权重<br>` +
        `W[${r}][:] = [${num2(w0)}, ${num2(w1)}] = w<sup>(${r + 1})</sup><br>` +
        `z<sub>${r + 1}</sub> = ${num2(w0)}×${num2(x[0])} + ${num2(w1)}×${num2(x[1])} + (${num2(b[r])}) = <b>${num2(z)}</b><br>` +
        `<span class="m2-highlight">整行 · x = 这个神经元的点积（场景一做过的事）</span>`;
    }
  }

  function buildScene3SvgHtml() {
    let s = `<defs><marker id="ch3m2-p3-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#888"/></marker></defs>`;
    s += `<text x="100" y="60" class="m2-p3-layer-title" text-anchor="middle">输入 x[3]</text>`;
    s += `<text x="280" y="60" class="m2-p3-layer-title" text-anchor="middle">W₁ [4×3]</text>`;
    s += `<text x="410" y="60" class="m2-p3-layer-title" text-anchor="middle">隐藏 h[4]</text>`;
    s += `<text x="590" y="60" class="m2-p3-layer-title" text-anchor="middle">W₂ [2×4]</text>`;
    s += `<text x="740" y="60" class="m2-p3-layer-title" text-anchor="middle">输出 y[2]</text>`;
    [[120,170,218,170],[340,170,388,170],[430,170,508,170],[670,170,718,170]].forEach(([x1,y1,x2,y2]) => {
      s += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="m2-p3-arrow-line" marker-end="url(#ch3m2-p3-arrow)"/>`;
    });
    for (let i = 0; i < 3; i++) {
      const x0 = 80, y0 = 130 + i * 40;
      s += `<rect data-cell="x-${i}" x="${x0}" y="${y0}" width="40" height="40" rx="3" class="m2-p3-cell"/>`;
      s += `<text data-text="x-${i}" x="${x0+20}" y="${y0+20}" class="m2-p3-text"></text>`;
    }
    for (let r = 0; r < 4; r++) for (let c = 0; c < 3; c++) {
      const x0 = 220 + c*40, y0 = 90 + r*40;
      s += `<rect data-cell="W1-${r}-${c}" x="${x0}" y="${y0}" width="40" height="40" rx="3" class="m2-p3-cell"/>`;
      s += `<text data-text="W1-${r}-${c}" x="${x0+20}" y="${y0+20}" class="m2-p3-text"></text>`;
    }
    for (let i = 0; i < 4; i++) {
      const x0 = 390, y0 = 90 + i * 40;
      s += `<rect data-cell="h-${i}" x="${x0}" y="${y0}" width="40" height="40" rx="3" class="m2-p3-cell"/>`;
      s += `<text data-text="h-${i}" x="${x0+20}" y="${y0+20}" class="m2-p3-text"></text>`;
    }
    for (let r = 0; r < 2; r++) for (let c = 0; c < 4; c++) {
      const x0 = 510 + c*40, y0 = 130 + r*40;
      s += `<rect data-cell="W2-${r}-${c}" x="${x0}" y="${y0}" width="40" height="40" rx="3" class="m2-p3-cell"/>`;
      s += `<text data-text="W2-${r}-${c}" x="${x0+20}" y="${y0+20}" class="m2-p3-text"></text>`;
    }
    for (let i = 0; i < 2; i++) {
      const x0 = 720, y0 = 130 + i * 40;
      s += `<rect data-cell="y-${i}" x="${x0}" y="${y0}" width="40" height="40" rx="3" class="m2-p3-cell"/>`;
      s += `<text data-text="y-${i}" x="${x0+20}" y="${y0+20}" class="m2-p3-text"></text>`;
    }
    return s;
  }

  function cacheScene3Elements() {
    const sc = el.p3SvgContent;
    const q = (a, k) => sc.querySelector(`[data-${a}="${k}"]`);
    el.p3XCell = [0,1,2].map(i => q('cell', `x-${i}`));
    el.p3XText = [0,1,2].map(i => q('text', `x-${i}`));
    el.p3W1Cell = []; el.p3W1Text = [];
    for (let r = 0; r < 4; r++) {
      el.p3W1Cell[r] = []; el.p3W1Text[r] = [];
      for (let c = 0; c < 3; c++) {
        el.p3W1Cell[r][c] = q('cell', `W1-${r}-${c}`);
        el.p3W1Text[r][c] = q('text', `W1-${r}-${c}`);
      }
    }
    el.p3HCell = [0,1,2,3].map(i => q('cell', `h-${i}`));
    el.p3HText = [0,1,2,3].map(i => q('text', `h-${i}`));
    el.p3W2Cell = []; el.p3W2Text = [];
    for (let r = 0; r < 2; r++) {
      el.p3W2Cell[r] = []; el.p3W2Text[r] = [];
      for (let c = 0; c < 4; c++) {
        el.p3W2Cell[r][c] = q('cell', `W2-${r}-${c}`);
        el.p3W2Text[r][c] = q('text', `W2-${r}-${c}`);
      }
    }
    el.p3YCell = [0,1].map(i => q('cell', `y-${i}`));
    el.p3YText = [0,1].map(i => q('text', `y-${i}`));
  }

  function deriveP3State(idx) {
    const st = {
      x: Array(3).fill('pending'), w1: Array(4).fill('pending'),
      h: Array(4).fill('pending'), w2: Array(2).fill('pending'),
      y: Array(2).fill('pending'),
      hShow: Array(4).fill(false), yShow: Array(2).fill(false),
      hActEach: false
    };
    if (idx < 0) return st;
    for (let i = 0; i <= idx; i++) {
      const t = P3_TICKS[i];
      if (t.type === 'x_ready') st.x.fill('active');
      else if (t.type === 'h_start') {
        for (let r = 0; r < t.row; r++) { st.w1[r] = 'done'; st.h[r] = 'done'; }
        st.w1[t.row] = 'active'; st.h[t.row] = 'active';
      } else if (t.type === 'h_activate') st.hShow[t.row] = true;
      else if (t.type === 'h_done' || t.type === 'h_act_each') {
        st.x.fill('done'); st.w1.fill('done'); st.h.fill('active'); st.hShow.fill(true);
      } else if (t.type === 'y_start') {
        for (let r = 0; r < t.row; r++) { st.w2[r] = 'done'; st.y[r] = 'done'; }
        st.w2[t.row] = 'active'; st.y[t.row] = 'active';
        st.h.fill('active'); st.hShow.fill(true);
      } else if (t.type === 'y_activate') st.yShow[t.row] = true;
      else if (t.type === 'done' || t.type === 'loss') { 
        st.w2.fill('done'); st.y.fill('active'); st.yShow.fill(true); 
      }
    }
    st.hActEach = P3_TICKS[idx].type === 'h_act_each';
    return st;
  }

  function computeP3Values() {
    const x = S.p3_x, fn = ACT_FNS[S.p3_act].f;
    const hRaw = P3_W1.map((row, i) => row[0]*x[0] + row[1]*x[1] + row[2]*x[2] + P3_b1[i]);
    const h = hRaw.map(fn);
    const yRaw = P3_W2.map((row, i) => row[0]*h[0] + row[1]*h[1] + row[2]*h[2] + row[3]*h[3] + P3_b2[i]);
    const y = yRaw.map(fn);
    return { hRaw, h, yRaw, y };
  }

  function p3NeuronStatus(j, z) {
    if (S.p3_act === 'relu' && z <= 0) return { cls: 'is-dead', msg: '死神经元：梯度 = 0' };
    if (S.p3_act === 'sigmoid' && Math.abs(z) > 4) return { cls: 'is-sat', msg: '饱和：梯度 ≈ 0' };
    if (S.p3_act === 'tanh' && Math.abs(z) > 3) return { cls: 'is-sat', msg: '饱和：梯度 ≈ 0' };
    return { cls: '', msg: '' };
  }

  function buildHActEachFormula(idx, vals) {
    const sym = ACT_SYM[S.p3_act] || 'σ';
    const rows = vals.hRaw.map((z, j) => {
      const st = p3NeuronStatus(j, z);
      return `
        <div class="p3-elem-row ${st.cls}">
          <span class="p3-elem-idx">z<sub>${j + 1}</sub></span>
          <span class="p3-elem-num">${num3(z)}</span>
          <span class="p3-op">→</span>
          <span class="p3-elem-act">${sym}</span>
          <span class="p3-op">→</span>
          <span class="p3-elem-out">h<sub>${j + 1}</sub> = ${num3(vals.h[j])}</span>
          <span class="p3-elem-note">${st.msg}</span>
        </div>`;
    }).join('');

    return `
      <div class="p3-formula-title">激活是逐元素的：z[4] 的每个分量各自过 σ → h[4]</div>
      <div class="p3-elem-table">${rows}</div>
      <div class="p3-formula-sum">
        维度不变：z ∈ ℝ⁴ → h ∈ ℝ⁴。σ 不做任何跨分量混合，混合只发生在 W₁ 里。
      </div>
      <div class="p3-formula-note">
        所以「一层」= 一次矩阵乘法（混合）+ 一次逐元素激活（弯折）。
      </div>`;
  }

  function buildHFormula(tick) {
    const r = tick.row, x = S.p3_x, row = P3_W1[r], bias = P3_b1[r];
    const act = ACT_FNS[S.p3_act];
    const terms = [0,1,2].map(c => ({ w: row[c], xv: x[c], prod: row[c]*x[c] }));
    const sum = terms.reduce((s, t) => s + t.prod, 0) + bias;
    const out = act.f(sum);
    let sub = -1;
    if (tick.type === 'h_term') sub = tick.term;
    else if (tick.type === 'h_bias') sub = 3;
    else if (tick.type === 'h_activate') sub = 4;
    let h = '';
    for (let c = 0; c < 3; c++) {
      const t = terms[c]; let cls = 'p3-term ', val = '';
      if (c < sub) { cls += 'p3-term-done'; val = `<span class="p3-term-val">${t.w.toFixed(2)}×${t.xv.toFixed(2)} = ${t.prod.toFixed(2)}</span>`; }
      else if (c === sub) { cls += 'p3-term-active'; val = `<span class="p3-term-val">${t.w.toFixed(2)}×${t.xv.toFixed(2)} = ${t.prod.toFixed(2)}</span>`; }
      else cls += 'p3-term-pending';
      h += `<div class="${cls}"><span class="p3-term-sym">w<sub>${r+1}${c+1}</sub>·x<sub>${c+1}</sub></span>${val}</div>`;
      if (c < 2) h += `<div class="p3-op">+</div>`;
    }
    let bCls = 'p3-term ', bVal = '';
    if (sub > 3) { bCls += 'p3-term-done'; bVal = `<span class="p3-term-val">= ${bias.toFixed(2)}</span>`; }
    else if (sub === 3) { bCls += 'p3-term-active'; bVal = `<span class="p3-term-val">= ${bias.toFixed(2)}</span>`; }
    else bCls += 'p3-term-pending';
    h += `<div class="p3-op">+</div><div class="${bCls}"><span class="p3-term-sym">b<sub>${r+1}</sub></span>${bVal}</div>`;
    let sumH = '', actH = '';
    if (sub >= 3) sumH = `<div class="p3-formula-sum">Σ = ${terms.map(t => t.prod.toFixed(2)).join(' + ')} + ${bias.toFixed(2)} = <b>${sum.toFixed(2)}</b></div>`;
    if (sub >= 4) actH = `<div class="p3-formula-activate">h<sub>${r+1}</sub> = ${act.sym}(${sum.toFixed(2)}) = <b style="color:#27ae60;">${out.toFixed(3)}</b></div>`;
    return `<div class="p3-formula-title">计算 h<sub>${r+1}</sub> · W₁ 第 ${r+1} 行 × 输入</div>
      <div class="p3-formula-eq"><span class="p3-eq-lhs">h<sub>${r+1}</sub> = ${act.sym}(</span>${h}<span class="p3-eq-rhs">)</span></div>${sumH}${actH}`;
  }

  function buildYFormula(tick, hVals) {
    const r = tick.row, row = P3_W2[r], bias = P3_b2[r];
    const act = ACT_FNS[S.p3_act];
    const terms = [0,1,2,3].map(c => ({ w: row[c], hv: hVals[c], prod: row[c]*hVals[c] }));
    const sum = terms.reduce((s, t) => s + t.prod, 0) + bias;
    const out = act.f(sum);
    let sub = -1;
    if (tick.type === 'y_term') sub = tick.term;
    else if (tick.type === 'y_bias') sub = 4;
    else if (tick.type === 'y_activate') sub = 5;
    let h = '';
    for (let c = 0; c < 4; c++) {
      const t = terms[c]; let cls = 'p3-term ', val = '';
      if (c < sub) { cls += 'p3-term-done'; val = `<span class="p3-term-val">${t.w.toFixed(2)}×${t.hv.toFixed(2)} = ${t.prod.toFixed(2)}</span>`; }
      else if (c === sub) { cls += 'p3-term-active'; val = `<span class="p3-term-val">${t.w.toFixed(2)}×${t.hv.toFixed(2)} = ${t.prod.toFixed(2)}</span>`; }
      else cls += 'p3-term-pending';
      h += `<div class="${cls}"><span class="p3-term-sym">w<sub>${r+1}${c+1}</sub>·h<sub>${c+1}</sub></span>${val}</div>`;
      if (c < 3) h += `<div class="p3-op">+</div>`;
    }
    let bCls = 'p3-term ', bVal = '';
    if (sub > 4) { bCls += 'p3-term-done'; bVal = `<span class="p3-term-val">= ${bias.toFixed(2)}</span>`; }
    else if (sub === 4) { bCls += 'p3-term-active'; bVal = `<span class="p3-term-val">= ${bias.toFixed(2)}</span>`; }
    else bCls += 'p3-term-pending';
    h += `<div class="p3-op">+</div><div class="${bCls}"><span class="p3-term-sym">b<sub>${r+1}</sub></span>${bVal}</div>`;
    let sumH = '', actH = '';
    if (sub >= 4) sumH = `<div class="p3-formula-sum">Σ = ${terms.map(t => t.prod.toFixed(2)).join(' + ')} + ${bias.toFixed(2)} = <b>${sum.toFixed(2)}</b></div>`;
    if (sub >= 5) actH = `<div class="p3-formula-activate">y<sub>${r+1}</sub> = ${act.sym}(${sum.toFixed(2)}) = <b style="color:#c0392b;">${out.toFixed(3)}</b></div>`;
    return `<div class="p3-formula-title">计算 y<sub>${r+1}</sub> · W₂ 第 ${r+1} 行 × 隐藏 h</div>
      <div class="p3-formula-eq"><span class="p3-eq-lhs">y<sub>${r+1}</sub> = ${act.sym}(</span>${h}<span class="p3-eq-rhs">)</span></div>${sumH}${actH}`;
  }

  function buildP3LossFormula(vals) {
    const { y } = vals;
    const loss = 0.5 * (Math.pow(y[0] - P4_TARGET[0], 2) + Math.pow(y[1] - P4_TARGET[1], 2));
    return `<div class="p3-formula-title">前向完成，评估“错了多少” (Loss)</div>
      <div class="p3-formula-eq" style="flex-direction: column; align-items: flex-start; gap: 8px;">
        <div>y = [${num3(y[0])}, ${num3(y[1])}]，目标 y* = [${P4_TARGET[0].toFixed(1)}, ${P4_TARGET[1].toFixed(1)}]</div>
        <div style="font-family: 'Times New Roman'; font-size: 15px;">L = ½ Σ(y_i − y_i*)²</div>
        <div style="margin-left: 20px;">= ½[(y₁ − y₁*)² + (y₂ − y₂*)²]</div>
        <div style="margin-left: 20px;">= ½[(${num3(y[0])} − ${P4_TARGET[0].toFixed(1)})² + (${num3(y[1])} − ${P4_TARGET[1].toFixed(1)})²]</div>
        <div style="margin-left: 20px;">= ½[${num3(Math.pow(y[0]-P4_TARGET[0],2))} + ${num3(Math.pow(y[1]-P4_TARGET[1],2))}]</div>
        <div style="margin-left: 20px; font-weight: bold; color: #c0392b; font-size: 15px;">= ${num3(loss)}</div>
      </div>
      <div class="p3-formula-note">
        L 越小 = 网络输出越接近目标。<br>训练的目标就是让 L → 0。怎么调 W？下一站讲。
      </div>`;
  }

  function buildP3Formula(idx, vals) {
    if (idx < 0) return `<div class="p3-formula-empty">点 <b>▶ 播放</b> 或 <b>下一步</b> 开始逐步演示。<br><span style="font-size:11px; color:#888;">当前激活函数：<b style="color:#2980b9;">${ACT_FNS[S.p3_act].name}</b></span></div>`;
    const t = P3_TICKS[idx];
    if (t.type === 'x_ready') return `<div class="p3-formula-empty"><div style="color:#2980b9; font-weight:bold; margin-bottom:6px;">输入就绪</div>x = [${S.p3_x.map(v => v.toFixed(2)).join(', ')}]<br><span style="font-size:11px; color:#888; margin-top:8px; display:block;">下一步：W₁ 的 4 行分别与 x 做点积 → 4 个隐藏神经元。</span></div>`;
    if (t.type === 'h_done') {
      const act = ACT_FNS[S.p3_act];
      let dead = 0;
      if (S.p3_act === 'relu') dead = vals.hRaw.filter(v => v <= 0).length;
      const warn = dead > 0 ? `<div style="margin-top:6px; color:#c0392b; font-size:11px;">⚠ 检测到 <b>${dead}</b> 个死神经元（ReLU 负半轴）</div>` : '';
      return `<div class="p3-formula-empty"><div style="color:#27ae60; font-weight:bold; margin-bottom:6px;">隐藏层完成</div>h = [${vals.h.map(v => v.toFixed(2)).join(', ')}]<br><span style="font-size:11px; color:#888; margin-top:8px; display:block;">激活函数：<b>${act.name}</b>。下一步：W₂ 把 4 维压到 2 维。</span>${warn}</div>`;
    }
    if (t.type === 'h_act_each') return buildHActEachFormula(idx, vals);
    if (t.type === 'loss') return buildP3LossFormula(vals);
    if (t.type === 'done') return `<div class="p3-formula-empty"><div style="color:#c0392b; font-weight:bold; margin-bottom:6px;">输出完成</div>y = [${vals.y.map(v => v.toFixed(2)).join(', ')}]<br><span style="font-size:11px; color:#888; margin-top:8px; display:block;">整个网络 = 两次「矩阵 × 向量 + 偏置 → 激活」的串联。</span></div>`;
    if (t.type.startsWith('h_')) return buildHFormula(t);
    if (t.type.startsWith('y_')) return buildYFormula(t, vals.h);
    return '';
  }

  function setCellCls(node, base, extra) {
    node.setAttribute('class', base + (extra ? ' ' + extra : ''));
  }

  function renderP3AtTick(idx) {
    P3.tickIdx = idx;
    const st = deriveP3State(idx);
    const vals = computeP3Values();
    const { hRaw, h, y } = vals;
    const x = S.p3_x, actKey = S.p3_act;

    for (let i = 0; i < 3; i++) {
      setCellCls(el.p3XCell[i], 'm2-p3-cell', st.x[i] === 'active' ? 'hl-x' : st.x[i] === 'done' ? 'done' : '');
      el.p3XText[i].textContent = x[i].toFixed(2);
    }
    for (let r = 0; r < 4; r++) {
      const rc = st.w1[r] === 'active' ? 'hl-w' : st.w1[r] === 'done' ? 'done' : '';
      for (let c = 0; c < 3; c++) {
        setCellCls(el.p3W1Cell[r][c], 'm2-p3-cell', rc);
        el.p3W1Text[r][c].textContent = P3_W1[r][c].toFixed(2);
      }
    }
    for (let i = 0; i < 4; i++) {
      let base = 'm2-p3-cell';
      if (st.h[i] === 'active') base += ' hl-h';
      else if (st.h[i] === 'done') base += ' done';
      let mk = '';
      if (st.hShow[i]) {
        if (actKey === 'relu' && hRaw[i] <= 0) mk = 'dead';
        else if ((actKey === 'sigmoid' || actKey === 'tanh') && Math.abs(hRaw[i]) > (actKey === 'sigmoid' ? 4 : 3)) mk = 'saturated';
      }
      setCellCls(el.p3HCell[i], base, mk);
      el.p3HText[i].textContent = st.hShow[i] ? h[i].toFixed(2) : '';
    }
    for (let r = 0; r < 2; r++) {
      const rc = st.w2[r] === 'active' ? 'hl-w' : st.w2[r] === 'done' ? 'done' : '';
      for (let c = 0; c < 4; c++) {
        setCellCls(el.p3W2Cell[r][c], 'm2-p3-cell', rc);
        el.p3W2Text[r][c].textContent = P3_W2[r][c].toFixed(2);
      }
    }
    for (let i = 0; i < 2; i++) {
      setCellCls(el.p3YCell[i], 'm2-p3-cell', st.y[i] === 'active' ? 'hl-y' : st.y[i] === 'done' ? 'done' : '');
      el.p3YText[i].textContent = st.yShow[i] ? y[i].toFixed(2) : '';
    }
    el.p3Formula.innerHTML = buildP3Formula(idx, vals);
    updateP3Warn(hRaw);
    updateP3Controls();

    for (let j = 0; j < 4; j++) {
      const node = el.p3HCell[j];
      if (!node) continue;
      node.classList.remove('act-each');
      node.style.animationDelay = '';
      if (st.hActEach) {
        void node.offsetWidth;
        node.style.animationDelay = (j * 110) + 'ms';
        node.classList.add('act-each');
      }
    }
  }

  function updateP3Warn(hRaw) {
    const act = ACT_FNS[S.p3_act];
    let w = '';
    if (S.p3_act === 'relu') {
      const dead = hRaw.map((v, i) => v <= 0 ? i : -1).filter(i => i >= 0);
      if (dead.length) w = `<div class="p3-warn-item" style="color:#c0392b;">⚠ <b>${dead.length} 个死神经元</b>：h<sub>${dead.map(i=>i+1).join(',')}</sub> 在 ReLU 负半轴被永久压制。</div>`;
    }
    if (S.p3_act === 'sigmoid' || S.p3_act === 'tanh') {
      const th = S.p3_act === 'sigmoid' ? 4 : 3;
      const sat = hRaw.map((v, i) => Math.abs(v) > th ? i : -1).filter(i => i >= 0);
      if (sat.length) w = `<div class="p3-warn-item" style="color:#e67e22;">⚠ <b>${sat.length} 个饱和神经元</b>：h<sub>${sat.map(i=>i+1).join(',')}</sub> 输入落进了梯度≈0 的区域。</div>`;
    }
    if (!w && P3.tickIdx >= 0) w = `<div class="p3-warn-item" style="color:#27ae60;">✓ 当前激活函数 <b>${act.name}</b> 下，所有神经元均处于活跃区。</div>`;
    el.p3WarnBox.innerHTML = w || `<div class="p3-warn-item" style="color:#888;">点击播放后，这里会显示每层的神经元健康状况。</div>`;
  }

  function updateP3Controls() {
    const last = P3_TICKS.length - 1, idx = P3.tickIdx;
    if (el.p3PlayBtn) el.p3PlayBtn.textContent = P3.playing ? '■ 暂停' : '▶ 播放';
    if (el.p3PrevBtn) el.p3PrevBtn.disabled = P3.playing || idx <= -1;
    if (el.p3NextBtn) el.p3NextBtn.disabled = P3.playing || idx >= last;
    if (el.p3StepButtons) el.p3StepButtons.forEach(b => b.disabled = P3.playing);
    if (el.p3SpeedBtns) el.p3SpeedBtns.forEach(b => b.classList.toggle('is-active', Number(b.dataset.speed) === P3.speed));
    if (el.p3ActBtns) el.p3ActBtns.forEach(b => b.classList.toggle('is-active', b.dataset.p3Act === S.p3_act));
  }
  function p3TickNext() {
    const last = P3_TICKS.length - 1;
    if (P3.tickIdx >= last) { P3.playing = false; updateP3Controls(); return; }
    renderP3AtTick(P3.tickIdx + 1);
    if (P3.playing) {
      if (P3.tickIdx >= last) { P3.playing = false; updateP3Controls(); }
      else P3.timer = setTimeout(p3TickNext, P3.interval);
    }
  }
  function p3TogglePlay() {
    if (P3.playing) { clearTimeout(P3.timer); P3.timer = null; P3.playing = false; updateP3Controls(); return; }
    if (P3.tickIdx >= P3_TICKS.length - 1) renderP3AtTick(-1);
    P3.playing = true; updateP3Controls();
    P3.timer = setTimeout(p3TickNext, P3.interval);
  }
  function p3Next() { if (P3.tickIdx < P3_TICKS.length - 1) renderP3AtTick(P3.tickIdx + 1); }
  function p3Prev() { if (P3.tickIdx > -1) renderP3AtTick(P3.tickIdx - 1); }
  function p3Reset() {
    if (P3.timer) { clearTimeout(P3.timer); P3.timer = null; }
    P3.playing = false; renderP3AtTick(-1);
  }
  function p3SetSpeed(sp) {
    P3.speed = sp; P3.interval = Math.round(400 / sp);
    if (P3.playing) { clearTimeout(P3.timer); P3.timer = setTimeout(p3TickNext, P3.interval); }
    updateP3Controls();
  }
  function p3SetAct(key) { if (S.p3_act === key) return; S.p3_act = key; p3Reset(); }

  /* =========================================================
   *  场景 4（极客推导版）
   * ========================================================= */
  function buildScene4SvgHtml() {
    let s = `<defs>
      <marker id="ch3m2-p4-arrow-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#c0392b"/>
      </marker>
    </defs>`;
    s += `<text x="${P4_X_X}" y="40" class="m2-p3-layer-title" text-anchor="middle">输入 x[3]</text>`;
    s += `<text x="${P4_H_X}" y="40" class="m2-p3-layer-title" text-anchor="middle">隐藏 h[4]</text>`;
    s += `<text x="${P4_Y_X}" y="40" class="m2-p3-layer-title" text-anchor="middle">输出 y[2]</text>`;
    s += `<text x="${P4_X_X}" y="470" class="m2-p3-layer-title" text-anchor="middle" style="font-size:11px; fill:#c0392b;">◀ 误差终点</text>`;
    s += `<text x="${P4_Y_X}" y="470" class="m2-p3-layer-title" text-anchor="middle" style="font-size:11px; fill:#c0392b;">误差起点 ▶</text>`;

    for (let i = 0; i < 2; i++) for (let j = 0; j < 4; j++) {
      const w = P3_W2[i][j];
      const sw = 1 + Math.abs(w) * 1.5;
      s += `<line data-line="W2-${i}-${j}" x1="${P4_H_X}" y1="${P4_H_YS[j]}" x2="${P4_Y_X}" y2="${P4_Y_YS[i]}"
        stroke="#d0d0d0" stroke-width="${sw}" class="m2-p4-line"/>`;
      s += `<text data-line-text="W2-${i}-${j}" x="0" y="0" font-size="10px" fill="#c0392b" font-family="Consolas, monospace" font-weight="bold" text-anchor="middle" dominant-baseline="middle" style="display:none; opacity:0.85; pointer-events:none;">0.00</text>`;
    }
    for (let j = 0; j < 4; j++) for (let k = 0; k < 3; k++) {
      const w = P3_W1[j][k];
      const sw = 1 + Math.abs(w) * 1.5;
      s += `<line data-line="W1-${j}-${k}" x1="${P4_X_X}" y1="${P4_X_YS[k]}" x2="${P4_H_X}" y2="${P4_H_YS[j]}"
        stroke="#d0d0d0" stroke-width="${sw}" class="m2-p4-line"/>`;
      s += `<text data-line-text="W1-${j}-${k}" x="0" y="0" font-size="10px" fill="#c0392b" font-family="Consolas, monospace" font-weight="bold" text-anchor="middle" dominant-baseline="middle" style="display:none; opacity:0.85; pointer-events:none;">0.00</text>`;
    }

    for (let k = 0; k < 3; k++) {
      const cy = P4_X_YS[k];
      s += `<circle data-node="x-${k}" cx="${P4_X_X}" cy="${cy}" r="${P4_R}" class="m2-p4-node"/>`;
      s += `<text x="${P4_X_X}" y="${cy}" class="m2-p4-label">x<sub>${k+1}</sub></text>`;
      s += `<text data-val="x-${k}" x="${P4_X_X}" y="${cy + P4_R + 16}" class="m2-p4-val"></text>`;
      s += `<text data-delta="x-${k}" x="${P4_X_X}" y="${cy - P4_R - 8}" class="m2-p4-delta"></text>`;
    }
    for (let j = 0; j < 4; j++) {
      const cy = P4_H_YS[j];
      s += `<circle data-node="h-${j}" cx="${P4_H_X}" cy="${cy}" r="${P4_R}" class="m2-p4-node"/>`;
      s += `<text x="${P4_H_X}" y="${cy}" class="m2-p4-label">h<sub>${j+1}</sub></text>`;
      s += `<text data-val="h-${j}" x="${P4_H_X}" y="${cy + P4_R + 16}" class="m2-p4-val"></text>`;
      s += `<text data-delta="h-${j}" x="${P4_H_X}" y="${cy - P4_R - 8}" class="m2-p4-delta"></text>`;
    }
    for (let i = 0; i < 2; i++) {
      const cy = P4_Y_YS[i];
      s += `<circle data-node="y-${i}" cx="${P4_Y_X}" cy="${cy}" r="${P4_R + 2}" class="m2-p4-node"/>`;
      s += `<text x="${P4_Y_X}" y="${cy}" class="m2-p4-label" style="font-size:16px;">y<sub>${i+1}</sub></text>`;
      s += `<text data-val="y-${i}" x="${P4_Y_X}" y="${cy + P4_R + 18}" class="m2-p4-val" style="font-size:11.5px; fill:#333;">现 0.00</text>`;
      s += `<text data-target="y-${i}" x="${P4_Y_X + P4_R + 10}" y="${cy - 6}" class="m2-p4-target" style="display:none; text-anchor:start;">y${i+1}* = ${P4_TARGET[i].toFixed(1)}</text>`;
      s += `<text data-delta="y-${i}" x="${P4_Y_X + P4_R + 10}" y="${cy + 14}" class="m2-p4-delta" style="text-anchor:start;">δ = 0</text>`;
    }
    return s;
  }

  function cacheScene4Elements() {
    const sc = el.p4SvgContent;
    const q = (a, k) => sc.querySelector(`[data-${a}="${k}"]`);
    el.p4NodeX = [0,1,2].map(k => q('node', `x-${k}`));
    el.p4NodeH = [0,1,2,3].map(j => q('node', `h-${j}`));
    el.p4NodeY = [0,1].map(i => q('node', `y-${i}`));
    el.p4ValX = [0,1,2].map(k => q('val', `x-${k}`));
    el.p4ValH = [0,1,2,3].map(j => q('val', `h-${j}`));
    el.p4ValY = [0,1].map(i => q('val', `y-${i}`));
    el.p4DeltaX = [0,1,2].map(k => q('delta', `x-${k}`));
    el.p4DeltaH = [0,1,2,3].map(j => q('delta', `h-${j}`));
    el.p4DeltaY = [0,1].map(i => q('delta', `y-${i}`));
    el.p4TargetY = [0,1].map(i => q('target', `y-${i}`));
    el.p4LinesW1 = []; el.p4TextW1 = [];
    for (let j = 0; j < 4; j++) {
      el.p4LinesW1[j] = []; el.p4TextW1[j] = [];
      for (let k = 0; k < 3; k++) {
        el.p4LinesW1[j][k] = sc.querySelector(`[data-line="W1-${j}-${k}"]`);
        el.p4TextW1[j][k] = sc.querySelector(`[data-line-text="W1-${j}-${k}"]`);
      }
    }
    el.p4LinesW2 = []; el.p4TextW2 = [];
    for (let i = 0; i < 2; i++) {
      el.p4LinesW2[i] = []; el.p4TextW2[i] = [];
      for (let j = 0; j < 4; j++) {
        el.p4LinesW2[i][j] = sc.querySelector(`[data-line="W2-${i}-${j}"]`);
        el.p4TextW2[i][j] = sc.querySelector(`[data-line-text="W2-${i}-${j}"]`);
      }
    }
  }

  function deriveP4State(idx) {
    const st = {
      targetVisible: false, activeY: -1, yFlowProgress: [0, 0],
      hAccumDone: false, hFlowProgress: [0, 0, 0, 0], 
      lossCalcDone: false, deltaYDone: false, deltaZ2Done: false,
      gradW2Done: false, deltaZ1Done: false, gradW1Done: false,
      updateFormulaDone: false, updateDemoDone: false, updateDone: false,
      loopForwardDone: false, loopLossDone: false, loopSummaryDone: false
    };
    if (idx < 0) return st;
    for (let i = 0; i <= idx; i++) {
      const t = P4_TICKS[i];
      if (t.type === 'init') st.targetVisible = true;
      else if (t.type === 'loss_calc') st.lossCalcDone = true;
      else if (t.type === 'grad_y') st.deltaYDone = true;
      else if (t.type === 'y1_flow') { st.yFlowProgress[0] = t.term + 1; st.activeY = 0; }
      else if (t.type === 'y2_flow') { st.yFlowProgress[1] = t.term + 1; st.activeY = 1; }
      else if (t.type === 'grad_z2') st.deltaZ2Done = true;
      else if (t.type === 'grad_W2') st.gradW2Done = true;
      else if (t.type === 'h_accum') { st.hAccumDone = true; st.activeY = -1; }
      else if (t.type === 'h1_flow') st.hFlowProgress[0] = t.term + 1;
      else if (t.type === 'h2_flow') st.hFlowProgress[1] = t.term + 1;
      else if (t.type === 'h3_flow') st.hFlowProgress[2] = t.term + 1;
      else if (t.type === 'h4_flow') st.hFlowProgress[3] = t.term + 1;
      else if (t.type === 'grad_z1') st.deltaZ1Done = true;
      else if (t.type === 'grad_W1') st.gradW1Done = true;
      else if (t.type === 'update_formula') st.updateFormulaDone = true;
      else if (t.type === 'update_demo') st.updateDemoDone = true;
      else if (t.type === 'update_done') st.updateDone = true;
      else if (t.type === 'loop_forward') st.loopForwardDone = true;
      else if (t.type === 'loop_loss') st.loopLossDone = true;
      else if (t.type === 'loop_summary') st.loopSummaryDone = true;
    }
    return st;
  }

  function computeP4Values() {
    const { h, hRaw, yRaw, y } = computeP3Values();
    const df = ACT_FNS[S.p3_act].df;
    const deltaY = [y[0] - P4_TARGET[0], y[1] - P4_TARGET[1]];
    const deltaZ2 = deltaY.map((dy, i) => dy * df(yRaw[i]));
    const gradW2 = [];
    for(let i=0; i<2; i++){
      gradW2[i] = [];
      for(let j=0; j<4; j++) gradW2[i][j] = deltaZ2[i] * h[j];
    }
    const deltaH = [0,1,2,3].map(j => {
      let sum = 0;
      for(let i=0; i<2; i++) sum += deltaZ2[i] * P3_W2[i][j];
      return sum;
    });
    const deltaZ1 = deltaH.map((dh, j) => dh * df(hRaw[j]));
    const gradW1 = [];
    for(let j=0; j<4; j++){
      gradW1[j] = [];
      for(let k=0; k<3; k++) gradW1[j][k] = deltaZ1[j] * S.p3_x[k];
    }
    const deltaX = [0,0,0];
    return { h, y, hRaw, yRaw, deltaY, deltaH, deltaX, gradW2, gradW1, deltaZ2, deltaZ1 };
  }

  // --- 公式生成 (严格带索引与说明) ---
  function buildP4InitFormula(vals) {
    return `<div class="p4-formula-title">1. 初始化：认识各节点的前向值</div>
      <div class="p4-formula-row">
        <span class="p4-formula-lhs">隐藏层 h (已过激活)</span>
        <span class="p4-formula-calc">[${vals.h.map(v=>v.toFixed(2)).join(', ')}]</span>
      </div>
      <div class="p4-formula-row">
        <span class="p4-formula-lhs">输出层 z₂ (加权和)</span>
        <span class="p4-formula-calc">[${vals.yRaw.map(v=>v.toFixed(2)).join(', ')}]</span>
      </div>
      <div class="p4-formula-row">
        <span class="p4-formula-lhs">输出 y = σ(z₂)</span>
        <span class="p4-formula-calc">[${vals.y[0].toFixed(3)}, ${vals.y[1].toFixed(3)}]</span>
      </div>
      <div class="p4-formula-row">
        <span class="p4-formula-lhs">目标输出 y*</span>
        <span class="p4-formula-calc">[${P4_TARGET[0].toFixed(1)}, ${P4_TARGET[1].toFixed(1)}]</span>
      </div>
      <div class="p4-formula-note">注：z₂ 是没过激活函数的原始值，y 是过了激活的最终输出。</div>`;
  }

  function buildP4LossCalcFormula(vals) {
    const L = computeLoss(vals.y, P4_TARGET);
    return `<div class="p4-formula-title">2. 计算 Loss（差距大小）</div>
      <div class="p4-formula-row">
        <span class="p4-formula-lhs">L = ½ Σ(y - y*)²</span>
        <span class="p4-formula-calc">½[(${vals.y[0].toFixed(2)} - 1)² + (${vals.y[1].toFixed(2)} - 0)²] = <b>${L.toFixed(4)}</b></span>
      </div>
      <div class="p4-formula-note">Loss 告诉我们现在有多差。下一步，我们要找到改变 W 的方向，让这个数字变小。</div>`;
  }

  function buildP4GradYFormula(vals) {
    return `<div class="p4-formula-title">3. 对 y 求导：∂L/∂y = y − y*</div>
      <div class="p4-formula-row">
        <span class="p4-formula-lhs">δy₁</span>
        <span class="p4-formula-calc">${vals.y[0].toFixed(3)} − ${P4_TARGET[0].toFixed(1)} = <b>${vals.deltaY[0].toFixed(3)}</b></span>
      </div>
      <div class="p4-formula-row">
        <span class="p4-formula-lhs">δy₂</span>
        <span class="p4-formula-calc">${vals.y[1].toFixed(3)} − ${P4_TARGET[1].toFixed(1)} = <b>${vals.deltaY[1].toFixed(3)}</b></span>
      </div>
      <div class="p4-formula-note">这就是最初的误差信号，它即将开始倒流。</div>`;
  }

  function buildP4YFlowFormula(yi, term, vals) {
    const row = P3_W2[yi];
    const hIdx = term + 1;
    const w = row[term];
    const prod = vals.deltaY[yi] * w;
    let prev = '';
    for (let k = 0; k < term; k++) {
      const wk = row[k], pk = vals.deltaY[yi] * wk;
      prev += `<div class="p4-formula-row p4-formula-done">
        <span class="p4-formula-lhs">→ h${k+1}</span>
        <span class="p4-formula-calc">δy${yi+1} × ${wk.toFixed(2)} = ${pk.toFixed(3)}</span>
      </div>`;
    }
    return `<div class="p4-formula-title">误差倒流：y${yi+1} 传给 h₁~h₄</div>
      <div class="p4-formula-subhead">沿 W₂ 第 ${yi+1} 行按权重分配：</div>
      ${prev}
      <div class="p4-formula-row p4-formula-active">
        <span class="p4-formula-lhs">→ h${hIdx} (连线上标注)</span>
        <span class="p4-formula-calc">${vals.deltaY[yi].toFixed(3)} × ${w.toFixed(2)} = <b>${prod.toFixed(3)}</b></span>
      </div>`;
  }

  function buildP4GradZ2Formula(vals) {
    const df = ACT_FNS[S.p3_act].df;
    const { y, deltaY, deltaZ2 } = vals;
    return `<div class="p4-formula-title">★ 过激活函数的导数 ∂L/∂z₂ = ∂L/∂y · σ'(z₂)</div>
      <div class="p4-formula-note" style="margin-top:0; padding-top:0; border:none; margin-bottom:10px; color:#c0392b;">
        y = σ(z₂)，所以误差穿回去时，必须乘上激活函数在 z₂ 处的斜率。
      </div>
      <div class="p4-formula-row">
        <span class="p4-formula-lhs">z₂ 的导数 σ'(z₂)</span>
        <span class="p4-formula-calc">y(1−y) = ${y[0].toFixed(3)}×(1−${y[0].toFixed(3)}) = ${df(y[0]).toFixed(3)}<br>${y[1].toFixed(3)}×(1−${y[1].toFixed(3)}) = ${df(y[1]).toFixed(3)}</span>
      </div>
      <div class="p4-formula-row p4-formula-active">
        <span class="p4-formula-lhs">δz₂（真正传回 W₂ 的误差）</span>
        <span class="p4-formula-calc">δy · σ'(z₂) = <b>[${deltaZ2[0].toFixed(3)}, ${deltaZ2[1].toFixed(3)}]</b></span>
      </div>`;
  }

  function buildP4GradW2Formula(vals) {
    const { gradW2 } = vals;
    let html = `<div class="p4-formula-title">计算 W₂ 梯度：∂L/∂W₂ = δz₂ · hᵀ</div>
      <div class="p4-grad-outer">
        <div style="font-size:12px; color:#888; text-align:center; margin-bottom:4px;">外积：(2×1) × (1×4) → (2×4)</div>
        <div class="p4-grad-table">`;

    // 表头
    html += `<div class="p4-grad-cell p4-grad-header"></div>`;
    for(let j=0; j<4; j++) html += `<div class="p4-grad-cell p4-grad-header">h<sub>${j+1}</sub></div>`;
    // 数据
    for(let i=0; i<2; i++) {
      html += `<div class="p4-grad-cell p4-grad-header">y<sub>${i+1}</sub></div>`;
      for(let j=0; j<4; j++) {
        const val = gradW2[i][j];
        const cls = val >= 0 ? 'pos' : 'neg';
        html += `<div class="p4-grad-cell ${cls}">${val.toFixed(3)}</div>`;
      }
    }
    html += `</div></div>`;

    html += `<div class="p4-formula-row" style="margin-top:10px;">
        <span class="p4-formula-lhs">例：∂L/∂W₂[0][0] (即 <b>y₁</b> 对 <b>h₁</b>)</span>
        <span class="p4-formula-calc">δz₂₀ × h₀ = ${vals.deltaZ2[0].toFixed(3)} × ${vals.h[0].toFixed(2)} = <b>${gradW2[0][0].toFixed(3)}</b></span>
      </div>
      <div class="p4-formula-note" style="border-left-color:#c0392b; background:#fff5f5; padding:8px;">
        <b>这个矩阵怎么读？</b><br>
        行代表输出 y，列代表隐藏层 h。<br>
        W₂[i][j] 的梯度 = 第 i 个输出的误差 δz₂ᵢ × 第 j 个隐藏层的输入 hⱼ。<br>
        连线上标注的就是这些乘积的值。
      </div>`;
    return html;
  }

  function buildP4HAccumFormula(vals) {
    let rows = '';
    for (let j = 0; j < 4; j++) {
      const w0 = P3_W2[0][j], w1 = P3_W2[1][j];
      const p0 = vals.deltaZ2[0] * w0, p1 = vals.deltaZ2[1] * w1;
      rows += `<div class="p4-formula-row">
        <span class="p4-formula-lhs">δh${j+1}</span>
        <span class="p4-formula-calc">${vals.deltaZ2[0].toFixed(3)}×${w0.toFixed(2)} + ${vals.deltaZ2[1].toFixed(3)}×${w1.toFixed(2)} = <b>${vals.deltaH[j].toFixed(3)}</b></span>
      </div>`;
    }
    return `<div class="p4-formula-title">隐藏层误差累加：δh = W₂ᵀ · δz₂</div>
      ${rows}
      <div class="p4-formula-note">h₁~h₄ 分别受到 y₁ 和 y₂ 的加权影响，把两条路径的误差加起来。</div>`;
  }

  function buildP4HFlowFormula(hj, term, vals) {
    const row = P3_W1[hj];
    const xIdx = term + 1;
    const w = row[term];
    const prod = vals.deltaH[hj] * w;
    let prev = '';
    for (let k = 0; k < term; k++) {
      const wk = row[k], pk = vals.deltaH[hj] * wk;
      prev += `<div class="p4-formula-row p4-formula-done">
        <span class="p4-formula-lhs">→ x${k+1}</span>
        <span class="p4-formula-calc">δh${hj+1} × ${wk.toFixed(2)} = ${pk.toFixed(3)}</span>
      </div>`;
    }
    return `<div class="p4-formula-title">误差继续倒流：h${hj+1} 传给 x₁~x₃</div>
      <div class="p4-formula-subhead">沿 W₁ 第 ${hj+1} 行，分给 x₁~x₃ (连线上标注)：</div>
      ${prev}
      <div class="p4-formula-row p4-formula-active">
        <span class="p4-formula-lhs">→ x${xIdx}</span>
        <span class="p4-formula-calc">${vals.deltaH[hj].toFixed(3)} × ${w.toFixed(2)} = <b>${prod.toFixed(3)}</b></span>
      </div>`;
  }

  function buildP4GradZ1Formula(vals) {
    const df = ACT_FNS[S.p3_act].df;
    const { h, deltaH, deltaZ1 } = vals;
    return `<div class="p4-formula-title">★ 过隐藏层激活的导数 ∂L/∂z₁ = ∂L/∂h · σ'(z₁)</div>
      <div class="p4-formula-row">
        <span class="p4-formula-lhs">h 的导数 σ'(z₁)</span>
        <span class="p4-formula-calc">h(1−h) = ... = [${df(h[0]).toFixed(2)}, ${df(h[1]).toFixed(2)}, ${df(h[2]).toFixed(2)}, ${df(h[3]).toFixed(2)}]</span>
      </div>
      <div class="p4-formula-row p4-formula-active">
        <span class="p4-formula-lhs">δz₁（传给 W₁ 的误差）</span>
        <span class="p4-formula-calc">δh · σ'(z₁) = <b>[${deltaZ1.map(v=>v.toFixed(3)).join(', ')}]</b></span>
      </div>`;
  }

  function buildP4GradW1Formula(vals) {
    const { gradW1 } = vals;
    let html = `<div class="p4-formula-title">计算 W₁ 梯度：∂L/∂W₁ = δz₁ · xᵀ</div>
      <div class="p4-grad-outer">
        <div style="font-size:12px; color:#888; text-align:center; margin-bottom:4px;">外积：(4×1) × (1×3) → (4×3)</div>
        <div class="p4-grad-table" style="grid-template-columns: repeat(4, 1fr);">`;

    // 表头
    html += `<div class="p4-grad-cell p4-grad-header"></div>`;
    for(let k=0; k<3; k++) html += `<div class="p4-grad-cell p4-grad-header">x<sub>${k+1}</sub></div>`;
    // 数据
    for(let j=0; j<4; j++) {
      html += `<div class="p4-grad-cell p4-grad-header">h<sub>${j+1}</sub></div>`;
      for(let k=0; k<3; k++) {
        const val = gradW1[j][k];
        const cls = val >= 0 ? 'pos' : 'neg';
        html += `<div class="p4-grad-cell ${cls}">${val.toFixed(3)}</div>`;
      }
    }
    html += `</div></div>`;

    html += `<div class="p4-formula-row" style="margin-top:10px;">
        <span class="p4-formula-lhs">例：∂L/∂W₁[0][0] (即 <b>h₁</b> 对 <b>x₁</b>)</span>
        <span class="p4-formula-calc">δz₁₀ × x₀ = ${vals.deltaZ1[0].toFixed(3)} × ${S.p3_x[0].toFixed(2)} = <b>${gradW1[0][0].toFixed(3)}</b></span>
      </div>
      <div class="p4-formula-note" style="border-left-color:#c0392b; background:#fff5f5; padding:8px;">
        <b>这个矩阵怎么读？</b><br>
        行代表隐藏层 h，列代表输入 x。<br>
        W₁[j][k] 的梯度 = 第 j 个隐藏层的误差 δz₁ⱼ × 第 k 个输入 xₖ。
      </div>`;
    return html;
  }

  function buildP4UpdateFormula() {
    return `<div class="p4-formula-title">更新规则：w ← w − η · 梯度</div>
      <div style="font-size:13px; line-height:1.8; margin:10px 0;">
        当前学习率 <b style="color:#2980b9; font-size:15px;">η = ${S.p4_eta.toFixed(2)}</b><br>
        η 决定了每次朝梯度反方向迈多大的一步。<br>
        下一步，我们以 W₂[0][0] 为例算一遍。
      </div>
      <div class="p4-formula-note">
        小步走，才不会跨过谷底；但不能太小，否则太慢。
      </div>`;
  }

  function buildP4UpdateDemo(vals) {
    const wOld = P3_W2[0][0];
    const grad = vals.gradW2[0][0];
    const delta = -S.p4_eta * grad;
    const wNew = wOld + delta;
    return `<div class="p4-formula-title">代入计算：W₂[0][0] 的更新</div>
      <div class="p4-formula-row">
        <span class="p4-formula-lhs">w_old</span>
        <span class="p4-formula-calc">${wOld.toFixed(4)}</span>
      </div>
      <div class="p4-formula-row">
        <span class="p4-formula-lhs">梯度 (∂L/∂w)</span>
        <span class="p4-formula-calc">${grad.toFixed(4)}</span>
      </div>
      <div class="p4-formula-row p4-formula-active">
        <span class="p4-formula-lhs">w_new = w_old − η·grad</span>
        <span class="p4-formula-calc">${wOld.toFixed(4)} − ${S.p4_eta.toFixed(2)} × (${grad.toFixed(4)}) = <b>${wNew.toFixed(4)}</b></span>
      </div>
      <div class="p4-formula-note">
        看看右侧真实的 Loss 曲线切片，小球是不是正在沿着梯度的反方向下坡？
      </div>`;
  }

  function buildP4UpdateDone() {
    return `<div class="p4-formula-title">✓ 更新完成</div>
      <div style="font-size:13px; line-height:1.8;">
        所有的 W₁ 和 W₂ 元素都已按公式更新。<br>
        注意看右侧 SVG 的连线粗细 —— 它们已经跟着 w 的绝对值变了！<br><br>
        <b>W 变了，接下来会发生什么？</b><br>
        用新的 W 再前向算一次，Loss 应该会变小。
      </div>`;
  }

  function buildP4LoopForwardFormula(vals) {
    const fwd = computeForwardWithW(P4.updatedW1 || P3_W1, P4.updatedW2 || P3_W2, P3_b1, P3_b2, S.p3_x, S.p3_act);
    return `<div class="p4-formula-title">闭环：用新 W 再做一次前向</div>
      <div class="p4-formula-row">
        <span class="p4-formula-lhs">旧 W 的 y</span>
        <span class="p4-formula-calc">[${vals.y[0].toFixed(3)}, ${vals.y[1].toFixed(3)}]</span>
      </div>
      <div class="p4-formula-row p4-formula-active">
        <span class="p4-formula-lhs">新 W 的 y'</span>
        <span class="p4-formula-calc"><b>[${fwd.y[0].toFixed(3)}, ${fwd.y[1].toFixed(3)}]</b></span>
      </div>
      <div class="p4-formula-note">
        目标 y* 是 [1.0, 0.0]。新的 y 已经更接近目标了。
      </div>`;
  }

  function buildP4LoopLossFormula(vals) {
    const fwd = computeForwardWithW(P4.updatedW1 || P3_W1, P4.updatedW2 || P3_W2, P3_b1, P3_b2, S.p3_x, S.p3_act);
    const lossOld = computeLoss(vals.y, P4_TARGET);
    const lossNew = computeLoss(fwd.y, P4_TARGET);
    const diff = lossOld - lossNew;
    return `<div class="p4-formula-title">闭环：Loss 下降了</div>
      <div class="p4-formula-row">
        <span class="p4-formula-lhs">L_old (旧 W)</span>
        <span class="p4-formula-calc">${lossOld.toFixed(4)}</span>
      </div>
      <div class="p4-formula-row p4-formula-active">
        <span class="p4-formula-lhs">L_new (新 W)</span>
        <span class="p4-formula-calc"><b>${lossNew.toFixed(4)}</b> (下降 ${(diff>0?'+':'')}${diff.toFixed(4)})</span>
      </div>
      <div class="p4-loss-bar">
        <div class="p4-loss-bar-row">
          <span style="width:50px;">旧 L</span>
          <div class="p4-loss-bar-fill" style="width:${lossOld*100}%; background:#e67e22;"></div>
        </div>
        <div class="p4-loss-bar-row">
          <span style="width:50px;">新 L</span>
          <div class="p4-loss-bar-fill" style="width:${lossNew*100}%; background:#27ae60;"></div>
        </div>
      </div>
      <div class="p4-formula-note" style="margin-top:15px;">
        这就是机器学习的本质：稍微调一下 W，Loss 就降一点。重复千百万次。
      </div>`;
  }

  function buildP4LoopSummaryFormula() {
    return `<div class="p4-formula-title">一次完整训练 = 前向 + 反向 + 更新</div>
      <div class="p4-loop-diagram">
        <div class="p4-loop-step">1. 前向<br>y = f(Wx)</div>
        <div class="p4-loop-arrow">→</div>
        <div class="p4-loop-step">2. 算 Loss<br>L(y, y*)</div>
        <div class="p4-loop-arrow">→</div>
        <div class="p4-loop-step">3. 反向<br>算 δ 和梯度</div>
        <div class="p4-loop-arrow">→</div>
        <div class="p4-loop-step">4. 更新 W<br>w ← w − η·grad</div>
      </div>
      <div style="text-align:center; font-size:20px; color:#888; margin-top:10px;">↓ 循环往复 ↓</div>
      <div class="p4-formula-note">
        走到这里，恭喜你！你已经看懂了神经网络最核心的运转机制。
      </div>`;
  }

  function buildP4Formula(idx, vals) {
    if (idx < 0) return `<div class="p3-formula-empty">
      点击 <b>▶ 播放</b> 开始，逐步观察误差如何倒流，并最终让网络"学会"。
    </div>`;
    const t = P4_TICKS[idx];
    if (t.type === 'init') return buildP4InitFormula(vals);
    if (t.type === 'loss_calc') return buildP4LossCalcFormula(vals);
    if (t.type === 'grad_y') return buildP4GradYFormula(vals);
    if (t.type === 'y1_flow') return buildP4YFlowFormula(0, t.term, vals);
    if (t.type === 'y2_flow') return buildP4YFlowFormula(1, t.term, vals);
    if (t.type === 'grad_z2') return buildP4GradZ2Formula(vals);
    if (t.type === 'grad_W2') return buildP4GradW2Formula(vals);
    if (t.type === 'h_accum') return buildP4HAccumFormula(vals);
    if (t.type === 'h1_flow') return buildP4HFlowFormula(0, t.term, vals);
    if (t.type === 'h2_flow') return buildP4HFlowFormula(1, t.term, vals);
    if (t.type === 'h3_flow') return buildP4HFlowFormula(2, t.term, vals);
    if (t.type === 'h4_flow') return buildP4HFlowFormula(3, t.term, vals);
    if (t.type === 'grad_z1') return buildP4GradZ1Formula(vals);
    if (t.type === 'grad_W1') return buildP4GradW1Formula(vals);
    if (t.type === 'update_formula') return buildP4UpdateFormula();
    if (t.type === 'update_demo') return buildP4UpdateDemo(vals);
    if (t.type === 'update_done') return buildP4UpdateDone();
    if (t.type === 'loop_forward') return buildP4LoopForwardFormula(vals);
    if (t.type === 'loop_loss') return buildP4LoopLossFormula(vals);
    if (t.type === 'loop_summary') return buildP4LoopSummaryFormula();
    return `<div class="p4-formula-title">✓ 训练演示结束</div>`;
  }

  function renderP4AtTick(idx) {
    P4.tickIdx = idx;
    const st = deriveP4State(idx);
    const vals = computeP4Values();
    const { h, y, deltaY, deltaH, deltaX } = vals;

    const useNewW = st.updateDone;
    const curW1 = useNewW ? (P4.updatedW1 || P3_W1) : P3_W1;
    const curW2 = useNewW ? (P4.updatedW2 || P3_W2) : P3_W2;

    let currentY = y;
    if (st.loopForwardDone) {
      const fwd = computeForwardWithW(curW1, curW2, P3_b1, P3_b2, S.p3_x, S.p3_act);
      currentY = fwd.y;
    }

    for (let k = 0; k < 3; k++) {
      el.p4NodeX[k].setAttribute('class', 'm2-p4-node' + (st.xAccumDone ? ' received' : ''));
      el.p4ValX[k].textContent = S.p3_x[k].toFixed(2);
      el.p4DeltaX[k].textContent = st.xAccumDone ? `δ=${deltaX[k].toFixed(3)}` : '';
    }
    for (let j = 0; j < 4; j++) {
      el.p4NodeH[j].setAttribute('class', 'm2-p4-node' + (st.hAccumDone ? ' received' : ''));
      el.p4ValH[j].textContent = h[j].toFixed(2);
      el.p4DeltaH[j].textContent = st.hAccumDone ? `δ=${deltaH[j].toFixed(3)}` : '';
    }
    for (let i = 0; i < 2; i++) {
      let cls = 'm2-p4-node';
      if (st.targetVisible) cls += ' target-on';
      if (st.activeY === i) cls += ' active';
      el.p4NodeY[i].setAttribute('class', cls);
      el.p4ValY[i].textContent = `现 ${currentY[i].toFixed(2)}`;
      el.p4TargetY[i].style.display = st.targetVisible ? '' : 'none';
      el.p4DeltaY[i].textContent = st.targetVisible ? `δ = ${deltaY[i] >= 0 ? '+' : ''}${deltaY[i].toFixed(2)}` : '';
      el.p4DeltaY[i].style.fill = deltaY[i] >= 0 ? '#c0392b' : '#2980b9';
    }

    // 误差倒流的线上带数值（通过静态节点控制显隐和位置）
    for (let i = 0; i < 2; i++) for (let j = 0; j < 4; j++) {
      const line = el.p4LinesW2[i][j];
      const txt = el.p4TextW2[i][j];
      const prog = st.yFlowProgress[i];
      let cls = 'm2-p4-line';
      if (prog > j) cls += ' arrived';
      else if (prog > 0) cls += ' flowing';
      line.setAttribute('class', cls);
      line.setAttribute('stroke-width', 1 + Math.abs(curW2[i][j]) * 1.5);
      
      if (prog > 0 && txt) {
         const x1 = Number(line.getAttribute('x1')), y1 = Number(line.getAttribute('y1'));
         const x2 = Number(line.getAttribute('x2')), y2 = Number(line.getAttribute('y2'));
         const t = Math.min(prog, 4) / 4;
         txt.setAttribute('x', x1 + (x2-x1) * t);
         txt.setAttribute('y', y1 + (y2-y1) * t - 4);
         txt.textContent = (deltaY[i] * P3_W2[i][j]).toFixed(3);
         txt.style.display = '';
      } else if (txt) {
         txt.style.display = 'none';
      }
    }

    for (let j = 0; j < 4; j++) for (let k = 0; k < 3; k++) {
      const line = el.p4LinesW1[j][k];
      const txt = el.p4TextW1[j][k];
      const prog = st.hFlowProgress[j];
      let cls = 'm2-p4-line';
      if (prog > k) cls += ' arrived';
      else if (prog > 0) cls += ' flowing';
      line.setAttribute('class', cls);
      line.setAttribute('stroke-width', 1 + Math.abs(curW1[j][k]) * 1.5);

      if (prog > 0 && txt) {
         const x1 = Number(line.getAttribute('x1')), y1 = Number(line.getAttribute('y1'));
         const x2 = Number(line.getAttribute('x2')), y2 = Number(line.getAttribute('y2'));
         const t = Math.min(prog, 3) / 3;
         txt.setAttribute('x', x1 + (x2-x1) * t);
         txt.setAttribute('y', y1 + (y2-y1) * t - 4);
         txt.textContent = (deltaH[j] * P3_W1[j][k]).toFixed(3);
         txt.style.display = '';
      } else if (txt) {
         txt.style.display = 'none';
      }
    }

    el.p4Hint.innerHTML = buildP4Formula(idx, vals);
    updateP4Controls();
    
    const t = P4_TICKS[idx];
    if (t && t.type === 'update_demo' && geoIter === 0) {
      startGeoAnim();
    }
  }

  function updateP4Controls() {
    const last = P4_TICKS.length - 1, idx = P4.tickIdx;
    if (el.p4PlayBtn) el.p4PlayBtn.textContent = P4.playing ? '■ 暂停' : '▶ 播放';
    if (el.p4PrevBtn) el.p4PrevBtn.disabled = P4.playing || idx <= -1;
    if (el.p4NextBtn) el.p4NextBtn.disabled = P4.playing || idx >= last;
    if (el.p4SpeedBtns) el.p4SpeedBtns.forEach(b => b.classList.toggle('is-active', Number(b.dataset.speed) === P4.speed));
  }
  
  function p4TickNext() {
    const last = P4_TICKS.length - 1;
    if (P4.tickIdx >= last) { P4.playing = false; updateP4Controls(); return; }
    
    const nextTick = P4_TICKS[P4.tickIdx + 1];
    if (nextTick.type === 'update_done') {
      const vals = computeP4Values();
      P4.updatedW2 = P3_W2.map((row, i) => row.map((w, j) => w - S.p4_eta * vals.gradW2[i][j]));
      P4.updatedW1 = P3_W1.map((row, j) => row.map((w, k) => w - S.p4_eta * vals.gradW1[j][k]));
      geoW = P4.updatedW2[0][0];
      drawGeoFrame();
    }

    renderP4AtTick(P4.tickIdx + 1);
    if (P4.playing) {
      if (P4.tickIdx >= last) { P4.playing = false; updateP4Controls(); }
      else P4.timer = setTimeout(p4TickNext, P4.interval);
    }
  }

  function p4TogglePlay() {
    if (P4.playing) { clearTimeout(P4.timer); P4.timer = null; P4.playing = false; updateP4Controls(); return; }
    if (P4.tickIdx >= P4_TICKS.length - 1) renderP4AtTick(-1);
    P4.playing = true; updateP4Controls();
    P4.timer = setTimeout(p4TickNext, P4.interval);
  }
  
  function p4Next() { 
    if (P4.tickIdx >= P4_TICKS.length - 1) return;
    const nextTick = P4_TICKS[P4.tickIdx + 1];
    if (nextTick.type === 'update_done') {
      const vals = computeP4Values();
      P4.updatedW2 = P3_W2.map((row, i) => row.map((w, j) => w - S.p4_eta * vals.gradW2[i][j]));
      P4.updatedW1 = P3_W1.map((row, j) => row.map((w, k) => w - S.p4_eta * vals.gradW1[j][k]));
      geoW = P4.updatedW2[0][0];
      drawGeoFrame();
    }
    renderP4AtTick(P4.tickIdx + 1); 
  }

  function p4Prev() { 
    if (P4.tickIdx > -1) renderP4AtTick(P4.tickIdx - 1); 
  }

  function p4Reset() {
    if (P4.timer) { clearTimeout(P4.timer); P4.timer = null; }
    P4.playing = false;
    P4.updatedW1 = null;
    P4.updatedW2 = null;
    resetGeo();
    renderP4AtTick(-1);
  }

  function p4SetSpeed(sp) {
    P4.speed = sp; P4.interval = Math.round(500 / sp);
    if (P4.playing) { clearTimeout(P4.timer); P4.timer = setTimeout(p4TickNext, P4.interval); }
    updateP4Controls();
  }

  function render() {
    if (S.tab === 'part1') renderScene1();
    else if (S.tab === 'part2') renderScene2();
  }

  /* =========================================================
   *  Lab 注册
   * ========================================================= */
  Lab.register({
    id: 'ch3-m2', index: '01',
    title: '前馈传播与空间变换',
    desc: '从造一个神经元，到让网络学会学习 —— 一条龙看懂神经网络。',

    template: function () {
      return `
        <div class="m2-wrap">
          <div class="m2-tabs">
            <button class="m2-tab-btn is-active" data-tab="part1">① 造神经元</button>
            <button class="m2-tab-btn"           data-tab="part2">② 并排放成矩阵</button>
            <button class="m2-tab-btn"           data-tab="part3">③ 串起来前向传播</button>
            <button class="m2-tab-btn"           data-tab="part4">④ 让网络学会学习</button>
          </div>

          <div class="m2-storyline">
            <div class="m2-story-node is-active" data-story="1">
              <div class="m2-story-num">1</div>
              <div class="m2-story-label">造神经元</div>
              <div class="m2-story-sub">点积 + 激活</div>
            </div>
            <div class="m2-story-arrow">→</div>
            <div class="m2-story-node" data-story="2">
              <div class="m2-story-num">2</div>
              <div class="m2-story-label">并排放</div>
              <div class="m2-story-sub">n 个神经元 = 矩阵 W</div>
            </div>
            <div class="m2-story-arrow">→</div>
            <div class="m2-story-node" data-story="3">
              <div class="m2-story-num">3</div>
              <div class="m2-story-label">串起来</div>
              <div class="m2-story-sub">多层堆叠 = 网络</div>
            </div>
            <div class="m2-story-arrow">→</div>
            <div class="m2-story-node" data-story="4">
              <div class="m2-story-num">4</div>
              <div class="m2-story-label">会学习</div>
              <div class="m2-story-sub">误差倒流 + 更新</div>
            </div>
          </div>

          <!-- ========== 场景 1 ========== -->
          <div class="m2-scene" data-role="scene1">
            <div class="m2-sidebar">
              <div class="m2-nav-card">
                <div class="m2-nav-prev">起点</div>
                <div class="m2-nav-cur">📍 第 1 步：先造出一个神经元</div>
                <div class="m2-nav-next">下一站 → 把 4 个这样的神经元并排放，就得到矩阵 W</div>
              </div>
              <div class="m2-card">
                <div class="m2-title">① 点积：坐标的直接映射</div>
                <div class="m2-algebra">
                  <div class="m2-line"><span>x 代数点积</span><span data-role="p1-math-proj"></span></div>
                  <div class="m2-line"><span>投影长度 L</span><b data-role="p1-math-proj-len" style="color:#c0392b;"></b></div>
                  <div class="m2-line"><span>权重长度 |w|</span><b data-role="p1-math-len-w"   style="color:#c0392b;"></b></div>
                  <div class="m2-line m2-highlight"><span>z = L × |w|</span><span data-role="p1-math-z" style="color:#c0392b;"></span></div>
                </div>
              </div>
              <div class="m2-card">
                <div class="m2-title">② 偏置：平移门槛</div>
                <div class="m2-algebra"><div class="m2-line"><span>z + b =</span><span data-role="p1-math-zb"></span></div></div>
                <div class="m2-stepper-group" style="margin-top:12px;">
                  <span class="m2-label">调节偏置 b</span>
                  <div class="stepper m2-stepper" data-role="p1-step-b"><button data-step="-1">-</button><span data-role="p1-b-val">-5.0</span><button data-step="1">+</button></div>
                </div>
              </div>
              <div class="m2-card">
                <div class="m2-title" data-role="p1-act-title">③ 激活：非线性阀门</div>
                <div class="p1-act-btns" data-role="p1-act-btns">
                  <button class="p1-act-btn is-active" data-act="sigmoid">Sigmoid</button>
                  <button class="p1-act-btn"          data-act="tanh">Tanh</button>
                  <button class="p1-act-btn"          data-act="relu">ReLU</button>
                  <button class="p1-act-btn"          data-act="linear">Linear</button>
                </div>
                <div class="m2-algebra" style="margin-top:10px;"><div class="m2-line"><span>a = f(z+b)</span><b data-role="p1-math-a"></b></div></div>
                <div class="p1-act-info">
                  <div class="p1-act-info-row"><span class="p1-act-info-label">当前</span><b data-role="p1-act-name" style="color:#27ae60;">Sigmoid</b></div>
                  <div class="p1-act-info-row"><span class="p1-act-info-label">输出范围</span><code data-role="p1-act-range">[0, 1]</code></div>
                  <div class="p1-act-info-row"><span class="p1-act-info-label">梯度</span><code data-role="p1-act-grad">最大 0.25</code></div>
                  <div class="p1-act-info-row p1-act-info-pros"><span class="p1-act-info-label">✓ 优点</span><span data-role="p1-act-pros"></span></div>
                  <div class="p1-act-info-row p1-act-info-cons"><span class="p1-act-info-label">⚠ 缺点</span><span data-role="p1-act-cons"></span></div>
                  <div class="p1-act-info-row p1-act-info-use"><span class="p1-act-info-label">📍 适用</span><span data-role="p1-act-use"></span></div>
                </div>
                <div class="m2-desc" style="border-left-color:#c0392b; background:#fff5f5; margin-top:10px;">
                  <b>为什么要激活？</b>没有它，多层等于一层。永远只是 Wx + b。
                  <div style="margin-top:6px; display:flex; align-items:center; gap:8px;">
                    <svg width="80" height="60" viewBox="0 0 80 60" style="background:#fff; border:1px solid #eee; border-radius:4px;">
                      <circle cx="40" cy="30" r="15" fill="none" stroke="#2980b9" stroke-width="1.2"/>
                      <circle cx="40" cy="30" r="2.5" fill="#c0392b"/>
                      <circle cx="20" cy="15" r="2.5" fill="#2980b9"/>
                      <circle cx="60" cy="45" r="2.5" fill="#2980b9"/>
                      <circle cx="20" cy="45" r="2.5" fill="#2980b9"/>
                      <circle cx="60" cy="15" r="2.5" fill="#2980b9"/>
                      <line x1="8" y1="52" x2="72" y2="8" stroke="#c0392b" stroke-width="1.2" stroke-dasharray="2 2"/>
                    </svg>
                    <div style="font-size:11.5px; color:#666; line-height:1.4;">直线 <b style="color:#c0392b;">永远分不开</b> 环形点<br>需 <b style="color:#2980b9;">非线性边界</b></div>
                  </div>
                </div>
              </div>
              <div class="m2-card">
                <div class="m2-title">调节特征方向 w</div>
                <div class="m2-stepper-group"><span class="m2-label">w1 (x方向)</span><div class="stepper m2-stepper" data-role="p1-step-w1"><button data-step="-1">-</button><span data-role="p1-w1-val">1.5</span><button data-step="1">+</button></div></div>
                <div class="m2-stepper-group"><span class="m2-label">w2 (y方向)</span><div class="stepper m2-stepper" data-role="p1-step-w2"><button data-step="-1">-</button><span data-role="p1-w2-val">1.0</span><button data-step="1">+</button></div></div>
              </div>
            </div>
            <div class="m2-canvas-box">
              <svg class="m2-svg" viewBox="0 0 800 500" preserveAspectRatio="xMidYMid meet">
                <text x="30"  y="30"  class="m2-zone-label">① 向量空间 (投影)</text>
                <text x="30"  y="270" class="m2-zone-label">② 一维数轴 (平移)</text>
                <text x="580" y="270" class="m2-zone-label">③ 激活曲线 (压缩)</text>
                <text x="760" y="248" class="m2-svg-label" fill="#aaa" text-anchor="end" font-size="10" data-role="p1-act-range-label">[0, 1]</text>
                <g data-role="p1-grid"></g>
                <line data-role="p1-axis-line" stroke="#888" stroke-width="2"/>
                <g data-role="p1-axis-ticks"></g>
                <line data-role="p1-data-flow-line" stroke="#bdc3c7" stroke-width="2"   stroke-dasharray="5 5"/>
                <line data-role="p1-act-input-line" stroke="#bdc3c7" stroke-width="1.5" stroke-dasharray="4 4"/>
                <rect data-role="p1-sat-left"  x="600" y="260" width="40" height="100" fill="#fee2e2" opacity="0.6" style="display:none;"/>
                <rect data-role="p1-sat-right" x="720" y="260" width="40" height="100" fill="#fee2e2" opacity="0.6" style="display:none;"/>
                <text data-role="p1-sat-left-label"  x="620" y="270" font-size="9" fill="#c0392b" text-anchor="middle" style="display:none;">梯度≈0</text>
                <text data-role="p1-sat-right-label" x="740" y="270" font-size="9" fill="#c0392b" text-anchor="middle" style="display:none;">梯度≈0</text>
                <line   data-role="p1-vec-w"     stroke="#c0392b" stroke-width="2.5"/>
                <polygon data-role="p1-arrow-w"  fill="#c0392b"/>
                <text   data-role="p1-label-w"   class="m2-svg-label" fill="#c0392b" font-weight="bold">w</text>
                <line data-role="p1-highlight-w" stroke="#c0392b" stroke-width="6" opacity="0.3" stroke-linecap="round"/>
                <line   data-role="p1-vec-x"     stroke="#2980b9" stroke-width="2.5"/>
                <polygon data-role="p1-arrow-x"  fill="#2980b9"/>
                <text   data-role="p1-label-x"   class="m2-svg-label" fill="#2980b9" font-weight="bold">x</text>
                <line data-role="p1-proj-line"  stroke="#95a5a6" stroke-width="1.5" stroke-dasharray="4 4"/>
                <text data-role="p1-label-proj" class="m2-svg-label" fill="#95a5a6" text-anchor="middle">投影</text>
                <circle data-role="p1-dot-z" r="6" fill="#c0392b" stroke="#fff" stroke-width="2"/>
                <text   data-role="p1-label-z" class="m2-svg-label" fill="#c0392b" text-anchor="middle">z</text>
                <g data-role="p1-bias-group" style="display:none;">
                  <line   data-role="p1-bias-arrow"       stroke="#27ae60" stroke-width="2"/>
                  <polygon data-role="p1-bias-arrow-head" fill="#27ae60"/>
                  <circle  data-role="p1-dot-zb"  r="6" fill="#27ae60" stroke="#fff" stroke-width="2"/>
                  <text    data-role="p1-label-zb" class="m2-svg-label" fill="#27ae60" text-anchor="middle">z+b</text>
                </g>
                <g>
                  <line x1="680" y1="360" x2="680" y2="240" stroke="#ccc" stroke-width="1.5"/>
                  <line x1="600" y1="360" x2="760" y2="360" stroke="#ccc" stroke-width="1.5"/>
                  <line x1="600" y1="310" x2="760" y2="310" stroke="#e0e0e0" stroke-dasharray="2 2"/>
                  <text x="595" y="314" class="m2-svg-label" fill="#aaa" text-anchor="end" font-size="10">0.5</text>
                  <path   data-role="p1-act-curve"     fill="none" stroke="#95a5a6" stroke-width="2.5"/>
                  <line   data-role="p1-act-in-vline"  stroke="#95a5a6" stroke-width="1.5" stroke-dasharray="3 3"/>
                  <line   data-role="p1-act-out-hline" stroke="#95a5a6" stroke-width="1.5" stroke-dasharray="3 3"/>
                  <circle data-role="p1-act-dot" r="5" fill="#95a5a6" stroke="#fff" stroke-width="2"/>
                  <text   data-role="p1-label-a" class="m2-svg-label" fill="#95a5a6" text-anchor="end">a</text>
                </g>
              </svg>
            </div>
          </div>

          <!-- ========== 场景 2 ========== -->
          <div class="m2-scene" data-role="scene2" style="display:none;">
            <div class="m2-sidebar">
              <div class="m2-nav-card">
                <div class="m2-nav-prev">← 上一站：一个神经元的点积 + 激活</div>
                <div class="m2-nav-cur">📍 第 2 步：把 n 个神经元并排放 = 矩阵 W</div>
                <div class="m2-nav-next">下一站 → 把 W 一层层串起来，就得到前馈网络</div>
              </div>
              <div class="m2-card">
                <div class="m2-title">矩阵是空间的“形变扳手”</div>
                <div class="m2-formula-display">z = <span class="m2-mat">W</span> x + <span class="m2-vec">b</span></div>
                <div class="m2-mat-container"><span class="m2-mat-label">W =</span><div class="m2-matrix" data-role="p2-mat-w"></div></div>
                <div class="m2-desc"><b>W 的每一行</b>都是一个神经元的方向 w。<b>矩阵 W</b> 就是「4 个神经元并排」。<br>W 把平面拉伸 / 剪切；b 平移空间。</div>
              </div>
              <div class="m2-card m2-det-card">
                <div class="m2-title">矩阵好坏衡量：行列式 det(W)</div>
                <div class="m2-det-display"><div><span class="m2-det-label">det(W) =</span><b data-role="p2-det-val" style="color:#27ae60;">1.00</b></div><div data-role="p2-det-level" class="m2-det-level">✓ 健康</div></div>
                <div class="m2-det-msg" data-role="p2-det-msg"></div>
                <div class="m2-det-scale">
                  <div class="m2-det-scale-bar">
                    <div class="m2-det-zone" style="flex:0.5; background:#fee2e2; color:#c0392b;">塌缩</div>
                    <div class="m2-det-zone" style="flex:2;   background:#fef3c7; color:#e67e22;">压缩</div>
                    <div class="m2-det-zone" style="flex:4;   background:#d1fae5; color:#27ae60;">健康</div>
                    <div class="m2-det-zone" style="flex:2;   background:#ede9fe; color:#8e44ad;">拉伸</div>
                  </div>
                  <div class="m2-det-scale-labels"><span>0</span><span>0.5</span><span>2</span><span>∞</span></div>
                </div>
                <div class="m2-det-formula">det(W) = w₁₁·w₂₂ − w₁₂·w₂₁</div>
              </div>
              <div class="m2-card"><div class="m2-title">代数结果（行乘列）</div><div class="m2-algebra" data-role="p2-math-calc"></div></div>
              <div class="m2-card">
                <div class="m2-title">调节权重矩阵 W</div>
                <div class="m2-stepper-group"><span class="m2-label">w₁₁</span><div class="stepper m2-stepper" data-role="p2-step-w11"><button data-step="-1">-</button><span data-role="p2-w11-val">1.0</span><button data-step="1">+</button></div></div>
                <div class="m2-stepper-group"><span class="m2-label">w₁₂</span><div class="stepper m2-stepper" data-role="p2-step-w12"><button data-step="-1">-</button><span data-role="p2-w12-val">0.5</span><button data-step="1">+</button></div></div>
                <div class="m2-stepper-group"><span class="m2-label">w₂₁</span><div class="stepper m2-stepper" data-role="p2-step-w21"><button data-step="-1">-</button><span data-role="p2-w21-val">0.5</span><button data-step="1">+</button></div></div>
                <div class="m2-stepper-group"><span class="m2-label">w₂₂</span><div class="stepper m2-stepper" data-role="p2-step-w22"><button data-step="-1">-</button><span data-role="p2-w22-val">1.0</span><button data-step="1">+</button></div></div>
              </div>
              <div class="m2-card">
                <div class="m2-title">调节偏置 b (平移)</div>
                <div class="m2-stepper-group"><span class="m2-label">b₁</span><div class="stepper m2-stepper" data-role="p2-step-b1"><button data-step="-1">-</button><span data-role="p2-b1-val">0.0</span><button data-step="1">+</button></div></div>
                <div class="m2-stepper-group"><span class="m2-label">b₂</span><div class="stepper m2-stepper" data-role="p2-step-b2"><button data-step="-1">-</button><span data-role="p2-b2-val">0.0</span><button data-step="1">+</button></div></div>
              </div>

              <div class="m2-card m2-map-card">
                <div class="m2-title">神经元 → 行 → 矩阵</div>
                <div class="m2-desc">
                  一个神经元有自己的 <b>w</b>。把 M 个神经元并排，每个 w 占一行，就是矩阵 W。
                </div>
                <div class="m2-map-grid" data-role="p2-map-box"></div>
                <div class="m2-stepper-group">
                  <span class="m2-label">高亮神经元</span>
                  <div class="m2-stepper">
                    <button data-role="p2-map-row-dec" type="button">−</button>
                    <span data-role="p2-map-row-val">1</span>
                    <button data-role="p2-map-row-inc" type="button">+</button>
                  </div>
                </div>
                <div class="m2-algebra" data-role="p2-map-calc"></div>
              </div>
            </div>
            <div class="m2-canvas-box">
              <svg class="m2-svg" viewBox="0 0 800 500" preserveAspectRatio="xMidYMid meet">
                <g data-role="p2-orig-grid"></g>
                <g data-role="p2-trans-grid"></g>
                <line data-role="p2-vec-x" stroke="#27ae60" stroke-width="3"/>
                <text data-role="p2-label-x" class="m2-svg-label" fill="#27ae60" font-weight="bold">x</text>
                <circle data-role="p2-bias-point" r="5" fill="#c0392b"/>
                <line   data-role="p2-vec-z" stroke="#c0392b" stroke-width="3" stroke-dasharray="6 3"/>
                <text   data-role="p2-label-z" class="m2-svg-label" fill="#c0392b" font-weight="bold">z</text>
              </svg>
            </div>
          </div>

          <!-- ========== 场景 3 ========== -->
          <div class="m2-scene" data-role="scene3" style="display:none;">
            <div class="m2-sidebar">
              <div class="m2-nav-card">
                <div class="m2-nav-prev">← 上一站：n 个神经元并排 = 矩阵</div>
                <div class="m2-nav-cur">📍 第 3 步：把矩阵串起来 = 前馈网络</div>
                <div class="m2-nav-next">下一站 → 怎么把 loss 降下去</div>
              </div>

              <div class="m2-card m2-layer-card">
                <div class="m2-title">一层 = 矩阵 + 激活</div>
                <div class="m2-desc">神经元并排成矩阵 W，再对结果做一次逐元素激活 σ，就构成一层。两层首尾相接，就是前馈网络。</div>
                <div class="m2-layer-diagram">
                  <div class="m2-layer-block">
                    <div class="m2-layer-tag">输入</div>
                    <div class="m2-layer-box">x<sup>[3]</sup></div>
                  </div>
                  <div class="m2-layer-arrow">→</div>
                  <div class="m2-layer-block">
                    <div class="m2-layer-tag">第 1 层</div>
                    <div class="m2-layer-box">z₁ = W₁x + b₁<br>h = σ(z₁)</div>
                  </div>
                  <div class="m2-layer-arrow">→</div>
                  <div class="m2-layer-block">
                    <div class="m2-layer-tag">第 2 层</div>
                    <div class="m2-layer-box">z₂ = W₂h + b₂<br>y = σ(z₂)</div>
                  </div>
                  <div class="m2-layer-arrow">→</div>
                  <div class="m2-layer-block">
                    <div class="m2-layer-tag">输出</div>
                    <div class="m2-layer-box">y<sup>[2]</sup></div>
                  </div>
                </div>
                <div class="m2-layer-note">矩阵负责“混合”，激活负责“弯折”。缺一不可。</div>
              </div>

              <div class="m2-card">
                <div class="m2-title">选择激活函数</div>
                <div class="p3-act-btns" data-role="p3-act-btns">
                  <button class="p3-act-btn is-active" data-p3-act="sigmoid">Sigmoid</button>
                  <button class="p3-act-btn"          data-p3-act="tanh">Tanh</button>
                  <button class="p3-act-btn"          data-p3-act="relu">ReLU</button>
                  <button class="p3-act-btn"          data-p3-act="linear">Linear</button>
                </div>
              </div>
              <div class="m2-card"><div class="m2-title">神经元健康检查</div><div class="p3-warn-box" data-role="p3-warn-box"></div></div>
              <div class="m2-card"><div class="m2-title">当前计算</div><div data-role="p3-formula" class="p3-formula"></div></div>
              <div class="m2-card">
                <div class="m2-title">调节输入 x</div>
                <div class="m2-stepper-group"><span class="m2-label">x₁</span><div class="stepper m2-stepper" data-role="p3-step-x1"><button data-step="-1">-</button><span data-role="p3-x1-val">1.0</span><button data-step="1">+</button></div></div>
                <div class="m2-stepper-group"><span class="m2-label">x₂</span><div class="stepper m2-stepper" data-role="p3-step-x2"><button data-step="-1">-</button><span data-role="p3-x2-val">-0.5</span><button data-step="1">+</button></div></div>
                <div class="m2-stepper-group"><span class="m2-label">x₃</span><div class="stepper m2-stepper" data-role="p3-step-x3"><button data-step="-1">-</button><span data-role="p3-x3-val">1.0</span><button data-step="1">+</button></div></div>
              </div>
              <div class="m2-card">
                <div class="m2-title">播放控制</div>
                <div class="p3-controls">
                  <button data-role="p3-prev" class="m2-btn-secondary">◀ 上一步</button>
                  <button data-role="p3-play" class="m2-btn-primary">▶ 播放</button>
                  <button data-role="p3-next" class="m2-btn-secondary">下一步 ▶</button>
                </div>
                <div class="p3-controls" style="margin-top:8px;">
                  <span class="m2-label" style="min-width:38px;">速度</span>
                  <div class="p3-speed-group">
                    <button data-speed="0.5" class="p3-speed-btn">慢</button>
                    <button data-speed="1"   class="p3-speed-btn is-active">中</button>
                    <button data-speed="2"   class="p3-speed-btn">快</button>
                  </div>
                  <button data-role="p3-reset" class="m2-btn-secondary">↺</button>
                </div>
              </div>
            </div>
            <div class="m2-canvas-box">
              <svg class="m2-svg" viewBox="0 0 800 500" preserveAspectRatio="xMidYMid meet">
                <g data-role="p3-svg-content"></g>
              </svg>
            </div>
          </div>

          <!-- ========== 场景 4 ========== -->
          <div class="m2-scene" data-role="scene4" style="display:none;">
            <div class="m2-sidebar">
              <div class="m2-nav-card">
                <div class="m2-nav-prev">← 上一站：前向传播</div>
                <div class="m2-nav-cur">📍 第 4 步：用误差教网络「学习」</div>
                <div class="m2-nav-next">终点 · 循环往复，直到 Loss 趋于 0</div>
              </div>

              <div class="m2-card" style="border-left:3px solid #c0392b;">
                <div class="m2-title">W₁ / W₂ 从哪来？</div>
                <div class="m2-desc" style="background:#fff5f5; border-left-color:#c0392b; font-size:12px;">
                  <b style="color:#c0392b;">不是天上掉的，是“初始化”出来的。</b><br>
                  <b>为什么必须随机？</b>如果全为 0，所有神经元输出相同，梯度相同，永远学不到不同特征（对称性破缺）。<br>
                  <b>两种经典初始化：</b>
                  <table class="p4-init-table" style="width:100%; margin-top:6px; font-size:11px; text-align:left;">
                    <tr><th>方法</th><th>适合激活</th><th>方差</th></tr>
                    <tr><td>Xavier</td><td>sigmoid / tanh</td><td>1/n_in</td></tr>
                    <tr><td>He (Kaiming)</td><td>ReLU</td><td>2/n_in</td></tr>
                  </table>
                  <span style="color:#888; margin-top:6px; display:block;">当前 W 就是从各自分布随机采样后固定下来的。</span>
                </div>
              </div>

              <!-- 真实 Loss 切片 -->
              <div class="m2-card" style="border-left:3px solid #2980b9;">
                <div class="m2-title">真实的梯度下降：Loss 随 W₂[0][0] 的变化</div>
                
                <div class="p4-eta-geo">
                  <svg viewBox="0 0 240 100" class="p4-eta-svg">
                    <defs>
                      <marker id="p4-geo-arrow-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                        <path d="M 0 0 L 10 5 L 0 10 z" fill="#27ae60"/>
                      </marker>
                    </defs>
                    <path id="p4-geo-path" d="" fill="none" stroke="#ccc" stroke-width="2"/>
                    <text x="120" y="95" font-size="10" fill="#888" text-anchor="middle" font-family="Consolas, monospace">固定其他参数，只动 W₂[0][0]</text>
                    
                    <circle id="p4-geo-ball" cx="20" cy="26" r="6" fill="#c0392b" style="transition: all 0.3s ease-out;"/>
                    <line id="p4-geo-arrow" x1="20" y1="26" x2="30" y2="30" stroke="#27ae60" stroke-width="2" marker-end="url(#p4-geo-arrow-head)" style="transition: all 0.3s ease-out;"/>
                  </svg>
                </div>

                <div class="m2-stepper-group" style="margin-bottom:12px; flex-direction:column; align-items:flex-start; gap:6px;">
                  <span class="m2-label" id="p4-geo-eta-label" style="font-family:Consolas, monospace; font-size:11px;"></span>
                  <div style="display:flex; justify-content:space-between; width:100%;">
                    <span class="m2-label">学习率 η</span>
                    <div class="m2-stepper">
                      <button data-role="p4-eta-dec" type="button">−</button>
                      <span data-role="p4-eta-val">0.10</span>
                      <button data-role="p4-eta-inc" type="button">+</button>
                    </div>
                  </div>
                </div>

                <div class="m2-desc" style="background:#f0f8ff; border-left-color:#2980b9; font-size:11.5px; padding:8px;">
                  <b>这不是假想的抛物线</b>，是固定其他参数、实测出的真实 Loss 曲面切片。<br><br>
                  <b>梯度</b> = 脚下的坡度（画成绿色箭头）。<br>
                  <b>η</b> = 步子大小。点击加减号看看，起点是不动的，但下一步会跨多远。
                </div>
              </div>
              
              <!-- 考点提炼卡 -->
              <div class="m2-card" style="border-left:3px solid #27ae60;">
                <div class="m2-title">📐 考试规律总结（前向 vs 反向）</div>
                <div class="m2-desc" style="background:#f0fdf4; border-left-color:#27ae60; font-size:12px;">
                  <b>前向传播（矩阵 × 向量 + 激活）：</b><br>
                  z<sup>[l]</sup> = W<sup>[l]</sup> · a<sup>[l-1]</sup> + b<sup>[l]</sup><br>
                  a<sup>[l]</sup> = σ(z<sup>[l]</sup>)<br><br>
                  
                  <b>反向传播（链式法则）：</b><br>
                  1. 算误差：δ<sup>[L]</sup> = ∇<sub>a</sub>L ⊙ σ'(z<sup>[L]</sup>)<br>
                  2. 传误差：δ<sup>[l]</sup> = (W<sup>[l+1]</sup>)ᵀ · δ<sup>[l+1]</sup> ⊙ σ'(z<sup>[l]</sup>)<br>
                  3. 算梯度：∂L/∂W<sup>[l]</sup> = δ<sup>[l]</sup> · (a<sup>[l-1]</sup>)ᵀ<br><br>
                  
                  <b style="color:#c0392b;">⚠️ 易错点：</b><br>
                  - 反向传播算误差时，<b>必须乘激活导数</b> σ'(z)。<br>
                  - 梯度矩阵的维度与权重矩阵 W 完全一致。<br>
                  - 转置符号 (ᵀ) 容易漏：向前传用 W，向后传用 Wᵀ。
                </div>
              </div>

              <div class="m2-card"><div class="m2-title">当前计算（核心机制）</div><div data-role="p4-hint" class="p4-formula" style="min-height:220px;"></div></div>
              
              <div class="m2-card">
                <div class="m2-title">播放控制</div>
                <div class="p3-controls">
                  <button data-role="p4-prev" class="m2-btn-secondary">◀ 上一步</button>
                  <button data-role="p4-play" class="m2-btn-primary">▶ 播放</button>
                  <button data-role="p4-next" class="m2-btn-secondary">下一步 ▶</button>
                </div>
                <div class="p3-controls" style="margin-top:8px;">
                  <span class="m2-label" style="min-width:38px;">速度</span>
                  <div class="p3-speed-group">
                    <button data-speed="0.5" class="p3-speed-btn">慢</button>
                    <button data-speed="1"   class="p3-speed-btn is-active">中</button>
                    <button data-speed="2"   class="p3-speed-btn">快</button>
                  </div>
                  <button data-role="p4-reset" class="m2-btn-secondary">↺</button>
                </div>
              </div>
            </div>
            <div class="m2-canvas-box">
              <svg class="m2-svg" viewBox="0 0 800 500" preserveAspectRatio="xMidYMid meet">
                <g data-role="p4-svg-content"></g>
              </svg>
            </div>
          </div>
        </div>
      `;
    },

    init: function (root) {
      el.root = root;
      el.tabs = Array.from(root.querySelectorAll('.m2-tab-btn'));
      el.scene1 = root.querySelector('[data-role="scene1"]');
      el.scene2 = root.querySelector('[data-role="scene2"]');
      el.scene3 = root.querySelector('[data-role="scene3"]');
      el.scene4 = root.querySelector('[data-role="scene4"]');
      el.storyNodes = Array.from(root.querySelectorAll('.m2-story-node'));

      /* 场景 1 */
      [
        ['p1MathProj','p1-math-proj'],['p1MathLenW','p1-math-len-w'],['p1MathProjLen','p1-math-proj-len'],
        ['p1MathZ','p1-math-z'],['p1MathZb','p1-math-zb'],['p1MathA','p1-math-a'],
        ['p1ActTitle','p1-act-title'],['p1Grid','p1-grid'],['p1AxisLine','p1-axis-line'],
        ['p1AxisTicks','p1-axis-ticks'],['p1DataFlowLine','p1-data-flow-line'],
        ['p1ActInputLine','p1-act-input-line'],['p1DotZ','p1-dot-z'],['p1LabelZ','p1-label-z'],
        ['p1VecW','p1-vec-w'],['p1ArrowW','p1-arrow-w'],['p1LabelW','p1-label-w'],
        ['p1ProjHighlight','p1-highlight-w'],['p1VecX','p1-vec-x'],['p1ArrowX','p1-arrow-x'],
        ['p1LabelX','p1-label-x'],['p1ProjLine','p1-proj-line'],['p1LabelProj','p1-label-proj'],
        ['p1BiasGroup','p1-bias-group'],['p1BiasArrow','p1-bias-arrow'],
        ['p1BiasArrowHead','p1-bias-arrow-head'],['p1DotZb','p1-dot-zb'],['p1LabelZb','p1-label-zb'],
        ['p1ActCurve','p1-act-curve'],['p1ActInVLine','p1-act-in-vline'],
        ['p1ActOutHLine','p1-act-out-hline'],['p1ActDot','p1-act-dot'],['p1LabelA','p1-label-a'],
        ['p1ActName','p1-act-name'],['p1ActRange','p1-act-range'],['p1ActGrad','p1-act-grad'],
        ['p1ActPros','p1-act-pros'],['p1ActCons','p1-act-cons'],['p1ActUse','p1-act-use'],
        ['p1SatLeft','p1-sat-left'],['p1SatRight','p1-sat-right'],
        ['p1SatLeftLabel','p1-sat-left-label'],['p1SatRightLabel','p1-sat-right-label'],
        ['p1ActRangeLabel','p1-act-range-label']
      ].forEach(([k, r]) => el[k] = root.querySelector(`[data-role="${r}"]`));
      el.p1ActBtns = Array.from(root.querySelectorAll('.p1-act-btn'));

      /* 场景 2 */
      el.p2MatW = root.querySelector('[data-role="p2-mat-w"]');
      el.p2MathCalc = root.querySelector('[data-role="p2-math-calc"]');
      el.p2OrigGrid = root.querySelector('[data-role="p2-orig-grid"]');
      el.p2TransGrid = root.querySelector('[data-role="p2-trans-grid"]');
      el.p2VecX = root.querySelector('[data-role="p2-vec-x"]');
      el.p2LabelX = root.querySelector('[data-role="p2-label-x"]');
      el.p2VecZ = root.querySelector('[data-role="p2-vec-z"]');
      el.p2LabelZ = root.querySelector('[data-role="p2-label-z"]');
      el.p2BiasPoint = root.querySelector('[data-role="p2-bias-point"]');
      el.p2DetVal = root.querySelector('[data-role="p2-det-val"]');
      el.p2DetLevel = root.querySelector('[data-role="p2-det-level"]');
      el.p2DetMsg = root.querySelector('[data-role="p2-det-msg"]');
      el.p2MapBox = root.querySelector('[data-role="p2-map-box"]');
      el.p2MapRowVal = root.querySelector('[data-role="p2-map-row-val"]');
      el.p2MapCalc = root.querySelector('[data-role="p2-map-calc"]');

      /* 场景 3 */
      el.p3SvgContent = root.querySelector('[data-role="p3-svg-content"]');
      el.p3SvgContent.innerHTML = buildScene3SvgHtml();
      cacheScene3Elements();
      el.p3Formula = root.querySelector('[data-role="p3-formula"]');
      el.p3WarnBox = root.querySelector('[data-role="p3-warn-box"]');
      el.p3PlayBtn = root.querySelector('[data-role="p3-play"]');
      el.p3PrevBtn = root.querySelector('[data-role="p3-prev"]');
      el.p3NextBtn = root.querySelector('[data-role="p3-next"]');
      el.p3ResetBtn = root.querySelector('[data-role="p3-reset"]');
      el.p3ActBtns = Array.from(root.querySelectorAll('.p3-act-btn'));
      el.p3StepButtons = [
        ...root.querySelectorAll('[data-role="p3-step-x1"] button'),
        ...root.querySelectorAll('[data-role="p3-step-x2"] button'),
        ...root.querySelectorAll('[data-role="p3-step-x3"] button')
      ];
      const scene3El = root.querySelector('[data-role="scene3"]');
      el.p3SpeedBtns = Array.from(scene3El.querySelectorAll('.p3-speed-btn'));

      /* 场景 4 */
      el.p4SvgContent = root.querySelector('[data-role="p4-svg-content"]');
      el.p4SvgContent.innerHTML = buildScene4SvgHtml();
      cacheScene4Elements();
      el.p4Hint = root.querySelector('[data-role="p4-hint"]');
      el.p4PlayBtn = root.querySelector('[data-role="p4-play"]');
      el.p4PrevBtn = root.querySelector('[data-role="p4-prev"]');
      el.p4NextBtn = root.querySelector('[data-role="p4-next"]');
      el.p4ResetBtn = root.querySelector('[data-role="p4-reset"]');
      el.p4EtaVal = root.querySelector('[data-role="p4-eta-val"]');
      
      el.geoPath = root.querySelector('#p4-geo-path');
      el.geoBall = root.querySelector('#p4-geo-ball');
      el.geoArrow = root.querySelector('#p4-geo-arrow');
      el.geoEtaLabel = root.querySelector('#p4-geo-eta-label');

      const scene4El = root.querySelector('[data-role="scene4"]');
      el.p4SpeedBtns = Array.from(scene4El.querySelectorAll('.p3-speed-btn'));

      renderScene1Grid();
      render();
      renderP2Map();
      renderP3AtTick(-1);
      renderP4AtTick(-1);
      updateStoryline(S.tab);
      
      geoCurve = buildGeoCurve();
      if (el.geoPath) el.geoPath.setAttribute('d', geoCurve.pathD);
      resetGeo();

      /* Tab 切换 */
      el.tabs.forEach(btn => {
        btn.addEventListener('click', e => {
          const tab = e.target.dataset.tab;
          if (!tab || S.tab === tab) return;
          if (S.tab === 'part3' && P3.playing) { clearTimeout(P3.timer); P3.timer = null; P3.playing = false; }
          if (S.tab === 'part4' && P4.playing) { clearTimeout(P4.timer); P4.timer = null; P4.playing = false; }
          S.tab = tab;
          el.tabs.forEach(b => b.classList.toggle('is-active', b === e.target));
          el.scene1.style.display = tab === 'part1' ? 'flex' : 'none';
          el.scene2.style.display = tab === 'part2' ? 'flex' : 'none';
          el.scene3.style.display = tab === 'part3' ? 'flex' : 'none';
          el.scene4.style.display = tab === 'part4' ? 'flex' : 'none';
          updateStoryline(tab);
          if (tab === 'part1' || tab === 'part2') render();
          else if (tab === 'part3') updateP3Controls();
          else if (tab === 'part4') updateP4Controls();
        });
      });

      const bind = (role, getV, setV, step, min, max, onAfter) => {
        const sEl = root.querySelector(`[data-role="${role}"]`);
        if (!sEl) return;
        const vEl = sEl.querySelector('span');
        sEl.addEventListener('click', e => {
          const btn = e.target.closest('[data-step]');
          if (!btn) return;
          let v = getV() + Number(btn.dataset.step) * step;
          v = Math.min(max, Math.max(min, v));
          setV(v);
          vEl.textContent = v.toFixed(1);
          if (onAfter) onAfter(); else render();
        });
        vEl.textContent = getV().toFixed(1);
      };
      bind('p1-step-b',  () => S.p1_b,    v => S.p1_b = v, 0.5, -6, 6);
      bind('p1-step-w1', () => S.p1_w[0], v => S.p1_w[0] = v, 0.5, -2, 2);
      bind('p1-step-w2', () => S.p1_w[1], v => S.p1_w[1] = v, 0.5, -2, 2);
      bind('p2-step-w11', () => S.p2_W[0], v => S.p2_W[0] = v, 0.5, -1.5, 1.5);
      bind('p2-step-w12', () => S.p2_W[1], v => S.p2_W[1] = v, 0.5, -1.5, 1.5);
      bind('p2-step-w21', () => S.p2_W[2], v => S.p2_W[2] = v, 0.5, -1.5, 1.5);
      bind('p2-step-w22', () => S.p2_W[3], v => S.p2_W[3] = v, 0.5, -1.5, 1.5);
      bind('p2-step-b1',  () => S.p2_b[0], v => S.p2_b[0] = v, 0.5, -3, 3);
      bind('p2-step-b2',  () => S.p2_b[1], v => S.p2_b[1] = v, 0.5, -3, 3);
      bind('p3-step-x1', () => S.p3_x[0], v => S.p3_x[0] = v, 0.5, -2, 2, p3Reset);
      bind('p3-step-x2', () => S.p3_x[1], v => S.p3_x[1] = v, 0.5, -2, 2, p3Reset);
      bind('p3-step-x3', () => S.p3_x[2], v => S.p3_x[2] = v, 0.5, -2, 2, p3Reset);

      const mapInc = root.querySelector('[data-role="p2-map-row-inc"]');
      const mapDec = root.querySelector('[data-role="p2-map-row-dec"]');
      if (mapInc) mapInc.addEventListener('click', () => { S.p2_map_row = (S.p2_map_row + 1) % 2; renderP2Map(); });
      if (mapDec) mapDec.addEventListener('click', () => { S.p2_map_row = (S.p2_map_row + 1) % 2; renderP2Map(); });

      // 学习率 eta 绑定
      const etaInc = root.querySelector('[data-role="p4-eta-inc"]');
      const etaDec = root.querySelector('[data-role="p4-eta-dec"]');
      if (etaInc) etaInc.addEventListener('click', () => {
        S.p4_eta = Math.min(0.5, Number((S.p4_eta + 0.05).toFixed(2)));
        el.p4EtaVal.textContent = S.p4_eta.toFixed(2);
        drawGeoFrame();
        if (P4.tickIdx >= 0) renderP4AtTick(P4.tickIdx);
      });
      if (etaDec) etaDec.addEventListener('click', () => {
        S.p4_eta = Math.max(0.01, Number((S.p4_eta - 0.05).toFixed(2)));
        el.p4EtaVal.textContent = S.p4_eta.toFixed(2);
        drawGeoFrame();
        if (P4.tickIdx >= 0) renderP4AtTick(P4.tickIdx);
      });

      el.p1ActBtns.forEach(btn => {
        btn.addEventListener('click', () => {
          S.p1_act = btn.dataset.act;
          renderScene1();
        });
      });

      el.p3PlayBtn.addEventListener('click', p3TogglePlay);
      el.p3PrevBtn.addEventListener('click', p3Prev);
      el.p3NextBtn.addEventListener('click', p3Next);
      el.p3ResetBtn.addEventListener('click', p3Reset);
      el.p3SpeedBtns.forEach(b => b.addEventListener('click', () => p3SetSpeed(Number(b.dataset.speed))));
      el.p3ActBtns.forEach(b => b.addEventListener('click', () => p3SetAct(b.dataset.p3Act)));

      el.p4PlayBtn.addEventListener('click', p4TogglePlay);
      el.p4PrevBtn.addEventListener('click', p4Prev);
      el.p4NextBtn.addEventListener('click', p4Next);
      el.p4ResetBtn.addEventListener('click', p4Reset);
      el.p4SpeedBtns.forEach(b => b.addEventListener('click', () => p4SetSpeed(Number(b.dataset.speed))));
    },

    onHide: function () {
      if (P3.timer) { clearTimeout(P3.timer); P3.timer = null; P3.playing = false; }
      if (P4.timer) { clearTimeout(P4.timer); P4.timer = null; P4.playing = false; }
      cancelAnimationFrame(geoAnimId);
    }
  });
})();