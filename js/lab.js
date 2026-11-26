/* ================================================================
   lab.js — 主框架（无模块逻辑）
   ----------------------------------------------------------------
   Lab.register(mod)  —— 模块注册（包含防重复注册）
   Lab.start({stage, nav}) —— 生成导航并挂载第一个模块
   ================================================================ */
var Lab = (function () {
  'use strict';

  const modules = [];
  let active = null;
  let stageEl = null;
  let navEl = null;

  function register(mod) {
    if (!mod || !mod.id) return;
    
    // 防呆：检查是否重复注册
    const exists = modules.find(function (m) { return m.id === mod.id; });
    if (exists) {
      console.warn('警告：模块 ' + mod.id + ' 已经被注册过了，已跳过重复注册。请检查是否引入了两次相同的模块文件。');
      return;
    }

    modules.push(mod);
    modules.sort(function (a, b) {
      return String(a.index).localeCompare(String(b.index));
    });
  }

  function buildNav() {
    navEl.innerHTML = '';
    modules.forEach(function (mod) {
      const a = document.createElement('a');
      a.className = 'nav-item';
      a.dataset.id = mod.id;
      a.innerHTML = '<span class="nav-index">' + mod.index + '</span>' + mod.title;
      a.addEventListener('click', function () { activate(mod.id); });
      navEl.appendChild(a);
    });
  }

  function highlightNav(id) {
    Array.prototype.forEach.call(navEl.querySelectorAll('.nav-item'), function (a) {
      a.classList.toggle('active', a.dataset.id === id);
    });
  }

  function activate(id) {
    const mod = modules.find(function (m) { return m.id === id; });
    if (!mod) return;
    if (active && active.mod === mod) return;

    if (active) {
      active.section.classList.remove('is-active');
      if (typeof active.mod.onHide === 'function') {
        active.mod.onHide(active.section);
      }
    }

    if (!mod._section) {
      const section = document.createElement('section');
      section.className = 'module';
      section.id = mod.id;

      const head = document.createElement('div');
      head.className = 'module-head';
      head.innerHTML =
        '<h2>' + mod.index + ' · ' + mod.title + '</h2>' +
        '<p>' + (mod.desc || '') + '</p>';
      section.appendChild(head);

      const body = document.createElement('div');
      body.className = 'module-body';
      body.innerHTML = mod.template();
      section.appendChild(body);

      stageEl.appendChild(section);
      mod._section = section;

      if (typeof mod.init === 'function') mod.init(section);
    }

    mod._section.classList.add('is-active');
    active = { mod: mod, section: mod._section };

    if (typeof mod.onShow === 'function') mod.onShow(mod._section);

    highlightNav(id);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function start(opts) {
    stageEl = opts.stage;
    navEl = opts.nav;
    buildNav();
    if (modules.length) activate(modules[0].id);
  }

  return { register: register, start: start };
})();