// 博客左侧栏：手机模式下作为抽屉，点击汉堡 / 遮罩 / Esc / 链接均可开合
(function () {
  var toggle = document.getElementById('sidebarToggle');
  var sidebar = document.getElementById('sidebar');
  var overlay = document.getElementById('sidebarOverlay');
  if (!toggle || !sidebar || !overlay) return;

  var mobile = window.matchMedia('(max-width: 899px)');

  function isOpen() { return sidebar.classList.contains('is-open'); }

  function open() {
    sidebar.classList.add('is-open');
    overlay.hidden = false;
    // 下一帧再加可见类，触发过渡
    requestAnimationFrame(function () { overlay.classList.add('is-visible'); });
    toggle.setAttribute('aria-expanded', 'true');
    document.body.classList.add('sidebar-open');
  }

  function close() {
    sidebar.classList.remove('is-open');
    overlay.classList.remove('is-visible');
    toggle.setAttribute('aria-expanded', 'false');
    document.body.classList.remove('sidebar-open');
    setTimeout(function () { overlay.hidden = true; }, 250);
  }

  toggle.addEventListener('click', function () { isOpen() ? close() : open(); });
  overlay.addEventListener('click', close);
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && isOpen()) close(); });

  // 手机上点了某篇文章后自动收起抽屉
  sidebar.addEventListener('click', function (e) {
    if (e.target.closest('a') && mobile.matches) close();
  });

  // 从手机切回桌面时复位
  mobile.addEventListener('change', function (e) { if (!e.matches) close(); });
})();
