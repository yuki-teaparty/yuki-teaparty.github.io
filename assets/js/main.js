// 滚动淡入（无障碍：尊重 prefers-reduced-motion）
(function () {
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var els = document.querySelectorAll('.reveal');

  if (reduce || !('IntersectionObserver' in window)) {
    els.forEach(function (el) { el.classList.add('is-visible'); });
    return;
  }

  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) {
        e.target.classList.add('is-visible');
        io.unobserve(e.target);
      }
    });
  }, { threshold: 0.12 });

  els.forEach(function (el) { io.observe(el); });
})();
