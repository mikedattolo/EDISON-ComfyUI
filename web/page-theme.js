/* EDISON simplified page theme bootstrap.
 *
 * Standalone workspace pages are pinned to a single readable palette:
 * light background, black text, blue actions.
 */
(function () {
  'use strict';

  function applyTheme() {
    var b = document.body;
    if (!b) return;

    var removeClasses = [
      'dark-theme', 'light-theme',
      'color-default', 'color-blue', 'color-purple', 'color-cyan',
      'color-emerald', 'color-rose', 'color-orange', 'color-pink',
      'color-red', 'color-green', 'color-yellow'
    ];

    removeClasses.forEach(function (c) { b.classList.remove(c); });
    b.classList.add('light-theme', 'color-blue');

    try {
      localStorage.setItem('edison-theme', 'light');
      localStorage.setItem('edison_theme', 'light');
      localStorage.setItem('edison-color', 'blue');
      localStorage.setItem('edison_color_theme', 'blue');
    } catch (_) {
      // Ignore storage write failures.
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', applyTheme);
  } else {
    applyTheme();
  }

  window.addEventListener('storage', function () {
    applyTheme();
  });

  window.EdisonThemeBootstrap = { apply: applyTheme };
})();
