/* EDISON unified theme bootstrap.
 *
 * Standalone tab pages (printing, branding, projects, etc.) don't share
 * <body> classes with the main shell. This script reads the same
 * localStorage keys the main app writes and applies the matching
 * `light-theme` / `color-*` classes to <body> so page-theme.css
 * resolves to the right palette.
 */
(function () {
  'use strict';

  function readSettings() {
    try { return JSON.parse(localStorage.getItem('edison_settings') || '{}') || {}; }
    catch (_) { return {}; }
  }

  function applyTheme() {
    try {
      var settings = readSettings();
      var theme = localStorage.getItem('edison-theme')
        || localStorage.getItem('edison_theme')
        || settings.theme
        || (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
      var color = localStorage.getItem('edison-color')
        || localStorage.getItem('edison_color_theme')
        || settings.colorTheme
        || settings.color
        || '';

      var b = document.body;
      if (!b) return;

      b.classList.remove('light-theme', 'dark-theme');
      b.classList.add(String(theme).toLowerCase() === 'light' ? 'light-theme' : 'dark-theme');

      var colorClasses = ['color-default', 'color-blue', 'color-purple',
                          'color-cyan', 'color-emerald', 'color-rose',
                          'color-orange', 'color-pink', 'color-red',
                          'color-green', 'color-yellow'];
      colorClasses.forEach(function (c) { b.classList.remove(c); });
      var colorStr = String(color).toLowerCase();
      if (colorStr && colorStr !== 'default' && /^[a-z]+$/.test(colorStr)) {
        b.classList.add('color-' + colorStr);
      }
    } catch (e) {
      if (window.console && console.warn) {
        console.warn('[edison-theme] bootstrap failed:', e);
      }
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', applyTheme);
  } else {
    applyTheme();
  }

  window.addEventListener('storage', function (e) {
    if (!e.key) return;
    if (e.key === 'edison-theme' || e.key === 'edison-color'
        || e.key === 'edison_theme' || e.key === 'edison_color_theme'
        || e.key === 'edison_settings') {
      applyTheme();
    }
  });

  window.EdisonThemeBootstrap = { apply: applyTheme };
})();
/* EDISON unified theme bootstrap.
 *
 * Standalone tab pages (printing, branding, projects, etc.) don't share
 * <body> classes with the main shell. This script reads the same
 * localStorage keys the main app writes ("edison_theme",
 * "edison_color_theme", and "edison_settings.theme/color") and applies
 * the matching `light-theme` / `color-*` classes to <body> so
 * page-theme.css resolves to the right palette.
 */
(function () {
  'use strict';

  function applyTheme() {
    try {
      var settings = {};
      try { settings = JSON.parse(localStorage.getItem('edison_settings') || '{}'); }
      catch (_) { settings = {}; }

      var theme = localStorage.getItem('edison_theme')
        || settings.theme
        || (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
      var color = localStorage.getItem('edison_color_theme')
        || settings.colorTheme
        || settings.color
        || '';

      var b = document.body;
      if (!b) return;

      // Theme class
      b.classList.remove('light-theme', 'dark-theme');
      if (String(theme).toLowerCase() === 'light') {
        b.classList.add('light-theme');
      } else {
        b.classList.add('dark-theme');
      }

      // Color accent class
      var colorClasses = ['color-blue', 'color-purple', 'color-cyan',
                          'color-orange', 'color-pink', 'color-red',
                          'color-green', 'color-yellow'];
      colorClasses.forEach(function (c) { b.classList.remove(c); });
      if (color && /^[a-z]+$/i.test(color)) {
        b.classList.add('color-' + String(color).toLowerCase());
      }
    } catch (e) {
      // Theme bootstrap is non-critical.
      console && console.warn && console.warn('[edison-theme] bootstrap failed:', e);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', applyTheme);
  } else {
    applyTheme();
  }

  // Re-apply when other tabs change the theme.
  window.addEventListener('storage', function (e) {
    if (!e.key) return;
    if (e.key === 'edison_theme' || e.key === 'edison_color_theme' || e.key === 'edison_settings') {
      applyTheme();
    }
  });

  window.EdisonThemeBootstrap = { apply: applyTheme };
})();
