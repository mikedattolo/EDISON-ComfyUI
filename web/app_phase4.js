/**
 * EDISON Phase 4 UI — image studio panel + queue ETA badge.
 *
 * Surfaces the /api/phase4/* endpoints inside the existing chat tab
 * without touching the chat composer wiring.  Adds:
 *
 *   - footer button "Image Studio" that opens a side panel with
 *       aspect presets, prompt rewriter, seed-variation helper,
 *       and a one-click "insert into chat" action.
 *   - small ETA badge next to the send button that polls
 *       /api/phase4/queue/eta?lane=image when an image gen is in flight.
 *
 * No build step.  Loaded as a plain script.
 */
(function () {
  'use strict';

  const API = (window.EDISON_API_BASE || '').replace(/\/$/, '');
  const $ = (sel, root = document) => root.querySelector(sel);
  const h = (tag, attrs = {}, ...kids) => {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') el.className = v;
      else if (k === 'style') Object.assign(el.style, v);
      else if (k.startsWith('on') && typeof v === 'function') {
        el.addEventListener(k.slice(2).toLowerCase(), v);
      } else el.setAttribute(k, v);
    }
    for (const kid of kids) {
      if (kid == null) continue;
      el.append(kid.nodeType ? kid : document.createTextNode(kid));
    }
    return el;
  };

  async function api(path, opts = {}) {
    const r = await fetch(`${API}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...opts,
    });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
  }

  // ── Modal shell ────────────────────────────────────────────────

  let modalEl = null;
  function ensureModal() {
    if (modalEl) return modalEl;
    modalEl = h('div', {
      id: 'phase4Modal',
      style: {
        position: 'fixed', inset: '0', display: 'none',
        background: 'rgba(0,0,0,0.55)', zIndex: '9999',
        alignItems: 'center', justifyContent: 'center',
        fontFamily: 'inherit',
      },
    });
    modalEl.addEventListener('click', (e) => {
      if (e.target === modalEl) closeModal();
    });
    document.body.appendChild(modalEl);
    return modalEl;
  }
  function openModal(content) {
    const m = ensureModal();
    m.innerHTML = '';
    const card = h('div', {
      style: {
        background: 'var(--bg-secondary, #1e1e1e)',
        color: 'var(--text-primary, #eee)',
        borderRadius: '12px', padding: '20px',
        width: 'min(720px, 92vw)', maxHeight: '88vh', overflow: 'auto',
        boxShadow: '0 16px 48px rgba(0,0,0,0.6)',
      },
    });
    card.appendChild(content);
    m.appendChild(card);
    m.style.display = 'flex';
  }
  function closeModal() { if (modalEl) modalEl.style.display = 'none'; }

  // ── Image Studio panel ────────────────────────────────────────

  async function openImageStudio() {
    const root = h('div');
    root.appendChild(h('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' } },
      h('h2', { style: { margin: 0, fontSize: '18px' } }, 'Image Studio'),
      h('button', { onclick: closeModal, style: { background: 'transparent', color: 'inherit', border: 0, fontSize: '20px', cursor: 'pointer' } }, '×'),
    ));

    const status = h('div', { style: { fontSize: '12px', opacity: '0.7', marginBottom: '12px' } }, 'Loading presets…');
    root.appendChild(status);

    const presetGrid = h('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '8px', marginBottom: '14px' } });
    root.appendChild(presetGrid);

    // Prompt rewriter
    const promptInput = h('textarea', {
      rows: '3',
      placeholder: 'Describe the image…',
      style: { width: '100%', padding: '8px', borderRadius: '6px', border: '1px solid #333', background: 'var(--bg-primary, #111)', color: 'inherit', fontFamily: 'inherit' },
    });
    const familySel = h('select', { style: { padding: '6px', marginRight: '8px' } });
    ['sdxl', 'flux_dev', 'flux_schnell', 'sdxl_turbo', 'sd15'].forEach(f =>
      familySel.appendChild(h('option', { value: f }, f)));
    const projectInput = h('input', { type: 'text', placeholder: 'project_id (optional, applies style sheet)', style: { padding: '6px', flex: '1', background: 'var(--bg-primary, #111)', color: 'inherit', border: '1px solid #333', borderRadius: '6px' } });
    const rewriteOut = h('pre', { style: { background: 'var(--bg-primary, #111)', padding: '10px', borderRadius: '6px', maxHeight: '180px', overflow: 'auto', fontSize: '12px', whiteSpace: 'pre-wrap' } }, '');
    const rewriteBtn = h('button', { class: 'btn-primary', style: { padding: '8px 14px', cursor: 'pointer' } }, 'Rewrite prompt');
    const insertBtn = h('button', { style: { padding: '8px 14px', marginLeft: '8px', cursor: 'pointer' } }, 'Insert into chat');
    insertBtn.disabled = true;

    rewriteBtn.addEventListener('click', async () => {
      rewriteBtn.disabled = true; rewriteBtn.textContent = '…';
      try {
        const body = {
          prompt: promptInput.value || '',
          model_family: familySel.value,
          project_id: projectInput.value.trim() || null,
        };
        const out = await api('/api/phase4/image/rewrite', { method: 'POST', body: JSON.stringify(body) });
        rewriteOut.textContent =
          `POSITIVE:\n${out.positive}\n\n` +
          (out.negative ? `NEGATIVE:\n${out.negative}\n\n` : '') +
          (out.intent ? `INTENT: ${out.intent}\n` : '');
        insertBtn.disabled = false;
        insertBtn.dataset.payload = out.positive;
      } catch (e) {
        rewriteOut.textContent = 'Error: ' + e.message;
      } finally {
        rewriteBtn.disabled = false; rewriteBtn.textContent = 'Rewrite prompt';
      }
    });

    insertBtn.addEventListener('click', () => {
      const ti = document.getElementById('messageInput');
      if (!ti) return;
      const txt = `generate an image: ${insertBtn.dataset.payload || promptInput.value}`;
      ti.value = txt;
      ti.dispatchEvent(new Event('input', { bubbles: true }));
      ti.focus();
      closeModal();
    });

    root.appendChild(h('h3', { style: { fontSize: '14px', margin: '14px 0 6px' } }, 'Prompt rewriter'));
    root.appendChild(h('div', { style: { display: 'flex', gap: '6px', marginBottom: '6px' } }, familySel, projectInput));
    root.appendChild(promptInput);
    root.appendChild(h('div', { style: { margin: '8px 0' } }, rewriteBtn, insertBtn));
    root.appendChild(rewriteOut);

    // Seed variations + ETA
    const utilRow = h('div', { style: { display: 'flex', gap: '8px', marginTop: '14px', flexWrap: 'wrap' } });
    const seedInput = h('input', { type: 'number', value: '12345', style: { width: '120px', padding: '6px', background: 'var(--bg-primary, #111)', color: 'inherit', border: '1px solid #333', borderRadius: '6px' } });
    const seedBtn = h('button', { style: { padding: '6px 10px', cursor: 'pointer' } }, 'More like this (4 seeds)');
    const seedOut = h('span', { style: { fontSize: '12px', opacity: '0.85' } });
    seedBtn.addEventListener('click', async () => {
      try {
        const r = await api(`/api/phase4/image/variations?seed=${encodeURIComponent(seedInput.value)}&count=4`);
        seedOut.textContent = '→ ' + r.seeds.join(', ');
      } catch (e) { seedOut.textContent = 'Error: ' + e.message; }
    });
    utilRow.append(seedInput, seedBtn, seedOut);
    root.appendChild(utilRow);

    const etaRow = h('div', { style: { marginTop: '10px', fontSize: '12px', opacity: '0.85' } }, 'Image queue ETA: …');
    root.appendChild(etaRow);
    api('/api/phase4/queue/eta?lane=image').then((r) => {
      etaRow.textContent = `Image lane → queued ${r.queued}, in-flight ${r.in_flight}, avg ${Math.round((r.avg_duration_ms || 0) / 100) / 10}s${r.eta_ms ? `, ETA ~${Math.round(r.eta_ms / 1000)}s` : ''}`;
    }).catch(() => { etaRow.textContent = 'Image queue ETA: unavailable'; });

    openModal(root);

    // Load presets
    try {
      const { presets } = await api('/api/phase4/image/presets');
      status.textContent = `${presets.length} presets — click to apply size hint to your prompt`;
      presets.forEach((p) => {
        const card = h('button', {
          style: {
            textAlign: 'left', padding: '8px', border: '1px solid #333',
            borderRadius: '6px', background: 'var(--bg-primary, #111)',
            color: 'inherit', cursor: 'pointer', fontSize: '12px',
          },
          onclick: () => {
            const ti = document.getElementById('messageInput');
            const tag = `[${p.name} ${p.width}x${p.height}]`;
            if (ti) {
              ti.value = (ti.value ? ti.value + ' ' : 'generate an image ') + tag;
              ti.dispatchEvent(new Event('input', { bubbles: true }));
              ti.focus();
              closeModal();
            }
          },
        },
          h('strong', null, p.name),
          h('br'),
          `${p.width}×${p.height}`,
          h('br'),
          h('span', { style: { opacity: '0.65' } }, p.hint || ''),
        );
        presetGrid.appendChild(card);
      });
    } catch (e) {
      status.textContent = 'Failed to load presets: ' + e.message;
    }
  }

  // ── Inject footer button ──────────────────────────────────────

  function injectButton() {
    const footer = document.querySelector('.input-footer');
    if (!footer || footer.querySelector('#phase4Btn')) return;
    const target = footer.querySelector('div[style*="display:flex"]') || footer;
    const btn = h('button', {
      id: 'phase4Btn',
      class: 'input-footer-btn',
      title: 'Image Studio — presets, prompt rewriter, queue ETA',
      onclick: openImageStudio,
    });
    btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="1" y="2" width="12" height="10" rx="1"/><circle cx="5" cy="6" r="1"/><path d="M1 10l4-3 4 3 4-4"/></svg> Image Studio';
    target.appendChild(btn);
  }

  // ── Boot ──────────────────────────────────────────────────────

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', injectButton);
  } else {
    injectButton();
  }
  // Re-inject if app re-renders the footer
  const mo = new MutationObserver(() => injectButton());
  mo.observe(document.body, { childList: true, subtree: true });

  window.EDISONImageStudio = { open: openImageStudio };
})();
