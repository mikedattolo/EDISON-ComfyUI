(function () {
    const pages = [
        { key: 'chat', label: 'Chat', href: '/', icon: 'C', matches: ['/', '/index.html'] },
        {
            key: 'branding',
            label: 'Branding',
            href: '/branding',
            icon: 'B',
            matches: ['/branding', '/branding.html'],
            heroKicker: 'Brand Systems',
            heroTitle: 'Keep client identity work, asset review, and approvals in one visual workspace.',
            heroDescription: 'Use branding as the handoff point between client files, style exploration, and the downstream project or fabrication work that follows.',
            highlights: ['Client asset hubs', 'Revision-friendly folders', 'Project-ready deliverables'],
            stats: [
                { label: 'Best for', value: 'Client identity work' },
                { label: 'Next handoff', value: 'Projects + Printing' }
            ],
            prompt: 'Review the current branding workspace and help me create a stronger brand package workflow with approvals, file organization, and deliverables.',
            mode: 'agent',
            secondaryHref: '/projects',
            secondaryLabel: 'Open Projects'
        },
        {
            key: 'projects',
            label: 'Projects',
            href: '/projects',
            icon: 'P',
            matches: ['/projects', '/projects.html'],
            heroKicker: 'Delivery Control',
            heroTitle: 'Track client scope, deadlines, and output paths without losing the chat-first workflow.',
            heroDescription: 'Projects should bridge branding, video, fabrication, and connector work. This page is where those threads become actual delivery status.',
            highlights: ['Client-linked records', 'Due dates + approvals', 'Cross-workspace visibility'],
            stats: [
                { label: 'Use for', value: 'Operational tracking' },
                { label: 'Pairs with', value: 'Branding + Video' }
            ],
            prompt: 'Audit my project workspace and suggest the highest-impact improvements for delivery tracking, approvals, and connected outputs.',
            mode: 'work',
            secondaryHref: '/branding',
            secondaryLabel: 'Open Branding'
        },
        {
            key: 'video',
            label: 'Video',
            href: '/video-editor',
            icon: 'V',
            matches: ['/video-editor', '/video_editor.html'],
            heroKicker: 'Production Studio',
            heroTitle: 'Move from clip inspection to production planning with less context switching.',
            heroDescription: 'The video workspace should support browsing, edits, exports, and campaign planning. Use chat to turn raw clips into structured production work.',
            highlights: ['Clip browsing', 'Edit operations', 'Campaign export context'],
            stats: [
                { label: 'Focus', value: 'Editing + review' },
                { label: 'Next step', value: 'Projects + Connectors' }
            ],
            prompt: 'Help me turn this video workspace into a stronger production flow with better clip organization, edit review, and export planning.',
            mode: 'work',
            secondaryHref: '/projects',
            secondaryLabel: 'Open Projects'
        },
        {
            key: 'printing',
            label: 'Print',
            href: '/printing',
            icon: '3D',
            matches: ['/printing', '/printing.html'],
            heroKicker: 'Fabrication Ops',
            heroTitle: 'Keep machine status, fabrication prep, and print jobs tied to the same client workflow.',
            heroDescription: 'Printing should feel like part of the business pipeline, not a detached device screen. Use this workspace for job readiness and fabrication context.',
            highlights: ['Machine visibility', 'Job prep context', 'Fabrication handoff'],
            stats: [
                { label: 'Use for', value: 'Queued fabrication work' },
                { label: 'Pairs with', value: 'Branding + Nodes' }
            ],
            prompt: 'Review the 3D printing workspace and tell me how to make print job setup, queueing, and fabrication status clearer.',
            mode: 'agent',
            secondaryHref: '/nodes',
            secondaryLabel: 'Open Nodes'
        },
        {
            key: 'connectors',
            label: 'Connectors',
            href: '/connectors',
            icon: 'API',
            matches: ['/connectors', '/connectors.html'],
            heroKicker: 'API + Social Control',
            heroTitle: 'Manage external services with a cleaner handoff between setup, testing, and automation.',
            heroDescription: 'Connectors should support both business platforms and internal automation. Use chat to reason about API setup before or after testing here.',
            highlights: ['OAuth + token setup', 'API test loops', 'Automation-friendly connections'],
            stats: [
                { label: 'Best for', value: 'External integrations' },
                { label: 'Feeds', value: 'Assistants + Projects' }
            ],
            prompt: 'Inspect the connectors workspace and recommend improvements for API setup, testing, and scheduled business actions.',
            mode: 'code',
            secondaryHref: '/assistants',
            secondaryLabel: 'Open Assistants'
        },
        {
            key: 'assistants',
            label: 'Assistants',
            href: '/assistants',
            icon: 'AI',
            matches: ['/assistants', '/assistants.html'],
            heroKicker: 'Custom AI Layer',
            heroTitle: 'Build reusable assistants and triggers that feel native to the main chat experience.',
            heroDescription: 'Custom AI should act like saved expertise, not a separate toy. This workspace is where roles, prompts, and automation hooks become reusable tools.',
            highlights: ['Role templates', 'Automation triggers', 'Connector-backed actions'],
            stats: [
                { label: 'Use for', value: 'Reusable specialist flows' },
                { label: 'Pairs with', value: 'Connectors + Chat' }
            ],
            prompt: 'Review the custom AI workspace and help me make assistants and automations feel more integrated with the main chat.',
            mode: 'code',
            secondaryHref: '/connectors',
            secondaryLabel: 'Open Connectors'
        },
        {
            key: 'nodes',
            label: 'Nodes',
            href: '/nodes',
            icon: 'N',
            matches: ['/nodes', '/nodes.html'],
            heroKicker: 'Distributed Workstations',
            heroTitle: 'Track remote workers, CAD nodes, and hardware capacity as part of the same operating system.',
            heroDescription: 'Nodes should surface readiness, ownership, and job routing clearly, especially when fabrication and Rhino work depend on remote machines.',
            highlights: ['Worker status', 'Remote hardware map', 'Distributed task readiness'],
            stats: [
                { label: 'Focus', value: 'Remote execution' },
                { label: 'Supports', value: 'Printing + CAD tasks' }
            ],
            prompt: 'Review the nodes workspace and recommend UI improvements for node status, health, and distributed job control.',
            mode: 'code',
            secondaryHref: '/printing',
            secondaryLabel: 'Open Printing'
        },
        {
            key: 'help',
            label: 'Help',
            href: '/help',
            icon: '?',
            matches: ['/help', '/help.html'],
            heroKicker: 'Reference Layer',
            heroTitle: 'Use help as a launch surface into the right workspace instead of a dead-end documentation page.',
            heroDescription: 'Documentation should shorten the path from question to action. This page now acts as a clearer entry point into the business workspaces.',
            highlights: ['Feature lookup', 'Workflow jump-off', 'Fast orientation'],
            stats: [
                { label: 'Best for', value: 'System orientation' },
                { label: 'Next move', value: 'Jump back into work' }
            ],
            prompt: 'Summarize the most important EDISON workspaces and tell me which one I should use next for branding, video, printing, or connectors.',
            mode: 'chat',
            secondaryHref: '/projects',
            secondaryLabel: 'Open Projects'
        },
    ];

    function normalizePath(pathname) {
        if (!pathname) return '/';
        const normalized = pathname.replace(/\/+$/, '');
        return normalized || '/';
    }

    function resolvePage(pathname) {
        return pages.find((page) => page.matches.includes(pathname)) || pages[0];
    }

    function renderNav(activeKey) {
        return pages.map((page) => {
            const active = page.key === activeKey ? ' active' : '';
            return [
                '<a class="ws-shell-link', active, '" href="', page.href, '">',
                '<span class="ws-shell-link-icon">', page.icon, '</span>',
                '<span>', page.label, '</span>',
                '</a>'
            ].join('');
        }).join('');
    }

    function renderHeroHighlights(page) {
        return (page.highlights || []).map((item) => {
            return ['<span class="ws-page-highlight">', item, '</span>'].join('');
        }).join('');
    }

    function renderHeroStats(page) {
        return (page.stats || []).map((item) => {
            return [
                '<div class="ws-page-stat">',
                '<span class="ws-page-stat-label">', item.label, '</span>',
                '<strong>', item.value, '</strong>',
                '</div>'
            ].join('');
        }).join('');
    }

    function buildChatHref(prompt, mode) {
        const url = new URL('/', window.location.origin);
        if (prompt) {
            url.searchParams.set('prompt', prompt);
        }
        if (mode) {
            url.searchParams.set('mode', mode);
        }
        return url.pathname + url.search;
    }

    function renderPageHero(page) {
        if (!page || page.key === 'chat') {
            return '';
        }

        return [
            '<section class="ws-page-hero" aria-label="Workspace overview">',
            '<div class="ws-page-hero-main">',
            '<span class="ws-page-kicker">', page.heroKicker || page.label, '</span>',
            '<h2>', page.heroTitle || page.label, '</h2>',
            '<p>', page.heroDescription || '', '</p>',
            '<div class="ws-page-highlights">', renderHeroHighlights(page), '</div>',
            '<div class="ws-page-actions">',
            '<a class="ws-page-btn primary" href="', buildChatHref(page.prompt, page.mode), '">Ask In Chat</a>',
            '<a class="ws-page-btn" href="/?panel=file-manager">Open Files</a>',
            page.secondaryHref && page.secondaryLabel
                ? ['<a class="ws-page-btn subtle" href="', page.secondaryHref, '">', page.secondaryLabel, '</a>'].join('')
                : '',
            '</div>',
            '</div>',
            '<aside class="ws-page-hero-side">',
            '<div class="ws-page-side-head">',
            '<span>Workspace focus</span>',
            '<strong>', page.label, '</strong>',
            '</div>',
            '<div class="ws-page-stats">', renderHeroStats(page), '</div>',
            '</aside>',
            '</section>'
        ].join('');
    }

    function renderWorkspaceRibbon(page) {
        if (!page || page.key === 'chat') {
            return '';
        }

        return [
            '<section class="ws-workspace-ribbon" aria-label="Workspace quick actions">',
            '<div class="ws-workspace-ribbon-main">',
            '<span class="ws-workspace-ribbon-kicker">EDISON Workspace Layer <b>v3</b></span>',
            '<h2>', page.label, ' workspace is active</h2>',
            '<p>Launch actions directly from this page or jump back to chat with a prefilled prompt.</p>',
            '</div>',
            '<div class="ws-workspace-ribbon-actions">',
            '<a class="ws-page-btn primary" href="', buildChatHref(page.prompt, page.mode), '">Ask In Chat</a>',
            '<a class="ws-page-btn" href="/?panel=file-manager">Open Files</a>',
            '<a class="ws-page-btn subtle" href="/?new=1">Start New Chat</a>',
            '</div>',
            '</section>'
        ].join('');
    }

    function cleanupQueryParam(name) {
        const url = new URL(window.location.href);
        if (!url.searchParams.has(name)) return;
        url.searchParams.delete(name);
        window.history.replaceState({}, '', url.toString());
    }

    function openFileManager() {
        if (typeof window.toggleFileManager === 'function') {
            window.toggleFileManager();
            cleanupQueryParam('panel');
            return true;
        }
        return false;
    }

    function createNewChat() {
        if (window.edisonApp && typeof window.edisonApp.createNewChat === 'function') {
            window.edisonApp.createNewChat();
            cleanupQueryParam('new');
            return true;
        }
        return false;
    }

    function focusChatInput() {
        if (window.edisonApp && window.edisonApp.messageInput) {
            window.edisonApp.messageInput.focus();
            return true;
        }
        return false;
    }

    function applyChatPrompt(prompt, mode) {
        if (!prompt || !window.edisonApp || !window.edisonApp.messageInput) {
            return false;
        }

        if (mode && typeof window.edisonApp.setMode === 'function') {
            window.edisonApp.setMode(mode);
        }

        window.edisonApp.messageInput.value = prompt;
        if (typeof window.edisonApp.handleInputChange === 'function') {
            window.edisonApp.handleInputChange();
        }
        window.edisonApp.messageInput.focus();
        window.edisonApp.messageInput.setSelectionRange(
            window.edisonApp.messageInput.value.length,
            window.edisonApp.messageInput.value.length
        );
        cleanupQueryParam('prompt');
        cleanupQueryParam('mode');
        return true;
    }

    function waitForChatPrompt(prompt, mode, attempt) {
        const tries = attempt || 0;
        if (applyChatPrompt(prompt, mode) || tries >= 40) {
            return;
        }

        window.setTimeout(function () {
            waitForChatPrompt(prompt, mode, tries + 1);
        }, 250);
    }

    function waitForAction(action, attempt) {
        const tries = attempt || 0;
        let handled = false;

        if (action === 'file-manager') {
            handled = openFileManager();
        } else if (action === 'new-chat') {
            handled = createNewChat();
        } else if (action === 'focus-chat') {
            handled = focusChatInput();
        }

        if (handled || tries >= 40) {
            return;
        }

        window.setTimeout(function () {
            waitForAction(action, tries + 1);
        }, 250);
    }

    function handleQueryActions(currentPage) {
        if (currentPage.key !== 'chat') return;

        const params = new URLSearchParams(window.location.search);
        const prompt = params.get('prompt');
        const mode = params.get('mode') || 'auto';
        if (params.get('panel') === 'file-manager') {
            waitForAction('file-manager', 0);
        }
        if (params.get('new') === '1') {
            waitForAction('new-chat', 0);
        }
        if (prompt) {
            waitForChatPrompt(prompt, mode, 0);
        }
    }

    function handleShortcut(event) {
        if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey) {
            return;
        }

        const target = event.target;
        const isTyping = target && (
            target.tagName === 'INPUT' ||
            target.tagName === 'TEXTAREA' ||
            target.tagName === 'SELECT' ||
            target.isContentEditable
        );

        if (isTyping) {
            return;
        }

        const currentPath = normalizePath(window.location.pathname);
        if (currentPath === '/' || currentPath === '/index.html') {
            if (event.key === '/') {
                event.preventDefault();
                waitForAction('focus-chat', 0);
            }
            if (event.shiftKey && event.key.toLowerCase() === 'f') {
                event.preventDefault();
                if (!openFileManager()) {
                    window.location.href = '/?panel=file-manager';
                }
            }
        }
    }

    function initShell() {
        if (!document.body || document.body.dataset.workspaceShell === 'off' || document.querySelector('.ws-shell-rail')) {
            return;
        }

        const currentPath = normalizePath(window.location.pathname);
        const currentPage = resolvePage(currentPath);

        document.body.classList.add('ws-shell-enabled');
        document.body.dataset.wsPage = currentPage.key;

        const rail = document.createElement('div');
        rail.className = 'ws-shell-rail';
        rail.innerHTML = [
            '<div class="ws-shell-inner">',
            '<a class="ws-shell-brand" href="/">',
            '<span class="ws-shell-brand-mark">E</span>',
            '<span class="ws-shell-brand-copy">',
            '<strong>EDISON Workspace</strong>',
            '<span>Chat, files, branding, projects, media, fabrication, and automation in one flow</span>',
            '</span>',
            '</a>',
            '<nav class="ws-shell-nav" aria-label="Workspace navigation">',
            renderNav(currentPage.key),
            '</nav>',
            '<div class="ws-shell-tools">',
            '<button class="ws-shell-tool" type="button" data-ws-action="file-manager">',
            '<span class="ws-shell-tool-icon">F</span><span>Files</span>',
            '</button>',
            '<button class="ws-shell-tool primary" type="button" data-ws-action="new-chat">',
            '<span class="ws-shell-tool-icon">+</span><span>New Chat</span>',
            '</button>',
            '</div>',
            '</div>'
        ].join('');

        rail.addEventListener('click', function (event) {
            const control = event.target.closest('[data-ws-action]');
            if (!control) return;

            const action = control.getAttribute('data-ws-action');
            if (action === 'file-manager') {
                if (currentPage.key === 'chat') {
                    if (!openFileManager()) {
                        waitForAction('file-manager', 0);
                    }
                } else {
                    window.location.href = '/?panel=file-manager';
                }
            }

            if (action === 'new-chat') {
                if (currentPage.key === 'chat') {
                    if (!createNewChat()) {
                        waitForAction('new-chat', 0);
                    }
                } else {
                    window.location.href = '/?new=1';
                }
            }
        });

        document.body.prepend(rail);
        if (currentPage.key !== 'chat') {
            const topbar = document.querySelector('.topbar');
            if (topbar && !document.querySelector('.ws-workspace-ribbon')) {
                topbar.insertAdjacentHTML('afterend', renderWorkspaceRibbon(currentPage));
            }
        }
        if (currentPage.key !== 'chat') {
            const main = document.querySelector('main.shell, main');
            if (main && !main.querySelector('.ws-page-hero')) {
                main.insertAdjacentHTML('afterbegin', renderPageHero(currentPage));
            }
        }
        document.addEventListener('keydown', handleShortcut);
        handleQueryActions(currentPage);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initShell);
    } else {
        initShell();
    }
})();