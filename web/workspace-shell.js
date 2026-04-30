(function () {
    const pages = [
        { key: 'chat', label: 'Chat', href: '/', icon: 'C', matches: ['/', '/index.html'] },
        { key: 'branding', label: 'Branding', href: '/branding', icon: 'B', matches: ['/branding', '/branding.html'] },
        { key: 'projects', label: 'Projects', href: '/projects', icon: 'P', matches: ['/projects', '/projects.html'] },
        { key: 'video', label: 'Video', href: '/video-editor', icon: 'V', matches: ['/video-editor', '/video_editor.html'] },
        { key: 'printing', label: 'Print', href: '/printing', icon: '3D', matches: ['/printing', '/printing.html'] },
        { key: 'connectors', label: 'Connectors', href: '/connectors', icon: 'API', matches: ['/connectors', '/connectors.html'] },
        { key: 'assistants', label: 'Assistants', href: '/assistants', icon: 'AI', matches: ['/assistants', '/assistants.html'] },
        { key: 'nodes', label: 'Nodes', href: '/nodes', icon: 'N', matches: ['/nodes', '/nodes.html'] },
        { key: 'help', label: 'Help', href: '/help', icon: '?', matches: ['/help', '/help.html'] },
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
        if (params.get('panel') === 'file-manager') {
            waitForAction('file-manager', 0);
        }
        if (params.get('new') === '1') {
            waitForAction('new-chat', 0);
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
        document.addEventListener('keydown', handleShortcut);
        handleQueryActions(currentPage);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initShell);
    } else {
        initShell();
    }
})();