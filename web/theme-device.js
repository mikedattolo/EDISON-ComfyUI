/**
 * EDISON Theme & Device Detection System
 * Handles dark/light mode, color schemes, and responsive mobile UI
 */

console.log('🎨 theme-device.js loading...');

class ThemeManager {
    constructor() {
        this.isMobile = this.detectDevice();
        this.currentTheme = this.loadTheme();
        this.currentColor = this.loadColor();
        this.init();
    }

    /**
     * Detect device type and capabilities
     */
    detectDevice() {
        const userAgent = navigator.userAgent.toLowerCase();
        
        const isMobileDevice = {
            isAndroid: /android/.test(userAgent),
            isIOS: /iphone|ipad|ipod/.test(userAgent),
            isTablet: /tablet|ipad|kindle|playbook|silk|nexus 7|nexus 10|xoom/.test(userAgent),
            isPhone: /mobile|android|iphone/.test(userAgent),
            isTouchDevice: () => {
                return (
                    (navigator.maxTouchPoints && navigator.maxTouchPoints > 2) ||
                    (navigator.msMaxTouchPoints && navigator.msMaxTouchPoints > 2)
                );
            }
        };

        const isMobile = isMobileDevice.isPhone || 
                        (window.innerWidth <= 768) ||
                        isMobileDevice.isTouchDevice();

        console.log('📱 Device Detection:', {
            isMobile,
            isAndroid: isMobileDevice.isAndroid,
            isIOS: isMobileDevice.isIOS,
            isTablet: isMobileDevice.isTablet,
            screenWidth: window.innerWidth,
            screenHeight: window.innerHeight
        });

        return isMobile;
    }

    /**
     * Initialize UI based on device type
     */
    init() {
        this.applyTheme(this.currentTheme);
        this.applyColor(this.currentColor);
        this.setupMobileUI();
        this.setupEventListeners();
        this.updateDeviceInfo();
        this.handleWindowResize();
    }

    /**
     * Setup mobile-specific UI layout
     */
    setupMobileUI() {
        const sidebar = document.querySelector('.sidebar');
        const mobileHeader = document.getElementById('mobileHeader');
        const mainContent = document.querySelector('.main-content');

        if (this.isMobile && mobileHeader && sidebar) {
            // Show mobile header
            mobileHeader.style.display = 'flex';
            sidebar.classList.add('mobile-layout');
            
            console.log('📱 Mobile UI activated');
        } else if (mobileHeader) {
            // Hide mobile header on desktop
            mobileHeader.style.display = 'none';
            console.log('🖥️ Desktop UI activated');
        }
    }

    /**
     * Setup all event listeners
     */
    setupEventListeners() {
        // Theme buttons
        const darkBtn = document.getElementById('darkThemeBtn');
        const lightBtn = document.getElementById('lightThemeBtn');
        if (darkBtn) darkBtn.addEventListener('click', () => this.setTheme('dark'));
        if (lightBtn) lightBtn.addEventListener('click', () => this.setTheme('light'));

        // Color buttons
        document.querySelectorAll('.color-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const color = btn.dataset.color;
                this.setColor(color);
            });
        });

        // Settings panel controls
        const settingsBtn = document.getElementById('settingsBtn');
        const settingCloseBtn = document.getElementById('settingCloseBtn');
        const mobileSettingsBtn = document.getElementById('mobileSettingsBtn');
        const settingsPanel = document.getElementById('settingsPanel');

        if (settingsBtn) settingsBtn.addEventListener('click', () => this.toggleSettings());
        if (settingCloseBtn) settingCloseBtn.addEventListener('click', () => this.closeSettings());
        if (mobileSettingsBtn) mobileSettingsBtn.addEventListener('click', () => this.toggleSettings());

        // Mobile menu
        const mobileMenuBtn = document.getElementById('mobileMenuBtn');
        if (mobileMenuBtn) {
            mobileMenuBtn.addEventListener('click', () => this.toggleMobileSidebar());
        }

        // Close settings when clicking outside
        if (settingsPanel) {
            settingsPanel.addEventListener('click', (e) => {
                if (e.target === settingsPanel) {
                    this.closeSettings();
                }
            });
        }

        // Window resize handler
        window.addEventListener('resize', () => this.handleWindowResize());

        // Orientation change
        window.addEventListener('orientationchange', () => {
            console.log('📱 Orientation changed');
            this.updateDeviceInfo();
        });
    }

    /**
     * Handle window resize for responsive layout
     */
    handleWindowResize() {
        const wasMobile = this.isMobile;
        this.isMobile = this.detectDevice();

        if (wasMobile !== this.isMobile) {
            console.log(`🔄 Layout switching: ${wasMobile ? 'mobile' : 'desktop'} → ${this.isMobile ? 'mobile' : 'desktop'}`);
            this.setupMobileUI();
        }

        this.updateDeviceInfo();
    }

    /**
     * Toggle settings panel
     */
    toggleSettings() {
        const settingsPanel = document.getElementById('settingsPanel');
        if (settingsPanel) {
            settingsPanel.classList.toggle('open');
            console.log('⚙️ Settings toggled');
        }
    }

    /**
     * Close settings panel
     */
    closeSettings() {
        const settingsPanel = document.getElementById('settingsPanel');
        if (settingsPanel) {
            settingsPanel.classList.remove('open');
        }
    }

    /**
     * Toggle mobile sidebar
     */
    toggleMobileSidebar() {
        const sidebar = document.querySelector('.sidebar');
        if (sidebar) {
            sidebar.classList.toggle('mobile-open');
            console.log('📱 Mobile sidebar toggled');
        }
    }

    /**
     * Apply theme to body
     */
    applyTheme(theme) {
        document.body.classList.remove('dark-theme', 'light-theme');
        if (theme === 'light') {
            document.body.classList.add('light-theme');
        }
        console.log(`🎨 Theme applied: ${theme}`);
    }

    /**
     * Set and save theme
     */
    setTheme(theme) {
        this.currentTheme = theme;
        this.applyTheme(theme);
        localStorage.setItem('edison-theme', theme);

        // Update button states
        document.querySelectorAll('.theme-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.theme === theme);
        });

        console.log(`✅ Theme saved: ${theme}`);
    }

    /**
     * Load theme from localStorage or system preference
     */
    loadTheme() {
        const saved = localStorage.getItem('edison-theme');
        if (saved) return saved;

        // Check system preference
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
            return 'light';
        }
        return 'dark';
    }

    /**
     * Apply color scheme to body
     */
    applyColor(color) {
        // Remove all color classes
        document.body.classList.remove(
            'color-default', 'color-blue', 'color-purple', 
            'color-cyan', 'color-emerald', 'color-rose'
        );

        if (color !== 'default') {
            document.body.classList.add(`color-${color}`);
        }
        console.log(`🎨 Color applied: ${color}`);
    }

    /**
     * Set and save color scheme
     */
    setColor(color) {
        this.currentColor = color;
        this.applyColor(color);
        localStorage.setItem('edison-color', color);

        // Update button states
        document.querySelectorAll('.color-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.color === color);
        });

        console.log(`✅ Color saved: ${color}`);
    }

    /**
     * Load color from localStorage
     */
    loadColor() {
        return localStorage.getItem('edison-color') || 'default';
    }

    /**
     * Update device info display
     */
    updateDeviceInfo() {
        const screenSize = `${window.innerWidth}x${window.innerHeight}`;
        const deviceType = this.isMobile ? (
            navigator.userAgent.includes('Tablet') || navigator.userAgent.includes('iPad') 
                ? 'Tablet' 
                : 'Mobile'
        ) : 'Desktop';
        const orientation = window.innerHeight > window.innerWidth ? 'Portrait' : 'Landscape';

        // Update display
        const deviceTypeEl = document.getElementById('deviceType');
        const screenSizeEl = document.getElementById('screenSize');
        const orientationEl = document.getElementById('deviceOrientation');

        if (deviceTypeEl) deviceTypeEl.textContent = deviceType;
        if (screenSizeEl) screenSizeEl.textContent = screenSize;
        if (orientationEl) orientationEl.textContent = orientation;

        console.log(`📊 Device Info: ${deviceType} | ${screenSize} | ${orientation}`);
    }
}

// Initialize theme manager when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.themeManager = new ThemeManager();
    });
} else {
    window.themeManager = new ThemeManager();
}

console.log('✅ theme-device.js loaded');
