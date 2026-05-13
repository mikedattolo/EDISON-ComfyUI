/**
 * EDISON Theme & Device Detection System
 * Handles dark/light mode, color schemes, and responsive mobile UI
 */

console.log('🎨 theme-device.js loading...');

class ThemeManager {
    constructor() {
        this.isMobile = this.detectDevice();
        this.currentTheme = 'light';
        this.currentColor = 'blue';
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
        // Theme/color customization is intentionally disabled in simplified UI mode.
        
        // Mobile menu
        const mobileMenuBtn = document.getElementById('mobileMenuBtn');
        if (mobileMenuBtn) {
            mobileMenuBtn.addEventListener('click', () => this.toggleMobileSidebar());
        }
        
        // Mobile backdrop to close sidebar
        const mobileBackdrop = document.getElementById('mobileBackdrop');
        if (mobileBackdrop) {
            mobileBackdrop.addEventListener('click', () => this.toggleMobileSidebar());
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
        const backdrop = document.getElementById('mobileBackdrop');
        if (sidebar) {
            const isOpen = sidebar.classList.toggle('open');
            if (backdrop) {
                backdrop.style.display = isOpen ? 'block' : 'none';
            }
            console.log('📱 Mobile sidebar toggled:', isOpen ? 'open' : 'closed');
        }
    }

    /**
     * Apply theme to body
     */
    applyTheme(theme) {
        document.body.classList.remove('dark-theme', 'light-theme');
        document.body.classList.add('light-theme');
        console.log('🎨 Theme applied: light');
    }

    /**
     * Set and save theme
     */
    setTheme(theme) {
        this.currentTheme = 'light';
        this.applyTheme('light');
        localStorage.setItem('edison-theme', 'light');
        localStorage.setItem('edison_theme', 'light');
        console.log('✅ Theme saved: light');
    }

    /**
     * Load theme from localStorage or system preference
     */
    loadTheme() {
        return 'light';
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
        document.body.classList.add('color-blue');
        console.log('🎨 Color applied: blue');
    }

    /**
     * Set and save color scheme
     */
    setColor(color) {
        this.currentColor = 'blue';
        this.applyColor('blue');
        localStorage.setItem('edison-color', 'blue');
        localStorage.setItem('edison_color_theme', 'blue');
        console.log('✅ Color saved: blue');
    }

    /**
     * Load color from localStorage
     */
    loadColor() {
        return 'blue';
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
