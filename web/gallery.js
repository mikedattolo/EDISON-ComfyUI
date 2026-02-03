/**
 * EDISON Image Gallery
 * Manages the display and interaction with generated images
 */

class ImageGallery {
    constructor() {
        this.galleryPanel = null;
        this.galleryGrid = null;
        this.galleryLoading = null;
        this.galleryEmpty = null;
        this.images = [];
        this.apiEndpoint = localStorage.getItem('apiEndpoint') || 'http://192.168.1.26:8811';
        this.escapeListenerAdded = false;
        this.isOpen = false;
        this.init();
    }

    init() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.initElements());
        } else {
            this.initElements();
        }
    }

    initElements() {
        // Get DOM elements
        this.galleryPanel = document.getElementById('galleryPanel');
        this.galleryGrid = document.getElementById('galleryGrid');
        this.galleryLoading = document.getElementById('galleryLoading');
        this.galleryEmpty = document.getElementById('galleryEmpty');
        
        if (!this.galleryPanel) {
            console.error('Gallery panel not found in DOM');
            return;
        }

        // Ensure gallery panel is attached directly to body (avoid layout/stacking issues)
        if (this.galleryPanel.parentElement !== document.body) {
            document.body.appendChild(this.galleryPanel);
        }
        
        // Ensure gallery starts closed (CSS handles positioning via .active class)
        this.galleryPanel.classList.remove('active');
        this.isOpen = false;
        console.log('Gallery initialized - starts closed');
        
        // Set up event listeners
        this.setupEventListeners();
        console.log('Gallery initialized successfully');
    }

    setupEventListeners() {
        // Gallery button now uses inline onclick in HTML - no JS listener needed
        // This avoids double-firing issues

        // Close button - also uses inline onclick in HTML
        // No JS listeners needed for open/close buttons

        // Close on escape key - only add once
        if (!this.escapeListenerAdded) {
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' && this.isOpen) {
                    this.close();
                }
            });
            this.escapeListenerAdded = true;
        }
    }

    async toggle() {
        console.log('Toggle called, current state:', this.isOpen);
        if (this.isOpen) {
            this.close();
        } else {
            this.open();
        }
    }

    async open() {
        console.log('Opening gallery');
        console.log('Gallery panel element:', this.galleryPanel);

        if (!this.galleryPanel) {
            console.error('NO GALLERY PANEL!');
            alert('Gallery panel not found!');
            return;
        }

        // Force critical styles to guarantee visibility and stacking
        Object.assign(this.galleryPanel.style, {
            position: 'fixed',
            top: '0',
            right: '0',
            left: 'auto',
            bottom: '0',
            width: '600px',
            maxWidth: '90vw',
            height: '100vh',
            zIndex: '99999',
            display: 'flex',
            opacity: '1',
            pointerEvents: 'auto'
        });

        // Set inline styles for open state
        this.galleryPanel.style.transform = 'translateX(0)';
        this.galleryPanel.style.visibility = 'visible';
        this.galleryPanel.classList.add('active');
        this.isOpen = true;

        console.log('Gallery panel styles after open:', {
            transform: this.galleryPanel.style.transform,
            visibility: this.galleryPanel.style.visibility,
            display: this.galleryPanel.style.display,
            zIndex: this.galleryPanel.style.zIndex,
            position: this.galleryPanel.style.position
        });

        await this.loadImages();
    }

    close() {
        console.log('Closing gallery');
        // Remove active class and set inline styles for closed state
        this.galleryPanel.classList.remove('active');
        this.galleryPanel.style.transform = 'translateX(100%)';
        this.galleryPanel.style.visibility = 'hidden';
        this.galleryPanel.style.pointerEvents = 'none';
        this.isOpen = false;
    }

    async loadImages() {
        // Show loading state
        this.galleryLoading.style.display = 'flex';
        this.galleryEmpty.style.display = 'none';
        this.galleryGrid.style.display = 'none';

        try {
            const response = await fetch(`${this.apiEndpoint}/gallery/list`);
            if (!response.ok) throw new Error('Failed to load images');
            
            const data = await response.json();
            this.images = data.images || [];

            // Hide loading
            this.galleryLoading.style.display = 'none';

            if (this.images.length === 0) {
                this.galleryEmpty.style.display = 'flex';
            } else {
                this.galleryGrid.style.display = 'grid';
                this.renderImages();
            }
        } catch (error) {
            console.error('Failed to load gallery:', error);
            this.galleryLoading.style.display = 'none';
            this.galleryEmpty.style.display = 'flex';
            this.galleryEmpty.querySelector('p').textContent = 'Failed to load images';
        }
    }

    renderImages() {
        this.galleryGrid.innerHTML = '';

        this.images.forEach(image => {
            const item = this.createGalleryItem(image);
            this.galleryGrid.appendChild(item);
        });
    }

    createGalleryItem(image) {
        const item = document.createElement('div');
        item.className = 'gallery-item';
        item.setAttribute('data-id', image.id);

        // Format date
        const date = new Date(image.timestamp * 1000);
        const dateStr = date.toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });

        // Truncate prompt
        const prompt = image.prompt.length > 100 
            ? image.prompt.substring(0, 100) + '...' 
            : image.prompt;

        item.innerHTML = `
            <img class="gallery-item-image" 
                 src="${this.apiEndpoint}${image.url}" 
                 alt="${image.prompt}"
                 loading="lazy">
            <div class="gallery-item-info">
                <div class="gallery-item-prompt" title="${image.prompt}">${prompt}</div>
                <div class="gallery-item-meta">
                    <span class="gallery-item-date">
                        <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
                            <circle cx="6" cy="6" r="5" stroke="currentColor" stroke-width="1" fill="none"/>
                            <path d="M6 3v3l2 2" stroke="currentColor" stroke-width="1" fill="none"/>
                        </svg>
                        ${dateStr}
                    </span>
                    <div class="gallery-item-actions">
                        <button class="gallery-item-btn download" title="Download">
                            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                                <path d="M7 1v8M3 6l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                                <path d="M1 11h12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                            </svg>
                        </button>
                        <button class="gallery-item-btn delete" title="Delete">
                            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                                <path d="M2 3h10M5 1h4M5 5v6M9 5v6M3 3l1 9h6l1-9" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                            </svg>
                        </button>
                    </div>
                </div>
            </div>
        `;

        // Add click handler for full view
        const img = item.querySelector('.gallery-item-image');
        img.addEventListener('click', () => this.showFullImage(image));

        // Add download handler
        const downloadBtn = item.querySelector('.download');
        downloadBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.downloadImage(image);
        });

        // Add delete handler
        const deleteBtn = item.querySelector('.delete');
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.deleteImage(image.id);
        });

        return item;
    }

    showFullImage(image) {
        // Create modal if it doesn't exist
        let modal = document.getElementById('galleryModal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'galleryModal';
            modal.className = 'gallery-modal';
            document.body.appendChild(modal);
        }

        // Format settings
        const settings = image.settings || {};
        const settingsStr = Object.entries(settings)
            .map(([key, value]) => `${key}: ${value}`)
            .join(' • ');

        modal.innerHTML = `
            <div class="gallery-modal-content">
                <button class="gallery-modal-close">×</button>
                <img class="gallery-modal-image" src="${this.apiEndpoint}${image.url}" alt="${image.prompt}">
                <div class="gallery-modal-info">
                    <div class="gallery-modal-prompt">${image.prompt}</div>
                    <div class="gallery-modal-meta">
                        <span>Size: ${image.width}x${image.height}</span>
                        <span>Model: ${image.model || 'SDXL'}</span>
                        ${settingsStr ? `<span>${settingsStr}</span>` : ''}
                    </div>
                </div>
            </div>
        `;

        // Show modal
        modal.classList.add('active');

        // Close handlers
        const closeBtn = modal.querySelector('.gallery-modal-close');
        closeBtn.addEventListener('click', () => {
            modal.classList.remove('active');
        });

        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.classList.remove('active');
            }
        });
    }

    async downloadImage(image) {
        try {
            const response = await fetch(`${this.apiEndpoint}${image.url}`);
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `edison-${image.id}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Failed to download image:', error);
            alert('Failed to download image');
        }
    }

    async deleteImage(id) {
        if (!confirm('Are you sure you want to delete this image?')) {
            return;
        }

        try {
            const response = await fetch(`${this.apiEndpoint}/gallery/delete/${id}`, {
                method: 'DELETE'
            });

            if (!response.ok) throw new Error('Failed to delete image');

            // Remove from local array
            this.images = this.images.filter(img => img.id !== id);

            // Re-render
            if (this.images.length === 0) {
                this.galleryGrid.style.display = 'none';
                this.galleryEmpty.style.display = 'flex';
            } else {
                this.renderImages();
            }
        } catch (error) {
            console.error('Failed to delete image:', error);
            alert('Failed to delete image');
        }
    }

    // Method to add a new image (called after generation)
    addImage(image) {
        this.images.unshift(image); // Add to beginning
        
        if (this.galleryPanel.classList.contains('active')) {
            // If gallery is open, re-render
            this.galleryGrid.style.display = 'grid';
            this.galleryEmpty.style.display = 'none';
            this.renderImages();
        }
    }
}

// Initialize gallery - handle both cases: before and after DOMContentLoaded
if (!window.imageGallery) {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            console.log('Initializing ImageGallery singleton (on DOMContentLoaded)');
            window.imageGallery = new ImageGallery();
        });
    } else {
        console.log('Initializing ImageGallery singleton (DOM already ready)');
        window.imageGallery = new ImageGallery();
    }
}
