// ===========================================
// EDISON New Features v1
// 3D Model Generation, Minecraft Tools, File Manager
// ===========================================

console.log('🧊 app_new_features.js v1 loading...');

(function() {
    'use strict';

    // ========================================
    // API endpoint resolution
    // ========================================
    function getApiEndpoint() {
        try {
            const saved = localStorage.getItem('edison_settings');
            if (saved) {
                const settings = JSON.parse(saved);
                if (settings.apiEndpoint) return settings.apiEndpoint;
            }
        } catch (e) { /* ignore */ }
        const protocol = window.location.protocol || 'http:';
        const host = window.location.hostname || 'localhost';
        return `${protocol}//${host}:8811`;
    }

    const API = getApiEndpoint();

    // ========================================
    // Utility helpers
    // ========================================
    function formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function formatDate(ts) {
        if (!ts) return '';
        const d = new Date(ts * 1000);
        return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    }

    function showStatus(el, msg, type) {
        if (!el) return;
        el.style.display = 'block';
        el.className = 'feature-status ' + (type || '');
        el.innerHTML = msg;
    }

    function hideStatus(el) {
        if (el) el.style.display = 'none';
    }

    const HAS_THREE = typeof window.THREE !== 'undefined';

    // ========================================
    // 3D Viewer (normal generation)
    // ========================================
    const threeDViewer = {
        initialized: false,
        scene: null,
        camera: null,
        renderer: null,
        controls: null,
        modelRoot: null,
        draftRoot: null,
        canvas: null,
        wrap: null,
        autoRotate: true,
        wireframe: false,
        progressTimer: null,
        progressValue: 0,
        isGenerating: false,
    };

    function initThreeDViewer() {
        if (threeDViewer.initialized || !HAS_THREE) return;
        const canvas = document.getElementById('threeDCanvas');
        const wrap = document.getElementById('threeDCanvasWrap');
        if (!canvas || !wrap) return;

        const THREE = window.THREE;
        threeDViewer.canvas = canvas;
        threeDViewer.wrap = wrap;
        threeDViewer.scene = new THREE.Scene();
        threeDViewer.scene.background = new THREE.Color(0x11131a);

        threeDViewer.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
        threeDViewer.camera.position.set(2.6, 1.8, 3.2);

        threeDViewer.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
        threeDViewer.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        threeDViewer.renderer.outputEncoding = THREE.sRGBEncoding;

        const ambient = new THREE.AmbientLight(0xffffff, 0.7);
        const keyLight = new THREE.DirectionalLight(0xffffff, 0.9);
        keyLight.position.set(4, 6, 4);
        const fillLight = new THREE.DirectionalLight(0x88aaff, 0.45);
        fillLight.position.set(-4, 2, -4);
        threeDViewer.scene.add(ambient, keyLight, fillLight);

        const grid = new THREE.GridHelper(8, 16, 0x3b4252, 0x2e3440);
        grid.position.y = -1.05;
        threeDViewer.scene.add(grid);

        if (THREE.OrbitControls) {
            threeDViewer.controls = new THREE.OrbitControls(threeDViewer.camera, threeDViewer.renderer.domElement);
            threeDViewer.controls.enableDamping = true;
            threeDViewer.controls.dampingFactor = 0.08;
            threeDViewer.controls.target.set(0, 0.2, 0);
            threeDViewer.controls.update();
        }

        threeDViewer.initialized = true;
        resizeThreeDViewer();
        animateThreeDViewer();
    }

    function resizeThreeDViewer() {
        if (!threeDViewer.initialized || !threeDViewer.wrap) return;
        const w = Math.max(100, threeDViewer.wrap.clientWidth);
        const h = Math.max(180, threeDViewer.wrap.clientHeight);
        threeDViewer.camera.aspect = w / h;
        threeDViewer.camera.updateProjectionMatrix();
        threeDViewer.renderer.setSize(w, h, false);
    }

    function animateThreeDViewer() {
        if (!threeDViewer.initialized) return;
        requestAnimationFrame(animateThreeDViewer);

        if (threeDViewer.controls) {
            if (threeDViewer.autoRotate && !threeDViewer.isGenerating) {
                threeDViewer.controls.autoRotate = true;
                threeDViewer.controls.autoRotateSpeed = 1.4;
            } else {
                threeDViewer.controls.autoRotate = false;
            }
            threeDViewer.controls.update();
        } else if (threeDViewer.modelRoot && threeDViewer.autoRotate) {
            threeDViewer.modelRoot.rotation.y += 0.006;
        }

        if (threeDViewer.draftRoot && threeDViewer.isGenerating) {
            const t = performance.now() * 0.001;
            threeDViewer.draftRoot.rotation.y += 0.02;
            threeDViewer.draftRoot.rotation.x = Math.sin(t * 1.2) * 0.2;
            const pulse = 0.9 + Math.sin(t * 3.2) * 0.08;
            threeDViewer.draftRoot.scale.setScalar(pulse);
        }

        threeDViewer.renderer.render(threeDViewer.scene, threeDViewer.camera);
    }

    function clearThreeDObjects() {
        if (!threeDViewer.scene) return;
        if (threeDViewer.modelRoot) {
            threeDViewer.scene.remove(threeDViewer.modelRoot);
            threeDViewer.modelRoot = null;
        }
        if (threeDViewer.draftRoot) {
            threeDViewer.scene.remove(threeDViewer.draftRoot);
            threeDViewer.draftRoot = null;
        }
    }

    function set3DOverlayVisible(visible) {
        const overlay = document.getElementById('threeDOverlay');
        if (!overlay) return;
        overlay.style.display = visible ? 'flex' : 'none';
    }

    function set3DProgress(percent, text) {
        const pctEl = document.getElementById('threeDProgressPct');
        const textEl = document.getElementById('threeDProgressText');
        const ring = document.getElementById('threeDProgressRing');
        const pct = Math.max(0, Math.min(100, Math.round(percent)));
        if (pctEl) pctEl.textContent = `${pct}%`;
        if (textEl && text) textEl.textContent = text;
        if (ring) {
            const radius = 45;
            const circumference = 2 * Math.PI * radius;
            ring.style.strokeDasharray = `${circumference}`;
            ring.style.strokeDashoffset = `${circumference * (1 - pct / 100)}`;
        }
    }

    function createDraftGenerationMesh(promptText) {
        if (!HAS_THREE) return;
        const THREE = window.THREE;
        const root = new THREE.Group();
        const material = new THREE.MeshStandardMaterial({
            color: 0x7aa2ff,
            emissive: 0x223355,
            roughness: 0.35,
            metalness: 0.3,
            transparent: true,
            opacity: 0.9,
            wireframe: threeDViewer.wireframe,
        });

        const count = 56;
        for (let i = 0; i < count; i++) {
            const gType = i % 3;
            let geom;
            if (gType === 0) geom = new THREE.BoxGeometry(0.24, 0.24, 0.24);
            else if (gType === 1) geom = new THREE.IcosahedronGeometry(0.14, 1);
            else geom = new THREE.ConeGeometry(0.12, 0.24, 6);
            const m = material.clone();
            const mesh = new THREE.Mesh(geom, m);
            mesh.position.set((Math.random() - 0.5) * 1.8, (Math.random() - 0.5) * 1.8, (Math.random() - 0.5) * 1.8);
            mesh.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI);
            const s = 0.6 + Math.random() * 0.7;
            mesh.scale.setScalar(s);
            root.add(mesh);
        }

        const textHint = new THREE.Mesh(
            new THREE.TorusKnotGeometry(0.45, 0.12, 90, 14),
            new THREE.MeshStandardMaterial({
                color: 0x9c7bff,
                emissive: 0x281f52,
                roughness: 0.2,
                metalness: 0.55,
                wireframe: threeDViewer.wireframe,
            })
        );
        textHint.userData.prompt = promptText || '';
        root.add(textHint);

        root.position.y = 0.1;
        return root;
    }

    function fitThreeDCameraToObject(object3D) {
        if (!object3D || !threeDViewer.camera) return;
        const THREE = window.THREE;
        const box = new THREE.Box3().setFromObject(object3D);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z, 0.8);
        const fov = threeDViewer.camera.fov * (Math.PI / 180);
        const distance = (maxDim / 2) / Math.tan(fov / 2) * 1.6;

        threeDViewer.camera.position.set(center.x + distance * 0.9, center.y + distance * 0.55, center.z + distance * 0.95);
        if (threeDViewer.controls) {
            threeDViewer.controls.target.copy(center);
            threeDViewer.controls.update();
        } else {
            threeDViewer.camera.lookAt(center);
        }
    }

    function startLive3DProgress(promptText) {
        if (!HAS_THREE) return;
        const container = document.getElementById('threeDViewerContainer');
        if (container) container.style.display = 'block';
        initThreeDViewer();
        clearThreeDObjects();
        threeDViewer.isGenerating = true;
        threeDViewer.progressValue = 0;
        set3DOverlayVisible(true);
        set3DProgress(0, 'Initializing generation pipeline...');

        const draft = createDraftGenerationMesh(promptText);
        threeDViewer.draftRoot = draft;
        if (draft) {
            threeDViewer.scene.add(draft);
            fitThreeDCameraToObject(draft);
        }

        if (threeDViewer.progressTimer) {
            clearInterval(threeDViewer.progressTimer);
        }
        threeDViewer.progressTimer = setInterval(() => {
            if (!threeDViewer.isGenerating) return;
            if (threeDViewer.progressValue < 95) {
                // Slow down progress as it gets higher to avoid appearing stuck
                let step;
                if (threeDViewer.progressValue < 30) step = 3;
                else if (threeDViewer.progressValue < 60) step = 1.5;
                else if (threeDViewer.progressValue < 80) step = 0.5;
                else step = 0.2;
                threeDViewer.progressValue = Math.min(95, threeDViewer.progressValue + step);
                let text;
                if (threeDViewer.progressValue < 25) text = 'Loading models & initializing...';
                else if (threeDViewer.progressValue < 50) text = 'Sampling latent space (may use CPU)...';
                else if (threeDViewer.progressValue < 75) text = 'Refining geometry and topology...';
                else if (threeDViewer.progressValue < 90) text = 'Decoding mesh (CPU mode takes a few minutes)...';
                else text = 'Finalizing mesh — almost done...';
                set3DProgress(threeDViewer.progressValue, text);
            }
        }, 800);
    }

    function completeLive3DProgress() {
        threeDViewer.isGenerating = false;
        if (threeDViewer.progressTimer) {
            clearInterval(threeDViewer.progressTimer);
            threeDViewer.progressTimer = null;
        }
        set3DProgress(100, 'Model generated. Loading final mesh...');
    }

    function failLive3DProgress(message) {
        threeDViewer.isGenerating = false;
        if (threeDViewer.progressTimer) {
            clearInterval(threeDViewer.progressTimer);
            threeDViewer.progressTimer = null;
        }
        set3DProgress(0, message || 'Generation failed');
    }

    async function load3DModelIntoViewer(url, formatHint) {
        if (!HAS_THREE) return;
        initThreeDViewer();
        clearThreeDObjects();
        const THREE = window.THREE;
        const ext = (formatHint || '').toLowerCase();
        let model = null;

        const normalizeRoot = (root) => {
            root.traverse((child) => {
                if (child.isMesh && child.material) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach((mat) => {
                            if ('wireframe' in mat) mat.wireframe = threeDViewer.wireframe;
                        });
                    } else if ('wireframe' in child.material) {
                        child.material.wireframe = threeDViewer.wireframe;
                    }
                    child.castShadow = true;
                    child.receiveShadow = true;
                }
            });
            return root;
        };

        try {
            if (ext === 'glb' && THREE.GLTFLoader) {
                model = await new Promise((resolve, reject) => {
                    const loader = new THREE.GLTFLoader();
                    loader.load(url, (gltf) => resolve(gltf.scene), undefined, reject);
                });
            } else if (ext === 'obj' && THREE.OBJLoader) {
                model = await new Promise((resolve, reject) => {
                    const loader = new THREE.OBJLoader();
                    loader.load(url, resolve, undefined, reject);
                });
            } else if (ext === 'ply' && THREE.PLYLoader) {
                const geometry = await new Promise((resolve, reject) => {
                    const loader = new THREE.PLYLoader();
                    loader.load(url, resolve, undefined, reject);
                });
                geometry.computeVertexNormals();
                const material = new THREE.MeshStandardMaterial({
                    color: 0x8fb6ff,
                    roughness: 0.4,
                    metalness: 0.25,
                    wireframe: threeDViewer.wireframe,
                });
                model = new THREE.Mesh(geometry, material);
            }
        } catch (e) {
            console.error('3D model loading failed:', e);
        }

        if (!model) {
            // Show error in viewer instead of a fake mesh
            const info = document.getElementById('threeDViewerInfo');
            if (info) {
                info.textContent = `Failed to load 3D model (${ext.toUpperCase() || 'unknown format'}). The file may be corrupted or the format unsupported.`;
            }
            set3DProgress(0, 'Model loading failed. Try a different format.');
            setTimeout(() => set3DOverlayVisible(false), 2000);
            return;
        }

        threeDViewer.modelRoot = normalizeRoot(model);
        threeDViewer.scene.add(threeDViewer.modelRoot);
        fitThreeDCameraToObject(threeDViewer.modelRoot);

        const info = document.getElementById('threeDViewerInfo');
        if (info) {
            info.textContent = `Interactive preview loaded · ${ext.toUpperCase() || 'MODEL'} · Drag to orbit · Scroll to zoom`;
        }

        setTimeout(() => set3DOverlayVisible(false), 240);
    }

    function applyThreeDWireframeState() {
        const apply = (obj) => {
            if (!obj) return;
            obj.traverse((child) => {
                if (child.isMesh && child.material) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach((mat) => {
                            if ('wireframe' in mat) mat.wireframe = threeDViewer.wireframe;
                        });
                    } else if ('wireframe' in child.material) {
                        child.material.wireframe = threeDViewer.wireframe;
                    }
                }
            });
        };
        apply(threeDViewer.modelRoot);
        apply(threeDViewer.draftRoot);
    }

    window.viewer3DResetCamera = function() {
        if (!threeDViewer.initialized) return;
        const target = threeDViewer.modelRoot || threeDViewer.draftRoot;
        if (target) fitThreeDCameraToObject(target);
    };

    window.viewer3DToggleWireframe = function() {
        threeDViewer.wireframe = !threeDViewer.wireframe;
        applyThreeDWireframeState();
    };

    window.viewer3DToggleAutoRotate = function() {
        threeDViewer.autoRotate = !threeDViewer.autoRotate;
    };

    window.viewer3DFullscreen = function() {
        if (!threeDViewer.wrap) return;
        const element = threeDViewer.wrap;
        if (document.fullscreenElement) {
            document.exitFullscreen?.();
            return;
        }
        element.requestFullscreen?.();
    };

    // ========================================
    // Minecraft 3D Preview Viewer
    // ========================================
    const mcViewer = {
        initialized: false,
        scene: null,
        camera: null,
        renderer: null,
        controls: null,
        modelRoot: null,
        canvas: null,
        wrap: null,
        autoRotate: true,
    };

    function initMCViewer() {
        if (mcViewer.initialized || !HAS_THREE) return;
        const canvas = document.getElementById('mcPreviewCanvas');
        const wrap = document.getElementById('mcCanvasWrap');
        if (!canvas || !wrap) return;

        const THREE = window.THREE;
        mcViewer.canvas = canvas;
        mcViewer.wrap = wrap;
        mcViewer.scene = new THREE.Scene();
        mcViewer.scene.background = new THREE.Color(0x131722);

        mcViewer.camera = new THREE.PerspectiveCamera(52, 1, 0.1, 1000);
        mcViewer.camera.position.set(2.5, 2.0, 2.5);

        mcViewer.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
        mcViewer.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        mcViewer.renderer.outputEncoding = THREE.sRGBEncoding;

        const ambient = new THREE.AmbientLight(0xffffff, 0.85);
        const lightA = new THREE.DirectionalLight(0xffffff, 0.8);
        lightA.position.set(4, 6, 3);
        const lightB = new THREE.DirectionalLight(0xaad8ff, 0.35);
        lightB.position.set(-3, 2, -4);
        mcViewer.scene.add(ambient, lightA, lightB);

        if (THREE.OrbitControls) {
            mcViewer.controls = new THREE.OrbitControls(mcViewer.camera, mcViewer.renderer.domElement);
            mcViewer.controls.enableDamping = true;
            mcViewer.controls.dampingFactor = 0.08;
            mcViewer.controls.target.set(0, 0.4, 0);
            mcViewer.controls.update();
        }

        const grid = new THREE.GridHelper(6, 12, 0x3a4666, 0x253047);
        grid.position.y = -1.02;
        mcViewer.scene.add(grid);

        mcViewer.initialized = true;
        resizeMCViewer();
        animateMCViewer();
    }

    function resizeMCViewer() {
        if (!mcViewer.initialized || !mcViewer.wrap) return;
        const w = Math.max(100, mcViewer.wrap.clientWidth);
        const h = Math.max(160, mcViewer.wrap.clientHeight);
        mcViewer.camera.aspect = w / h;
        mcViewer.camera.updateProjectionMatrix();
        mcViewer.renderer.setSize(w, h, false);
    }

    function animateMCViewer() {
        if (!mcViewer.initialized) return;
        requestAnimationFrame(animateMCViewer);

        if (mcViewer.controls) {
            mcViewer.controls.autoRotate = mcViewer.autoRotate;
            mcViewer.controls.autoRotateSpeed = 1.5;
            mcViewer.controls.update();
        } else if (mcViewer.modelRoot && mcViewer.autoRotate) {
            mcViewer.modelRoot.rotation.y += 0.01;
        }

        mcViewer.renderer.render(mcViewer.scene, mcViewer.camera);
    }

    function clearMCPreviewModel() {
        if (!mcViewer.scene || !mcViewer.modelRoot) return;
        mcViewer.scene.remove(mcViewer.modelRoot);
        mcViewer.modelRoot = null;
    }

    function createMinecraftGeometry(modelType) {
        const THREE = window.THREE;
        const group = new THREE.Group();
        const addBox = (w, h, d, x, y, z) => {
            const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d));
            mesh.position.set(x || 0, y || 0, z || 0);
            group.add(mesh);
        };
        const addPlane = (w, h, x, y, z, ry) => {
            const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h));
            mesh.position.set(x || 0, y || 0, z || 0);
            if (typeof ry === 'number') mesh.rotation.y = ry;
            group.add(mesh);
        };

        switch (modelType) {
            case 'item':
                addPlane(1.4, 1.4, 0, 0.2, 0, 0);
                break;
            case 'crop':
            case 'cross':
                addPlane(1.3, 1.3, 0, 0, 0, Math.PI / 4);
                addPlane(1.3, 1.3, 0, 0, 0, -Math.PI / 4);
                break;
            case 'slab':
                addBox(1.6, 0.8, 1.6, 0, -0.4, 0);
                break;
            case 'stairs':
                addBox(1.6, 0.8, 1.6, 0, -0.4, 0);
                addBox(1.6, 0.8, 0.8, 0, 0.4, -0.4);
                break;
            case 'fence':
                addBox(0.36, 1.6, 0.36, 0, 0, 0);
                addBox(1.4, 0.22, 0.22, 0, 0.42, 0);
                addBox(1.4, 0.22, 0.22, 0, -0.1, 0);
                break;
            case 'wall':
                addBox(1.1, 1.2, 0.45, 0, -0.2, 0);
                break;
            case 'pane':
                addBox(0.22, 1.5, 1.5, 0, 0, 0);
                break;
            case 'block':
            default:
                addBox(1.6, 1.6, 1.6, 0, -0.2, 0);
                break;
        }

        return group;
    }

    async function renderMinecraftModelPreview(textureUrl, modelType) {
        if (!HAS_THREE) return;
        const panel = document.getElementById('mc3DPreview');
        if (panel) panel.style.display = 'block';

        initMCViewer();
        clearMCPreviewModel();
        const THREE = window.THREE;
        const texture = await new Promise((resolve, reject) => {
            const loader = new THREE.TextureLoader();
            loader.load(textureUrl, resolve, undefined, reject);
        });

        texture.magFilter = THREE.NearestFilter;
        texture.minFilter = THREE.NearestFilter;
        texture.generateMipmaps = false;
        texture.wrapS = THREE.ClampToEdgeWrapping;
        texture.wrapT = THREE.ClampToEdgeWrapping;

        const material = new THREE.MeshStandardMaterial({
            map: texture,
            roughness: 0.65,
            metalness: 0.08,
            transparent: true,
            side: THREE.DoubleSide,
        });

        const root = createMinecraftGeometry(modelType || 'block');
        root.traverse((child) => {
            if (child.isMesh) {
                child.material = material;
                child.castShadow = true;
                child.receiveShadow = true;
            }
        });

        mcViewer.modelRoot = root;
        mcViewer.scene.add(root);

        const box = new THREE.Box3().setFromObject(root);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z, 1.2);
        const dist = maxDim * 1.9;
        mcViewer.camera.position.set(center.x + dist * 0.8, center.y + dist * 0.65, center.z + dist * 0.8);
        if (mcViewer.controls) {
            mcViewer.controls.target.copy(center);
            mcViewer.controls.update();
        }
    }

    window.mcResetCamera = function() {
        if (!mcViewer.initialized || !mcViewer.modelRoot) return;
        const THREE = window.THREE;
        const box = new THREE.Box3().setFromObject(mcViewer.modelRoot);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z, 1.2);
        const dist = maxDim * 1.9;
        mcViewer.camera.position.set(center.x + dist * 0.8, center.y + dist * 0.65, center.z + dist * 0.8);
        if (mcViewer.controls) {
            mcViewer.controls.target.copy(center);
            mcViewer.controls.update();
        }
    };

    window.mcToggleAutoRotate = function() {
        mcViewer.autoRotate = !mcViewer.autoRotate;
    };

    // ========================================
    // FEATURE 1: 3D Model Generation
    // ========================================
    let threeDPanelOpen = false;
    let threeDMode = 'text';
    let threeDImageData = null;

    window.toggle3DPanel = function() {
        const panel = document.getElementById('threeDPanel');
        if (!panel) return;
        threeDPanelOpen = !threeDPanelOpen;
        if (threeDPanelOpen) {
            panel.style.visibility = 'visible';
            panel.style.transform = 'translateX(0)';
            initThreeDViewer();
            resizeThreeDViewer();
            load3DModelsList();
        } else {
            panel.style.transform = 'translateX(100%)';
            setTimeout(() => { panel.style.visibility = 'hidden'; }, 300);
        }
    };

    window.set3DMode = function(mode) {
        threeDMode = mode;
        document.querySelectorAll('[data-3d-mode]').forEach(btn => {
            btn.classList.toggle('active', btn.dataset['3dMode'] === mode);
        });
        const textInput = document.getElementById('threeDTextInput');
        const imgInput = document.getElementById('threeDImageInput');
        if (textInput) textInput.style.display = mode === 'text' ? 'block' : 'none';
        if (imgInput) imgInput.style.display = mode === 'image' ? 'block' : 'none';
    };

    // Dropzone for 3D image input
    function init3DDropzone() {
        const dropzone = document.getElementById('threeDDropzone');
        const fileInput = document.getElementById('threeDImageFile');
        const preview = document.getElementById('threeDImagePreview');
        if (!dropzone || !fileInput) return;

        dropzone.addEventListener('click', () => fileInput.click());
        dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
        dropzone.addEventListener('drop', e => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleThreeDFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', e => {
            if (e.target.files.length) handleThreeDFile(e.target.files[0]);
        });
    }

    function handleThreeDFile(file) {
        if (!file.type.startsWith('image/')) { alert('Please select an image file'); return; }
        const reader = new FileReader();
        reader.onload = e => {
            threeDImageData = e.target.result;
            const preview = document.getElementById('threeDImagePreview');
            const dropzone = document.getElementById('threeDDropzone');
            if (preview) {
                preview.innerHTML = `<img src="${threeDImageData}" alt="Reference"><button class="feature-img-remove" onclick="window.remove3DImage()">✕</button>`;
                preview.style.display = 'block';
            }
            if (dropzone) dropzone.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    window.remove3DImage = function() {
        threeDImageData = null;
        const preview = document.getElementById('threeDImagePreview');
        const dropzone = document.getElementById('threeDDropzone');
        if (preview) { preview.innerHTML = ''; preview.style.display = 'none'; }
        if (dropzone) dropzone.style.display = 'flex';
        const fileInput = document.getElementById('threeDImageFile');
        if (fileInput) fileInput.value = '';
    };

    window.generate3DModel = async function() {
        const prompt = document.getElementById('threeDPrompt')?.value?.trim() || '';
        const format = document.getElementById('threeDFormat')?.value || 'glb';
        const steps = parseInt(document.getElementById('threeDSteps')?.value || '64');
        const guidance = parseFloat(document.getElementById('threeDGuidance')?.value || '15');
        const statusEl = document.getElementById('threeDStatus');
        const btn = document.getElementById('threeDGenerateBtn');

        if (threeDMode === 'text' && !prompt) {
            showStatus(statusEl, '⚠️ Please enter a description for the 3D model.', 'warning');
            return;
        }
        if (threeDMode === 'image' && !threeDImageData) {
            showStatus(statusEl, '⚠️ Please upload a reference image.', 'warning');
            return;
        }

        showStatus(statusEl, '🔄 Generating 3D model... This may take a few minutes.', 'loading');
        if (HAS_THREE) startLive3DProgress(prompt || 'Image reference');
        if (btn) btn.disabled = true;

        try {
            const body = {
                prompt: prompt,
                format: format,
                num_steps: steps,
                guidance_scale: guidance,
            };
            if (threeDMode === 'image' && threeDImageData) {
                body.image = threeDImageData;
            }

            // 10 minute timeout for CPU fallback scenarios
            const controller = new AbortController();
            const fetchTimeout = setTimeout(() => controller.abort(), 600000);

            const resp = await fetch(`${API}/generate-3d`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body),
                signal: controller.signal
            });
            clearTimeout(fetchTimeout);

            if (!resp.ok) {
                const err = await resp.json().catch(() => ({detail: 'Server error'}));
                throw new Error(err.detail || `HTTP ${resp.status}`);
            }

            const data = await resp.json();
            showStatus(statusEl, `✅ ${data.message}`, 'success');
            add3DResult(data);
            if (HAS_THREE) {
                completeLive3DProgress();
                await load3DModelIntoViewer(`${API}${data.download_url}`, data.format);
            }
            load3DModelsList();
        } catch (err) {
            const msg = err.name === 'AbortError'
                ? 'Generation timed out after 10 minutes. Try fewer steps or a simpler prompt.'
                : err.message;
            if (HAS_THREE) failLive3DProgress(`Generation failed: ${msg}`);
            showStatus(statusEl, `❌ Error: ${err.message}`, 'error');
        } finally {
            if (btn) btn.disabled = false;
        }
    };

    function add3DResult(data) {
        const container = document.getElementById('threeDResults');
        if (!container) return;
        const card = document.createElement('div');
        card.className = 'threeD-result-card';
        card.innerHTML = `
            <div class="threeD-result-info">
                <span class="threeD-result-icon">🧊</span>
                <div>
                    <div class="threeD-result-name">${data.model_id}.${data.format}</div>
                    <div class="threeD-result-meta">${data.format.toUpperCase()} format</div>
                </div>
            </div>
            <div class="threeD-result-actions">
                <button class="threeD-preview-btn" onclick="window.preview3DModel('${data.download_url}', '${data.format}')">👁 Preview</button>
                <a href="${API}${data.download_url}" class="threeD-download-btn" download>⬇ Download</a>
            </div>
        `;
        container.prepend(card);
    }

    window.preview3DModel = async function(downloadUrl, format) {
        if (!HAS_THREE) return;
        const container = document.getElementById('threeDViewerContainer');
        if (container) container.style.display = 'block';
        set3DOverlayVisible(true);
        set3DProgress(100, 'Loading saved model preview...');
        await load3DModelIntoViewer(`${API}${downloadUrl}`, format || 'obj');
    };

    async function load3DModelsList() {
        const container = document.getElementById('threeDResults');
        if (!container) return;
        try {
            const resp = await fetch(`${API}/3d-models/list`);
            if (!resp.ok) return;
            const data = await resp.json();
            if (data.models && data.models.length > 0) {
                container.innerHTML = '';
                data.models.forEach(m => {
                    const card = document.createElement('div');
                    card.className = 'threeD-result-card';
                    const format = (m.format || '').toLowerCase();
                    card.innerHTML = `
                        <div class="threeD-result-info">
                            <span class="threeD-result-icon">🧊</span>
                            <div>
                                <div class="threeD-result-name">${m.filename}</div>
                                <div class="threeD-result-meta">${m.format.toUpperCase()} · ${formatBytes(m.size_bytes)}</div>
                            </div>
                        </div>
                        <div class="threeD-result-actions">
                            <button class="threeD-preview-btn" onclick="window.preview3DModel('${m.download_url}', '${format}')">👁 Preview</button>
                            <a href="${API}${m.download_url}" class="threeD-download-btn" download>⬇ Download</a>
                        </div>
                    `;
                    container.appendChild(card);
                });
            }
        } catch (e) {
            console.error('Failed to load 3D models list:', e);
        }
    }


    // ========================================
    // FEATURE 2 & 3: Minecraft Tools
    // ========================================
    let mcPanelOpen = false;
    let mcImageData = null;
    let mcGeneratedTextures = [];

    window.toggleMinecraftPanel = function() {
        const panel = document.getElementById('minecraftPanel');
        if (!panel) return;
        mcPanelOpen = !mcPanelOpen;
        if (mcPanelOpen) {
            panel.style.visibility = 'visible';
            panel.style.transform = 'translateX(0)';
            initMCViewer();
            resizeMCViewer();
            checkMCInstallStatus();
            loadMCGallery();
        } else {
            panel.style.transform = 'translateX(100%)';
            setTimeout(() => { panel.style.visibility = 'hidden'; }, 300);
        }
    };

    window.switchMCTab = function(tab) {
        document.querySelectorAll('.mc-tab').forEach(t => t.classList.toggle('active', t.dataset.mcTab === tab));
        document.getElementById('mcTexturesTab').style.display = tab === 'textures' ? 'block' : 'none';
        document.getElementById('mcModelsTab').style.display = tab === 'models' ? 'block' : 'none';
        document.getElementById('mcGalleryTab').style.display = tab === 'gallery' ? 'block' : 'none';
        if (tab === 'gallery') loadMCGallery();
        if (tab === 'models') loadMCTextureSelect();
    };

    // Minecraft dropzone
    function initMCDropzone() {
        const dropzone = document.getElementById('mcDropzone');
        const fileInput = document.getElementById('mcImageFile');
        if (!dropzone || !fileInput) return;

        dropzone.addEventListener('click', () => fileInput.click());
        dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
        dropzone.addEventListener('drop', e => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleMCFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', e => {
            if (e.target.files.length) handleMCFile(e.target.files[0]);
        });
    }

    function handleMCFile(file) {
        if (!file.type.startsWith('image/')) { alert('Please select an image file'); return; }
        const reader = new FileReader();
        reader.onload = e => {
            mcImageData = e.target.result;
            const preview = document.getElementById('mcImagePreview');
            const dropzone = document.getElementById('mcDropzone');
            if (preview) {
                preview.innerHTML = `<img src="${mcImageData}" alt="Reference" style="max-width:100%;image-rendering:pixelated;"><button class="feature-img-remove" onclick="window.removeMCImage()">✕</button>`;
                preview.style.display = 'block';
            }
            if (dropzone) dropzone.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    window.removeMCImage = function() {
        mcImageData = null;
        const preview = document.getElementById('mcImagePreview');
        const dropzone = document.getElementById('mcDropzone');
        if (preview) { preview.innerHTML = ''; preview.style.display = 'none'; }
        if (dropzone) dropzone.style.display = 'flex';
        const fileInput = document.getElementById('mcImageFile');
        if (fileInput) fileInput.value = '';
    };

    async function checkMCInstallStatus() {
        const statusEl = document.getElementById('mcInstallStatus');
        if (!statusEl) return;
        try {
            const resp = await fetch(`${API}/minecraft/install-status`);
            if (!resp.ok) return;
            const data = await resp.json();
            let html = '<div class="mc-status-grid">';
            html += `<div class="mc-status-item ${data.details.comfyui === 'running' ? 'ok' : 'warn'}">
                <span>${data.details.comfyui === 'running' ? '✅' : '⚠️'}</span> ComfyUI: ${data.details.comfyui || 'unknown'}
            </div>`;
            html += `<div class="mc-status-item ${data.details.pillow === 'installed' ? 'ok' : 'warn'}">
                <span>${data.details.pillow === 'installed' ? '✅' : '⚠️'}</span> Pillow: ${data.details.pillow || 'unknown'}
            </div>`;
            html += `<div class="mc-status-item ${data.model_gen ? 'ok' : 'warn'}">
                <span>${data.model_gen ? '✅' : 'ℹ️'}</span> 3D Model Gen: ${data.model_gen ? 'available' : 'optional'}
            </div>`;
            html += '</div>';
            if (data.details.comfyui !== 'running') {
                html += '<p class="mc-install-note">⚠️ ComfyUI must be running for texture generation. Start it first.</p>';
            }
            statusEl.innerHTML = html;
        } catch (e) {
            statusEl.innerHTML = '<p class="mc-install-note">⚠️ Could not check install status.</p>';
        }
    }

    window.generateMinecraftTexture = async function() {
        const prompt = document.getElementById('mcPrompt')?.value?.trim() || '';
        const textureType = document.getElementById('mcTextureType')?.value || 'block';
        const style = document.getElementById('mcStyle')?.value || 'pixel_art';
        const resolution = parseInt(document.getElementById('mcResolution')?.value || '16');
        const statusEl = document.getElementById('mcTextureStatus');
        const btn = document.getElementById('mcGenerateBtn');

        if (!prompt) {
            showStatus(statusEl, '⚠️ Please describe the texture you want to generate.', 'warning');
            return;
        }

        showStatus(statusEl, '🔄 Submitting texture generation request...', 'loading');
        if (btn) btn.disabled = true;

        try {
            const body = {
                prompt: prompt,
                texture_type: textureType,
                style: style,
                size: resolution,
            };
            if (mcImageData) body.image = mcImageData;

            const resp = await fetch(`${API}/minecraft/generate-texture`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body)
            });

            if (!resp.ok) {
                const err = await resp.json().catch(() => ({detail: 'Server error'}));
                throw new Error(err.detail || `HTTP ${resp.status}`);
            }

            const data = await resp.json();
            if (data.status === 'generating' && data.prompt_id) {
                showStatus(statusEl, '⏳ Generating texture with ComfyUI... Polling for result.', 'loading');
                pollMCTexture(data.prompt_id, statusEl, btn);
            } else {
                showStatus(statusEl, `✅ ${data.message || 'Texture generated!'}`, 'success');
                if (btn) btn.disabled = false;
            }
        } catch (err) {
            showStatus(statusEl, `❌ Error: ${err.message}`, 'error');
            if (btn) btn.disabled = false;
        }
    };

    async function pollMCTexture(promptId, statusEl, btn, attempts) {
        attempts = attempts || 0;
        if (attempts > 120) { // 2 min max
            showStatus(statusEl, '⚠️ Texture generation timed out. Check ComfyUI.', 'warning');
            if (btn) btn.disabled = false;
            return;
        }
        try {
            const resp = await fetch(`${API}/minecraft/texture-status/${promptId}`);
            const data = await resp.json();
            if (data.status === 'complete') {
                showStatus(statusEl, `✅ Texture generated! Type: ${data.texture_type}, Size: ${data.target_size}×${data.target_size}`, 'success');
                if (data.download_url) {
                    const imgHtml = `<div class="mc-texture-result">
                        <img src="${API}${data.download_url}" alt="Generated texture" class="mc-texture-img">
                        <div class="mc-texture-actions">
                            <a href="${API}${data.download_url}" download class="mc-action-btn">⬇ Download ${data.target_size}×${data.target_size}</a>
                            ${data.full_res_url ? `<a href="${API}${data.full_res_url}" download class="mc-action-btn">⬇ Full Res</a>` : ''}
                        </div>
                    </div>`;
                    statusEl.innerHTML += imgHtml;
                }
                mcGeneratedTextures.push(data);
                loadMCTextureSelect();
                if (btn) btn.disabled = false;
                return;
            } else if (data.status === 'error') {
                showStatus(statusEl, `❌ Error: ${data.detail || 'Unknown error'}`, 'error');
                if (btn) btn.disabled = false;
                return;
            }
        } catch (e) {
            // Network error, keep polling
        }
        setTimeout(() => pollMCTexture(promptId, statusEl, btn, attempts + 1), 1000);
    }

    async function loadMCTextureSelect() {
        const select = document.getElementById('mcModelTexture');
        if (!select) return;
        try {
            const resp = await fetch(`${API}/minecraft/textures/list`);
            if (!resp.ok) return;
            const data = await resp.json();
            select.innerHTML = '<option value="">-- Select a texture --</option>';
            if (data.textures) {
                data.textures.forEach(t => {
                    const opt = document.createElement('option');
                    opt.value = t.filename;
                    opt.textContent = `${t.texture_type} - ${t.filename}`;
                    select.appendChild(opt);
                });
            }
        } catch (e) {
            console.error('Failed to load textures list:', e);
        }
    }

    async function loadMCGallery() {
        const grid = document.getElementById('mcGalleryGrid');
        if (!grid) return;
        try {
            const resp = await fetch(`${API}/minecraft/textures/list`);
            if (!resp.ok) return;
            const data = await resp.json();
            if (!data.textures || data.textures.length === 0) {
                grid.innerHTML = '<p class="mc-empty">No Minecraft assets generated yet. Create some textures!</p>';
                return;
            }
            grid.innerHTML = '';
            data.textures.forEach(t => {
                const card = document.createElement('div');
                card.className = 'mc-gallery-card';
                card.innerHTML = `
                    <img src="${API}${t.download_url}" alt="${t.filename}" class="mc-gallery-img">
                    <div class="mc-gallery-info">
                        <span class="mc-gallery-type">${t.texture_type}</span>
                        <a href="${API}${t.download_url}" download class="mc-gallery-dl">⬇</a>
                    </div>
                `;
                grid.appendChild(card);
            });
        } catch (e) {
            grid.innerHTML = '<p class="mc-empty">Could not load gallery.</p>';
        }
    }

    window.generateMinecraftModel = async function() {
        const textureFile = document.getElementById('mcModelTexture')?.value || '';
        const modelType = document.getElementById('mcModelType')?.value || 'block';
        const modelName = document.getElementById('mcModelName')?.value?.trim() || 'custom_block';
        const statusEl = document.getElementById('mcModelStatus');
        const btn = document.getElementById('mcModelGenBtn');

        if (!textureFile) {
            showStatus(statusEl, '⚠️ Please select a generated texture first.', 'warning');
            return;
        }

        showStatus(statusEl, '🔄 Generating Minecraft model package...', 'loading');
        if (btn) btn.disabled = true;

        try {
            const resp = await fetch(`${API}/minecraft/generate-model`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    texture_filename: textureFile,
                    model_type: modelType,
                    name: modelName,
                })
            });

            if (!resp.ok) {
                const err = await resp.json().catch(() => ({detail: 'Server error'}));
                throw new Error(err.detail || `HTTP ${resp.status}`);
            }

            const data = await resp.json();
            let html = `✅ ${data.message}<br>`;
            html += `<div class="mc-model-result">`;
            html += `<pre class="mc-model-json">${JSON.stringify(data.model_json, null, 2)}</pre>`;
            if (data.blockstate_json) {
                html += `<pre class="mc-model-json">${JSON.stringify(data.blockstate_json, null, 2)}</pre>`;
            }
            html += `<a href="${API}${data.download_url}" download class="mc-action-btn">📦 Download ZIP Package</a>`;
            html += `</div>`;
            showStatus(statusEl, html, 'success');
            if (HAS_THREE && textureFile) {
                await renderMinecraftModelPreview(`${API}/minecraft/texture/${textureFile}`, modelType);
            }
        } catch (err) {
            showStatus(statusEl, `❌ Error: ${err.message}`, 'error');
        } finally {
            if (btn) btn.disabled = false;
        }
    };


    // ========================================
    // FEATURE 4: File Manager / Data Console
    // ========================================
    let fmPanelOpen = false;
    let fmCurrentPath = '';
    let fmDeleteTarget = null;
    let fmAllItems = [];

    window.toggleFileManager = function() {
        const panel = document.getElementById('fileManagerPanel');
        if (!panel) return;
        fmPanelOpen = !fmPanelOpen;
        if (fmPanelOpen) {
            panel.style.visibility = 'visible';
            panel.style.transform = 'translateX(0)';
            loadStorageInfo();
            fmNavigate('');
        } else {
            panel.style.transform = 'translateX(100%)';
            setTimeout(() => { panel.style.visibility = 'hidden'; }, 300);
        }
    };

    window.fmNavigate = async function(path) {
        fmCurrentPath = path;
        const fileList = document.getElementById('fmFileList');
        if (!fileList) return;
        fileList.innerHTML = '<div class="fm-loading">Loading...</div>';
        updateBreadcrumb(path);

        try {
            const resp = await fetch(`${API}/files/browse?path=${encodeURIComponent(path)}`);
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({detail: 'Error'}));
                throw new Error(err.detail || `HTTP ${resp.status}`);
            }
            const data = await resp.json();

            if (data.type === 'file') {
                // Single file info
                fileList.innerHTML = renderFileDetail(data);
                return;
            }

            fmAllItems = data.items || [];
            renderFileList(fmAllItems);
        } catch (err) {
            fileList.innerHTML = `<div class="fm-error">❌ ${err.message}</div>`;
        }
    };

    function renderFileList(items) {
        const fileList = document.getElementById('fmFileList');
        if (!fileList) return;

        if (items.length === 0) {
            fileList.innerHTML = '<div class="fm-empty">📂 This folder is empty</div>';
            return;
        }

        let html = '';
        // Back button if not root
        if (fmCurrentPath) {
            const parentPath = fmCurrentPath.split('/').slice(0, -1).join('/');
            html += `<div class="fm-item fm-item-back" onclick="window.fmNavigate('${parentPath}')">
                <span class="fm-item-icon">⬆️</span>
                <span class="fm-item-name">..</span>
                <span class="fm-item-meta">Parent folder</span>
            </div>`;
        }

        items.forEach(item => {
            const isDir = item.type === 'directory';
            const icon = isDir ? '📁' : getFileIcon(item.extension || '');
            const size = isDir ? `${item.child_count || 0} items` : formatBytes(item.size_bytes);
            const deletable = item.can_delete;

            html += `<div class="fm-item ${isDir ? 'fm-item-dir' : 'fm-item-file'}" 
                ${isDir ? `onclick="window.fmNavigate('${item.path}')"` : ''}>
                <span class="fm-item-icon">${icon}</span>
                <span class="fm-item-name" title="${item.path}">${item.name}</span>
                <span class="fm-item-meta">${size}</span>
                <span class="fm-item-date">${formatDate(item.modified)}</span>
                ${deletable ? 
                    `<button class="fm-delete-btn" onclick="event.stopPropagation(); window.fmRequestDelete('${item.path}', '${item.name}', ${isDir})" title="Delete">🗑️</button>` :
                    `<span class="fm-protected" title="${item.delete_reason || 'Protected'}">🔒</span>`
                }
            </div>`;
        });

        fileList.innerHTML = html;
    }

    function renderFileDetail(file) {
        return `<div class="fm-file-detail">
            <div class="fm-detail-icon">${getFileIcon(file.extension)}</div>
            <h3>${file.name}</h3>
            <p>Size: ${formatBytes(file.size_bytes)}</p>
            <p>Modified: ${formatDate(file.modified)}</p>
            <p>Type: ${file.extension || 'Unknown'}</p>
            <p>Status: ${file.can_delete ? '🗑️ Deletable' : '🔒 Protected'}</p>
            <button class="fm-btn fm-btn-back" onclick="window.fmNavigate('${fmCurrentPath}')">← Back</button>
            ${file.can_delete ? `<button class="fm-btn fm-btn-delete" onclick="window.fmRequestDelete('${file.path}', '${file.name}', false)">🗑️ Delete</button>` : ''}
        </div>`;
    }

    function getFileIcon(ext) {
        const icons = {
            '.png': '🖼️', '.jpg': '🖼️', '.jpeg': '🖼️', '.gif': '🖼️', '.webp': '🖼️', '.svg': '🖼️', '.bmp': '🖼️',
            '.mp4': '🎬', '.avi': '🎬', '.mkv': '🎬', '.mov': '🎬', '.webm': '🎬',
            '.mp3': '🎵', '.wav': '🎵', '.ogg': '🎵', '.flac': '🎵', '.m4a': '🎵',
            '.obj': '🧊', '.glb': '🧊', '.ply': '🧊', '.stl': '🧊', '.fbx': '🧊',
            '.zip': '📦', '.tar': '📦', '.gz': '📦', '.7z': '📦',
            '.py': '🐍', '.js': '📜', '.html': '🌐', '.css': '🎨', '.json': '📋',
            '.md': '📝', '.txt': '📄', '.pdf': '📕', '.csv': '📊',
            '.yaml': '⚙️', '.yml': '⚙️', '.sh': '⚡',
        };
        return icons[ext?.toLowerCase()] || '📄';
    }

    function updateBreadcrumb(path) {
        const bc = document.getElementById('fmBreadcrumb');
        if (!bc) return;
        let html = `<span class="fm-crumb" onclick="window.fmNavigate('')">🏠 EDISON</span>`;
        if (path) {
            const parts = path.split('/');
            let accumulated = '';
            parts.forEach((part, i) => {
                accumulated += (i === 0 ? '' : '/') + part;
                const p = accumulated;
                html += ` <span class="fm-crumb-sep">/</span> <span class="fm-crumb" onclick="window.fmNavigate('${p}')">${part}</span>`;
            });
        }
        bc.innerHTML = html;
    }

    window.fmSearchFiles = function(query) {
        if (!query.trim()) {
            renderFileList(fmAllItems);
            return;
        }
        const filtered = fmAllItems.filter(item => 
            item.name.toLowerCase().includes(query.toLowerCase())
        );
        renderFileList(filtered);
    };

    // Delete flow
    window.fmRequestDelete = function(path, name, isDir) {
        fmDeleteTarget = { path, name, isDir };
        const dialog = document.getElementById('fmDeleteConfirm');
        const msg = document.getElementById('fmDeleteMsg');
        if (dialog) dialog.style.display = 'flex';
        if (msg) msg.textContent = `Are you sure you want to delete "${name}"${isDir ? ' and all its contents' : ''}?`;
    };

    window.fmCancelDelete = function() {
        fmDeleteTarget = null;
        const dialog = document.getElementById('fmDeleteConfirm');
        if (dialog) dialog.style.display = 'none';
    };

    window.fmConfirmDelete = async function() {
        if (!fmDeleteTarget) return;
        const dialog = document.getElementById('fmDeleteConfirm');
        if (dialog) dialog.style.display = 'none';

        try {
            const resp = await fetch(`${API}/files/delete`, {
                method: 'DELETE',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ path: fmDeleteTarget.path })
            });

            if (!resp.ok) {
                const err = await resp.json().catch(() => ({detail: 'Error'}));
                alert(`Cannot delete: ${err.detail || 'Unknown error'}`);
                return;
            }

            const data = await resp.json();
            // Show brief toast
            showFMToast(`Deleted "${fmDeleteTarget.name}" (freed ${formatBytes(data.size_freed)})`);
            // Refresh
            fmNavigate(fmCurrentPath);
        } catch (err) {
            alert(`Delete failed: ${err.message}`);
        } finally {
            fmDeleteTarget = null;
        }
    };

    function showFMToast(msg) {
        const toast = document.createElement('div');
        toast.className = 'fm-toast';
        toast.textContent = msg;
        document.body.appendChild(toast);
        setTimeout(() => toast.classList.add('show'), 10);
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    // Storage info
    async function loadStorageInfo() {
        const container = document.getElementById('fmDrives');
        if (!container) return;
        try {
            const resp = await fetch(`${API}/files/storage`);
            if (!resp.ok) throw new Error('Failed to load storage');
            const data = await resp.json();

            if (!data.drives || data.drives.length === 0) {
                container.innerHTML = '<div class="fm-no-drives">No drives detected</div>';
                return;
            }

            let html = '';
            data.drives.forEach(d => {
                const pct = d.percent_used;
                const barColor = pct > 90 ? '#e74c3c' : pct > 70 ? '#f39c12' : '#2ecc71';
                html += `<div class="fm-drive" onclick="window.fmNavigate('${d.mountpoint}')" style="cursor: pointer;" title="Browse ${d.mountpoint}">
                    <div class="fm-drive-header">
                        <span class="fm-drive-icon">💾</span>
                        <span class="fm-drive-name">${d.mountpoint}</span>
                        <span class="fm-drive-type">${d.fstype}</span>
                    </div>
                    <div class="fm-drive-bar">
                        <div class="fm-drive-fill" style="width: ${pct}%; background: ${barColor};"></div>
                    </div>
                    <div class="fm-drive-stats">
                        <span>${d.used_gb} GB used of ${d.total_gb} GB</span>
                        <span class="fm-drive-free">${d.free_gb} GB free</span>
                    </div>
                </div>`;
            });
            container.innerHTML = html;
        } catch (e) {
            container.innerHTML = '<div class="fm-error">Could not load storage info</div>';
        }
    }


    // ========================================
    // Initialization
    // ========================================
    function initNewFeatures() {
        console.log('🧊 Initializing new features...');
        init3DDropzone();
        initMCDropzone();
        if (HAS_THREE) {
            initThreeDViewer();
            initMCViewer();
            window.addEventListener('resize', () => {
                resizeThreeDViewer();
                resizeMCViewer();
            });
        }

        // Close panels on Escape
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                if (threeDPanelOpen) window.toggle3DPanel();
                if (mcPanelOpen) window.toggleMinecraftPanel();
                if (fmPanelOpen) window.toggleFileManager();
            }
        });

        console.log('✅ New features initialized: 3D Models, Minecraft Tools, File Manager');
    }

    // Wait for DOM
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initNewFeatures);
    } else {
        initNewFeatures();
    }

})();
