"""
Minecraft 1.7.10 Texture & Model Generation Utilities
=====================================================

Provides authentic Minecraft-style texture post-processing and
proper 1.7.10-format JSON model generation with full element definitions.

Key features:
- Minecraft-authentic color palette quantization (from vanilla textures)
- Pixel-art post-processing (dithering, edge sharpening, tileability)
- Procedural texture generation fallback (noise-based stone, wood, ore, etc.)
- Full Minecraft 1.7.10 JSON model definitions (not just parent references)
- OBJ export for 3D preview
- Proper resource pack ZIP packaging

Inspired by:
- Random832/mctexgen (color palette + texture algorithms)
- Faithful AI / python-faithful-AI (texture upscaling concepts)
- Minecraft wiki model format documentation
"""

import io
import json
import math
import random
import uuid
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

# ============================================================
# Minecraft Vanilla Color Palette (extracted from 1.7.10 textures)
# Grouped by material type for better quantization
# ============================================================

# Core palette - the most common colors in vanilla Minecraft textures
MC_PALETTE_CORE = [
    # Stone/cobblestone grays
    (125, 125, 125), (140, 140, 140), (106, 106, 106), (85, 85, 85),
    (150, 150, 150), (169, 169, 169), (75, 75, 75), (95, 95, 95),
    (115, 115, 115), (160, 160, 160),
    # Dirt/wood browns
    (134, 96, 67), (149, 108, 76), (120, 85, 58), (100, 70, 46),
    (168, 135, 97), (185, 152, 113), (87, 60, 36), (110, 78, 50),
    (155, 120, 85), (75, 51, 28),
    # Grass/leaf greens  
    (89, 125, 39), (109, 153, 48), (67, 94, 29), (127, 178, 56),
    (55, 77, 22), (78, 110, 35), (100, 140, 44), (45, 63, 18),
    (118, 166, 52), (60, 85, 25),
    # Sand/sandstone yellows
    (218, 210, 158), (237, 232, 190), (200, 192, 138), (180, 172, 118),
    (228, 222, 172), (195, 185, 128), (210, 202, 148),
    # Water/ice blues
    (64, 68, 150), (59, 63, 140), (72, 76, 165), (47, 51, 120),
    (85, 89, 180), (95, 175, 210), (80, 160, 195),
    # Lava/redstone reds
    (207, 54, 21), (188, 45, 15), (225, 65, 30), (160, 35, 10),
    (180, 40, 12), (240, 80, 40), (150, 30, 8),
    # Gold/glowstone yellows
    (250, 238, 77), (220, 195, 50), (235, 215, 60), (200, 170, 35),
    (240, 225, 65), (190, 155, 30),
    # Iron/snow whites
    (220, 220, 220), (200, 200, 200), (235, 235, 235), (245, 245, 245),
    (190, 190, 190), (210, 210, 210),
    # Coal/obsidian blacks
    (20, 18, 20), (30, 28, 30), (40, 37, 40), (50, 47, 50),
    (15, 13, 15), (25, 23, 25), (35, 32, 35),
    # Nether reds/maroons
    (100, 0, 0), (125, 15, 8), (80, 0, 0), (145, 25, 12),
    (110, 8, 4), (90, 4, 2),
    # Diamond/lapis blues
    (45, 200, 210), (55, 215, 225), (35, 180, 190), (62, 68, 160),
    (50, 55, 145), (70, 75, 175),
    # Emerald greens
    (17, 160, 54), (20, 180, 62), (14, 140, 46), (24, 200, 70),
    # Prismarine teals
    (75, 125, 120), (65, 115, 105), (85, 140, 130),
    # Terracotta/clay
    (152, 94, 67), (140, 84, 60), (165, 105, 75), (120, 72, 48),
    (175, 115, 82), (130, 78, 55),
    # Wool/concrete dyes
    (180, 80, 65), (85, 50, 140), (50, 75, 175), (255, 215, 55),
    (125, 185, 30), (210, 130, 170), (65, 65, 65), (155, 160, 155),
    (55, 130, 130), (130, 55, 165), (45, 55, 135), (85, 55, 35),
    (55, 75, 30), (165, 50, 40), (25, 25, 25),
]

# Material-specific palettes for better quantization per texture type
MC_PALETTE_STONE = [
    (125, 125, 125), (140, 140, 140), (106, 106, 106), (85, 85, 85),
    (150, 150, 150), (169, 169, 169), (75, 75, 75), (95, 95, 95),
    (115, 115, 115), (160, 160, 160), (130, 130, 130), (100, 100, 100),
    (110, 110, 110), (145, 145, 145), (120, 120, 120), (90, 90, 90),
]

MC_PALETTE_WOOD = [
    (134, 96, 67), (149, 108, 76), (120, 85, 58), (100, 70, 46),
    (168, 135, 97), (185, 152, 113), (87, 60, 36), (110, 78, 50),
    (155, 120, 85), (75, 51, 28), (140, 100, 70), (95, 65, 42),
    (160, 128, 90), (125, 90, 62), (170, 140, 100), (80, 55, 32),
]

MC_PALETTE_ORE = MC_PALETTE_STONE + [
    (207, 54, 21), (250, 238, 77), (220, 195, 50), (45, 200, 210),
    (17, 160, 54), (62, 68, 160), (20, 18, 20), (220, 220, 220),
    (180, 40, 12), (55, 215, 225), (20, 180, 62), (50, 55, 145),
]


def get_palette_for_type(texture_type: str, prompt: str = "") -> List[Tuple[int, int, int]]:
    """Select the best color palette based on texture type and prompt keywords."""
    prompt_lower = prompt.lower()
    
    if any(w in prompt_lower for w in ['stone', 'cobble', 'gravel', 'andesite', 'granite', 'diorite']):
        return MC_PALETTE_STONE + MC_PALETTE_CORE[:10]
    elif any(w in prompt_lower for w in ['wood', 'plank', 'log', 'oak', 'birch', 'spruce', 'jungle', 'acacia']):
        return MC_PALETTE_WOOD + MC_PALETTE_CORE[:10]
    elif any(w in prompt_lower for w in ['ore', 'diamond', 'iron', 'gold', 'coal', 'lapis', 'emerald', 'redstone', 'ruby', 'sapphire']):
        return MC_PALETTE_ORE
    elif any(w in prompt_lower for w in ['dirt', 'mud', 'soul']):
        return MC_PALETTE_WOOD[:8] + MC_PALETTE_CORE[:10]
    elif any(w in prompt_lower for w in ['grass', 'leaf', 'leaves', 'vine', 'bush', 'fern']):
        return [c for c in MC_PALETTE_CORE if c[1] > c[0] and c[1] > c[2]] + MC_PALETTE_CORE[:10]
    elif any(w in prompt_lower for w in ['sand', 'sandstone', 'desert']):
        return MC_PALETTE_CORE[30:37] + MC_PALETTE_CORE[:10]
    
    return MC_PALETTE_CORE


# ============================================================
# Texture Post-Processing Pipeline
# ============================================================

def quantize_to_palette(image: Image.Image, palette: List[Tuple[int, int, int]], 
                        dither: bool = True) -> Image.Image:
    """Quantize an image to match Minecraft's color palette.
    
    Uses Floyd-Steinberg dithering for more natural-looking pixel art.
    """
    img_array = np.array(image.convert('RGB'), dtype=np.float64)
    h, w, _ = img_array.shape
    palette_array = np.array(palette, dtype=np.float64)
    
    result = img_array.copy()
    
    for y in range(h):
        for x in range(w):
            old_pixel = result[y, x].copy()
            # Find nearest palette color (Euclidean distance)
            distances = np.sqrt(np.sum((palette_array - old_pixel) ** 2, axis=1))
            nearest_idx = np.argmin(distances)
            new_pixel = palette_array[nearest_idx]
            result[y, x] = new_pixel
            
            if dither:
                # Floyd-Steinberg error diffusion
                error = old_pixel - new_pixel
                if x + 1 < w:
                    result[y, x + 1] += error * 7.0 / 16.0
                if y + 1 < h:
                    if x - 1 >= 0:
                        result[y + 1, x - 1] += error * 3.0 / 16.0
                    result[y + 1, x] += error * 5.0 / 16.0
                    if x + 1 < w:
                        result[y + 1, x + 1] += error * 1.0 / 16.0
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def make_tileable(image: Image.Image, blend_width: int = 0) -> Image.Image:
    """Make a texture seamlessly tileable by blending edges.
    
    For small textures (16x16), uses pixel mirroring.
    For larger textures, uses gradient blending at edges.
    """
    w, h = image.size
    if blend_width == 0:
        blend_width = max(1, w // 8)
    
    img = image.copy()
    pixels = np.array(img, dtype=np.float64)
    
    if w <= 16:
        # For tiny textures, average edge pixels with opposite edge
        for y in range(h):
            avg = (pixels[y, 0].astype(float) + pixels[y, -1].astype(float)) / 2
            pixels[y, 0] = avg
            pixels[y, -1] = avg
        for x in range(w):
            avg = (pixels[0, x].astype(float) + pixels[-1, x].astype(float)) / 2
            pixels[0, x] = avg
            pixels[-1, x] = avg
    else:
        # Gradient blend for larger textures
        for i in range(blend_width):
            alpha = i / blend_width
            # Left-right blend
            for y in range(h):
                pixels[y, i] = pixels[y, i] * alpha + pixels[y, w - blend_width + i] * (1 - alpha)
                pixels[y, w - 1 - i] = pixels[y, w - 1 - i] * alpha + pixels[y, blend_width - 1 - i] * (1 - alpha)
            # Top-bottom blend
            for x in range(w):
                pixels[i, x] = pixels[i, x] * alpha + pixels[h - blend_width + i, x] * (1 - alpha)
                pixels[h - 1 - i, x] = pixels[h - 1 - i, x] * alpha + pixels[blend_width - 1 - i, x] * (1 - alpha)
    
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def enhance_pixel_art(image: Image.Image, target_size: int = 16) -> Image.Image:
    """Post-process an AI-generated image into authentic Minecraft pixel art.
    
    Pipeline:
    1. Resize to target with LANCZOS (better than NEAREST for initial down)
    2. Sharpen to crisp up features
    3. Quantize colors
    4. Apply slight contrast boost
    5. Make tileable
    """
    # Step 1: Smart downscale - use LANCZOS for initial resize to get good color averaging
    img = image.convert('RGB')
    
    # If image is much larger, do a two-step resize for better quality
    w, h = img.size
    if w > target_size * 4:
        intermediate = target_size * 4
        img = img.resize((intermediate, intermediate), Image.LANCZOS)
    
    # Final resize to target
    img = img.resize((target_size, target_size), Image.LANCZOS)
    
    # Step 2: Boost contrast and saturation for that Minecraft pop
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = ImageEnhance.Color(img).enhance(1.2)
    
    return img


def process_minecraft_texture(image: Image.Image, target_size: int = 16, 
                               texture_type: str = "block", style: str = "pixel_art",
                               prompt: str = "", make_tile: bool = True,
                               use_palette: bool = True) -> Image.Image:
    """Full Minecraft texture processing pipeline.
    
    Takes an AI-generated image and converts it to an authentic Minecraft texture.
    """
    # Step 1: Smart resize
    img = enhance_pixel_art(image, target_size)
    
    # Step 2: Palette quantization (optional, makes it look more vanilla)
    if use_palette and style == 'pixel_art':
        palette = get_palette_for_type(texture_type, prompt)
        # Use dithering only for larger textures
        img = quantize_to_palette(img, palette, dither=(target_size >= 32))
    elif use_palette and style == 'faithful':
        # Faithful style uses wider palette with no dithering
        img = quantize_to_palette(img, MC_PALETTE_CORE, dither=False)
    
    # Step 3: Make tileable for block textures
    if make_tile and texture_type in ('block', 'environment', 'crop'):
        img = make_tileable(img)
    
    # Step 4: Handle transparency for items and crops
    if texture_type in ('item', 'crop', 'particle', 'cross'):
        img = img.convert('RGBA')
    
    return img


# ============================================================
# Procedural Texture Generation (Fallback / No-AI mode)
# ============================================================

def _noise_2d(w: int, h: int, scale: float = 1.0, seed: int = 0) -> np.ndarray:
    """Simple value noise for procedural textures."""
    rng = np.random.RandomState(seed)
    # Generate random values at grid points
    grid_w = max(2, int(w / scale))
    grid_h = max(2, int(h / scale))
    grid = rng.rand(grid_h + 1, grid_w + 1)
    
    # Interpolate
    result = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            gx = x * grid_w / w
            gy = y * grid_h / h
            x0, y0 = int(gx), int(gy)
            x1, y1 = min(x0 + 1, grid_w), min(y0 + 1, grid_h)
            fx, fy = gx - x0, gy - y0
            # Bilinear interpolation
            top = grid[y0, x0] * (1 - fx) + grid[y0, x1] * fx
            bot = grid[y1, x0] * (1 - fx) + grid[y1, x1] * fx
            result[y, x] = top * (1 - fy) + bot * fy
    
    return result


def _fractal_noise(w: int, h: int, octaves: int = 4, seed: int = 0) -> np.ndarray:
    """Multi-octave fractal noise."""
    result = np.zeros((h, w))
    amplitude = 1.0
    total_amp = 0.0
    for i in range(octaves):
        scale = max(1.0, 2.0 ** i)
        noise = _noise_2d(w, h, scale=w / scale, seed=seed + i * 1000)
        result += noise * amplitude
        total_amp += amplitude
        amplitude *= 0.5
    return result / total_amp


def generate_procedural_texture(texture_type: str, name: str = "", size: int = 16,
                                 seed: int = 0, style: str = "pixel_art") -> Image.Image:
    """Generate a Minecraft texture procedurally without AI.
    
    Useful as a fallback when ComfyUI is unavailable, or for
    generating basic textures that work well procedurally.
    """
    if seed == 0:
        seed = random.randint(1, 999999)
    
    rng = np.random.RandomState(seed)
    name_lower = name.lower() if name else ""
    
    # Determine base colors from name
    base_color, detail_color = _pick_colors(name_lower, texture_type)
    
    if texture_type == 'block':
        return _gen_block_texture(size, base_color, detail_color, name_lower, seed, rng)
    elif texture_type == 'item':
        return _gen_item_texture(size, base_color, detail_color, name_lower, seed, rng)
    elif texture_type == 'crop':
        return _gen_crop_texture(size, base_color, seed, rng)
    else:
        return _gen_block_texture(size, base_color, detail_color, name_lower, seed, rng)


def _pick_colors(name: str, texture_type: str) -> Tuple[Tuple[int,int,int], Tuple[int,int,int]]:
    """Pick appropriate base and detail colors from the name."""
    color_map = {
        'ruby': ((180, 40, 50), (220, 60, 70)),
        'sapphire': ((40, 50, 180), (60, 70, 220)),
        'emerald': ((20, 160, 55), (30, 200, 70)),
        'diamond': ((45, 200, 210), (80, 230, 240)),
        'gold': ((220, 195, 50), (250, 238, 77)),
        'iron': ((190, 190, 190), (220, 220, 220)),
        'coal': ((30, 28, 30), (50, 47, 50)),
        'copper': ((180, 110, 60), (210, 130, 70)),
        'tin': ((180, 180, 190), (200, 200, 210)),
        'silver': ((200, 200, 210), (220, 220, 230)),
        'obsidian': ((15, 10, 25), (30, 20, 45)),
        'netherrack': ((100, 0, 0), (125, 15, 8)),
        'stone': ((125, 125, 125), (140, 140, 140)),
        'cobble': ((106, 106, 106), (140, 140, 140)),
        'dirt': ((134, 96, 67), (149, 108, 76)),
        'wood': ((134, 96, 67), (168, 135, 97)),
        'oak': ((168, 135, 97), (185, 152, 113)),
        'birch': ((195, 185, 165), (215, 205, 185)),
        'spruce': ((87, 60, 36), (110, 78, 50)),
        'sand': ((218, 210, 158), (237, 232, 190)),
        'grass': ((89, 125, 39), (109, 153, 48)),
        'leaf': ((55, 77, 22), (89, 125, 39)),
        'lapis': ((50, 55, 145), (62, 68, 160)),
        'redstone': ((160, 35, 10), (207, 54, 21)),
        'amethyst': ((130, 55, 165), (160, 80, 200)),
        'crystal': ((180, 220, 240), (210, 240, 255)),
    }
    
    for key, colors in color_map.items():
        if key in name:
            return colors
    
    if texture_type == 'item':
        return ((180, 180, 180), (220, 220, 220))
    return ((125, 125, 125), (140, 140, 140))


def _gen_block_texture(size: int, base: Tuple, detail: Tuple, name: str, 
                        seed: int, rng: np.random.RandomState) -> Image.Image:
    """Generate a block texture with noise and optional ore specks."""
    pixels = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Base noise layer
    noise = _fractal_noise(size, size, octaves=3, seed=seed)
    
    for y in range(size):
        for x in range(size):
            t = noise[y, x]
            r = int(base[0] + (detail[0] - base[0]) * t + rng.randint(-8, 9))
            g = int(base[1] + (detail[1] - base[1]) * t + rng.randint(-8, 9))
            b = int(base[2] + (detail[2] - base[2]) * t + rng.randint(-8, 9))
            pixels[y, x] = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    
    img = Image.fromarray(pixels)
    
    # Add ore specks if it's an ore
    is_ore = any(w in name for w in ['ore', 'ruby', 'sapphire', 'diamond', 'gold', 'iron', 
                                       'coal', 'copper', 'tin', 'silver', 'emerald', 'lapis',
                                       'redstone', 'amethyst'])
    if is_ore:
        img = _add_ore_specks(img, detail, size, rng)
    
    # Add cracks for stone-like textures
    if any(w in name for w in ['stone', 'cobble', 'brick']):
        img = _add_stone_cracks(img, size, rng)
    
    return make_tileable(img)


def _add_ore_specks(img: Image.Image, ore_color: Tuple, size: int, 
                     rng: np.random.RandomState) -> Image.Image:
    """Add ore speck clusters to a stone-base texture."""
    draw = ImageDraw.Draw(img)
    # 2-4 clusters for 16x16
    num_clusters = max(2, size // 6)
    
    for _ in range(num_clusters):
        cx = rng.randint(2, size - 2)
        cy = rng.randint(2, size - 2)
        cluster_size = rng.randint(2, max(3, size // 6))
        
        for _ in range(cluster_size):
            ox = cx + rng.randint(-1, 2)
            oy = cy + rng.randint(-1, 2)
            if 0 <= ox < size and 0 <= oy < size:
                # Vary the ore color slightly
                r = max(0, min(255, ore_color[0] + rng.randint(-15, 16)))
                g = max(0, min(255, ore_color[1] + rng.randint(-15, 16)))
                b = max(0, min(255, ore_color[2] + rng.randint(-15, 16)))
                draw.point((ox, oy), fill=(r, g, b))
    
    return img


def _add_stone_cracks(img: Image.Image, size: int, rng: np.random.RandomState) -> Image.Image:
    """Add subtle crack lines to stone textures."""
    draw = ImageDraw.Draw(img)
    num_cracks = max(1, size // 8)
    
    for _ in range(num_cracks):
        x = rng.randint(0, size - 1)
        y = rng.randint(0, size - 1)
        length = rng.randint(2, max(3, size // 4))
        
        for _ in range(length):
            if 0 <= x < size and 0 <= y < size:
                # Dark crack pixel
                orig = img.getpixel((x, y))
                dark = tuple(max(0, c - 30) for c in orig[:3])
                draw.point((x, y), fill=dark)
            x += rng.choice([-1, 0, 1])
            y += rng.choice([0, 1])
    
    return img


def _gen_item_texture(size: int, base: Tuple, detail: Tuple, name: str,
                       seed: int, rng: np.random.RandomState) -> Image.Image:
    """Generate a simple item sprite with transparent background."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Determine item shape
    if any(w in name for w in ['sword', 'blade']):
        _draw_sword(draw, size, base, detail, rng)
    elif any(w in name for w in ['pickaxe', 'pick']):
        _draw_pickaxe(draw, size, base, detail, rng)
    elif any(w in name for w in ['gem', 'ruby', 'sapphire', 'diamond', 'emerald', 'crystal']):
        _draw_gem(draw, size, base, detail, rng)
    elif any(w in name for w in ['ingot', 'bar']):
        _draw_ingot(draw, size, base, detail, rng)
    else:
        _draw_generic_item(draw, size, base, detail, rng)
    
    return img


def _draw_sword(draw: ImageDraw.Draw, s: int, base: Tuple, detail: Tuple, rng):
    """Draw a simple sword shape."""
    # Handle (bottom center)
    hx = s // 2
    for y in range(s - 1, s - s//3 - 1, -1):
        c = (87 + rng.randint(-5, 6), 60 + rng.randint(-5, 6), 36 + rng.randint(-5, 6))
        draw.point((hx, y), fill=c + (255,))
    # Guard
    gy = s - s//3 - 1
    for x in range(hx - 1, hx + 2):
        draw.point((x, gy), fill=detail + (255,))
    # Blade (going up diagonally)
    bx = hx
    for y in range(gy - 1, max(0, s//6), -1):
        r = max(0, min(255, base[0] + rng.randint(-10, 11)))
        g = max(0, min(255, base[1] + rng.randint(-10, 11)))
        b = max(0, min(255, base[2] + rng.randint(-10, 11)))
        draw.point((bx, y), fill=(r, g, b, 255))
        if bx > 1:
            bx -= 1


def _draw_pickaxe(draw: ImageDraw.Draw, s: int, base: Tuple, detail: Tuple, rng):
    """Draw a simple pickaxe shape."""
    # Handle (diagonal)
    for i in range(s * 2 // 3):
        x = s // 4 + i * s // (s * 2)
        y = s - 2 - i
        if 0 <= x < s and 0 <= y < s:
            c = (87 + rng.randint(-5, 6), 60 + rng.randint(-5, 6), 36 + rng.randint(-5, 6))
            draw.point((x, y), fill=c + (255,))
    # Head
    head_y = s // 4
    for x in range(s // 6, s - s//6):
        for dy in range(max(1, s // 8)):
            if 0 <= head_y + dy < s:
                r = max(0, min(255, base[0] + rng.randint(-10, 11)))
                g = max(0, min(255, base[1] + rng.randint(-10, 11)))
                b = max(0, min(255, base[2] + rng.randint(-10, 11)))
                draw.point((x, head_y + dy), fill=(r, g, b, 255))


def _draw_gem(draw: ImageDraw.Draw, s: int, base: Tuple, detail: Tuple, rng):
    """Draw a simple gem/crystal shape."""
    cx, cy = s // 2, s // 2
    gem_r = s // 3
    for y in range(s):
        for x in range(s):
            dx, dy = x - cx, y - cy
            dist = abs(dx) + abs(dy)  # Diamond shape (Manhattan distance)
            if dist <= gem_r:
                brightness = 1.0 - (dist / gem_r) * 0.4
                # Add sparkle
                if rng.random() < 0.1:
                    brightness = min(1.5, brightness + 0.3)
                r = max(0, min(255, int(base[0] * brightness + rng.randint(-8, 9))))
                g = max(0, min(255, int(base[1] * brightness + rng.randint(-8, 9))))
                b = max(0, min(255, int(base[2] * brightness + rng.randint(-8, 9))))
                draw.point((x, y), fill=(r, g, b, 255))


def _draw_ingot(draw: ImageDraw.Draw, s: int, base: Tuple, detail: Tuple, rng):
    """Draw a simple ingot shape."""
    # Ingot is a rounded rectangle
    margin = max(1, s // 5)
    for y in range(margin + 1, s - margin):
        for x in range(margin, s - margin - 1):
            brightness = 0.8 + 0.4 * (1 - (y - margin) / (s - 2 * margin))
            r = max(0, min(255, int(base[0] * brightness + rng.randint(-5, 6))))
            g = max(0, min(255, int(base[1] * brightness + rng.randint(-5, 6))))
            b = max(0, min(255, int(base[2] * brightness + rng.randint(-5, 6))))
            draw.point((x, y), fill=(r, g, b, 255))


def _draw_generic_item(draw: ImageDraw.Draw, s: int, base: Tuple, detail: Tuple, rng):
    """Draw a generic item shape (rounded square)."""
    margin = max(1, s // 4)
    for y in range(margin, s - margin):
        for x in range(margin, s - margin):
            r = max(0, min(255, base[0] + rng.randint(-15, 16)))
            g = max(0, min(255, base[1] + rng.randint(-15, 16)))
            b = max(0, min(255, base[2] + rng.randint(-15, 16)))
            draw.point((x, y), fill=(r, g, b, 255))


def _gen_crop_texture(size: int, base: Tuple, seed: int, rng: np.random.RandomState) -> Image.Image:
    """Generate a simple crop/plant texture with transparency."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw stems
    stems = rng.randint(2, max(3, size // 4))
    for _ in range(stems):
        x = rng.randint(size // 4, 3 * size // 4)
        for y in range(size // 3, size):
            if rng.random() < 0.9:
                g = max(0, min(255, base[1] - 20 + rng.randint(-10, 11)))
                r = max(0, min(255, base[0] - 30 + rng.randint(-5, 6)))
                b = max(0, min(255, base[2] - 30 + rng.randint(-5, 6)))
                draw.point((x + rng.choice([-1, 0, 0, 0, 1]), y), fill=(r, g, b, 255))
    
    # Draw leaves/buds at top
    for _ in range(stems):
        lx = rng.randint(size // 4, 3 * size // 4)
        ly = rng.randint(size // 6, size // 2)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if 0 <= lx + dx < size and 0 <= ly + dy < size and rng.random() < 0.7:
                    r = max(0, min(255, base[0] + rng.randint(-10, 11)))
                    g = max(0, min(255, base[1] + rng.randint(-10, 11)))
                    b = max(0, min(255, base[2] + rng.randint(-10, 11)))
                    draw.point((lx + dx, ly + dy), fill=(r, g, b, 255))
    
    return img


# ============================================================
# Minecraft 1.7.10 Model JSON Generation (Full Element Defs)
# ============================================================

def generate_model_json(model_type: str, name: str, mod_id: str = "modid") -> Dict[str, Any]:
    """Generate a complete Minecraft 1.7.10 JSON model with full element definitions.
    
    Unlike simple parent-reference models, these include complete 'elements' arrays
    with proper face UV mappings, rotations, and culling - exactly what Minecraft
    1.7.10's model loader expects.
    """
    safe_name = name.lower().replace(' ', '_')
    
    generators = {
        'block': _model_full_block,
        'item': _model_item,
        'crop': _model_crop,
        'slab': _model_slab,
        'stairs': _model_stairs,
        'fence': _model_fence,
        'wall': _model_wall,
        'cross': _model_cross,
        'pane': _model_pane,
    }
    
    gen_func = generators.get(model_type, _model_full_block)
    return gen_func(safe_name, mod_id)


def _face(uv: List[float], texture: str, cullface: str = None, rotation: int = 0) -> Dict:
    """Helper to create a face definition."""
    face = {"uv": uv, "texture": texture}
    if cullface:
        face["cullface"] = cullface
    if rotation:
        face["rotation"] = rotation
    return face


def _model_full_block(name: str, mod_id: str) -> Dict:
    """Full cube block model with proper element definition."""
    tex = f"#{name}"
    return {
        "__comment": f"Minecraft 1.7.10 block model: {name} - Generated by EDISON",
        "textures": {
            "particle": f"{mod_id}:blocks/{name}",
            name: f"{mod_id}:blocks/{name}"
        },
        "elements": [
            {
                "from": [0, 0, 0],
                "to": [16, 16, 16],
                "faces": {
                    "down":  _face([0, 0, 16, 16], tex, "down"),
                    "up":    _face([0, 0, 16, 16], tex, "up"),
                    "north": _face([0, 0, 16, 16], tex, "north"),
                    "south": _face([0, 0, 16, 16], tex, "south"),
                    "west":  _face([0, 0, 16, 16], tex, "west"),
                    "east":  _face([0, 0, 16, 16], tex, "east"),
                }
            }
        ]
    }


def _model_slab(name: str, mod_id: str) -> Dict:
    """Half-slab model."""
    tex = f"#{name}"
    return {
        "__comment": f"Minecraft 1.7.10 slab model: {name} - Generated by EDISON",
        "textures": {
            "particle": f"{mod_id}:blocks/{name}",
            name: f"{mod_id}:blocks/{name}"
        },
        "elements": [
            {
                "from": [0, 0, 0],
                "to": [16, 8, 16],
                "faces": {
                    "down":  _face([0, 0, 16, 16], tex, "down"),
                    "up":    _face([0, 0, 16, 16], tex),
                    "north": _face([0, 8, 16, 16], tex, "north"),
                    "south": _face([0, 8, 16, 16], tex, "south"),
                    "west":  _face([0, 8, 16, 16], tex, "west"),
                    "east":  _face([0, 8, 16, 16], tex, "east"),
                }
            }
        ]
    }


def _model_stairs(name: str, mod_id: str) -> Dict:
    """Stair block model (two elements: bottom slab + top half)."""
    tex = f"#{name}"
    return {
        "__comment": f"Minecraft 1.7.10 stairs model: {name} - Generated by EDISON",
        "textures": {
            "particle": f"{mod_id}:blocks/{name}",
            name: f"{mod_id}:blocks/{name}"
        },
        "elements": [
            {
                "__comment": "Bottom slab",
                "from": [0, 0, 0],
                "to": [16, 8, 16],
                "faces": {
                    "down":  _face([0, 0, 16, 16], tex, "down"),
                    "up":    _face([0, 0, 16, 16], tex),
                    "north": _face([0, 8, 16, 16], tex, "north"),
                    "south": _face([0, 8, 16, 16], tex, "south"),
                    "west":  _face([0, 8, 16, 16], tex, "west"),
                    "east":  _face([0, 8, 16, 16], tex, "east"),
                }
            },
            {
                "__comment": "Top step",
                "from": [8, 8, 0],
                "to": [16, 16, 16],
                "faces": {
                    "up":    _face([8, 0, 16, 16], tex, "up"),
                    "north": _face([0, 0, 8, 8], tex, "north"),
                    "south": _face([8, 0, 16, 8], tex, "south"),
                    "west":  _face([0, 0, 16, 8], tex),
                    "east":  _face([0, 0, 16, 8], tex, "east"),
                }
            }
        ]
    }


def _model_fence(name: str, mod_id: str) -> Dict:
    """Fence post model."""
    tex = f"#{name}"
    return {
        "__comment": f"Minecraft 1.7.10 fence post model: {name} - Generated by EDISON",
        "textures": {
            "particle": f"{mod_id}:blocks/{name}",
            name: f"{mod_id}:blocks/{name}"
        },
        "elements": [
            {
                "__comment": "Post",
                "from": [6, 0, 6],
                "to": [10, 16, 10],
                "faces": {
                    "down":  _face([6, 6, 10, 10], tex, "down"),
                    "up":    _face([6, 6, 10, 10], tex, "up"),
                    "north": _face([6, 0, 10, 16], tex),
                    "south": _face([6, 0, 10, 16], tex),
                    "west":  _face([6, 0, 10, 16], tex),
                    "east":  _face([6, 0, 10, 16], tex),
                }
            }
        ],
        "fence_side": {
            "__comment": "Attach this when fence connects to a neighbor",
            "elements": [
                {
                    "__comment": "Top rail",
                    "from": [7, 12, 0],
                    "to": [9, 15, 6],
                    "faces": {
                        "up":    _face([7, 0, 9, 6], tex),
                        "down":  _face([7, 0, 9, 6], tex),
                        "north": _face([7, 12, 9, 15], tex, "north"),
                        "west":  _face([0, 12, 6, 15], tex),
                        "east":  _face([0, 12, 6, 15], tex),
                    }
                },
                {
                    "__comment": "Bottom rail",
                    "from": [7, 6, 0],
                    "to": [9, 9, 6],
                    "faces": {
                        "up":    _face([7, 0, 9, 6], tex),
                        "down":  _face([7, 0, 9, 6], tex),
                        "north": _face([7, 6, 9, 9], tex, "north"),
                        "west":  _face([0, 6, 6, 9], tex),
                        "east":  _face([0, 6, 6, 9], tex),
                    }
                }
            ]
        }
    }


def _model_wall(name: str, mod_id: str) -> Dict:
    """Wall post model."""
    tex = f"#{name}"
    return {
        "__comment": f"Minecraft 1.7.10 wall model: {name} - Generated by EDISON",
        "textures": {
            "particle": f"{mod_id}:blocks/{name}",
            name: f"{mod_id}:blocks/{name}"
        },
        "elements": [
            {
                "__comment": "Post",
                "from": [4, 0, 4],
                "to": [12, 16, 12],
                "faces": {
                    "down":  _face([4, 4, 12, 12], tex, "down"),
                    "up":    _face([4, 4, 12, 12], tex, "up"),
                    "north": _face([4, 0, 12, 16], tex),
                    "south": _face([4, 0, 12, 16], tex),
                    "west":  _face([4, 0, 12, 16], tex),
                    "east":  _face([4, 0, 12, 16], tex),
                }
            }
        ]
    }


def _model_cross(name: str, mod_id: str) -> Dict:
    """Cross/flower model (two intersecting planes)."""
    tex = f"#{name}"
    return {
        "__comment": f"Minecraft 1.7.10 cross model: {name} - Generated by EDISON",
        "ambientocclusion": False,
        "textures": {
            "particle": f"{mod_id}:blocks/{name}",
            name: f"{mod_id}:blocks/{name}"
        },
        "elements": [
            {
                "__comment": "Plane 1 (diagonal)",
                "from": [0.8, 0, 8],
                "to": [15.2, 16, 8],
                "rotation": {"origin": [8, 8, 8], "axis": "y", "angle": 45, "rescale": True},
                "shade": False,
                "faces": {
                    "north": {"uv": [0, 0, 16, 16], "texture": tex},
                    "south": {"uv": [0, 0, 16, 16], "texture": tex},
                }
            },
            {
                "__comment": "Plane 2 (diagonal)",
                "from": [8, 0, 0.8],
                "to": [8, 16, 15.2],
                "rotation": {"origin": [8, 8, 8], "axis": "y", "angle": 45, "rescale": True},
                "shade": False,
                "faces": {
                    "west": {"uv": [0, 0, 16, 16], "texture": tex},
                    "east": {"uv": [0, 0, 16, 16], "texture": tex},
                }
            }
        ]
    }


def _model_pane(name: str, mod_id: str) -> Dict:
    """Glass pane model (thin center panel)."""
    tex = f"#{name}"
    return {
        "__comment": f"Minecraft 1.7.10 pane model: {name} - Generated by EDISON",
        "textures": {
            "particle": f"{mod_id}:blocks/{name}",
            name: f"{mod_id}:blocks/{name}",
            f"{name}_edge": f"{mod_id}:blocks/{name}"
        },
        "elements": [
            {
                "__comment": "Pane post",
                "from": [7, 0, 7],
                "to": [9, 16, 9],
                "faces": {
                    "down":  _face([7, 7, 9, 9], tex, "down"),
                    "up":    _face([7, 7, 9, 9], tex, "up"),
                    "north": _face([7, 0, 9, 16], tex),
                    "south": _face([7, 0, 9, 16], tex),
                    "west":  _face([7, 0, 9, 16], tex),
                    "east":  _face([7, 0, 9, 16], tex),
                }
            }
        ]
    }


def _model_item(name: str, mod_id: str) -> Dict:
    """Item model (flat sprite with proper display transforms)."""
    return {
        "__comment": f"Minecraft 1.7.10 item model: {name} - Generated by EDISON",
        "parent": "builtin/generated",
        "textures": {
            "layer0": f"{mod_id}:items/{name}"
        },
        "display": {
            "thirdperson": {
                "rotation": [-90, 0, 0],
                "translation": [0, 1, -3],
                "scale": [0.55, 0.55, 0.55]
            },
            "firstperson": {
                "rotation": [0, -135, 25],
                "translation": [0, 4, 2],
                "scale": [1.7, 1.7, 1.7]
            }
        }
    }


def _model_crop(name: str, mod_id: str) -> Dict:
    """Crop model (hash pattern of 4 faces)."""
    tex = f"#{name}"
    return {
        "__comment": f"Minecraft 1.7.10 crop model: {name} - Generated by EDISON",
        "ambientocclusion": False,
        "textures": {
            "particle": f"{mod_id}:blocks/{name}",
            name: f"{mod_id}:blocks/{name}"
        },
        "elements": [
            {
                "from": [0, 0, 4],
                "to": [16, 16, 4],
                "shade": False,
                "faces": {
                    "north": {"uv": [0, 0, 16, 16], "texture": tex},
                    "south": {"uv": [0, 0, 16, 16], "texture": tex},
                }
            },
            {
                "from": [0, 0, 12],
                "to": [16, 16, 12],
                "shade": False,
                "faces": {
                    "north": {"uv": [0, 0, 16, 16], "texture": tex},
                    "south": {"uv": [0, 0, 16, 16], "texture": tex},
                }
            },
            {
                "from": [4, 0, 0],
                "to": [4, 16, 16],
                "shade": False,
                "faces": {
                    "west": {"uv": [0, 0, 16, 16], "texture": tex},
                    "east": {"uv": [0, 0, 16, 16], "texture": tex},
                }
            },
            {
                "from": [12, 0, 0],
                "to": [12, 16, 16],
                "shade": False,
                "faces": {
                    "west": {"uv": [0, 0, 16, 16], "texture": tex},
                    "east": {"uv": [0, 0, 16, 16], "texture": tex},
                }
            }
        ]
    }


# ============================================================
# Blockstate Generation
# ============================================================

def generate_blockstate_json(model_type: str, name: str, mod_id: str = "modid") -> Optional[Dict]:
    """Generate blockstate JSON for blocks that need rotation/variant handling."""
    safe_name = name.lower().replace(' ', '_')
    
    if model_type == 'block':
        return {
            "variants": {
                "normal": {"model": f"{mod_id}:{safe_name}"}
            }
        }
    elif model_type == 'slab':
        return {
            "variants": {
                "half=bottom": {"model": f"{mod_id}:{safe_name}"},
                "half=top": {"model": f"{mod_id}:{safe_name}", "x": 180, "uvlock": True}
            }
        }
    elif model_type == 'stairs':
        return {
            "variants": {
                "facing=east,half=bottom,shape=straight":  {"model": f"{mod_id}:{safe_name}"},
                "facing=west,half=bottom,shape=straight":  {"model": f"{mod_id}:{safe_name}", "y": 180, "uvlock": True},
                "facing=south,half=bottom,shape=straight": {"model": f"{mod_id}:{safe_name}", "y": 90, "uvlock": True},
                "facing=north,half=bottom,shape=straight": {"model": f"{mod_id}:{safe_name}", "y": 270, "uvlock": True},
                "facing=east,half=top,shape=straight":     {"model": f"{mod_id}:{safe_name}", "x": 180, "uvlock": True},
                "facing=west,half=top,shape=straight":     {"model": f"{mod_id}:{safe_name}", "x": 180, "y": 180, "uvlock": True},
                "facing=south,half=top,shape=straight":    {"model": f"{mod_id}:{safe_name}", "x": 180, "y": 90, "uvlock": True},
                "facing=north,half=top,shape=straight":    {"model": f"{mod_id}:{safe_name}", "x": 180, "y": 270, "uvlock": True},
            }
        }
    elif model_type == 'fence':
        return {
            "variants": {
                "normal": {"model": f"{mod_id}:{safe_name}_post"}
            }
        }
    elif model_type == 'wall':
        return {
            "variants": {
                "normal": {"model": f"{mod_id}:{safe_name}_post"}
            }
        }
    elif model_type in ('cross', 'crop'):
        return {
            "variants": {
                "normal": {"model": f"{mod_id}:{safe_name}"}
            }
        }
    elif model_type == 'pane':
        return {
            "variants": {
                "normal": {"model": f"{mod_id}:{safe_name}_post"}
            }
        }
    elif model_type == 'item':
        return None  # Items don't have blockstates
    
    return {"variants": {"normal": {"model": f"{mod_id}:{safe_name}"}}}


# ============================================================
# OBJ Export for 3D Preview
# ============================================================

def model_to_obj(model_json: Dict, name: str = "block") -> str:
    """Convert a Minecraft JSON model to OBJ format for 3D preview.
    
    Converts the 'elements' array into proper OBJ vertices and faces.
    """
    lines = [
        f"# Minecraft 1.7.10 Model: {name}",
        f"# Generated by EDISON",
        f"# https://github.com/mikedattolo/EDISON-ComfyUI",
        "",
        f"mtllib {name}.mtl",
        f"usemtl {name}_material",
        "",
    ]
    
    vertex_idx = 1
    elements = model_json.get("elements", [])
    
    if not elements:
        # Fallback: generate a simple cube
        elements = [{"from": [0, 0, 0], "to": [16, 16, 16], "faces": {
            "down": {}, "up": {}, "north": {}, "south": {}, "west": {}, "east": {}
        }}]
    
    for elem in elements:
        fr = elem.get("from", [0, 0, 0])
        to = elem.get("to", [16, 16, 16])
        faces = elem.get("faces", {})
        
        # Scale from MC units (0-16) to OBJ units (0-1)
        x0, y0, z0 = fr[0] / 16.0, fr[1] / 16.0, fr[2] / 16.0
        x1, y1, z1 = to[0] / 16.0, to[1] / 16.0, to[2] / 16.0
        
        # 8 vertices of the box
        verts = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),  # front
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),  # back
        ]
        
        for v in verts:
            lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")
        
        # Texture coordinates
        lines.append("vt 0.0 0.0")
        lines.append("vt 1.0 0.0")
        lines.append("vt 1.0 1.0")
        lines.append("vt 0.0 1.0")
        
        vi = vertex_idx  # base vertex index for this element
        ti = vertex_idx  # base tex coord index (simplified)
        
        # Faces (only emit faces that are defined in the model)
        face_map = {
            "north": (vi+0, vi+1, vi+2, vi+3),  # front face
            "south": (vi+5, vi+4, vi+7, vi+6),  # back face
            "west":  (vi+4, vi+0, vi+3, vi+7),  # left face
            "east":  (vi+1, vi+5, vi+6, vi+2),  # right face
            "down":  (vi+4, vi+5, vi+1, vi+0),  # bottom face
            "up":    (vi+3, vi+2, vi+6, vi+7),  # top face
        }
        
        for face_name, (a, b, c, d) in face_map.items():
            if face_name in faces:
                lines.append(f"f {a} {b} {c} {d}")
        
        vertex_idx += 8
    
    return "\n".join(lines)


def generate_mtl(name: str, texture_filename: str) -> str:
    """Generate a simple MTL material file."""
    return f"""# Minecraft Material: {name}
# Generated by EDISON

newmtl {name}_material
Ns 10.0
Ka 0.2 0.2 0.2
Kd 0.8 0.8 0.8
Ks 0.1 0.1 0.1
d 1.0
illum 2
map_Kd {texture_filename}
"""


# ============================================================
# Resource Pack ZIP Packaging
# ============================================================

def create_resource_pack_zip(
    texture_path: str,
    model_json: Dict,
    blockstate_json: Optional[Dict],
    model_type: str,
    name: str,
    mod_id: str = "modid",
    output_dir: str = ".",
) -> str:
    """Create a complete Minecraft 1.7.10 resource pack ZIP.
    
    Structure:
    assets/<mod_id>/
        textures/blocks/<name>.png  (or items/)
        models/block/<name>.json  (or item/)
        blockstates/<name>.json
    README.txt
    pack.mcmeta
    """
    safe_name = name.lower().replace(' ', '_')
    model_id = str(uuid.uuid4())[:8]
    zip_filename = f"mc_mod_{safe_name}_{model_id}.zip"
    zip_path = Path(output_dir) / zip_filename
    
    tex_folder = "blocks" if model_type != 'item' else "items"
    model_folder = "block" if model_type != 'item' else "item"
    
    with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:
        # pack.mcmeta
        pack_meta = {
            "pack": {
                "pack_format": 1,
                "description": f"EDISON Generated: {name}"
            }
        }
        zf.writestr("pack.mcmeta", json.dumps(pack_meta, indent=2))
        
        # Texture
        zf.write(texture_path, f"assets/{mod_id}/textures/{tex_folder}/{safe_name}.png")
        
        # Model JSON
        zf.writestr(
            f"assets/{mod_id}/models/{model_folder}/{safe_name}.json",
            json.dumps(model_json, indent=2)
        )
        
        # Blockstate JSON
        if blockstate_json:
            zf.writestr(
                f"assets/{mod_id}/blockstates/{safe_name}.json",
                json.dumps(blockstate_json, indent=2)
            )
        
        # OBJ preview
        obj_content = model_to_obj(model_json, safe_name)
        zf.writestr(f"preview/{safe_name}.obj", obj_content)
        mtl_content = generate_mtl(safe_name, f"{safe_name}.png")
        zf.writestr(f"preview/{safe_name}.mtl", mtl_content)
        
        # Java code template
        java_code = _generate_java_template(safe_name, model_type, mod_id)
        zf.writestr(f"src/{safe_name}.java", java_code)
        
        # README
        readme = _generate_readme(safe_name, model_type, mod_id, tex_folder, model_folder)
        zf.writestr("README.txt", readme)
    
    return zip_filename


def _generate_java_template(name: str, model_type: str, mod_id: str) -> str:
    """Generate a Java code template for registering the block/item in a Forge mod."""
    class_name = ''.join(w.capitalize() for w in name.split('_'))
    
    if model_type == 'item':
        return f"""package com.{mod_id}.items;

import net.minecraft.item.Item;
import net.minecraft.creativetab.CreativeTabs;
import cpw.mods.fml.common.registry.GameRegistry;

/**
 * {class_name} - Generated by EDISON
 * 
 * Minecraft 1.7.10 Forge Mod Item
 */
public class Item{class_name} extends Item {{

    public Item{class_name}() {{
        super();
        setUnlocalizedName("{name}");
        setTextureName("{mod_id}:{name}");
        setCreativeTab(CreativeTabs.tabMaterials);
        setMaxStackSize(64);
    }}

    // Register in your mod's init:
    // GameRegistry.registerItem(new Item{class_name}(), "{name}");
}}
"""
    else:
        return f"""package com.{mod_id}.blocks;

import net.minecraft.block.Block;
import net.minecraft.block.material.Material;
import net.minecraft.creativetab.CreativeTabs;
import cpw.mods.fml.common.registry.GameRegistry;

/**
 * Block{class_name} - Generated by EDISON
 * 
 * Minecraft 1.7.10 Forge Mod Block
 */
public class Block{class_name} extends Block {{

    public Block{class_name}() {{
        super(Material.rock);
        setBlockName("{name}");
        setBlockTextureName("{mod_id}:{name}");
        setCreativeTab(CreativeTabs.tabBlock);
        setHardness(3.0F);
        setResistance(5.0F);
        setStepSound(soundTypeStone);
        setHarvestLevel("pickaxe", 2);
    }}

    // Register in your mod's init:
    // GameRegistry.registerBlock(new Block{class_name}(), "{name}");
}}
"""


def _generate_readme(name: str, model_type: str, mod_id: str, 
                      tex_folder: str, model_folder: str) -> str:
    """Generate a README for the resource pack."""
    return f"""# Minecraft 1.7.10 Mod Asset: {name}
# Generated by EDISON (https://github.com/mikedattolo/EDISON-ComfyUI)
# Type: {model_type}

## Quick Install:
1. Copy the 'assets' folder into your mod's src/main/resources/
2. Replace '{mod_id}' with your actual mod ID in all files
3. Register the block/item in your mod code (see src/{name}.java)

## File Structure:
assets/{mod_id}/
    textures/{tex_folder}/{name}.png    - The texture
    models/{model_folder}/{name}.json   - Model definition (full elements)
    blockstates/{name}.json             - Block state variants

preview/
    {name}.obj  - 3D preview (OBJ format, open in Blender/etc.)
    {name}.mtl  - Material file for OBJ

src/
    {name}.java - Example Forge mod registration code

## Model Features:
- Full element definitions (not just parent references)
- Proper UV mappings for all faces
- Face culling for performance
- Compatible with Minecraft 1.7.10 model loader

## pack.mcmeta:
- pack_format: 1 (Minecraft 1.7.10)
"""


# ============================================================
# Improved ComfyUI Prompt Builder
# ============================================================

def build_minecraft_prompt(prompt: str, texture_type: str, style: str, 
                            size: int) -> Tuple[str, str]:
    """Build optimized positive and negative prompts for Minecraft texture generation.
    
    Returns (positive_prompt, negative_prompt) tuple.
    """
    # Style-specific prompt engineering
    style_prompts = {
        'pixel_art': (
            "pixel art, 16-bit retro game texture, crisp clean pixels, "
            "no anti-aliasing, sharp pixel edges, limited color palette, "
            "single flat texture, centered, game asset, "
        ),
        'faithful': (
            "detailed pixel art texture, high-detail game texture, "
            "faithful resource pack style, detailed shading, "
            "single flat texture, centered, game asset, "
        ),
        'painterly': (
            "hand-painted game texture, soft brushstrokes, painterly style, "
            "warm color palette, artistic game art, "
            "single flat texture, centered, game asset, "
        ),
    }
    
    # Type-specific prompt additions
    type_prompts = {
        'block': (
            "square tileable seamless texture, top-down flat view, "
            "uniform lighting, no perspective, no shadows on edges, "
            "minecraft block texture, game texture atlas, "
        ),
        'item': (
            "2D game item sprite, inventory icon, "
            "transparent background, clean outline, "
            "minecraft item icon, centered on canvas, "
        ),
        'crop': (
            "pixel art plant sprite, transparent background, "
            "small plant growth, farm crop texture, "
            "minecraft crop, game vegetation sprite, "
        ),
        'skin': (
            "minecraft player skin layout, 64x32 UV layout, "
            "character texture map, front back left right, "
            "pixel art character, game skin template, "
        ),
        'mob': (
            "pixel art creature texture map, entity sprite, "
            "game mob texture, UV unwrapped, "
            "minecraft style creature, pixel art animal, "
        ),
        'armor': (
            "pixel art armor texture overlay, equipment sprite, "
            "armor layer texture, minecraft armor, "
            "game equipment overlay, transparent areas, "
        ),
        'gui': (
            "game UI element, flat interface texture, "
            "clean pixel art button, menu element, "
            "minecraft GUI, game interface icon, "
        ),
        'particle': (
            "tiny particle sprite, glowing pixel effect, "
            "small game effect, transparent background, "
            "minecraft particle, animated frame, "
        ),
        'environment': (
            "seamless tileable environment texture, "
            "sky water lava terrain, natural pattern, "
            "minecraft environment, game world texture, "
        ),
    }
    
    style_p = style_prompts.get(style, style_prompts['pixel_art'])
    type_p = type_prompts.get(texture_type, type_prompts['block'])
    
    # Resolution hint
    res_hint = f"{size}x{size} pixel resolution, " if size <= 32 else ""
    
    positive = f"{style_p}{type_p}{res_hint}minecraft 1.7.10 style, {prompt}"
    
    # Strong negative prompt to avoid common AI generation issues
    negative = (
        "3d render, realistic, photographic, photograph, "
        "blurry, smooth gradients, anti-aliased, soft edges, "
        "watermark, text, logo, signature, artist name, "
        "multiple objects, collage, comparison, side by side, "
        "human face, person, portrait, "
        "high resolution details, noise, grain, "
        "perspective, depth of field, bokeh, "
        "nsfw, nude, naked, violence, gore, "
        "worst quality, low quality, jpeg artifacts, "
        "border, frame, vignette, "
    )
    
    return positive, negative


def create_minecraft_workflow(prompt: str, negative_prompt: str,
                               width: int, height: int,
                               steps: int = 28, guidance: float = 7.5,
                               checkpoint: str = "sd_xl_base_1.0.safetensors") -> Dict:
    """Create a ComfyUI workflow optimized for Minecraft texture generation.
    
    Key differences from the generic workflow:
    - Higher CFG guidance (7.5 vs 3.5) for more prompt-adherent results
    - More steps (28 vs 20) for better quality
    - Proper negative prompt to avoid non-pixel-art results
    - DPM++ 2M Karras sampler for better detail at small sizes
    """
    import random as rng
    seed = rng.randint(0, 2**32 - 1)
    
    return {
        "3": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": guidance,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler"
        },
        "4": {
            "inputs": {
                "ckpt_name": checkpoint
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "5": {
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        },
        "6": {
            "inputs": {
                "text": prompt,
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "7": {
            "inputs": {
                "text": negative_prompt,
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "8": {
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            },
            "class_type": "VAEDecode"
        },
        "9": {
            "inputs": {
                "filename_prefix": "EDISON_MC",
                "images": ["8", 0]
            },
            "class_type": "SaveImage"
        }
    }
