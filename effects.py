
import numpy as np
import random
import io
from typing import Optional, Tuple
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageDraw, ImageFont, ImageChops

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# ===================================================================================
# == UTILITY & REALISM FUNCTIONS
# ===================================================================================
def apply_jpeg_compression_effect(pil_img, quality_range=(60, 92), **kwargs):
    quality = random.randint(*quality_range)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')

def normalize_edgecolor(edge, facecolor=None):
    import matplotlib.colors as mcolors
    fallback = (0.0, 0.0, 0.0, 1.0)
    try:
        if edge is None: raise ValueError("No edge provided")
        rgba = mcolors.to_rgba(edge)
    except Exception:
        try:
            if facecolor is not None:
                r, g, b, _ = mcolors.to_rgba(facecolor)
                h, s, v = mcolors.rgb_to_hsv((r, g, b))
                darker_v = max(0, v * 0.6)
                rgba = mcolors.hsv_to_rgb((h, s, darker_v)); rgba = (*rgba, 1.0)
            else: return fallback
        except Exception: return fallback
    r, g, b, a = rgba
    if a < 0.3 or (r > 0.95 and g > 0.95 and b > 0.95): return fallback
    return (r, g, b, 1.0)

def apply_noise_effect(pil_img, sigma_range=(2, 8), **kwargs):
    img_array = np.array(pil_img)
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img_array.shape).astype(np.float32)
    noisy = np.clip(img_array.astype(np.float32) + noise, 0, 255)
    return Image.fromarray(noisy.astype(np.uint8))

def apply_blur_effect(pil_img, radius_range=(0.5, 1.8), **kwargs):
    radius = random.uniform(*radius_range)
    return pil_img.filter(ImageFilter.GaussianBlur(radius=radius))

def _manual_motion_blur(image, radius, angle):
    # Pillow has no built-in motion blur (ImageFilter.MotionBlur does not exist in any release).
    # NOTE: ImageFilter.Kernel only accepts 3x3 or 5x5 kernels, so any radius >= 2.5 (size >= 7)
    # raises ValueError here; apply_realism_effects logs that as a warning and skips the effect.

    # Ensure kernel size is an odd integer for ImageFilter.Kernel
    size = int(radius * 2) + 1
    if size % 2 == 0:
        size += 1
    
    if size <= 1: return image

    kernel = np.zeros((size, size))
    center = size // 2

    # Draw a line on the kernel
    angle_rad = np.deg2rad(angle)
    for i in range(size):
        x = int(center + (i - center) * np.cos(angle_rad))
        y = int(center + (i - center) * np.sin(angle_rad))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0
            
    # Handle cases where the line is short or off-kernel
    if kernel.sum() == 0:
        kernel[center, center] = 1.0
    
    # Normalize
    kernel /= kernel.sum()
    
    return image.filter(ImageFilter.Kernel((size, size), kernel.flatten()))

def apply_motion_blur_effect(pil_img, radius_range=(2, 5), angle_range=(0, 360), **kwargs):
    radius = random.uniform(*radius_range)
    angle = random.uniform(*angle_range)
    return _manual_motion_blur(pil_img, radius, angle)

def apply_low_res_effect(pil_img, scale_range=(0.25, 0.6), **kwargs):
    scale = random.uniform(*scale_range)
    w, h = pil_img.size
    small_w, small_h = max(1, int(w * scale)), max(1, int(h * scale))
    return pil_img.resize((small_w, small_h), resample=Image.BICUBIC).resize((w, h), resample=Image.BILINEAR)

def apply_pixelation_effect(pil_img, factor_options=[2,3,4], **kwargs):
    factor = random.choice(factor_options)
    if factor <= 1: return pil_img
    w, h = pil_img.size
    small = pil_img.resize((max(1, w // factor), max(1, h // factor)), resample=Image.BILINEAR)
    return small.resize((w, h), resample=Image.NEAREST)

def apply_posterize_effect(pil_img, color_options=[16,32,64], **kwargs):
    n_colors = random.choice(color_options)
    if n_colors >= 256: return pil_img
    return pil_img.convert('P', palette=Image.ADAPTIVE, colors=n_colors).convert('RGB')

def apply_color_variation_effect(pil_img, shift_range=(0.97, 1.03), **kwargs):
    arr = np.array(pil_img).astype(np.float32)
    channel_to_shift = random.randint(0, 2)
    shift_factor = random.uniform(*shift_range)
    arr[:, :, channel_to_shift] *= shift_factor
    arr = np.clip(arr, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))

def apply_ui_chrome_effect(pil_img, **kwargs):
    w, h = pil_img.size
    draw = ImageDraw.Draw(pil_img)
    chrome_h = int(max(24, h * 0.045))
    base_color = tuple(np.random.randint(230, 250, size=3))
    draw.rectangle([0, 0, w, chrome_h], fill=base_color)
    dot_r = max(4, int(chrome_h * 0.22))
    dot_y = chrome_h // 2
    gap = dot_r * 2 + 4
    dots_x = [12 + i * gap for i in range(3)]
    colors = [(255, 95, 86), (255, 189, 46), (39, 201, 63)]
    for i, x in enumerate(dots_x):
        draw.ellipse([x - dot_r, dot_y - dot_r, x + dot_r, dot_y + dot_r], fill=colors[i])
    return pil_img

def apply_watermark_effect(pil_img, opacity_range=(0.04, 0.12), **kwargs):
    text = random.choice(['CONFIDENTIAL', 'SAMPLE', 'PRELIMINARY', 'FOR INTERNAL USE ONLY', 'DRAFT'])
    opacity = random.uniform(*opacity_range)
    w, h = pil_img.size
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font_size = max(16, int(min(w, h) * 0.05))
        fnt = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        fnt = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=fnt)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    angle = random.uniform(-30, 30)
    text_img = Image.new('RGBA', (tw + 20, th + 20), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_img)
    text_draw.text((10, 10), text, fill=(150, 150, 150, int(255 * opacity)), font=fnt)
    rotated = text_img.rotate(angle, expand=1, resample=Image.BICUBIC)
    px, py = (w - rotated.width) // 2, (h - rotated.height) // 2
    overlay.paste(rotated, (px, py), rotated)
    return Image.alpha_composite(pil_img.convert('RGBA'), overlay).convert('RGB')
    
def apply_vignette_effect(pil_img, **kwargs):
    w, h = pil_img.size
    X, Y = np.ogrid[:h, :w]
    centerX, centerY = w / 2.0, h / 2.0
    radius = np.sqrt((X - centerX)**2 + (Y - centerY)**2)
    max_radius = np.sqrt(centerX**2 + centerY**2)
    vignette_factor = 1.0 - (radius / max_radius) ** 2
    vignette_factor = np.clip(vignette_factor, 0.3, 1.0)
    arr = np.array(pil_img).astype(np.float32)
    arr *= vignette_factor[:, :, np.newaxis]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def apply_scanner_streaks_effect(pil_img, **kwargs):
    pil_img_rgba = pil_img.convert('RGBA')
    draw = ImageDraw.Draw(pil_img_rgba)
    w, h = pil_img.size
    for _ in range(random.randint(1, 4)):
        y = random.randint(0, h - 1)
        opacity = random.randint(5, 20)
        color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255), opacity)
        draw.line([(0, y), (w, y)], fill=color, width=random.choice([1, 2]))
    return pil_img_rgba.convert('RGB')

def apply_clipping_effect(pil_img, clip_range_pct=(0.01, 0.04), **kwargs):
    w, h = pil_img.size
    side = random.choice(['top', 'left', 'right', 'bottom'])
    clip_amount = random.randint(int(min(w, h) * clip_range_pct[0]), int(min(w, h) * clip_range_pct[1]))
    bg_color = pil_img.getpixel((2, 2))
    new_img = Image.new('RGB', (w, h), bg_color)
    dx, dy = 0, 0
    if side == 'top':
        # When clipping the top, content is shifted up in Matplotlib's (bottom-left) coordinate system.
        # The offset to add to the y-coordinate should be positive.
        region = pil_img.crop((0, clip_amount, w, h)); new_img.paste(region, (0, 0)); dy = clip_amount
    elif side == 'left':
        # When clipping the left, content is shifted left. The offset is negative.
        region = pil_img.crop((clip_amount, 0, w, h)); new_img.paste(region, (0, 0)); dx = -clip_amount
    elif side == 'right':
        region = pil_img.crop((0, 0, w - clip_amount, h)); new_img.paste(region, (0, 0))
    else: # bottom
        region = pil_img.crop((0, 0, w, h - clip_amount)); new_img.paste(region, (0, 0))
    return new_img, dx, dy

def apply_printing_artifacts_effect(pil_img, texture_alpha=(0.05, 0.1), blur_radius=(0.2, 0.4), **kwargs):
    if random.random() < 0.5:
        w, h = pil_img.size; noise = np.random.randint(245, 256, size=(h, w), dtype=np.uint8)
        paper_texture = Image.fromarray(noise).filter(ImageFilter.GaussianBlur(radius=0.5))
        texture_rgb = Image.merge('RGB', (paper_texture, paper_texture, paper_texture))
        pil_img = Image.blend(pil_img, texture_rgb, alpha=random.uniform(*texture_alpha))
    if random.random() < 0.5: pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(*blur_radius)))
    return pil_img

def apply_mouse_cursor_effect(pil_img, **kwargs):
    pil_img_rgba = pil_img.convert('RGBA')
    w, h = pil_img.size
    cursor_pos = (random.randint(int(w * 0.1), int(w * 0.9)), random.randint(int(h * 0.1), int(h * 0.9)))
    draw = ImageDraw.Draw(pil_img_rgba)
    x, y = cursor_pos
    cursor_poly = [(x, y), (x, y + 16), (x + 3, y + 13), (x + 6, y + 19), (x + 8, y + 18), (x + 5, y + 12), (x + 11, y + 12), (x, y)]
    draw.polygon(cursor_poly, outline=(0, 0, 0, 200), fill=(255, 255, 255, 220))
    return pil_img_rgba.convert('RGB')

def apply_text_degradation_effect(pil_img, renderer, ax, blur_radius_range=(0.4, 1.2), pixelate_scale_options=[2, 3], **kwargs):
    text_items = list(getattr(ax, 'texts', [])) + list(ax.xaxis.get_ticklabels()) + list(ax.yaxis.get_ticklabels())
    text_items.extend([ax.xaxis.label, ax.yaxis.label, ax.title])
    if ax.get_legend(): text_items.extend(ax.get_legend().get_texts())
    for t in text_items:
        if not t or not t.get_visible(): continue
        try: bbox = t.get_window_extent(renderer)
        except (RuntimeError, AttributeError): continue
        if bbox.width <= 1 or bbox.height <= 1: continue
        pad = 2
        x0, y0, x1, y1 = map(int, (bbox.x0 - pad, bbox.y0 - pad, bbox.x1 + pad, bbox.y1 + pad))
        x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(pil_img.width, x1), min(pil_img.height, y1)
        if x1 <= x0 or y1 <= y0: continue
        region = pil_img.crop((x0, y0, x1, y1))
        if random.random() < 0.6:
            region = region.filter(ImageFilter.GaussianBlur(radius=random.uniform(*blur_radius_range)))
        else:
            scale = random.choice(pixelate_scale_options)
            small = region.resize((max(1, region.width // scale), max(1, region.height // scale)), resample=Image.BILINEAR)
            region = small.resize(region.size, resample=Image.NEAREST)
        pil_img.paste(region, (x0, y0))
    return pil_img

def apply_grid_occlusion_effect(pil_img, **kwargs):
    pil_img_rgba = pil_img.convert('RGBA')
    draw = ImageDraw.Draw(pil_img_rgba)
    w, h = pil_img.size
    for _ in range(random.randint(1,2)):
        y_center = random.uniform(0.2 * h, 0.9 * h); height = random.uniform(0.04 * h, 0.15 * h)
        x0 = random.uniform(0, 0.25 * w); x1 = random.uniform(0.75 * w, w)
        opacity = int(255 * random.uniform(0.06, 0.18)); color = (230, 230, 245, opacity)
        draw.rectangle([x0, y_center - height / 2, x1, y_center + height / 2], fill=color)
    return pil_img_rgba.convert('RGB')

def apply_scan_rotation_effect(pil_img, angle_range=(-1, 1), **kwargs):
    angle = random.uniform(*angle_range)
    rotated_img = pil_img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=pil_img.getpixel((0,0)))
    return rotated_img, angle

def apply_grayscale_effect(pil_img, **kwargs):
    return pil_img.convert('L').convert('RGB')

def apply_perspective_warp_effect(pil_img, distortion_factor=0.08, border_value=(255, 255, 255), return_homography=False, **kwargs):
    if not _HAS_CV2:
        raise RuntimeError("perspective warp requires opencv (see requirements.txt)")
    w, h = pil_img.size
    dx = w * distortion_factor
    dy = h * distortion_factor

    src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst_pts = np.float32([
        [random.uniform(0, dx), random.uniform(0, dy)],
        [random.uniform(w - 1 - dx, w - 1), random.uniform(0, dy)],
        [random.uniform(w - 1 - dx, w - 1), random.uniform(h - 1 - dy, h - 1)],
        [random.uniform(0, dx), random.uniform(h - 1 - dy, h - 1)]
    ])

    try:
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        out_img = Image.fromarray(cv2.warpPerspective(np.array(pil_img), H, (w, h), borderValue=border_value))
    except Exception:
        H, out_img = None, pil_img

    if return_homography:
        return out_img, H
    return out_img


def apply_uneven_lighting_effect(pil_img, intensity=0.6, gradient_type="radial", **kwargs):
    intensity = max(0.0, min(1.0, float(intensity)))
    arr = np.array(pil_img).astype(np.float32)
    h, w = arr.shape[:2]

    if gradient_type == "radial":
        X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        distance = np.sqrt(X ** 2 + Y ** 2)
        light_mask = 1.0 - (distance / np.max(distance)) * intensity
    elif gradient_type == "linear":
        gradient = np.linspace(1.0, 1.0 - intensity, w)
        light_mask = np.tile(gradient, (h, 1))
    else:
        return pil_img

    light_mask = np.clip(light_mask, 0.0, 1.0)
    light_mask_3d = np.dstack([light_mask] * 3)
    degraded = np.clip(arr * light_mask_3d, 0, 255).astype(np.uint8)
    return Image.fromarray(degraded)


def apply_chromatic_aberration_effect(pil_img, r_shift=(2, 0), b_shift=(-2, 0), **kwargs):
    r_dx, r_dy = r_shift
    b_dx, b_dy = b_shift
    arr = np.array(pil_img)
    h, w = arr.shape[:2]

    if _HAS_CV2:
        try:
            ch_r, ch_g, ch_b = cv2.split(arr)
            M_r = np.float32([[1, 0, r_dx], [0, 1, r_dy]])
            M_b = np.float32([[1, 0, b_dx], [0, 1, b_dy]])
            ch_r = cv2.warpAffine(ch_r, M_r, (w, h), borderMode=cv2.BORDER_REPLICATE)
            ch_b = cv2.warpAffine(ch_b, M_b, (w, h), borderMode=cv2.BORDER_REPLICATE)
            merged = cv2.merge((ch_r, ch_g, ch_b))
            return Image.fromarray(merged)
        except Exception:
            pass

    r, g, b = pil_img.split()
    r = ImageChops.offset(r, int(r_dx), int(r_dy))
    b = ImageChops.offset(b, int(b_dx), int(b_dy))
    return Image.merge("RGB", (r, g, b))


def apply_pdf_document_context_effect(pil_img, margin_px_range=(20, 50), caption_probability=0.7, body_text_probability=0.5, **kwargs):
    """
    Applies page-context realism effect (margins, optional figure caption, optional text blocks).
    Grows canvas and returns (new_pil_img, dx, dy) where dx and dy translate Matplotlib bboxes.
    """
    w, h = pil_img.size
    margin_left = random.randint(*margin_px_range)
    margin_right = random.randint(*margin_px_range)
    margin_top = random.randint(*margin_px_range)
    margin_bottom = random.randint(*margin_px_range)

    # Optional caption
    has_caption = (random.random() < caption_probability)
    caption_height = 0
    caption_text = ""
    if has_caption:
        fig_num = random.randint(1, 12)
        phrases = [
            "Comparison of performance metrics across experimental groups.",
            "Distribution of normalized values for key parameters.",
            "Overview of experimental measurements and observed trends.",
            "Summary of evaluation results under standard test conditions.",
            "Quantitative breakdown of observed data distributions.",
            "Analysis of key metrics relative to baseline performance.",
            "Experimental results demonstrating systematic behavior.",
            "Detailed view of measured indicators across categories."
        ]
        caption_text = f"Figure {fig_num}: {random.choice(phrases)}"
        caption_height = random.randint(30, 45)

    # Optional body text blocks
    has_body_text = (random.random() < body_text_probability)
    top_text_height = random.randint(30, 60) if has_body_text else 0
    bottom_text_height = random.randint(30, 60) if has_body_text else 0

    total_top_padding = margin_top + top_text_height
    total_bottom_padding = margin_bottom + caption_height + bottom_text_height

    new_w = w + margin_left + margin_right
    new_h = h + total_top_padding + total_bottom_padding

    # Page background (slightly off-white or white)
    bg_gray = random.randint(245, 255)
    bg_color = (bg_gray, bg_gray, bg_gray)
    new_img = Image.new('RGB', (new_w, new_h), bg_color)

    # Paste original chart canvas into padded region
    paste_x = margin_left
    paste_y = total_top_padding
    new_img.paste(pil_img, (paste_x, paste_y))

    draw = ImageDraw.Draw(new_img)
    try:
        font_size = max(11, int(min(new_w, new_h) * 0.022))
        fnt = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        fnt = ImageFont.load_default()

    # Draw body text above
    if has_body_text and top_text_height > 0:
        line_y = max(5, margin_top // 2)
        mock_line = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
        draw.text((margin_left, line_y), mock_line[:int(new_w * 0.12)], fill=(60, 60, 60), font=fnt)

    # Draw caption below figure
    if has_caption:
        caption_y = paste_y + h + 10
        draw.text((margin_left, caption_y), caption_text, fill=(30, 30, 30), font=fnt)

    # Draw body text below caption
    if has_body_text and bottom_text_height > 0:
        line_y = paste_y + h + caption_height + 15
        mock_line = "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
        draw.text((margin_left, line_y), mock_line[:int(new_w * 0.12)], fill=(60, 60, 60), font=fnt)

    dx = paste_x
    dy = total_bottom_padding

    return new_img, dx, dy


def apply_page_curl(
    pil_img: Image.Image,
    curl_axis: str = "y",
    amplitude: Optional[float] = None,
    wavelength: Optional[float] = None,
    phase: float = 0.0,
    border_value: Optional[Tuple[int, int, int]] = None,
    return_forward_map: bool = True,
    **kwargs
):
    """
    Apply a cylindrical page-curl / sine-mesh non-rigid displacement field.
    Returns:
      (warped_img, forward_map) if return_forward_map is True
      warped_img if return_forward_map is False
    """
    w, h = pil_img.size

    # Compute amplitude and wavelength defaults if not provided
    if amplitude is None:
        amp_ratio = float(kwargs.get("amplitude_ratio", random.uniform(0.02, 0.05)))
        amplitude = amp_ratio * (h if curl_axis == "y" else w)
    else:
        amplitude = float(amplitude)

    if wavelength is None:
        wave_ratio = float(kwargs.get("wavelength_ratio", random.uniform(0.8, 1.5)))
        wavelength = wave_ratio * (w if curl_axis == "y" else h)
    else:
        wavelength = float(wavelength)

    phase = float(phase)

    # Ensure wavelength is not zero
    if wavelength == 0:
        wavelength = float(w if curl_axis == "y" else h)

    if border_value is None:
        try:
            border_value = pil_img.getpixel((0, 0))
            if not isinstance(border_value, tuple):
                border_value = (border_value, border_value, border_value)
            elif len(border_value) > 3:
                border_value = border_value[:3]
        except Exception:
            border_value = (255, 255, 255)

    # 1. Forward coordinate mapping function: (x_src, y_src) -> (x_dst, y_dst)
    # in image coordinates (origin top-left)
    def forward_map(points_array):
        pts = np.asarray(points_array, dtype=np.float64)
        is_1d = (pts.ndim == 1)
        pts_2d = np.atleast_2d(pts)

        xs = pts_2d[:, 0]
        ys = pts_2d[:, 1]

        if curl_axis == "y":
            # Cylinder / spine along Y: curvature across X displaces Y
            dy = amplitude * np.sin(2.0 * np.pi * xs / wavelength + phase)
            warped_xs = xs
            warped_ys = ys + dy
        else:
            # Cylinder / spine along X: curvature across Y displaces X
            dx = amplitude * np.sin(2.0 * np.pi * ys / wavelength + phase)
            warped_xs = xs + dx
            warped_ys = ys

        warped = np.column_stack([warped_xs, warped_ys])
        return warped[0] if is_1d else warped

    # 2. Image remap using inverse coordinate mapping: (x_dst, y_dst) -> (x_src, y_src)
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    if curl_axis == "y":
        map_x = grid_x
        map_y = grid_y - np.float32(amplitude * np.sin(2.0 * np.pi * grid_x / wavelength + phase))
    else:
        map_x = grid_x - np.float32(amplitude * np.sin(2.0 * np.pi * grid_y / wavelength + phase))
        map_y = grid_y

    img_arr = np.array(pil_img)
    bv = border_value if isinstance(border_value, tuple) else (255, 255, 255)

    if _HAS_CV2:
        try:
            warped_arr = cv2.remap(
                img_arr,
                map_x,
                map_y,
                interpolation=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=bv
            )
            out_img = Image.fromarray(warped_arr)
        except Exception:
            out_img = pil_img
    else:
        try:
            from scipy.ndimage import map_coordinates
            coords = np.array([map_y, map_x])
            warped_arr = np.empty_like(img_arr)
            for c in range(min(3, img_arr.shape[2] if img_arr.ndim > 2 else 1)):
                fill_c = bv[c] if isinstance(bv, (tuple, list)) and c < len(bv) else 255
                if img_arr.ndim > 2:
                    warped_arr[:, :, c] = map_coordinates(
                        img_arr[:, :, c], coords, order=1, mode="constant", cval=fill_c
                    )
                else:
                    warped_arr = map_coordinates(img_arr, coords, order=1, mode="constant", cval=fill_c)
            out_img = Image.fromarray(warped_arr)
        except Exception:
            out_img = pil_img

    if return_forward_map:
        return out_img, forward_map
    return out_img

