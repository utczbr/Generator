import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
import matplotlib
from matplotlib import patches, rcParams, transforms, colormaps
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.colors import ListedColormap
from matplotlib import colormaps
from scipy import stats
from scipy.stats import random_correlation
from scipy.special import gamma, hyp1f1
from scipy.signal import find_peaks, lfilter
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist, pdist
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.linalg import eigh, block_diag, toeplitz
def _block_matrix(shape, n_row_groups, n_col_groups, block_pairs, noise, minval, maxval, shuffle, random_state):
    """Gaussian noise plus one uniform offset per (row_group, col_group) block, optionally shuffled."""
    if random_state is not None:
        np.random.seed(random_state)
    rows, cols = shape
    row_groups = np.array_split(np.arange(rows), max(1, n_row_groups))
    col_groups = np.array_split(np.arange(cols), max(1, n_col_groups))
    matrix = np.random.normal(0, noise, (rows, cols))
    for i, j in block_pairs(len(row_groups), len(col_groups)):
        matrix[np.ix_(row_groups[i], col_groups[j])] += np.random.uniform(minval, maxval)
    if shuffle:
        matrix = matrix[np.ix_(np.random.permutation(rows), np.random.permutation(cols))]
    return matrix, None, None


def make_biclusters(shape, n_clusters, noise=0.0, minval=-5.0, maxval=5.0, shuffle=True, random_state=None):
    """Diagonal blocks: row group k pairs with column group k."""
    n = int(n_clusters[0] if isinstance(n_clusters, (list, tuple)) else n_clusters)
    return _block_matrix(shape, n, n, lambda nr, nc: ((k, k % nc) for k in range(nr)),
                         noise, minval, maxval, shuffle, random_state)


def make_checkerboard(shape, n_clusters, noise=0.0, minval=-5.0, maxval=5.0, shuffle=True, random_state=None):
    """Alternating blocks: (row group i, column group j) is raised when i + j is even."""
    n_r, n_c = (int(n_clusters[0]), int(n_clusters[1])) if isinstance(n_clusters, (list, tuple)) else (int(n_clusters),) * 2
    return _block_matrix(shape, n_r, n_c, lambda nr, nc: ((i, j) for i in range(nr) for j in range(nc) if (i + j) % 2 == 0),
                         noise, minval, maxval, shuffle, random_state)


from themes import THEMES, FONT_FAMILIES
from synth.structural import (
    CONTEXT_CONFIGURATIONS,
    STRUCTURAL_THEMES,
    HEATMAP_ANNOTATION_FORMATS,
    COLORBAR_TITLES_SCIENTIFIC,
    COLORBAR_TITLES_BUSINESS,
    HEATMAP_CHART_TITLES,
    HEATMAP_XLABELS_SCIENTIFIC,
    HEATMAP_YLABELS_SCIENTIFIC,
    HEATMAP_XLABELS_BUSINESS,
    HEATMAP_YLABELS_BUSINESS,
)

DUAL_AXIS_PROBABILITY: float = 0.15
TREATMENT_KEY_PROBABILITY: float = 0.30

# ===================================================================================
# == DATA GENERATION & THEMES ==
# ===================================================================================

def generate_realistic_data(num_points, max_scale, allow_negative=False, pattern_type=None, domain='scientific'):
    """
    Generate statistically realistic data based on real-world scientific and business patterns.
    
    Critical improvements:
    - Domain-specific parameter constraints based on published literature
    - Heteroscedastic noise models matching measurement error characteristics
    - Realistic coefficient of variation (CV) ranges for biological systems
    - Enforced monotonicity and physical plausibility constraints
    - Measurement precision limitations
    """
    
    if pattern_type is None:
        # Weight patterns by actual frequency in scientific literature
        if domain == 'scientific':
            pattern_type = np.random.choice(
                ['dose_response', 'replicates', 'exponential_decay', 'power_law', 
                 'sigmoid_growth', 'linear_regression', 'gaussian_peak', 'enzyme_kinetics'],
                p=[0.25, 0.20, 0.15, 0.10, 0.10, 0.10, 0.05, 0.05]
            )
        else:  # business
            pattern_type = np.random.choice(
                ['seasonal_trend', 'pareto_distribution', 'exponential_growth', 
                 'market_saturation', 'random_walk_drift', 'step_intervention'],
                p=[0.30, 0.25, 0.15, 0.15, 0.10, 0.05]
            )
    
    data = np.zeros(num_points)
    
    # === SCIENTIFIC PATTERNS WITH REALISTIC CONSTRAINTS ===
    if pattern_type == 'dose_response':
        # Hill equation with literature-validated parameter ranges
        log_conc = np.linspace(-10, -3, num_points)  # pM to mM range (realistic drug concentrations)
        
        # Realistic EC50 values from drug databases (ChEMBL, PubChem)
        ec50 = np.random.uniform(-8.5, -4.5)  # 3nM to 30µM range
        
        # Hill slopes from literature (rarely exceed 4, typically 0.7-2.5)
        hill_slope = np.random.choice([
            np.random.uniform(0.7, 1.3),   # 60% - physiological range
            np.random.uniform(1.3, 2.0),   # 30% - cooperative binding
            np.random.uniform(2.0, 3.5)    # 10% - strong cooperativity
        ], p=[0.6, 0.3, 0.1])
        
        # Realistic baseline and maximum response
        baseline = np.random.uniform(0, 0.08) * max_scale  # 0-8% baseline activity
        max_response = np.random.uniform(0.85, 0.98) * max_scale  # 85-98% max response
        
        # Hill equation
        data = baseline + (max_response - baseline) / (1 + 10**((ec50 - log_conc) * hill_slope))
        
        # Heteroscedastic noise: CV increases at curve inflection points
        response_fraction = (data - baseline) / (max_response - baseline)
        cv = 0.05 + 0.10 * np.sqrt(response_fraction * (1 - response_fraction))
        noise = np.random.normal(0, data * cv, num_points)
        data += noise
        
        # Enforce non-negativity for biological measurements
        data = np.clip(data, 0, max_response * 1.05)
    
    elif pattern_type == 'replicates':
        # Biological replicates with realistic technical variation
        mean_val = np.random.uniform(0.25, 0.75) * max_scale
        
        # CV based on measurement type (qPCR: 5-15%, Western: 10-25%, Cell assays: 15-35%)
        measurement_type = np.random.choice(['qpcr', 'western', 'cell_assay'], p=[0.3, 0.3, 0.4])
        cv_ranges = {
            'qpcr': (0.05, 0.15),
            'western': (0.10, 0.25), 
            'cell_assay': (0.15, 0.35)
        }
        cv = np.random.uniform(*cv_ranges[measurement_type])
        
        # Log-normal distribution for biological variability (more realistic than normal)
        sigma = np.sqrt(np.log(1 + cv**2))
        mu = np.log(mean_val) - 0.5 * sigma**2
        data = np.random.lognormal(mu, sigma, num_points)
        
        # Clip to realistic bounds
        data = np.clip(data, 0, max_scale * 1.2)
    
    elif pattern_type == 'exponential_decay':
        # Realistic pharmacokinetic/radioactive decay parameters
        t = np.linspace(0, np.random.uniform(5, 20), num_points)
        
        # Half-life ranges based on actual pharmacokinetic data
        half_life = np.random.choice([
            np.random.uniform(0.5, 2),    # Fast clearance (minutes to hours)
            np.random.uniform(2, 12),     # Moderate clearance (hours)
            np.random.uniform(12, 72)     # Slow clearance (days)
        ], p=[0.3, 0.5, 0.2])
        
        decay_constant = np.log(2) / half_life
        
        initial_value = np.random.uniform(0.8, 0.95) * max_scale
        baseline = np.random.uniform(0, 0.1) * max_scale
        
        data = (initial_value - baseline) * np.exp(-decay_constant * t) + baseline
        
        # Proportional error model (higher error at higher concentrations)
        error_proportional = 0.08  # 8% proportional error
        error_additive = 0.02 * max_scale  # 2% additive error
        noise = np.random.normal(0, data * error_proportional + error_additive, num_points)
        data += noise
        
        data = np.clip(data, baseline * 0.8, initial_value * 1.1)
    
    elif pattern_type == 'enzyme_kinetics':
        # Michaelis-Menten kinetics with realistic parameters
        substrate = np.logspace(-1, 2, num_points)  # 0.1 to 100 units
        
        # Realistic Km values (µM to mM range for most enzymes)
        km = np.random.uniform(0.5, 50)
        vmax = np.random.uniform(0.7, 0.95) * max_scale
        
        # Michaelis-Menten equation
        data = (vmax * substrate) / (km + substrate)
        
        # Add realistic experimental noise (CV = 5-15% for enzyme assays)
        cv = np.random.uniform(0.05, 0.15)
        noise = np.random.normal(0, data * cv, num_points)
        data += noise
        
        data = np.clip(data, 0, vmax * 1.05)
    
    elif pattern_type == 'gaussian_peak':
        # Spectroscopy peak or chromatography data
        x = np.arange(num_points)
        
        # Peak position and width
        mu = np.random.uniform(num_points * 0.3, num_points * 0.7)
        sigma = np.random.uniform(num_points * 0.05, num_points * 0.20)
        
        amplitude = max_scale * np.random.uniform(0.80, 0.95)
        baseline = max_scale * np.random.uniform(0, 0.10)
        
        data = amplitude * np.exp(-((x - mu)**2) / (2 * sigma**2)) + baseline
        
        # Poisson-like noise (typical for photon counting/mass spectrometry)
        noise_factor = np.sqrt(np.abs(data - baseline))
        noise = np.random.normal(0, noise_factor * 0.3, num_points)
        data += noise
        
        data = np.clip(data, baseline * 0.9, amplitude * 1.1)
    
    # === BUSINESS PATTERNS WITH REALISTIC CONSTRAINTS ===
    elif pattern_type == 'seasonal_trend':
        # Realistic business seasonality with multiple components
        x = np.arange(num_points)
        
        # Multiple seasonality (annual + quarterly + monthly if enough points)
        components = []
        
        if num_points >= 12:  # Annual seasonality
            annual_freq = 2 * np.pi / 12
            annual_amp = max_scale * np.random.uniform(0.15, 0.30)
            annual_phase = np.random.uniform(0, 2*np.pi)
            components.append(annual_amp * np.cos(annual_freq * x + annual_phase))
        
        if num_points >= 4:   # Quarterly seasonality
            quarterly_freq = 2 * np.pi / 4
            quarterly_amp = max_scale * np.random.uniform(0.08, 0.15)
            quarterly_phase = np.random.uniform(0, 2*np.pi)
            components.append(quarterly_amp * np.cos(quarterly_freq * x + quarterly_phase))
        
        # Base level with trend
        base_level = max_scale * np.random.uniform(0.3, 0.5)
        trend_slope = max_scale * np.random.uniform(-0.05, 0.20) / num_points
        trend = x * trend_slope
        
        # Combine components
        seasonal = np.sum(components, axis=0) if components else np.zeros(num_points)
        data = base_level + trend + seasonal
        
        # Business-appropriate noise (higher during peak seasons)
        noise_base = max_scale * 0.03
        noise_seasonal = np.abs(seasonal) * 0.2
        noise = np.random.normal(0, noise_base + noise_seasonal, num_points)
        data += noise
        
        data = np.clip(data, 0, max_scale * 1.5)
    
    elif pattern_type == 'pareto_distribution':
        # Pareto principle (80/20 rule) - realistic for business data
        shape = np.random.uniform(1.05, 2.5)  # Literature range for business data
        
        # Generate Pareto samples
        samples = np.random.pareto(shape, num_points) + 1
        
        # Sort in descending order for typical business visualization
        data = np.sort(samples)[::-1]
        
        # Scale to max_scale with realistic ceiling
        data = (data / data.max()) * max_scale * np.random.uniform(0.6, 0.9)
        
        # Add small multiplicative noise (log-normal)
        noise = np.random.lognormal(0, 0.10, num_points)
        data *= noise
    
    elif pattern_type == 'exponential_growth':
        # Realistic business growth with saturation
        t = np.linspace(0, 1, num_points)
        
        # Growth rates based on actual business metrics
        growth_rate = np.random.choice([
            np.random.uniform(2, 5),      # Moderate growth
            np.random.uniform(5, 10),     # High growth
            np.random.uniform(10, 15)     # Exponential phase
        ], p=[0.5, 0.3, 0.2])
        
        initial_value = max_scale * np.random.uniform(0.05, 0.20)
        
        # Exponential with eventual saturation (logistic-like)
        data = initial_value * np.exp(growth_rate * t)
        
        # Apply market saturation
        carrying_capacity = max_scale * np.random.uniform(0.8, 1.0)
        saturation_factor = 1 / (1 + (data / carrying_capacity))
        data *= saturation_factor
        
        # Business noise (proportional to current value)
        cv = np.random.uniform(0.08, 0.20)
        noise = np.random.normal(0, data * cv, num_points)
        data += noise
        
        data = np.clip(data, initial_value * 0.8, carrying_capacity * 1.1)
    
    # === FALLBACK PATTERNS ===
    elif pattern_type == 'linear':
        start = max_scale * np.random.uniform(0.10, 0.40)
        end = max_scale * np.random.uniform(0.50, 0.90)
        
        if domain == 'scientific' and np.random.random() < 0.7:
            start, end = min(start, end), max(start, end)
        
        data = np.linspace(start, end, num_points)
        
        noise_cv = np.random.uniform(0.05, 0.15)
        noise = np.random.normal(0, data * noise_cv + max_scale * 0.01, num_points)
        data += noise
    
    elif pattern_type == 'plateau':
        p1, p2 = sorted(random.sample(range(num_points + 1), 2))
        low = max_scale * np.random.uniform(0.1, 0.3)
        high = max_scale * np.random.uniform(0.7, 0.9)
        
        data = np.concatenate([
            np.full(p1, low), 
            np.full(p2 - p1, high), 
            np.full(num_points - p2, low)
        ])
        
        noise = np.random.normal(0, max_scale * 0.05, num_points)
        data += noise
    
    else:  # Default: random_walk
        start = max_scale * np.random.uniform(0.3, 0.7)
        steps = np.random.normal(0, max_scale * 0.1, num_points)
        data = start + np.cumsum(steps)
        
        noise = np.random.normal(0, max_scale * 0.05, num_points)
        data += noise
    
    # === POST-PROCESSING FOR MEASUREMENT REALISM ===
    
    # Apply measurement constraints
    if not allow_negative:
        data = np.clip(data, 0, max_scale * 1.05)  # Strict 5% overshoot for realism without excess
    else:
        data = np.clip(data, -max_scale * 0.4, max_scale * 1.05)  # Strict 5% overshoot for realism without excess
    
    # Realistic measurement precision (instruments have limited precision)
    if max_scale >= 1000:
        precision = np.random.choice([0, 1], p=[0.7, 0.3])
    elif max_scale >= 100:
        precision = np.random.choice([1, 2], p=[0.6, 0.4])
    elif max_scale >= 10:
        precision = np.random.choice([2, 3], p=[0.7, 0.3])
    else:
        precision = 3
    
    data = np.round(data, precision)
    
    # Remove impossible values
    if domain == 'scientific' and not allow_negative:
        data = np.abs(data)
    
    return data

def apply_chart_theme(ax, theme_name, orientation='vertical'):
    theme = THEMES.get(theme_name, THEMES.get('default', {}))
    if not theme:
        theme = {'facecolor': 'white', 'grid_color': '#CCCCCC', 'grid_style': 'solid', 'font': 'Arial'}
    
    ax.set_facecolor(theme.get('facecolor', 'white'))
    
    grid_axis = 'y' if orientation == 'vertical' else 'x'
    
    if theme.get('grid_style', 'none') != 'none':
        ax.grid(axis=grid_axis, color=theme.get('grid_color', '#CCCCCC'), 
                linestyle=theme.get('grid_style', 'solid'),
                linewidth=theme.get('grid_linewidth', 1.0), zorder=0)
    
    try: 
        rcParams['font.sans-serif'] = [theme.get('font', 'Arial'), 'DejaVu Sans', 'Arial']
    except Exception: 
        pass
    
    for spine, visible in theme.get('spines', {}).items():
        if spine in ax.spines: 
            ax.spines[spine].set_visible(visible)
    
    scale = theme.get('typography_scale', 1.0)
    if theme.get('spine_width'):
        for spine in ax.spines.values():
            spine.set_linewidth(theme['spine_width'] * scale)
    
    if theme.get('tick_direction'): 
        ax.tick_params(axis='both', direction=theme['tick_direction'])
    
    return theme


def apply_tick_formatters(ax, orientation='vertical', active_profile=None, theme_config=None):
    """
    Apply publication-grade tick formatters and numeric clutter (FR-088, T234c).
    Under profile != 'legacy', draws from feature_rng to apply:
    - ScalarFormatter with scientific multiplier (useMathText=True, powerlimits=(-2, 2))
    - PercentFormatter
    - Currency formatters ($k, $M)
    - Metric / engineering scale suffixes (k, M)
    """
    if not active_profile or active_profile == 'legacy':
        return

    import matplotlib.ticker as ticker
    from feature_rng import feature_rng

    seed = ax.figure._generator_config.get('seed') if hasattr(ax, 'figure') and hasattr(ax.figure, '_generator_config') else 42
    image_idx = getattr(ax.figure, '_generator_image_idx', 0) if hasattr(ax, 'figure') else 0
    rng = feature_rng(seed if seed is not None else 42, image_idx, "tick_formatter")

    # Probability of applying a non-standard formatter (>= 25% requirement)
    p_format = 0.35
    if isinstance(theme_config, dict) and 'tick_formatter_probability' in theme_config:
        p_format = float(theme_config['tick_formatter_probability'])

    if rng.random() >= p_format:
        return

    actual_orientation = orientation
    if isinstance(theme_config, dict) and 'orientation' in theme_config:
        actual_orientation = theme_config['orientation']

    target_axis = ax.xaxis if actual_orientation == 'horizontal' else ax.yaxis

    fmt_type = rng.choices(
        ['scientific', 'percent', 'currency', 'metric_suffix'],
        weights=[0.35, 0.25, 0.20, 0.20],
        k=1
    )[0]

    if fmt_type == 'scientific':
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        target_axis.set_major_formatter(formatter)
    elif fmt_type == 'percent':
        target_axis.set_major_formatter(ticker.PercentFormatter(xmax=100.0, symbol='%'))
    elif fmt_type == 'currency':
        def _curr(x, pos):
            ax_val = abs(x)
            if ax_val >= 1e6:
                return f"${x*1e-6:.1f}M"
            elif ax_val >= 1e3:
                return f"${x*1e-3:.1f}k"
            else:
                return f"${x:,.0f}"
        target_axis.set_major_formatter(ticker.FuncFormatter(_curr))
    elif fmt_type == 'metric_suffix':
        def _metric(x, pos):
            ax_val = abs(x)
            if ax_val >= 1e6:
                return f"{x*1e-6:.1f}M"
            elif ax_val >= 1e3:
                return f"{x*1e-3:.1f}k"
            else:
                return f"{x:.0f}"
        target_axis.set_major_formatter(ticker.FuncFormatter(_metric))


def apply_typography_variation(ax, domain='scientific', theme_config=None, tick_rotation_cfg=None, orientation='vertical'):
    """
    Aplica diversas configurações de tipografia com base na análise do usuário.
    Varia a família da fonte, tamanhos, peso e rotação.
    """
    
    active_profile = None
    if isinstance(theme_config, dict):
        active_profile = theme_config.get('profile')
    if active_profile is None and hasattr(ax, 'figure') and hasattr(ax.figure, '_generator_config'):
        active_profile = ax.figure._generator_config.get('profile')

    if active_profile and active_profile != 'legacy':
        apply_tick_formatters(ax, orientation=orientation, active_profile=active_profile, theme_config=theme_config)
        import font_registry
        font_registry.ensure_fonts_registered()

        seed = None
        image_idx = 0
        if hasattr(ax, 'figure') and hasattr(ax.figure, '_generator_config'):
            seed = ax.figure._generator_config.get('seed')
            image_idx = getattr(ax.figure, '_generator_image_idx', 0)

        # Check if theme_config specifies a font (e.g. PUBLICATION_THEMES)
        if isinstance(theme_config, dict) and 'font' in theme_config:
            font_name = font_registry.validate_font_and_glyphs(
                theme_config['font'], active_profile=active_profile
            )
        else:
            from feature_rng import feature_rng
            rng = feature_rng(seed if seed is not None else 42, image_idx, "font")
            chosen = font_registry.sample_bundled_font(rng, domain=domain)
            font_name = chosen["family"]

        if hasattr(ax, 'figure'):
            ax.figure._chosen_font_family = font_name
    else:
        # Seleciona a família da fonte (legacy)
        if domain == 'scientific':
            family = np.random.choice(['sans-serif', 'serif'], p=[0.7, 0.3])
        else: # business
            family = np.random.choice(['sans-serif', 'serif'], p=[0.8, 0.2])
        font_name = np.random.choice(FONT_FAMILIES[family])
    
    # Tamanhos de fonte
    scale = theme_config.get('typography_scale', 1.0) if isinstance(theme_config, dict) else 1.0
    if isinstance(theme_config, dict) and 'title_size' in theme_config:
        title_size = max(6, int(round(theme_config['title_size'] * scale)))
        label_size = max(5, int(round(theme_config.get('label_size', 7) * scale)))
        tick_size = max(5, int(round(theme_config.get('font_size', 7) * scale)))
    else:
        title_size = max(6, int(round(np.random.randint(12, 17) * scale)))
        label_size = max(5, int(round(np.random.randint(10, 14) * scale)))
        tick_size = max(5, int(round(np.random.randint(8, 12) * scale)))
    
    try:
        # Aplica aos elementos dos eixos
        if ax.title:
            ax.title.set_fontsize(title_size)
            ax.title.set_fontfamily(font_name)
            if active_profile and active_profile != 'legacy':
                from feature_rng import feature_rng
                rng = feature_rng(seed if seed is not None else 42, image_idx, "font_weight")
                ax.title.set_fontweight(rng.choices(['normal', 'bold'], weights=[0.6, 0.4], k=1)[0])
            else:
                ax.title.set_fontweight(np.random.choice(['normal', 'bold'], p=[0.6, 0.4]))
        
        if ax.xaxis.label:
            ax.xaxis.label.set_fontsize(label_size)
            ax.xaxis.label.set_fontfamily(font_name)
            
        if ax.yaxis.label:
            ax.yaxis.label.set_fontsize(label_size)
            ax.yaxis.label.set_fontfamily(font_name)
        
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(tick_size)
            label.set_fontfamily(font_name)
        
        # Variações de rotação de texto (FR-088)
        if tick_rotation_cfg is None and isinstance(theme_config, dict):
            tick_rotation_cfg = theme_config.get('tick_rotation')
        if tick_rotation_cfg is None and hasattr(ax, 'figure') and hasattr(ax.figure, '_generator_config'):
            tick_rotation_cfg = ax.figure._generator_config.get('tick_rotation')
        if not isinstance(tick_rotation_cfg, dict):
            tick_rotation_cfg = {}

        mode = tick_rotation_cfg.get('mode', 'legacy')
        active_profile = None
        if isinstance(theme_config, dict):
            active_profile = theme_config.get('profile')
        if active_profile is None and hasattr(ax, 'figure') and hasattr(ax.figure, '_generator_config'):
            active_profile = ax.figure._generator_config.get('profile')
        if mode == 'legacy' and active_profile and active_profile != 'legacy' and 'mode' not in tick_rotation_cfg:
            mode = 'profile'

        if mode == 'legacy':
            # Note: label_angle is Vega-Lite-only under legacy mode (D1-07/D2-01)
            if np.random.random() < 0.3: # 30% de chance
                rotation = np.random.choice([0, 45, 90], p=[0.5, 0.3, 0.2])
                # Gira apenas os ticks do eixo x para evitar eixo y ilegível
                ax.tick_params(axis='x', labelrotation=rotation)
        else:
            # Profile mode: honor label_angle if specified, else draw from angles/weights
            label_angle = None
            if isinstance(theme_config, dict) and 'label_angle' in theme_config:
                label_angle = theme_config.get('label_angle')
            elif hasattr(ax, 'figure') and hasattr(ax.figure, '_generator_config'):
                label_angle = ax.figure._generator_config.get('label_angle')
            
            if label_angle is not None and label_angle != 0:
                rotation = int(label_angle)
            else:
                p_rotate = tick_rotation_cfg.get('p_rotate', 0.40)
                if np.random.random() < p_rotate:
                    angles = tick_rotation_cfg.get('angles', [30, 45, 60, 90, -30, -45, -60])
                    weights = tick_rotation_cfg.get('weights', [0.20, 0.25, 0.15, 0.10, 0.10, 0.10, 0.10])
                    if weights and len(weights) == len(angles):
                        norm_w = [w / sum(weights) for w in weights]
                        rotation = int(np.random.choice(angles, p=norm_w))
                    else:
                        rotation = int(np.random.choice(angles))
                else:
                    rotation = 0

            if rotation != 0:
                ax.tick_params(axis='x', labelrotation=rotation)
                if abs(rotation) == 90 or rotation == 0:
                    ha = 'center'
                elif rotation > 0:
                    ha = 'right'
                else:
                    ha = 'left'
                for label in ax.get_xticklabels():
                    label.set_horizontalalignment(ha)
            
    except Exception as e:
        # Captura erros caso as fontes não sejam encontradas no sistema
        if domain == 'scientific': # Checa contra uma variável usada no bloco try
            print(f"Aviso: Não foi possível aplicar a fonte {font_name}. Erro: {e}")
        pass # Continua com as fontes padrão

def apply_axis_scaling(ax, data_min=None, orientation='vertical', scale_type='auto'):
    """Aplica diferentes escalas de eixo (log, symlog) com base na análise."""
    
    if scale_type == 'auto':
        scale_type = np.random.choice(['linear', 'log', 'symlog'],
                                      p=[0.80, 0.15, 0.05])
    
    if scale_type == 'linear':
        return  # Não faz nada
    
    # CRÍTICO: Verifica os limites dos dados antes de aplicar a escala log
    if scale_type == 'log':
        # Se data_min não foi fornecido ou é <= 0, não podemos usar 'log'.
        # Muda para 'symlog', que lida com valores zero e negativos.
        if data_min is None or data_min <= 0:
            scale_type = 'symlog'
    
    try:
        if orientation == 'vertical':
            if scale_type == 'log':
                # Isso só será executado se data_min > 0
                ax.set_yscale('log')
            elif scale_type == 'symlog':
                # Usa linthresh 1.0 conforme sugerido na análise
                ax.set_yscale('symlog', linthresh=1.0)
        else:  # horizontal
            if scale_type == 'log':
                ax.set_xscale('log')
            elif scale_type == 'symlog':
                ax.set_xscale('symlog', linthresh=1.0)
    except Exception as e:
        # Captura quaisquer erros restantes
        print(f"AVISO: Não foi possível aplicar a escala de eixo '{scale_type}'. Erro: {e}")

# ===================================================================================
# == CHART ELEMENT ADDITIONS (CRITICAL FOR YOLO ANNOTATION) ==
# ===================================================================================

def add_bar_shadows(ax, bars, fig):
    """Add realistic drop shadows to bars"""
    for bar in bars:
        dx, dy = 2 / fig.dpi, -2 / fig.dpi
        shadow_transform = ax.transData + transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        shadow = patches.Rectangle(bar.get_xy(), bar.get_width(), bar.get_height(),
                                 transform=shadow_transform, facecolor='black',
                                 alpha=0.2, zorder=bar.get_zorder() - 0.1)
        ax.add_patch(shadow)

def add_significance_markers(ax, bar_info, y_max, orientation='vertical', error_tops=None, significance_prob=None):
    """Add statistical significance markers between bars"""
    annotations = []
    
    threshold = (1.0 - significance_prob) if significance_prob is not None else 0.4
    if len(bar_info) < 2 or random.random() < threshold: 
        return annotations
    
    mode = random.choice(['bracket', 'letters'])
    
    if mode == 'bracket':
        try:
            idx1, idx2 = random.sample(range(len(bar_info)), 2)
        except ValueError:
            return annotations
        
        bracket_style = random.choice(['standard', 'extended'])
        
        # Get bar positions
        pos1, pos2 = bar_info[idx1]['center'], bar_info[idx2]['center']
        text = random.choice(['*', '**', '***', 'ns'])
        
        start_idx = min(idx1, idx2)
        end_idx = max(idx1, idx2)
        
        max_height_in_range = 0
        
        if orientation == 'vertical':
            # Find maximum height of ANY bar or error bar within range
            for i in range(start_idx, end_idx + 1):
                bar_height = error_tops[i] if error_tops and i < len(error_tops) else bar_info[i]['height']
                if abs(bar_height) > max_height_in_range:
                    max_height_in_range = abs(bar_height)
            
            y_for_level = max_height_in_range
            level = max_height_in_range * (1 + random.uniform(0.10, 0.20))
            
            height1 = error_tops[idx1] if error_tops and idx1 < len(error_tops) else bar_info[idx1]['height']
            height2 = error_tops[idx2] if error_tops and idx2 < len(error_tops) else bar_info[idx2]['height']
            
            if bracket_style == 'extended':
                gap = y_for_level * 0.05
                start_y1 = height1 + gap
                start_y2 = height2 + gap
                ax.plot([pos1, pos1, pos2, pos2], [start_y1, level, level, start_y2], 
                       lw=1.2, c='black', zorder=15)
            else:  # 'standard'
                tip_height = y_for_level * 0.05
                ax.plot([pos1, pos1, pos2, pos2], 
                       [level - tip_height, level, level, level - tip_height], 
                       lw=1.2, c='black', zorder=15)
            
            txt = ax.text((pos1 + pos2) / 2, level, text, ha='center', va='bottom', 
                         color='black', fontsize=12, zorder=15)
        
        else:  # Horizontal orientation
            # Find maximum "height" (width) of ANY bar in range
            for i in range(start_idx, end_idx + 1):
                bar_width = error_tops[i] if error_tops and i < len(error_tops) else bar_info[i]['height']
                if abs(bar_width) > max_height_in_range:
                    max_height_in_range = abs(bar_width)
            
            x_for_level = max_height_in_range
            level = x_for_level * (1 + random.uniform(0.15, 0.30))
            
            height1 = error_tops[idx1] if error_tops and idx1 < len(error_tops) else bar_info[idx1]['height']
            height2 = error_tops[idx2] if error_tops and idx2 < len(error_tops) else bar_info[idx2]['height']
            
            if bracket_style == 'extended':
                gap = x_for_level * 0.05
                start_x1 = height1 + gap
                start_x2 = height2 + gap
                ax.plot([start_x1, level, level, start_x2], [pos1, pos1, pos2, pos2], 
                       lw=1.2, c='black', zorder=15)
            else:  # 'standard'
                tip_height = x_for_level * 0.05
                ax.plot([level - tip_height, level, level, level - tip_height], 
                       [pos1, pos1, pos2, pos2], lw=1.2, c='black', zorder=15)
            
            txt = ax.text(level, (pos1 + pos2) / 2, text, ha='left', va='center', 
                         color='black', fontsize=12, zorder=15)
        
        annotations.append(txt)
    
    elif mode == 'letters':
        letters = random.sample(['a', 'b', 'c', 'd'], k=min(len(bar_info), 4))
        
        for i, info in enumerate(bar_info):
            if i >= len(letters): 
                break
            
            pos, height = info['center'], info['height']
            base_y = error_tops[i] if error_tops and i < len(error_tops) else height
            offset = 0.05 * y_max
            y_pos = base_y + offset if height >= 0 else base_y - offset
            va = 'bottom' if height >= 0 else 'top'
            
            if orientation == 'vertical':
                txt = ax.text(pos, y_pos, letters[i], ha='center', va=va, fontsize=10)
            else:
                txt = ax.text(y_pos, pos, letters[i], ha='left', va='center', fontsize=10)
            
            annotations.append(txt)
    
    return annotations

def apply_legend_variation(ax, num_items):
    """Aplica diversas configurações de posicionamento e estilo de legenda."""
    
    # Posições internas
    inside_locs = ['upper right', 'upper left', 'lower left', 'lower right', 
                   'center', 'center right', 'center left']
    # Posição externa (à direita)
    outside_right = 'center left'
    
    # Escolhe a localização
    if num_items <= 4:
        # Legendas pequenas podem ir para dentro
        loc = np.random.choice(inside_locs + [outside_right], 
                             p=[0.12]*7 + [0.16])
    else:
        # Legendas grandes são melhores do lado de fora
        loc = np.random.choice(inside_locs + [outside_right],
                             p=[0.05]*7 + [0.65])
    
    # Configurações de moldura
    frameon = np.random.choice([True, False], p=[0.4, 0.6])
    
    # Múltiplas colunas para muitos itens
    ncol = 1
    if num_items > 6:
        ncol = int(np.random.choice([1, 2], p=[0.7, 0.3]))

    handles, labels = ax.get_legend_handles_labels()
    kwargs = {'loc': loc, 'frameon': frameon, 'ncols': ncol}
    if loc == 'center left':
        kwargs['bbox_to_anchor'] = (1.04, 0.5)

    if handles and labels:
        legend = ax.legend(handles=handles, labels=labels, **kwargs)
    else:
        import matplotlib.patches as mpatches
        pal = ['#2E86AB', '#D62728', '#2CA02C', '#FF7F0E']
        h_fall = [mpatches.Patch(color=pal[i % len(pal)], label=f'Series {i+1}') for i in range(num_items)]
        legend = ax.legend(handles=h_fall, **kwargs)
    
    return legend

from collections import namedtuple
if 'BoundingBox' not in globals():
    BoundingBox = namedtuple('BoundingBox', ['x0', 'y0', 'x1', 'y1'])

def extract_legend_elements(legend, ax, img_h=None, img_w=None):
    """
    Decompose a Matplotlib Legend into granular elements (FR-114, OC-9):
    - 'legend_title': legend title text (if present)
    - 'legend_label': each series label text
    - 'legend_marker': each series handle (marker / line / patch)

    A marker-less Line2D handle has zero-height window extent (E16: 119/119),
    so its legend_marker box is padded by half line-width (>= 2 px per side).
    All boxes have width and height >= 2 px.
    """
    if legend is None or not legend.get_visible():
        return []

    fig = ax.figure
    try:
        renderer = fig.canvas.get_renderer()
    except Exception:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    if img_h is None or img_w is None:
        canvas_w, canvas_h = fig.canvas.get_width_height()
        img_w = img_w or canvas_w
        img_h = img_h or canvas_h

    out = []

    # 1. Legend title
    title_artist = legend.get_title()
    if title_artist and title_artist.get_text() and title_artist.get_text().strip():
        try:
            tb = title_artist.get_window_extent(renderer)
            if tb.width > 0 and tb.height > 0:
                x0, y0, x1, y1 = float(tb.x0), float(img_h - tb.y1), float(tb.x1), float(img_h - tb.y0)
                out.append({
                    "class_name": "legend_title",
                    "bbox": BoundingBox(tb.x0, tb.y0, tb.x1, tb.y1),
                    "obb": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                    "polygon": [x0, y0, x1, y0, x1, y1, x0, y1],
                    "text": title_artist.get_text().strip(),
                    "role": "legend_title",
                })
        except Exception:
            pass

    # 2. Legend labels and markers
    texts = legend.get_texts()
    handles = getattr(legend, 'legend_handles', getattr(legend, 'legendHandles', []))

    for txt, handle in zip(texts, handles):
        # Label text
        try:
            text_str = txt.get_text().strip() if txt.get_text() else ""
            if text_str:
                tb = txt.get_window_extent(renderer)
                if tb.width > 0 and tb.height > 0:
                    x0, y0, x1, y1 = float(tb.x0), float(img_h - tb.y1), float(tb.x1), float(img_h - tb.y0)
                    out.append({
                        "class_name": "legend_label",
                        "bbox": BoundingBox(tb.x0, tb.y0, tb.x1, tb.y1),
                        "obb": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                        "polygon": [x0, y0, x1, y0, x1, y1, x0, y1],
                        "text": text_str,
                        "role": "legend_label",
                    })
        except Exception:
            pass

        # Handle marker
        try:
            hb = handle.get_window_extent(renderer)
            lw = getattr(handle, 'get_linewidth', lambda: 2.0)()
            if not isinstance(lw, (int, float)) or lw <= 0:
                lw = 2.0
            pad_y = max(lw / 2.0, 2.0)
            pad_x = 2.0 if hb.width < 2.0 else 0.0

            ymin = hb.y0 - pad_y if hb.height < 2.0 else hb.y0
            ymax = hb.y1 + pad_y if hb.height < 2.0 else hb.y1
            xmin = hb.x0 - pad_x if hb.width < 2.0 else hb.x0
            xmax = hb.x1 + pad_x if hb.width < 2.0 else hb.x1

            x0, y0, x1, y1 = float(xmin), float(img_h - ymax), float(xmax), float(img_h - ymin)
            out.append({
                "class_name": "legend_marker",
                "bbox": BoundingBox(xmin, ymin, xmax, ymax),
                "obb": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                "polygon": [x0, y0, x1, y0, x1, y1, x0, y1],
                "text": None,
                "role": "legend_marker",
            })
        except Exception:
            pass

    return out

def apply_pie_label_strategy(data, labels_text):
    """Implementa diversas estratégias de rotulagem para gráficos de pizza."""
    
    # Escolhe uma estratégia de rotulagem aleatória
    strategy = np.random.choice(
        ['default_leader', 'outside_pct_only', 'inside_pct_only', 'none'],
        p=[0.40, 0.30, 0.20, 0.10]
    )
    
    pie_params = {}

    if strategy == 'default_leader':
        # Estratégia 1: Rótulos de texto fora, porcentagens dentro (o seu original)
        pie_params['labels'] = labels_text
        pie_params['autopct'] = '%1.1f%%'
        pie_params['pctdistance'] = 0.7  # Porcentagem dentro
        pie_params['labeldistance'] = 1.1 # Rótulo de texto fora
    
    elif strategy == 'outside_pct_only':
        # Estratégia 2: Apenas porcentagens, fora da fatia. Sem rótulos de texto.
        pie_params['labels'] = None
        pie_params['autopct'] = '%1.1f%%'
        pie_params['pctdistance'] = 0.8 # Um pouco mais longe do centro
        pie_params['labeldistance'] = 1.15 # Posição da porcentagem
    
    elif strategy == 'inside_pct_only':
        # Estratégia 3: Apenas porcentagens, dentro da fatia. Sem rótulos de texto.
        pie_params['labels'] = None
        pie_params['autopct'] = '%1.1f%%'
        pie_params['pctdistance'] = 0.5 # Bem dentro
        pie_params['labeldistance'] = 1.1 # (Não usado)

    elif strategy == 'none':
        # Estratégia 4: Sem rótulos
        pie_params['labels'] = None
        pie_params['autopct'] = None
    
    return pie_params

def _safe_range(value, fallback):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return value[0], value[1]
    return fallback

def closure(values):
    arr = np.asarray(values, dtype=float)
    arr = np.clip(arr, 1e-12, None)
    total = float(np.sum(arr))
    if total <= 0:
        if arr.size == 0:
            return arr
        return np.full_like(arr, 1.0 / float(arr.size))
    return arr / total

def perturbation(x, y):
    return closure(np.asarray(x, dtype=float) * np.asarray(y, dtype=float))

def power_transform(a, x):
    return closure(np.power(np.asarray(x, dtype=float), a))

def aitchison_distance(x, y):
    x = closure(x)
    y = closure(y)
    x = np.clip(x, 1e-12, None)
    y = np.clip(y, 1e-12, None)
    gx = np.exp(np.mean(np.log(x)))
    gy = np.exp(np.mean(np.log(y)))
    return float(np.sqrt(np.sum((np.log(x / gx) - np.log(y / gy)) ** 2)))

def alr_transform(composition):
    comp = closure(composition)
    comp = np.clip(comp, 1e-12, None)
    if comp.size <= 1:
        return np.zeros(0, dtype=float)
    ref = comp[-1]
    return np.log(comp[:-1] / ref)

def inverse_alr_transform(alr_data):
    alr_data = np.asarray(alr_data, dtype=float)
    if alr_data.size == 0:
        return np.array([1.0], dtype=float)
    exp_data = np.exp(alr_data)
    reconstructed = np.concatenate([exp_data, [1.0]])
    return closure(reconstructed)

def apply_aln_noise(composition, variance_scale=0.05):
    comp = closure(composition)
    d_minus_1 = comp.size - 1
    if d_minus_1 <= 0:
        return comp
    alr = alr_transform(comp)
    cov = np.eye(d_minus_1) * float(variance_scale)
    noise = np.random.multivariate_normal(mean=np.zeros(d_minus_1), cov=cov)
    return inverse_alr_transform(alr + noise)

def inject_rounding_artifacts(composition, decimals=3):
    comp = closure(composition)
    rounded = np.round(comp, int(decimals))
    sum_val = float(np.sum(rounded))
    epsilon = 10.0 ** (-int(decimals))
    metrics = {
        "decimals": int(decimals),
        "sum": float(sum_val),
        "deviation": float(sum_val - 1.0),
        "is_under": bool(np.isclose(sum_val, 1.0 - epsilon)),
        "is_over": bool(np.isclose(sum_val, 1.0 + epsilon)),
        "is_perfect": bool(np.isclose(sum_val, 1.0))
    }
    return rounded, sum_val, metrics

def shannon_entropy(composition):
    comp = np.asarray(composition, dtype=float)
    comp = comp[comp > 0]
    if comp.size == 0:
        return 0.0
    return float(-np.sum(comp * np.log2(comp)))

def dynamic_other_aggregation(composition, labels, max_slices=5, entropy_loss_tolerance=0.5, other_label="Other"):
    comp = np.asarray(composition, dtype=float)
    labels = list(labels)
    meta = {
        "enabled": True,
        "original_count": int(comp.size),
        "final_count": int(comp.size),
        "other_fraction": 0.0,
        "entropy_original": shannon_entropy(comp),
        "entropy_aggregated": shannon_entropy(comp),
        "entropy_loss": 0.0,
        "max_slices": int(max_slices),
        "entropy_loss_tolerance": float(entropy_loss_tolerance)
    }

    if comp.size <= max_slices:
        return comp, labels, meta

    sort_idx = np.argsort(comp)[::-1]
    sorted_comp = comp[sort_idx]
    sorted_labels = [labels[i] for i in sort_idx]

    orig_entropy = shannon_entropy(comp)
    selected_comp = None
    selected_labels = None
    selected_entropy = None

    for k in range(max_slices - 1, 0, -1):
        head = sorted_comp[:k]
        tail_sum = float(np.sum(sorted_comp[k:]))
        agg_comp = np.append(head, tail_sum)
        agg_entropy = shannon_entropy(agg_comp)
        entropy_loss = orig_entropy - agg_entropy

        if entropy_loss > entropy_loss_tolerance and k < (max_slices - 1):
            k_prev = k + 1
            head_prev = sorted_comp[:k_prev]
            tail_sum_prev = float(np.sum(sorted_comp[k_prev:]))
            selected_comp = np.append(head_prev, tail_sum_prev)
            selected_labels = sorted_labels[:k_prev] + [other_label]
            selected_entropy = shannon_entropy(selected_comp)
            break

        if entropy_loss <= entropy_loss_tolerance:
            selected_comp = agg_comp
            selected_labels = sorted_labels[:k] + [other_label]
            selected_entropy = agg_entropy
            break

    if selected_comp is None:
        k = max_slices - 1
        head = sorted_comp[:k]
        tail_sum = float(np.sum(sorted_comp[k:]))
        selected_comp = np.append(head, tail_sum)
        selected_labels = sorted_labels[:k] + [other_label]
        selected_entropy = shannon_entropy(selected_comp)

    meta.update({
        "final_count": int(selected_comp.size),
        "other_fraction": float(selected_comp[-1]) if selected_comp.size > 0 else 0.0,
        "entropy_aggregated": float(selected_entropy),
        "entropy_loss": float(orig_entropy - selected_entropy)
    })

    return selected_comp, selected_labels, meta

def sort_clockwise_descending(values, labels, other_label="Other"):
    values = list(values)
    labels = list(labels)

    other_val = None
    if other_label in labels:
        idx = labels.index(other_label)
        other_val = values.pop(idx)
        labels.pop(idx)

    if values:
        sort_idx = list(np.argsort(values)[::-1])
        sorted_values = [values[i] for i in sort_idx]
        sorted_labels = [labels[i] for i in sort_idx]
    else:
        sorted_values = []
        sorted_labels = []

    if other_val is not None:
        sorted_values.append(other_val)
        sorted_labels.append(other_label)

    return np.asarray(sorted_values, dtype=float), sorted_labels

def compute_polar_coordinates(sorted_values):
    normalized = closure(sorted_values)
    angles = normalized * 2 * np.pi
    start_angles = np.cumsum(np.insert(angles, 0, 0))[:-1]
    end_angles = np.cumsum(angles)
    return np.column_stack((start_angles, end_angles))

def _build_pie_labels(n_components, prefix="Item"):
    return [f"{prefix} {i+1}" for i in range(int(n_components))]

def _generate_asymmetric_alpha(n_components, alpha_range, asymmetry_strength_range=None):
    a_min, a_max = _safe_range(alpha_range, (0.3, 8.0))
    base = np.random.uniform(float(a_min), float(a_max), int(n_components))
    if asymmetry_strength_range:
        s_min, s_max = _safe_range(asymmetry_strength_range, (0.0, 1.5))
        strength = random.uniform(float(s_min), float(s_max))
        trend = np.linspace(1.0 + strength, 1.0 - strength, int(n_components))
        base = base * trend
    return np.clip(base, 1e-3, None)

def _sample_dirichlet(n_components, cfg):
    alpha_range = cfg.get("dirichlet", {}).get("alpha_range", (0.3, 8.0))
    asym_range = cfg.get("dirichlet", {}).get("asymmetry_strength_range", (0.0, 1.5))
    alpha = _generate_asymmetric_alpha(n_components, alpha_range, asym_range)
    comp = np.random.dirichlet(alpha)
    return comp, {"alpha": alpha.tolist(), "n_components": int(n_components)}

def _sample_pareto_shares(n_components, cfg):
    pareto_cfg = cfg.get("pareto", {})
    alpha_min, alpha_max = _safe_range(pareto_cfg.get("alpha_shape_range"), (1.16, 3.5))
    alpha_shape = random.uniform(float(alpha_min), float(alpha_max))
    scale_min = float(pareto_cfg.get("scale_min", 1.0))
    raw = stats.pareto.rvs(b=alpha_shape, scale=scale_min, size=int(n_components))
    comp = closure(raw)
    return comp, {"alpha_shape": float(alpha_shape), "scale_min": float(scale_min), "n_components": int(n_components)}

def _sample_stick_breaking(cfg):
    stick_cfg = cfg.get("stick_breaking", {})
    n_min, n_max = _safe_range(stick_cfg.get("n_components_range"), (3, 10))
    n_components = random.randint(int(n_min), int(n_max))
    g_min, g_max = _safe_range(stick_cfg.get("gamma_dispersion_range"), (1.0, 8.0))
    gamma_disp = random.uniform(float(g_min), float(g_max))

    betas = stats.beta.rvs(1, gamma_disp, size=n_components)
    if n_components > 0:
        betas[-1] = 1.0
    weights = np.zeros_like(betas, dtype=float)
    if n_components > 0:
        weights[0] = betas[0]
        remaining = 1.0 - betas[0]
        for i in range(1, n_components):
            weights[i] = betas[i] * remaining
            remaining *= (1.0 - betas[i])

    comp = closure(weights)
    return comp, {"gamma_dispersion": float(gamma_disp), "n_components": int(n_components)}

def _sample_dominant_trace(cfg):
    dom_cfg = cfg.get("dominant_trace", {})
    d_min, d_max = _safe_range(dom_cfg.get("n_dominant_range"), (1, 2))
    t_min, t_max = _safe_range(dom_cfg.get("n_trace_range"), (8, 25))
    n_dominant = random.randint(int(d_min), int(d_max))
    n_trace = random.randint(int(t_min), int(t_max))

    a_min, a_max = _safe_range(dom_cfg.get("dominance_alpha_range"), (60.0, 120.0))
    b_min, b_max = _safe_range(dom_cfg.get("dominance_beta_range"), (5.0, 30.0))
    dominance_alpha = random.uniform(float(a_min), float(a_max))
    dominance_beta = random.uniform(float(b_min), float(b_max))
    phi = float(stats.beta.rvs(dominance_alpha, dominance_beta))

    alpha_dom = np.full(n_dominant, 10.0, dtype=float)
    alpha_trace = np.full(n_trace, 0.05, dtype=float)

    dom_comp = stats.dirichlet.rvs(alpha_dom, size=1).flatten()
    trace_comp = stats.dirichlet.rvs(alpha_trace, size=1).flatten()
    comp = np.hstack((dom_comp * phi, trace_comp * (1.0 - phi)))

    apply_poisson = bool(dom_cfg.get("apply_poisson", True))
    read_min, read_max = _safe_range(dom_cfg.get("poisson_read_depth_range"), (2000, 20000))
    read_depth = int(random.randint(int(read_min), int(read_max)))
    if apply_poisson:
        counts = stats.poisson.rvs(mu=comp * read_depth)
        comp = closure(counts)
    else:
        comp = closure(comp)

    meta = {
        "phi": float(phi),
        "n_dominant": int(n_dominant),
        "n_trace": int(n_trace),
        "apply_poisson": bool(apply_poisson),
        "poisson_read_depth": int(read_depth)
    }
    return comp, meta

def _kdga_normalizing_constant(alpha, lambda_param):
    if alpha.size < 2:
        return 1.0
    p = len(alpha) - 1
    alpha_p1 = float(alpha[-1])
    sum_alpha_p = float(np.sum(alpha[:-1]))
    gamma_prod = float(np.prod([gamma(a) for a in alpha[:-1]]) * gamma(alpha_p1))
    gamma_sum = float(gamma(sum_alpha_p + alpha_p1))
    hypergeo = float(hyp1f1(sum_alpha_p, sum_alpha_p + alpha_p1, -lambda_param))
    return (gamma_prod / gamma_sum) * hypergeo

def _sample_kdga(n_components, cfg):
    kdga_cfg = cfg.get("kdga", {})
    a_min, a_max = _safe_range(kdga_cfg.get("alpha_range"), (0.3, 5.0))
    l_min, l_max = _safe_range(kdga_cfg.get("lambda_range"), (-3.0, 3.0))
    alpha = np.random.uniform(float(a_min), float(a_max), int(n_components))
    lambda_param = random.uniform(float(l_min), float(l_max))
    raw = stats.gamma.rvs(a=alpha, scale=1.0)
    tilt = np.exp(np.clip(-lambda_param * raw, -50.0, 50.0))
    comp = closure(raw * tilt)
    try:
        c2 = float(_kdga_normalizing_constant(alpha, lambda_param))
    except Exception:
        c2 = None
    meta = {"alpha": alpha.tolist(), "lambda": float(lambda_param), "c2": c2, "n_components": int(n_components)}
    return comp, meta

def generate_pie_composition(pie_config=None, debug_mode=False):
    cfg = pie_config or {}
    dist_weights = cfg.get("distribution_weights", {})
    dist_names = list(dist_weights.keys())
    if not dist_names:
        dist_names = ["dirichlet"]
        dist_weights = {"dirichlet": 1}

    weights = [float(dist_weights.get(name, 0)) for name in dist_names]
    if sum(weights) <= 0:
        weights = [1.0] * len(dist_names)

    dist_name = random.choices(dist_names, weights=weights, k=1)[0]

    n_min, n_max = _safe_range(cfg.get("num_components_range"), (3, 12))
    n_min, n_max = int(n_min), int(n_max)
    if n_min > n_max:
        n_min, n_max = n_max, n_min
    n_components = random.randint(n_min, n_max)

    if dist_name == "pareto":
        comp_raw, dist_meta = _sample_pareto_shares(n_components, cfg)
    elif dist_name == "dominant_trace":
        comp_raw, dist_meta = _sample_dominant_trace(cfg)
        n_components = int(comp_raw.size)
    elif dist_name == "stick_breaking":
        comp_raw, dist_meta = _sample_stick_breaking(cfg)
        n_components = int(comp_raw.size)
    elif dist_name == "kdga":
        comp_raw, dist_meta = _sample_kdga(n_components, cfg)
    else:
        comp_raw, dist_meta = _sample_dirichlet(n_components, cfg)

    comp_raw = closure(comp_raw)
    labels_raw = _build_pie_labels(n_components, cfg.get("label_prefix", "Item"))

    entropy_raw = shannon_entropy(comp_raw)

    aln_cfg = cfg.get("aln_noise", {})
    aln_enabled = bool(aln_cfg.get("enabled", False))
    variance_scale = None
    comp_noisy = comp_raw
    if aln_enabled:
        v_min, v_max = _safe_range(aln_cfg.get("variance_scale_range"), (0.02, 0.08))
        variance_scale = random.uniform(float(v_min), float(v_max))
        comp_noisy = apply_aln_noise(comp_raw, variance_scale=variance_scale)

    distance_raw_noisy = aitchison_distance(comp_raw, comp_noisy) if aln_enabled else 0.0

    rounding_cfg = cfg.get("rounding", {})
    rounding_enabled = bool(rounding_cfg.get("enabled", False))
    rounded_values = None
    rounding_meta = {"enabled": False}
    comp_for_plot = comp_noisy
    if rounding_enabled:
        decimals = int(rounding_cfg.get("decimals", 3))
        rounded_values, rounded_sum, rounding_meta = inject_rounding_artifacts(comp_noisy, decimals=decimals)
        rounding_meta["enabled"] = True
        rounding_meta["rounded_sum"] = float(rounded_sum)
        comp_for_plot = closure(rounded_values)

    agg_cfg = cfg.get("aggregation", {})
    agg_enabled = bool(agg_cfg.get("enabled", True))
    other_label = cfg.get("sorting", {}).get("other_label", "Other")
    max_slices = int(cfg.get("max_slices", 5))
    entropy_tol = float(cfg.get("entropy_loss_tolerance", 0.5))

    if agg_enabled:
        comp_agg, labels_agg, agg_meta = dynamic_other_aggregation(
            comp_for_plot, labels_raw, max_slices=max_slices,
            entropy_loss_tolerance=entropy_tol, other_label=other_label
        )
    else:
        comp_agg, labels_agg = comp_for_plot, labels_raw
        agg_meta = {
            "enabled": False,
            "original_count": int(comp_for_plot.size),
            "final_count": int(comp_for_plot.size),
            "other_fraction": 0.0,
            "entropy_original": shannon_entropy(comp_for_plot),
            "entropy_aggregated": shannon_entropy(comp_for_plot),
            "entropy_loss": 0.0,
            "max_slices": int(max_slices),
            "entropy_loss_tolerance": float(entropy_tol)
        }

    sort_cfg = cfg.get("sorting", {})
    sort_enabled = bool(sort_cfg.get("enabled", True))
    clockwise = bool(sort_cfg.get("clockwise", True))

    if sort_enabled:
        comp_final, labels_final = sort_clockwise_descending(comp_agg, labels_agg, other_label=other_label)
    else:
        comp_final, labels_final = np.asarray(comp_agg, dtype=float), list(labels_agg)

    # Enforce minimum slice threshold of 3.5% so every slice is visually distinct with well-formed keypoints
    comp_final = np.maximum(0.035, comp_final)
    comp_final = comp_final / np.sum(comp_final)

    polar_coords = compute_polar_coordinates(comp_final)

    meta = {
        "distribution": {"name": dist_name, "params": dist_meta},
        "aln_noise": {
            "enabled": bool(aln_enabled),
            "variance_scale": float(variance_scale) if variance_scale is not None else None,
            "aitchison_distance": float(distance_raw_noisy)
        },
        "rounding": rounding_meta,
        "aggregation": agg_meta,
        "sorting": {"enabled": bool(sort_enabled), "clockwise": bool(clockwise), "other_label": str(other_label)},
        "entropy": {
            "raw": float(entropy_raw),
            "final": float(shannon_entropy(comp_final)),
            "loss": float(entropy_raw - shannon_entropy(comp_final))
        },
        "composition": {
            "labels": list(labels_final),
            "values": [float(v) for v in comp_final],
            "values_raw": [float(v) for v in comp_raw],
            "values_noisy": [float(v) for v in comp_noisy],
            "values_rounded": [float(v) for v in rounded_values] if rounded_values is not None else None,
            "labels_pre_aggregation": list(labels_raw)
        },
        "polar_coords": [[float(a), float(b)] for a, b in polar_coords]
    }

    if debug_mode:
        print(f"DEBUG [PIE-COMP] dist={dist_name}, slices={len(comp_final)}")

    return comp_final, labels_final, meta

def add_error_bars(ax, bar_info, orientation='vertical', measurement_type='biological', error_bar_prob=0.7):
    """Add realistic error bars with measurement-specific characteristics"""
    error_artists = []
    error_tops = [info['height'] for info in bar_info]  # Initialize with bar heights
    
    for i, info in enumerate(bar_info):
        if random.random() < error_bar_prob and info['height'] >= 0:  # chance for error bars
            center, value = info['center'], info['height']
            
            # Realistic error bar calculation based on measurement type
            if measurement_type == 'biological':
                # Biological replicates: SEM or SD
                error_type = np.random.choice(['sem', 'sd'], p=[0.7, 0.3])
                n_replicates = np.random.choice([3, 4, 5, 6, 8], p=[0.4, 0.3, 0.15, 0.10, 0.05])
                
                # CV based on assay type
                cv = np.random.uniform(0.10, 0.30)  # 10-30% CV for biological data
                sd = value * cv
                
                if error_type == 'sem':
                    error = sd / np.sqrt(n_replicates)
                else:
                    error = sd
                    
            elif measurement_type == 'analytical':
                # Analytical chemistry: typically smaller errors
                cv = np.random.uniform(0.02, 0.08)  # 2-8% CV
                error = value * cv
                
            elif measurement_type == 'survey':
                # Survey data: confidence intervals
                error = value * np.random.uniform(0.05, 0.15)  # 5-15% margin of error
                
            else:  # Default
                error = value * np.random.uniform(0.08, 0.20)
            
            # Create error bar
            if orientation == 'vertical':
                artist = ax.errorbar(center, value, yerr=error, fmt='none', 
                                   ecolor='black', capsize=4, elinewidth=1.2, 
                                   capthick=1.2, zorder=10)
                error_tops[i] = value + error
            else:
                artist = ax.errorbar(value, center, xerr=error, fmt='none', 
                                   ecolor='black', capsize=4, elinewidth=1.2, 
                                   capthick=1.2, zorder=10)
                error_tops[i] = value + error
                
            error_artists.append(artist)
    
    return error_artists, error_tops

def add_data_labels(ax, artists, orientation='vertical', chart_type='bar', 
                   error_tops=None, bar_info_list=None):
    """
    Add data labels to chart elements with correct positioning for stacked bars.
    
    CRITICAL FIX: Uses bar_info_list metadata to correctly position labels
    on stacked bar segments and match with error bars.
    """
    labels = []
    
    # Create lookup for bar info if available
    bar_info_centers = None
    if bar_info_list:
        bar_info_centers = [b['center'] for b in bar_info_list]
    
    for artist in artists:
        if isinstance(artist, patches.Rectangle) and chart_type in ['bar', 'histogram']:
            matched_idx = None
            matched_bar_info = None
            
            # Determine artist's center in data coordinates
            if orientation == 'vertical':
                artist_center = artist.get_x() + artist.get_width() / 2.0
                artist_height = artist.get_height()
                artist_bottom = artist.get_y()
            else:  # horizontal
                artist_center = artist.get_y() + artist.get_height() / 2.0
                artist_height = artist.get_width()
                artist_bottom = artist.get_x()
            
            # Match to bar_info using center AND bottom position (critical for stacked bars)
            if bar_info_list:
                min_distance = float('inf')
                for idx, b_info in enumerate(bar_info_list):
                    # Check center match
                    center_diff = abs(b_info['center'] - artist_center)
                    
                    # Check bottom position match (critical for stacked bars)
                    bottom_diff = abs(b_info.get('bottom', 0) - artist_bottom)
                    
                    # Combined distance metric
                    distance = center_diff + bottom_diff
                    
                    if distance < min_distance:
                        min_distance = distance
                        matched_idx = idx
                        matched_bar_info = b_info
            
            # Use segment height as label value (NOT cumulative for stacked)
            if matched_bar_info:
                label_value = matched_bar_info['height']
            else:
                if orientation == 'vertical':
                    label_value = artist_height if artist.get_y() >= 0 else -artist_height
                else:  # horizontal
                    label_value = artist_height if artist.get_x() >= 0 else -artist_height
            
            if abs(label_value) < 0.01: 
                continue
            
            label_text = f'{label_value:.1f}'
            
            # Position label at TOP of segment
            if orientation == 'vertical':
                x_pos = artist_center
                offset = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                
                if label_value >= 0:
                    # Use error bar top as anchor if available
                    anchor_y = error_tops[matched_idx] if matched_idx is not None and error_tops else (matched_bar_info['height'] if matched_bar_info else artist.get_y() + artist.get_height())
                    y_pos = anchor_y + offset
                    va = 'bottom'
                else:  # Negative bars
                    anchor_y = artist.get_y()
                    y_pos = anchor_y - offset
                    va = 'top'
                
                txt = ax.text(x_pos, y_pos, label_text, ha='center', va=va, 
                             fontsize=7, zorder=12)
                labels.append(txt)
            
            else:  # horizontal
                y_pos = artist_center
                offset = 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
                
                if label_value >= 0:
                    anchor_x = error_tops[matched_idx] if matched_idx is not None and error_tops else (matched_bar_info['height'] if matched_bar_info else artist.get_x() + artist.get_width())
                    x_pos = anchor_x + offset
                    ha = 'left'
                else:  # Negative bars
                    anchor_x = artist.get_x()
                    x_pos = anchor_x - offset
                    ha = 'right'
                
                txt = ax.text(x_pos, y_pos, label_text, ha=ha, va='center',
                             fontsize=7, zorder=12)
                labels.append(txt)
        
        elif isinstance(artist, patches.Wedge) and chart_type == 'pie':  # For pie charts
            ang = (artist.theta2 - artist.theta1)/2. + artist.theta1
            y = np.sin(np.deg2rad(ang)); x = np.cos(np.deg2rad(ang))
            value = (artist.theta2 - artist.theta1) / 360
            
            if value > 0.05:
                txt = ax.text(0.7 * x, 0.7 * y, f'{value:.0%}', ha='center', va='center', 
                             fontsize=8, color='white', zorder=12,
                             bbox=dict(boxstyle="round,pad=0.2", fc='black', ec="none", alpha=0.4))
                labels.append(txt)
    
    return labels

def add_treatment_key_xaxis(ax, bar_info_list, domain_category=None):
    """Add treatment key annotations below X-axis"""
    annotation_artists = []
    centers = [info['center'] for info in bar_info_list]
    
    if len(centers) != 4: 
        return annotation_artists  # Return empty list
    
    ax.set_xlabel(''); ax.set_xticklabels([]); ax.tick_params(axis='x', length=0)
    
    from synth.semantics.sampler import sample_comparative_pair
    treatment1, treatment2 = sample_comparative_pair(domain_category=domain_category)
    y_pos1, y_pos2 = -0.15, -0.25
    
    text1 = ax.text(-0.1, y_pos1, f"{treatment1}", transform=ax.transAxes, 
                   ha='right', va='center', fontsize=10)
    text2 = ax.text(-0.1, y_pos2, f"{treatment2}", transform=ax.transAxes, 
                   ha='right', va='center', fontsize=10)
    
    annotation_artists.extend([text1, text2])
    
    symbols = [('-', '-'), ('+', '-'), ('-', '+'), ('+', '+')]
    blended_transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    
    for i, center in enumerate(centers):
        ax.text(center, y_pos1, symbols[i][0], transform=blended_transform, 
               ha='center', va='center', fontsize=12)
        ax.text(center, y_pos2, symbols[i][1], transform=blended_transform, 
               ha='center', va='center', fontsize=12)
    
    return annotation_artists

# ===================================================================================
# == CHART-SPECIFIC GENERATOR FUNCTIONS (COMPLETE WITH BUG FIXES) ==
# ===================================================================================

def _generate_bar_chart(ax, theme_name, theme_config, style_config, debug_mode=False):
    """
    CRITICAL FIXES:
    - Complete stacked bar annotation metadata
    - Proper dual-axis handling
    - Consistent return structure for all code paths
    - Error bar matching for stacked segments
    """
    
    # Initialize return values at start
    data_artists = []
    other_artists = []
    bar_info_list = []
    orientation = style_config.get('orientation', 'vertical')
    error_tops = []
    axis_related_artists = []
    scale_axis_info = {'primary_scale_axis': 'y' if orientation == 'vertical' else 'x'}
    
    if debug_mode:
        print(f"DEBUG: _generate_bar_chart - Theme: {theme_name}, Style: {style_config}")
        print(f"DEBUG: _generate_bar_chart - Orientation: {orientation}")
    
    use_synthetic = style_config.get('use_synthetic_data_engine', False) or (isinstance(theme_config, dict) and theme_config.get('use_synthetic_data_engine', False))
    syn_domain = style_config.get('synthetic_domain', None) or (theme_config.get('synthetic_domain', None) if isinstance(theme_config, dict) else None)
    syn_table = None

    # --- DUAL Y-AXIS LOGIC ---
    dual_axis_prob = style_config.get('dual_axis_probability', DUAL_AXIS_PROBABILITY)
    force_dual = style_config.get('force_dual_axis') if 'force_dual_axis' in style_config else (random.random() < dual_axis_prob)
    if style_config.get('orientation', 'vertical') == 'vertical' and force_dual:
        print(" - Generating dual Y-axis bar chart with scientific styles")
        is_scientific = True
        if isinstance(theme_config, dict):
            if theme_config.get('semantic_domain') == "business" or not theme_config.get('semantic_domain'):
                sub_weights = theme_config.get('scientific_subdomain_weights', {'biomedical': 0.70, 'engineering': 0.30})
                subdomains = list(sub_weights.keys())
                weights = list(sub_weights.values())
                domain_cat = random.choices(subdomains, weights=weights, k=1)[0]
                theme_config['semantic_domain'] = domain_cat
            theme_config['effective_is_scientific'] = True
        ax2 = ax.twinx()
        
        num_bars_1 = random.randint(2, 5)
        num_bars_2 = random.randint(2, 5)
        max_scale_1 = random.choice([50, 100, 200])
        max_scale_2 = random.choice([500, 1000, 2000])
        
        if use_synthetic:
            from synth.tabular import sample_multivariate_table
            table_domain = syn_domain or (theme_config.get('semantic_domain') if isinstance(theme_config, dict) else None) or ('biomedical' if is_scientific else 'business')
            syn_table = sample_multivariate_table(num_rows=max(num_bars_1, num_bars_2), num_series=2, domain=table_domain)
            if isinstance(theme_config, dict) and syn_table is not None:
                theme_config['semantic_domain'] = syn_table.get('domain')
            data_1 = syn_table['data'][:num_bars_1, 0]
            data_2 = syn_table['data'][:num_bars_2, 1]
        else:
            data_1 = generate_realistic_data(num_bars_1, max_scale_1, domain='scientific')
            data_2 = generate_realistic_data(num_bars_2, max_scale_2, domain='scientific')
        
        if debug_mode:
            print(f"DEBUG: Dual-axis chart - Data sets: {len(data_1)} and {len(data_2)} values")
            print(f"DEBUG: Dual-axis chart - Max scales: {max_scale_1} and {max_scale_2}")
        
        bar_width = 0.8
        positions_1 = np.arange(num_bars_1)
        gap = 2
        positions_2 = np.arange(num_bars_2) + num_bars_1 + gap
        all_positions = np.concatenate([positions_1, positions_2])
        
        labels_1 = [f'Cond {i+1}' for i in range(num_bars_1)]
        labels_2 = [f'Treat {i+1}' for i in range(num_bars_2)]
        all_labels = labels_1 + labels_2
        
        bar_styles = [{'facecolor': 'white', 'edgecolor': 'black', 'hatch': h} 
                     for h in ['', '////', '....', 'xxxx']]
        random.shuffle(bar_styles)
        style_1, style_2 = random.sample(bar_styles, 2)
        
        rects1 = ax.bar(positions_1, data_1, width=bar_width, zorder=2, 
                       label='Group 1', **style_1)
        rects2 = ax2.bar(positions_2, data_2, width=bar_width, zorder=2, 
                        label='Group 2', **style_2)
        
        data_artists = list(rects1) + list(rects2)
        
        #  Store complete metadata for BOTH axis groups
        bar_info_list_1, bar_info_list_2 = [], []
        
        for i, r in enumerate(rects1):
            bar_info_list_1.append({
                'center': r.get_x() + r.get_width()/2, 
                'height': r.get_height(), 
                'width': r.get_width(),
                'bottom': r.get_y(),
                'top': r.get_y() + r.get_height(),
                'series_idx': 0,
                'bar_idx': i,
                'axis': 'primary'
            })
        
        for i, r in enumerate(rects2):
            bar_info_list_2.append({
                'center': r.get_x() + r.get_width()/2, 
                'height': r.get_height(), 
                'width': r.get_width(),
                'bottom': r.get_y(),
                'top': r.get_y() + r.get_height(),
                'series_idx': 1,
                'bar_idx': i + len(rects1),
                'axis': 'secondary'
            })
        
        err_prob = style_config.get('error_bar_probability', 0.70) if isinstance(style_config, dict) else 0.70
        error_artists_1, error_tops_1 = add_error_bars(ax, bar_info_list_1, orientation='vertical', error_bar_prob=err_prob)
        error_artists_2, error_tops_2 = add_error_bars(ax2, bar_info_list_2, orientation='vertical', error_bar_prob=err_prob)
        
        other_artists.extend(error_artists_1)
        other_artists.extend(error_artists_2)
        
        combined_error_tops = error_tops_1 + error_tops_2
        
        if use_synthetic and syn_table is not None:
            ax.set_xlabel(syn_table['x_label'])
            ax.set_ylabel(syn_table['series_names'][0])
            ax2.set_ylabel(syn_table['series_names'][1])
            ax.set_title(syn_table['title'], fontsize=14, pad=15)
        else:
            from synth.semantics.sampler import sample_axis_pair, sample_secondary_y
            sem_dom = theme_config.get('semantic_domain') if isinstance(theme_config, dict) else None
            x_label, y1_label, domain_cat = sample_axis_pair("bar", "vertical", is_scientific=True, semantic_domain=sem_dom)
            y2_label = sample_secondary_y(y1_label, is_scientific=True, domain_category=domain_cat)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y1_label)
            ax2.set_ylabel(y2_label)
            if isinstance(theme_config, dict):
                theme_config['semantic_domain'] = domain_cat
        
        #  Atomic position and label setting
        ax.set_xticks(all_positions)
        ax.set_xticklabels(all_labels, rotation=45, ha='right')
        
        ax.set_ylim(0, max_scale_1 * 1.2)
        ax2.set_ylim(0, max_scale_2 * 1.2)
        
        ax.grid(False)
        ax2.grid(False)
        
        drop_spine_prob = style_config.get('drop_bottom_spine_probability', 0.05) if isinstance(style_config, dict) else 0.05
        if random.random() < drop_spine_prob:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', length=0)
        
        scale_axis_info = {'primary_scale_axis': 'y', 'secondary_scale_axis': 'y2'}
        
        if isinstance(theme_config, dict):
            theme_config['series_count'] = 2
            theme_config['series_names'] = list(syn_table['series_names'][:2]) if (use_synthetic and syn_table is not None) else ['Group 1', 'Group 2']

        if debug_mode:
            print(f"DEBUG: Dual-axis chart completed - {len(data_artists)} data artists, {len(other_artists)} other artists")
        
        return data_artists, other_artists, bar_info_list_1 + bar_info_list_2, \
               orientation, combined_error_tops, axis_related_artists, scale_axis_info, None
    
    # --- STANDARD BAR CHART LOGIC ---
    is_scientific = style_config.get('is_scientific', False)
    style = style_config['style']
    pattern = style_config['pattern']
    
    HATCHES = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    
    has_treatment_axis = False
    
    num_bars = random.randint(3, 8)
    max_scale = random.choice([50, 100, 200, 500, 750, 1000, 2000])
    allow_neg_prob = style_config.get('allow_negative_probability', 0.15) if isinstance(style_config, dict) else 0.15
    allow_negative = random.random() < allow_neg_prob and not is_scientific
    data_pattern_type = 'diverging' if allow_negative else None
    
    horiz_prob = style_config.get('horizontal_probability', 0.40) if isinstance(style_config, dict) else 0.40
    if isinstance(style_config, dict) and style_config.get('orientation') in ('horizontal', 'vertical'):
        orientation = style_config['orientation']
    else:
        orientation = 'horizontal' if num_bars > 6 and random.random() < horiz_prob else 'vertical'
        if isinstance(style_config, dict):
            style_config['orientation'] = orientation

    from synth.semantics.sampler import sample_axis_pair
    x_label, y_label, domain_cat = sample_axis_pair(
        "bar", orientation, is_scientific, semantic_domain=theme_config.get('semantic_domain') if isinstance(theme_config, dict) else None
    )
    if isinstance(theme_config, dict):
        theme_config['semantic_domain'] = domain_cat
    
    if debug_mode:
        print(f"DEBUG: Standard bar chart - Style: {style}, Pattern: {pattern}, Scientific: {is_scientific}")
        print(f"DEBUG: Standard bar chart - Num bars: {num_bars}, Max scale: {max_scale}, Orientation: {orientation}")
    
    ticks_setter = ax.set_xticks if orientation == 'vertical' else ax.set_yticks
    
    if is_scientific:
        num_groups, bars_per_group = random.randint(2, 6), random.randint(1, 4)
        if use_synthetic:
            from synth.tabular import sample_multivariate_table
            table_domain = syn_domain or theme_config.get('semantic_domain') or 'biomedical'
            syn_table = sample_multivariate_table(num_rows=num_groups, num_series=bars_per_group, domain=table_domain)
            if isinstance(theme_config, dict) and syn_table is not None:
                theme_config['semantic_domain'] = syn_table.get('domain')
        bar_styles = [{'facecolor': 'white', 'edgecolor': 'black', 'hatch': h} 
                     for h in ['', '////', '....', 'xxxx']]
        random.shuffle(bar_styles)
        
        bar_width = 0.8 / bars_per_group; group_width = bar_width * bars_per_group
        
        for i in range(num_groups):
            group_center = i * (group_width + 0.4)
            if use_synthetic and syn_table is not None:
                data = syn_table['data'][i, :]
            else:
                data = generate_realistic_data(bars_per_group, max_scale, 
                                             allow_negative=False, domain='scientific')
            
            for j in range(bars_per_group):
                pos = group_center - (group_width / 2) + (j + 0.5) * bar_width
                value = data[j]
                bar_style = bar_styles[j % len(bar_styles)]
                
                if orientation == 'vertical':
                    bar_container = ax.bar(pos, value, width=bar_width, **bar_style, zorder=2)
                    bar_info_list.append({
                        'center': pos, 
                        'height': value, 
                        'width': bar_width,
                        'bottom': 0,
                        'top': value
                    })
                else:
                    bar_container = ax.barh(pos, value, height=bar_width, **bar_style, zorder=2)
                    bar_info_list.append({
                        'center': pos, 
                        'height': value, 
                        'width': bar_width,
                        'bottom': 0,
                        'top': value
                    })
                
                data_artists.extend(bar_container.patches)
        
        ticks_positions = [i * (group_width + 0.4) for i in range(num_groups)]
        if use_synthetic and syn_table is not None:
            tick_labels = syn_table['categories'][:num_groups]
        else:
            tick_labels = [f'Group {g+1}' for g in range(num_groups)]
        
        ticks_setter(ticks_positions, tick_labels)
        
        tk_prob = style_config.get('treatment_key_probability', TREATMENT_KEY_PROBABILITY)
        force_tk = style_config.get('force_treatment_key') if 'force_treatment_key' in style_config else (random.random() < tk_prob)
        if len(bar_info_list) == 4 and force_tk and orientation == 'vertical':
            axis_related_artists.extend(add_treatment_key_xaxis(ax, bar_info_list, domain_category=domain_cat))
            has_treatment_axis = True
        
        if isinstance(theme_config, dict):
            theme_config['series_count'] = bars_per_group
            theme_config['series_names'] = list(syn_table['series_names'][:bars_per_group]) if (use_synthetic and syn_table is not None) else [f'Series {j+1}' for j in range(bars_per_group)]
    
    else:  # Standard Styles
        palette_name = theme_config.get('palette', 'viridis')
        
        if isinstance(palette_name, list): 
            colors = [palette_name[i % len(palette_name)] for i in range(num_bars * 2)]
        else: 
            cmap = colormaps.get(palette_name); 
            colors = [cmap(i / (num_bars * 1.5)) for i in range(num_bars * 2)]
        
        if use_synthetic:
            from synth.tabular import sample_multivariate_table
            table_domain = syn_domain or theme_config.get('semantic_domain') or 'business'
            syn_table = sample_multivariate_table(num_rows=num_bars, num_series=2 if style in ('side_by_side', 'stacked') else 1, domain=table_domain)
            if isinstance(theme_config, dict) and syn_table is not None:
                theme_config['semantic_domain'] = syn_table.get('domain')
            categories = syn_table['categories'][:num_bars]
        else:
            categories = [f'Category {i+1}' for i in range(num_bars)]
            trunc_prob = style_config.get('truncate_category_probability', 0.2) if isinstance(style_config, dict) else 0.2
            if random.random() < trunc_prob: 
                categories = [c[:random.randint(5,8)] + '...' if len(c) > 8 else c for c in categories]
        
        if style == 'side_by_side':
            if use_synthetic and syn_table is not None:
                y_values1 = syn_table['data'][:, 0]
                y_values2 = syn_table['data'][:, 1]
                series_label1 = syn_table['series_names'][0]
                series_label2 = syn_table['series_names'][1]
            else:
                y_values1 = generate_realistic_data(num_bars, max_scale, allow_negative, data_pattern_type)
                y_values2 = generate_realistic_data(num_bars, max_scale, allow_negative, data_pattern_type)
                series_label1 = 'Series 1'
                series_label2 = 'Series 2'
            
            bar_width = 0.35; indices = np.arange(num_bars)
            
            if orientation == 'vertical':
                rects1 = ax.bar(indices - bar_width/2, y_values1, width=bar_width, 
                               label=series_label1, color=colors[0], zorder=3)
                rects2 = ax.bar(indices + bar_width/2, y_values2, width=bar_width, 
                               label=series_label2, color=colors[1], zorder=3)
                
                #  Store metadata for BOTH series
                for i in range(num_bars):
                    bar_info_list.append({
                        'center': indices[i] - bar_width/2, 
                        'height': y_values1[i], 
                        'width': bar_width,
                        'bottom': 0,
                        'top': y_values1[i],
                        'series_idx': 0
                    })
                    bar_info_list.append({
                        'center': indices[i] + bar_width/2, 
                        'height': y_values2[i], 
                        'width': bar_width,
                        'bottom': 0,
                        'top': y_values2[i],
                        'series_idx': 1
                    })
            else:
                pos1 = indices - bar_width/2
                pos2 = indices + bar_width/2
                rects1 = ax.barh(pos1, y_values1, height=bar_width, 
                                label='Series 1', color=colors[0], zorder=3)
                rects2 = ax.barh(pos2, y_values2, height=bar_width, 
                                label='Series 2', color=colors[1], zorder=3)
                
                for i in range(num_bars):
                    bar_info_list.append({
                        'center': pos1[i], 
                        'height': y_values1[i], 
                        'width': bar_width,
                        'bottom': 0,
                        'top': y_values1[i],
                        'series_idx': 0
                    })
                    bar_info_list.append({
                        'center': pos2[i], 
                        'height': y_values2[i], 
                        'width': bar_width,
                        'bottom': 0,
                        'top': y_values2[i],
                        'series_idx': 1
                    })
            
            ticks_setter(indices, categories)
            tk_prob = style_config.get('treatment_key_probability', TREATMENT_KEY_PROBABILITY)
            force_tk = style_config.get('force_treatment_key') if 'force_treatment_key' in style_config else (random.random() < tk_prob)
            if len(bar_info_list) == 4 and orientation == 'vertical' and force_tk:
                axis_related_artists.extend(add_treatment_key_xaxis(ax, bar_info_list, domain_category=domain_cat))
                has_treatment_axis = True
            if isinstance(theme_config, dict):
                theme_config['series_count'] = 2
                theme_config['series_names'] = [series_label1, series_label2]
            data_artists.extend(list(rects1) + list(rects2))
        
        elif style == 'stacked':
            if use_synthetic and syn_table is not None:
                y_values1 = syn_table['data'][:, 0]
                y_values2 = syn_table['data'][:, 1]
                portion_label1 = syn_table['series_names'][0]
                portion_label2 = syn_table['series_names'][1]
            else:
                y_values1 = generate_realistic_data(num_bars, max_scale/2, allow_negative=False)
                y_values2 = generate_realistic_data(num_bars, max_scale/2, allow_negative=False)
                portion_label1 = 'Portion 1'
                portion_label2 = 'Portion 2'
            
            bar_width = 0.7; indices = np.arange(num_bars)
            
            if orientation == 'vertical':
                rects1 = ax.bar(indices, y_values1, width=bar_width, 
                               label=portion_label1, color=colors[0], zorder=3)
                rects2 = ax.bar(indices, y_values2, width=bar_width, bottom=y_values1,
                               label=portion_label2, color=colors[1], zorder=3)
            else:
                rects1 = ax.barh(indices, y_values1, height=bar_width, 
                                label=portion_label1, color=colors[0], zorder=3)
                rects2 = ax.barh(indices, y_values2, height=bar_width, left=y_values1,
                                label=portion_label2, color=colors[1], zorder=3)
            
            ticks_setter(indices, categories)
            if isinstance(theme_config, dict):
                theme_config['series_count'] = 2
                theme_config['series_names'] = [portion_label1, portion_label2]
            data_artists.extend(list(rects1) + list(rects2))
            
            #  Store metadata for EACH stacked segment
            for i in range(num_bars):
                center_pos = indices[i]
                # Bottom segment
                bar_info_list.append({
                    'center': center_pos, 
                    'height': y_values1[i], 
                    'width': bar_width,
                    'bottom': 0,
                    'top': y_values1[i],
                    'series_idx': 0,
                    'bar_idx': i
                })
                # Top segment
                bar_info_list.append({
                    'center': center_pos, 
                    'height': y_values2[i], 
                    'width': bar_width,
                    'bottom': y_values1[i],  #  Bottom of top segment
                    'top': y_values1[i] + y_values2[i],  #  Cumulative top
                    'series_idx': 1,
                    'bar_idx': i
                })
        
        else:  # default, touching, 3d_effect
            if use_synthetic and syn_table is not None:
                y_values = syn_table['data'][:, 0]
            else:
                y_values = generate_realistic_data(num_bars, max_scale, allow_negative, data_pattern_type)
            
            bar_width = 0.95 if style == 'touching' else 0.8
            gap = 0.05 if style == 'touching' else 0.2
            indices = np.arange(num_bars) * (bar_width + gap)
            
            if orientation == 'vertical':
                rects = ax.bar(indices, y_values, width=bar_width, color=colors, zorder=3)
                for i in range(num_bars):
                    bar_info_list.append({
                        'center': indices[i], 
                        'height': y_values[i], 
                        'width': bar_width,
                        'bottom': 0,
                        'top': y_values[i]
                    })
            else:
                rects = ax.barh(indices, y_values, height=bar_width, color=colors, zorder=3)
                for i in range(num_bars):
                    bar_info_list.append({
                        'center': indices[i], 
                        'height': y_values[i], 
                        'width': bar_width,
                        'bottom': 0,
                        'top': y_values[i]
                    })
            
            ticks_setter(indices, categories)
            if isinstance(theme_config, dict):
                theme_config['series_count'] = 1
                theme_config['series_names'] = [syn_table['series_names'][0]] if (use_synthetic and syn_table is not None) else ['Series 1']
            data_artists.extend(list(rects))
    
    # Apply pattern styles
    if pattern != 'none':
        for bar in data_artists:
            fc = bar.get_facecolor()
            if pattern == 'hollow': 
                ec = bar.get_edgecolor()
                is_white_fc = hasattr(fc, '__iter__') and len(fc) >= 3 and all(c >= 0.9 for c in fc[:3])
                if is_scientific or is_white_fc:
                    edge_color = 'black'
                elif hasattr(ec, '__iter__') and len(ec) == 4 and ec[3] > 0 and not all(c >= 0.9 for c in ec[:3]):
                    edge_color = ec
                else:
                    edge_color = fc
                bar.set_facecolor('none')
                bar.set_edgecolor(edge_color)
                bar.set_hatch(None)
                bar.set_linewidth(1.5)
            elif pattern == 'dotted': 
                bar.set_hatch('..')
            elif pattern == 'striped': 
                bar.set_hatch('//')
            elif pattern == 'hatch': 
                bar.set_hatch(random.choice(HATCHES))
    
    if style == '3d_effect': 
        add_bar_shadows(ax, data_artists, ax.figure)
    
    jitter_prob = style_config.get('jitter_overlay_probability', 0.3) if isinstance(style_config, dict) else 0.3
    if not is_scientific and random.random() < jitter_prob and style != 'stacked':
        add_jitter_overlay(ax, bar_info_list, orientation)
    
    # --- Coordinated logic for error bars and significance markers ---
    error_bar_artists, error_tops = [], []
    
    err_prob = style_config.get('error_bar_probability', 0.70 if is_scientific else 0.30) if isinstance(style_config, dict) else (0.70 if is_scientific else 0.30)
    if (is_scientific or (not is_scientific and random.random() < err_prob)) and style != 'stacked':
        error_bar_artists, error_tops = add_error_bars(ax, bar_info_list, orientation, error_bar_prob=err_prob)
        other_artists.extend(error_bar_artists)
    else:
        error_tops = [b['height'] for b in bar_info_list]
    
    data_label_prob = style_config.get('data_label_probability', 0.1) if isinstance(style_config, dict) else 0.1
    if random.random() < data_label_prob and style != 'stacked':
        data_label_artists = add_data_labels(ax, data_artists, orientation, 'bar', 
                                            error_tops=error_tops, bar_info_list=bar_info_list)
        other_artists.extend(data_label_artists)
    
    if is_scientific:
        sig_prob = style_config.get('significance_probability', 0.40) if isinstance(style_config, dict) else 0.40
        y_max_limit = ax.get_ylim()[1] if orientation == 'vertical' else ax.get_xlim()[1]
        other_artists.extend(add_significance_markers(ax, bar_info_list, y_max_limit, 
                                                     orientation, error_tops=error_tops,
                                                     significance_prob=sig_prob))
    
    if allow_negative:
        if orientation == 'vertical': 
            ax.axhline(0, color='black', linewidth=0.8, zorder=1)
        else: 
            ax.axvline(0, color='black', linewidth=0.8, zorder=1)
    else:
        if orientation == 'vertical': 
            ax.set_ylim(bottom=0)
        else: 
            ax.set_xlim(left=0)
    
    if use_synthetic and syn_table is not None:
        ax.set_ylabel(syn_table['y_label'])
        if not has_treatment_axis:
            ax.set_xlabel(syn_table['x_label'])
        ax.set_title(syn_table['title'], fontsize=14, pad=15)
    else:
        ax.set_ylabel(y_label)
        if not has_treatment_axis:
            ax.set_xlabel(x_label)
    
    horiz_label_prob = style_config.get('horizontal_label_probability', 0.3) if isinstance(style_config, dict) else 0.3
    if random.random() < horiz_label_prob and orientation == 'vertical': 
        ax.tick_params(axis='x', labelrotation=0)
    
    drop_spine_prob = style_config.get('drop_bottom_spine_probability', 0.05) if isinstance(style_config, dict) else 0.05
    if is_scientific and orientation == 'vertical' and random.random() < drop_spine_prob:
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x', length=0)
    
    # Apply theme and typography variation
    theme = apply_chart_theme(ax, theme_name, orientation)
    apply_typography_variation(ax, domain='scientific' if is_scientific else 'business', theme_config=theme_config)

    if style in ('side_by_side', 'stacked'):
        active_profile = theme_config.get('profile') if isinstance(theme_config, dict) else None
        if active_profile is None and hasattr(ax, 'figure') and hasattr(ax.figure, '_generator_config'):
            active_profile = ax.figure._generator_config.get('profile')
        if active_profile and active_profile != 'legacy':
            seed = ax.figure._generator_config.get('seed') if hasattr(ax, 'figure') and hasattr(ax.figure, '_generator_config') else 42
            image_idx = getattr(ax.figure, '_generator_image_idx', 0) if hasattr(ax, 'figure') else 0
            from feature_rng import feature_rng
            rng = feature_rng(seed if seed is not None else 42, image_idx, "legend_spawn")
            if rng.random() < 0.70:
                apply_legend_variation(ax, 2)
    
    #  Ensure all code paths return complete metadata
    if not bar_info_list:
        # Emergency fallback: extract from data_artists
        for artist in data_artists:
            if isinstance(artist, patches.Rectangle):
                if orientation == 'vertical':
                    bar_info_list.append({
                        'center': artist.get_x() + artist.get_width() / 2,
                        'height': artist.get_height(),
                        'width': artist.get_width(),
                        'bottom': artist.get_y(),
                        'top': artist.get_y() + artist.get_height()
                    })
                else:
                    bar_info_list.append({
                        'center': artist.get_y() + artist.get_height() / 2,
                        'height': artist.get_width(),
                        'width': artist.get_height(),
                        'bottom': artist.get_x(),
                        'top': artist.get_x() + artist.get_width()
                    })
    
    if debug_mode:
        print(f"DEBUG: _generate_bar_chart returning - data_artists: {len(data_artists)}, other_artists: {len(other_artists)}, bar_info_list: {len(bar_info_list)}")
        print(f"DEBUG: Scale axis info: {scale_axis_info}")
    
    ax._bar_info_list = bar_info_list
    return data_artists, other_artists, bar_info_list, orientation, error_tops, \
           axis_related_artists, scale_axis_info, None

def _generate_line_chart(ax, theme_name, theme_config, is_scientific, debug_mode=False):
    """
    Generate a line chart with realistic trend data and keypoint detection.
    """
    theme = apply_chart_theme(ax, theme_name)
    line_cfg = theme_config.get('line_chart_config', {}) if isinstance(theme_config, dict) else {}
    num_series = random.randint(1, 4)
    num_points = random.randint(8, 25)
    max_scale = random.choice([50, 100, 500, 1000])
    
    data_artists = []
    other_artists = []
    keypoint_info = []
    
    x = np.arange(num_points)
    
    palette = theme.get('palette', 'viridis')
    
    if isinstance(palette, list):
        colors = [palette[i % len(palette)] for i in range(num_series)]
    else:
        try:
            cmap = colormaps.get(palette)
            colors = [cmap(i / max(1, num_series - 1)) for i in range(num_series)]
        except (ValueError, KeyError):
            cmap = colormaps.get('viridis')
            colors = [cmap(i / max(1, num_series - 1)) for i in range(num_series)]
    
    if debug_mode:
        print(f"DEBUG LINE: Generated {len(colors)} colors for {num_series} series")
        print(f"DEBUG LINE: Palette: {palette}, Type: {type(palette)}")
    
    if len(colors) < num_series:
        colors = colors * ((num_series // len(colors)) + 1)
    
    colors = colors[:num_series]
    
    use_synthetic = isinstance(theme_config, dict) and theme_config.get('use_synthetic_data_engine', False)
    syn_domain = theme_config.get('synthetic_domain', None) if isinstance(theme_config, dict) else None
    syn_table = None
    if use_synthetic:
        from synth.tabular import sample_multivariate_table
        table_domain = syn_domain or (theme_config.get('semantic_domain') if isinstance(theme_config, dict) else None) or ('biomedical' if is_scientific else 'business')
        syn_table = sample_multivariate_table(num_rows=num_points, num_series=num_series, domain=table_domain)
        if isinstance(theme_config, dict) and syn_table is not None:
            theme_config['semantic_domain'] = syn_table.get('domain')

    for series_idx in range(num_series):
        if use_synthetic and syn_table is not None:
            y_data = syn_table['data'][:, series_idx]
            series_label = syn_table['series_names'][series_idx]
        else:
            y_data = generate_realistic_data(num_points, max_scale, allow_negative=is_scientific,
                                            domain='scientific' if is_scientific else 'business')
            series_label = f'Series {series_idx+1}'
        
        # Determine marker visibility and style
        marker_prob = line_cfg.get('marker_probability', 0.6)
        has_markers = random.random() < marker_prob
        marker = random.choice(['o', 's', '^', 'v', 'D', 'p', '*']) if has_markers else None
        markersize = random.uniform(4.0, 7.0) if has_markers else 0.0
        
        # Determine line styling
        linestyle_prob = line_cfg.get('varied_linestyle_probability', 0.3)
        linestyle = random.choice(['-', '--', '-.', ':']) if random.random() < linestyle_prob else '-'
        linewidth = random.uniform(1.5, 3.0)
        
        line, = ax.plot(x, y_data, marker=marker, markersize=markersize, linewidth=linewidth,
                       linestyle=linestyle, color=colors[series_idx], label=series_label, zorder=3)
        data_artists.append(line)
        
        plotted = [(float(x[i]), float(y_data[i]), int(i)) for i in range(len(y_data))]
        
        inflection_pts = detect_inflection_points(x, y_data, threshold=0.1)
        prominence_factor = 0.08 if is_scientific else 0.05
        peaks, valleys = detect_extrema(x, y_data, prominence_factor=prominence_factor)
        
        if debug_mode:
            print(f"DEBUG [LINE] Series {series_idx}: Raw data points: {len(y_data)}")
            print(f"DEBUG [LINE] Series {series_idx}: Found {len(inflection_pts)} inflections, {len(peaks)} peaks, {len(valleys)} valleys")
            if peaks:
                print(f"DEBUG [LINE] Series {series_idx}: First peak - x: {peaks[0][0]:.2f}, y: {peaks[0][1]:.2f}")
            if valleys:
                print(f"DEBUG [LINE] Series {series_idx}: First valley - x: {valleys[0][0]:.2f}, y: {valleys[0][1]:.2f}")
            print(f"DEBUG [LINE] Series {series_idx}: Captured {len(plotted)} plotted points for pose construction")
        
        keypoint_info.append({
            'series_idx': series_idx,
            'start': (float(x[0]), float(y_data[0]), 0),
            'end': (float(x[-1]), float(y_data[-1]), len(x)-1),
            'inflections': [(float(x_val), float(y_val), int(idx)) for x_val, y_val, idx in inflection_pts],
            'peaks': [(float(x_val), float(y_val), int(idx)) for x_val, y_val, idx in peaks],
            'valleys': [(float(x_val), float(y_val), int(idx)) for x_val, y_val, idx in valleys],
            'boundary_points': [(float(x[i]), float(y_data[i]), int(i)) for i in range(len(x))],
            'all_points': [(float(x[i]), float(y_data[i]), int(i)) for i in range(len(x))],
            'plotted_points': plotted,
            'linewidth': linewidth,
            'linestyle': linestyle,
            'marker': marker,
            'markersize': markersize
        })
        
        if debug_mode:
            print(f"DEBUG [LINE] Series {series_idx}: Keypoint info stored - start={keypoint_info[-1]['start']}, end={keypoint_info[-1]['end']}")
            if keypoint_info[-1]['all_points']:
                print(f"DEBUG [LINE] Series {series_idx}: First point: ({keypoint_info[-1]['all_points'][0][0]:.2f}, {keypoint_info[-1]['all_points'][0][1]:.2f}), Last point: ({keypoint_info[-1]['all_points'][-1][0]:.2f}, {keypoint_info[-1]['all_points'][-1][1]:.2f})")
    
    if use_synthetic and syn_table is not None:
        ax.set_xlabel(syn_table['x_label'])
        ax.set_ylabel(syn_table['y_label'])
        ax.set_title(syn_table['title'], fontsize=14, pad=15)
    else:
        from synth.semantics.sampler import sample_axis_pair
        x_label, y_label, domain_cat = sample_axis_pair(
            "line", "vertical", is_scientific, semantic_domain=theme_config.get('semantic_domain') if isinstance(theme_config, dict) else None
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if isinstance(theme_config, dict):
            theme_config['semantic_domain'] = domain_cat
    
    legend_prob = line_cfg.get('legend_probability', 0.7)
    if num_series > 1 and random.random() < legend_prob:
        legend = apply_legend_variation(ax, num_series)
        other_artists.append(legend)
    
    apply_typography_variation(ax, domain='scientific' if is_scientific else 'business', theme_config=theme_config)
    
    # Collect minimum y value across all series for axis scaling
    all_y_vals = [pt[1] for kpi in keypoint_info for pt in kpi['plotted_points']]
    data_min = np.min(all_y_vals) if all_y_vals else 0
    apply_axis_scaling(ax, data_min=data_min, orientation='vertical')
    
    return data_artists, other_artists, [], 'vertical', None, [], {'primary_scale_axis': 'y'}, keypoint_info


def _generate_scatter_chart(ax, theme_name, theme_config, is_scientific, debug_mode=False):
    """Enhanced scatter with realistic correlation structures and sample sizes"""
    theme = apply_chart_theme(ax, theme_name)
    scatter_cfg = theme_config.get('scatter_chart_config', {}) if isinstance(theme_config, dict) else {}
    palette = theme.get('palette', 'viridis')
    
    # Realistic sample sizes based on publication analysis
    if is_scientific:
        num_points = np.random.choice([15, 20, 25, 30, 50, 75, 100], 
                                     p=[0.15, 0.25, 0.20, 0.15, 0.15, 0.05, 0.05])
    else:
        num_points = np.random.choice([50, 100, 200, 500], 
                                     p=[0.20, 0.40, 0.30, 0.10])
    
    max_scale = random.choice([50, 100, 200, 500, 1000])
    
    # Realistic correlation patterns with realistic R² values
    relationship = random.choices([
        'strong_positive',    # R² = 0.64-0.81
        'moderate_positive',  # R² = 0.36-0.64  
        'weak_positive',      # R² = 0.09-0.36
        'no_correlation',     # R² = 0-0.09
        'strong_negative',    # R² = 0.64-0.81
        'moderate_negative',  # R² = 0.36-0.64
        'nonlinear',         # Quadratic, exponential relationships
        'clustered'          # Multiple distinct groups
    ], weights=[0.20, 0.25, 0.20, 0.15, 0.05, 0.05, 0.05, 0.05])[0]
    
    data_artists = []
    other_artists = []
    
    use_synthetic = isinstance(theme_config, dict) and theme_config.get('use_synthetic_data_engine', False)
    syn_domain = theme_config.get('synthetic_domain', None) if isinstance(theme_config, dict) else None
    syn_table = None

    if use_synthetic:
        from synth.tabular import sample_multivariate_table
        table_domain = syn_domain or (theme_config.get('semantic_domain') if isinstance(theme_config, dict) else None) or ('biomedical' if is_scientific else 'business')
        syn_table = sample_multivariate_table(num_rows=num_points, num_series=2, domain=table_domain)
        if isinstance(theme_config, dict) and syn_table is not None:
            theme_config['semantic_domain'] = syn_table.get('domain')
        x_data = syn_table['data'][:, 0]
        y_data = syn_table['data'][:, 1]
    else:
        # Generate X data with realistic distribution
        x_data = generate_realistic_data(
            num_points, max_scale, 
            allow_negative=False,
            pattern_type='linear' if relationship != 'clustered' else None,
            domain='scientific' if is_scientific else 'business'
        )
    
    if relationship == 'clustered':
        # Generate realistic cluster data
        num_clusters = np.random.choice([2, 3, 4], p=[0.5, 0.3, 0.2])
        x_data, y_data = [], []
        
        for cluster_idx in range(num_clusters):
            cluster_size = max(5, num_points // num_clusters + np.random.randint(-3, 4))
            
            # Well-separated cluster centers
            mean_x = random.uniform(0.15, 0.85) * max_scale
            mean_y = random.uniform(0.15, 0.85) * max_scale
            
            # Realistic covariance (elliptical clusters)
            cov_xx = random.uniform(0.015, 0.08) * max_scale**2
            cov_yy = random.uniform(0.015, 0.08) * max_scale**2
            cov_xy = random.uniform(-0.5, 0.5) * np.sqrt(cov_xx * cov_yy)
            
            cov = [[cov_xx, cov_xy], [cov_xy, cov_yy]]
            cluster_data = np.random.multivariate_normal([mean_x, mean_y], cov, cluster_size)
            
            x_data.extend(cluster_data[:, 0])
            y_data.extend(cluster_data[:, 1])
        
        x_data = np.array(x_data)
        y_data = np.array(y_data)
    else:
        # Sort X for clearer trend visualization
        x_data = np.sort(x_data)
        
        # Generate Y based on X with realistic correlations
        if relationship in ['strong_positive', 'strong_negative']:
            # R² ≈ 0.75-0.90
            target_r_squared = np.random.uniform(0.64, 0.81)
            slope = np.random.uniform(0.5, 1.5) * (1 if 'positive' in relationship else -1)
            
        elif relationship in ['moderate_positive', 'moderate_negative']:
            # R² ≈ 0.40-0.65
            target_r_squared = np.random.uniform(0.36, 0.64)
            slope = np.random.uniform(0.3, 0.8) * (1 if 'positive' in relationship else -1)
            
        elif relationship in ['weak_positive', 'weak_negative']:
            # R² ≈ 0.10-0.35
            target_r_squared = np.random.uniform(0.09, 0.36)
            slope = np.random.uniform(0.1, 0.5) * (1 if 'positive' in relationship else -1)
            
        elif relationship == 'no_correlation':
            # R² ≈ 0-0.09
            target_r_squared = np.random.uniform(0.0, 0.09)
            slope = np.random.uniform(-0.2, 0.2)
        
        elif relationship == 'nonlinear':
            # Nonlinear relationships
            nonlinear_type = np.random.choice(['quadratic', 'exponential', 'logarithmic'])
            
            if nonlinear_type == 'quadratic':
                a = np.random.uniform(-0.002, 0.002)
                b = np.random.uniform(-0.5, 0.5)
                c = np.random.uniform(0.2, 0.5) * max_scale
                y_data = a * x_data**2 + b * x_data + c
                
            elif nonlinear_type == 'exponential':
                a = np.random.uniform(0.05, 0.15)
                b = np.random.uniform(0.01, 0.03)
                c = np.random.uniform(0, max_scale * 0.2)
                y_data = a * np.exp(b * x_data / max_scale) + c
                y_data = np.clip(y_data, 0, max_scale)
                
            elif nonlinear_type == 'logarithmic':
                a = np.random.uniform(10, 30)
                b = np.random.uniform(0, max_scale * 0.3)
                y_data = a * np.log(x_data + 1) + b
            
            # Add noise
            noise_cv = np.random.uniform(0.08, 0.15)
            noise = np.random.normal(0, np.abs(y_data) * noise_cv, num_points)
            y_data += noise
        else:
            # Default linear
            target_r_squared = 0.5
            slope = 0.5
        
        if relationship != 'nonlinear':
            # Generate linear relationship with specific R²
            intercept = max_scale * np.random.uniform(0.1, 0.3)
            y_perfect = slope * (x_data - np.mean(x_data)) + intercept
            
            # Add noise to achieve target R²
            noise_variance = np.var(y_perfect) * ((1 - target_r_squared) / target_r_squared)
            noise = np.random.normal(0, np.sqrt(noise_variance), num_points)
            y_data = y_perfect + noise
    
        # Ensure realistic bounds
        y_data = np.clip(y_data, 0 if not is_scientific else -max_scale*0.2, max_scale * 1.2)
        x_data = np.clip(x_data, 0, max_scale * 1.2)
    
    # Realistic point styling based on sample size
    scatter_kwargs = {
        'alpha': np.random.uniform(0.6, 0.8),
        'zorder': 2,
        'marker': np.random.choice(['o', 's', '^', 'D', '+'])
    }
    
    # Size scaling with sample size (larger datasets = smaller points, with support for bubble variations)
    bubble_prob = scatter_cfg.get('bubble_probability', 0.25)
    is_bubble = (random.random() < bubble_prob)
    if is_bubble:
        base_s = np.random.randint(40, 90)
        scatter_kwargs['s'] = np.random.uniform(base_s * 0.4, base_s * 2.2, len(x_data))
    else:
        if num_points < 30:
            scatter_kwargs['s'] = np.random.randint(70, 150)
        elif num_points < 100:
            scatter_kwargs['s'] = np.random.randint(40, 90)
        elif num_points < 500:
            scatter_kwargs['s'] = np.random.randint(20, 50)
        else:
            scatter_kwargs['s'] = np.random.randint(12, 30)
    
    # Store point size for radius calculation
    point_size = scatter_kwargs['s']
    
    # Color strategy
    if isinstance(palette, list) and palette:
        scatter_kwargs['c'] = palette[0]
    else:
        scatter_kwargs['c'] = '#2E86AB'
    
    scatter = ax.scatter(x_data, y_data, **scatter_kwargs)
    data_artists.append(scatter)
    
    # Add trend line for correlated data
    trendline_prob = scatter_cfg.get('trendline_probability', 0.70)
    if relationship not in ['no_correlation', 'clustered'] and np.random.random() < trendline_prob:
        # Fit trend line
        coeffs = np.polyfit(x_data, y_data, 1)
        trend_line = np.poly1d(coeffs)
        x_trend = np.linspace(np.min(x_data), np.max(x_data), 100)
        
        line_color = palette[1] if isinstance(palette, list) and len(palette) > 1 else '#D62728'
        line, = ax.plot(x_trend, trend_line(x_trend), '--', 
                       color=line_color, alpha=0.8, linewidth=2, zorder=1)
        other_artists.append(line)
    
    # Set labels
    if use_synthetic and syn_table is not None:
        ax.set_xlabel(syn_table['series_names'][0])
        ax.set_ylabel(syn_table['series_names'][1])
        ax.set_title(syn_table['title'], fontsize=14, pad=15)
    else:
        from synth.semantics.sampler import sample_axis_pair
        x_label, y_label, domain_cat = sample_axis_pair(
            "scatter", "vertical", is_scientific, semantic_domain=theme_config.get('semantic_domain') if isinstance(theme_config, dict) else None
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if isinstance(theme_config, dict):
            theme_config['semantic_domain'] = domain_cat
    
    # Realistic axis limits with padding
    x_range = np.max(x_data) - np.min(x_data)
    y_range = np.max(y_data) - np.min(y_data)
    
    ax.set_xlim(np.min(x_data) - 0.05*x_range, np.max(x_data) + 0.05*x_range)
    ax.set_ylim(np.min(y_data) - 0.05*y_range, np.max(y_data) + 0.05*y_range)
    
    # **NEW: Build scatter point metadata with coordinates and radius**
    # Convert matplotlib scatter point size to radius in data coordinates
    # scatter 's' parameter is in points^2, radius calculation requires axis transformation
    fig = ax.figure
    dpi = fig.dpi if fig else 72.0
    
    # Calculate approximate radius in data coordinates
    # Point size 's' is area in points^2, so radius in points = sqrt(s/pi)
    radius_points = np.sqrt(point_size / np.pi)
    
    # Transform radius from display coordinates to data coordinates
    # Get approximate data-to-display scaling factor from axis limits
    x_display_range = ax.transData.transform([[ax.get_xlim()[1], 0]])[0][0] - \
                      ax.transData.transform([[ax.get_xlim()[0], 0]])[0][0]
    x_data_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    x_scale_factor = x_data_range / x_display_range if x_display_range != 0 else 1.0
    
    y_display_range = ax.transData.transform([[0, ax.get_ylim()[1]]])[0][1] - \
                      ax.transData.transform([[0, ax.get_ylim()[0]]])[0][1]
    y_data_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    y_scale_factor = y_data_range / y_display_range if y_display_range != 0 else 1.0
    
    # Average radius in data coordinates (approximate)
    radius_data = radius_points * (x_scale_factor + y_scale_factor) / 2.0
    
    # Build scatter metadata structure
    scatter_metadata = {
        'relationship': relationship,
        'num_points': num_points,
        'point_size': point_size,
        'radius_data': radius_data,  # Approximate radius in data coordinates
        'points': [
            {
                'x': float(x_data[i]),
                'y': float(y_data[i]),
                'index': i,
                'radius': radius_data  # Same radius for all points (uniform size)
            }
            for i in range(len(x_data))
        ]
    }
    
    apply_typography_variation(ax, domain='scientific' if is_scientific else 'business', theme_config=theme_config)

    active_profile = theme_config.get('profile') if isinstance(theme_config, dict) else None
    if active_profile is None and hasattr(ax, 'figure') and hasattr(ax.figure, '_generator_config'):
        active_profile = ax.figure._generator_config.get('profile')
    if active_profile and active_profile != 'legacy' and relationship == 'clustered':
        seed = ax.figure._generator_config.get('seed') if hasattr(ax, 'figure') and hasattr(ax.figure, '_generator_config') else 42
        image_idx = getattr(ax.figure, '_generator_image_idx', 0) if hasattr(ax, 'figure') else 0
        from feature_rng import feature_rng
        rng = feature_rng(seed if seed is not None else 42, image_idx, "legend_spawn")
        if rng.random() < 0.70:
            import matplotlib.patches as mpatches
            pal = palette if isinstance(palette, list) and palette else ['#2E86AB', '#D62728', '#2CA02C', '#FF7F0E']
            handles = [mpatches.Patch(color=pal[i % len(pal)], label=f'Cluster {i+1}') for i in range(num_clusters)]
            ax.legend(handles=handles, loc='upper right')

    data_min = np.min(y_data)
    apply_axis_scaling(ax, data_min=data_min, orientation='vertical')
    
    return data_artists, other_artists, [], 'vertical', [], [], \
           {'primary_scale_axis': 'x', 'secondary_scale_axis': 'y'}, scatter_metadata

def _generate_boxplot_chart(ax, theme_name, theme_config, is_scientific, 
                           box_width=0.6, outlier_style='circle', show_significance=True, debug_mode=False):
    """Generate boxplot with realistic data and complete annotation metadata"""
    
    # Apply theme
    theme = apply_chart_theme(ax, theme_name)
    if not theme:
        theme = {'palette': 'viridis'}
    
    box_cfg = theme_config.get('box_plot_config', {}) if isinstance(theme_config, dict) else {}
    # Determine orientation (configurable chance for horizontal boxplot with more boxes)
    horizontal_prob = box_cfg.get('horizontal_probability', 0.15)
    is_horizontal = random.random() < horizontal_prob
    
    # Generate realistic data - more groups for horizontal orientation
    if is_horizontal:
        num_groups = random.randint(6, 12)
    else:
        num_groups = random.randint(3, 8)
    
    max_scale = 100
    
    datas = [generate_realistic_data(num_points=random.randint(20, 50), max_scale=max_scale,
                                   allow_negative=is_scientific, pattern_type=None)
             for _ in range(num_groups)]
    
    # Handle empty or invalid data
    if not datas or any(len(d) == 0 for d in datas):
        return [], [], [], 'vertical', [], [], {}, None
    
    # Create boxplot with orientation parameter
    bp = ax.boxplot(datas, patch_artist=True, widths=box_width,
                   vert=not is_horizontal,
                   flierprops={'marker': {'circle': 'o', 'star': '*', 'diamond': 'D'}.get(outlier_style, 'o'),
                              'markersize': 5, 'alpha': 0.6})
    
    # Apply styling
    data_artists = _apply_box_styles(bp, theme, is_scientific)
    _apply_line_styles(bp)  # Median, whisker, cap styles
    
    # Jitter overlay adapted for orientation
    jitter_prob = box_cfg.get('jitter_points_probability', 0.20)
    if is_scientific and random.random() < jitter_prob:
        for i, d in enumerate(datas):
            if is_horizontal:
                y_coords = np.random.normal(i + 1, 0.04, size=len(d))
                ax.plot(d, y_coords, '.', color='black', alpha=0.3, zorder=10)
            else:
                x_coords = np.random.normal(i + 1, 0.04, size=len(d))
                ax.plot(x_coords, d, '.', color='black', alpha=0.3, zorder=10)
    
    # Set axis labels based on orientation
    from synth.semantics.sampler import sample_axis_pair
    orientation_str = 'horizontal' if is_horizontal else 'vertical'
    x_label, y_label, domain_cat = sample_axis_pair(
        "box", orientation_str, is_scientific, semantic_domain=theme_config.get('semantic_domain') if isinstance(theme_config, dict) else None
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if is_horizontal:
        ax.set_yticklabels([f'G{i+1}' for i in range(num_groups)])
    else:
        ax.set_xticklabels([f'G{i+1}' for i in range(num_groups)])
    if isinstance(theme_config, dict):
        theme_config['semantic_domain'] = domain_cat
    
    # Collect error bar artists (whiskers and caps)
    error_groups = []
    for g in range(num_groups):
        group_artists = [
            bp['whiskers'][2*g], bp['whiskers'][2*g + 1],
            bp['caps'][2*g], bp['caps'][2*g + 1]
        ]
        error_groups.append(group_artists)
    
    # Add significance markers with correct orientation
    bar_info_list = []
    sig_artists = []
    orientation_str = 'horizontal' if is_horizontal else 'vertical'
    
    sig_prob = box_cfg.get('significance_probability', 0.50)
    if show_significance and random.random() < sig_prob:
        max_extent = 0
        error_tops = []
        
        for g in range(num_groups):
            if is_horizontal:
                # For horizontal: whiskers extend along x-axis
                top = bp['whiskers'][2*g+1].get_xdata()[1]
                max_extent = max(max_extent, top + abs(top) * 0.1)
                error_tops.append(top)
                center = g + 1
                bar_info_list.append({'center': center, 'height': top, 'width': box_width})
            else:
                # For vertical: whiskers extend along y-axis
                top = bp['whiskers'][2*g+1].get_ydata()[1]
                max_extent = max(max_extent, top + abs(top) * 0.1)
                error_tops.append(top)
                center = g + 1
                bar_info_list.append({'center': center, 'height': top, 'width': box_width})
        
        sig_artists = add_significance_markers(ax, bar_info_list, max_extent, orientation_str, error_tops)
    
    # Collect other artists
    other_artists_list = []
    for group in error_groups:
        other_artists_list.extend(group)
    
    if bp['fliers']:
        other_artists_list.extend(bp['fliers'])
    
    if sig_artists:
        other_artists_list.extend(sig_artists)
    
    # Extract median line coordinates adapted for orientation
    median_metadata = []
    
    for group_idx, median_line in enumerate(bp['medians']):
        x_coords = median_line.get_xdata()
        y_coords = median_line.get_ydata()
        
        if len(x_coords) >= 2 and len(y_coords) >= 2:
            if is_horizontal:
                lower_left = {'x': float(x_coords[0]), 'y': float(y_coords[0])}
                upper_right = {'x': float(x_coords[-1]), 'y': float(y_coords[-1])}
                median_value = float(x_coords[0])
                center_y = float((y_coords[0] + y_coords[-1]) / 2.0)
                line_length = float(y_coords[-1] - y_coords[0])
                
                median_metadata.append({
                    'group_index': group_idx,
                    'group_label': f'G{group_idx+1}',
                    'median_value': median_value,
                    'lower_left': lower_left,
                    'upper_right': upper_right,
                    'center_y': center_y,
                    'line_length': line_length,
                    'orientation': 'horizontal'
                })
            else:
                lower_left = {'x': float(x_coords[0]), 'y': float(y_coords[0])}
                upper_right = {'x': float(x_coords[-1]), 'y': float(y_coords[-1])}
                median_value = float(y_coords[0])
                center_x = float((x_coords[0] + x_coords[-1]) / 2.0)
                line_length = float(x_coords[-1] - x_coords[0])
                
                median_metadata.append({
                    'group_index': group_idx,
                    'group_label': f'G{group_idx+1}',
                    'median_value': median_value,
                    'lower_left': lower_left,
                    'upper_right': upper_right,
                    'center_x': center_x,
                    'line_length': line_length,
                    'orientation': 'vertical'
                })
    
    # Build complete boxplot metadata with orientation info
    boxplot_metadata = {
        'num_groups': num_groups,
        'box_width': box_width,
        'medians': median_metadata,
        'orientation': orientation_str
    }
    
    if is_horizontal:
        scale_axis_info = {'primary_scale_axis': 'x', 'boxplot_raw': bp}
    else:
        scale_axis_info = {'primary_scale_axis': 'y', 'boxplot_raw': bp}
    
    apply_typography_variation(ax, domain='scientific' if is_scientific else 'business', theme_config=theme_config)

    active_profile = theme_config.get('profile') if isinstance(theme_config, dict) else None
    if active_profile is None and hasattr(ax, 'figure') and hasattr(ax.figure, '_generator_config'):
        active_profile = ax.figure._generator_config.get('profile')
    if active_profile and active_profile != 'legacy' and num_groups >= 2:
        seed = ax.figure._generator_config.get('seed') if hasattr(ax, 'figure') and hasattr(ax.figure, '_generator_config') else 42
        image_idx = getattr(ax.figure, '_generator_image_idx', 0) if hasattr(ax, 'figure') else 0
        from feature_rng import feature_rng
        rng = feature_rng(seed if seed is not None else 42, image_idx, "legend_spawn")
        if rng.random() < 0.70:
            import matplotlib.patches as mpatches
            pal = theme.get('palette', ['#4472C4']) if isinstance(theme, dict) else ['#4472C4']
            if not isinstance(pal, list) or not pal:
                pal = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']
            handles = [mpatches.Patch(color=pal[i % len(pal)], label=f'G{i+1}') for i in range(num_groups)]
            ax.legend(handles=handles, loc='upper right')
    
    return data_artists, other_artists_list, bar_info_list, orientation_str, [], [], \
           scale_axis_info, boxplot_metadata


def _apply_box_styles(bp, theme, is_scientific):
    """Apply styling to boxplot boxes based on theme and context"""
    data_artists = bp['boxes']
    num_groups = len(data_artists)
    
    if is_scientific and random.random() < 0.9:
        scientific_style = random.choice(['hollow', 'grayscale'])
        
        if scientific_style == 'hollow':
            for patch in data_artists:
                patch.set_facecolor('none')
                patch.set_edgecolor('black')
                patch.set_linewidth(1.2)
        
        elif scientific_style == 'grayscale':
            colors = ['#FFFFFF', '#DDDDDD', '#BBBBBB', '#999999']
            for i, patch in enumerate(data_artists):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_edgecolor('black')
                patch.set_linewidth(1.2)
    
    else:
        palette = theme.get('palette', 'viridis')
        colors = []
        
        if isinstance(palette, list):
            colors = [palette[i % len(palette)] for i in range(num_groups)]
        elif isinstance(palette, str):
            try:
                cmap = colormaps.get(palette)
                colors = [cmap(i / num_groups) for i in range(num_groups)]
            except ValueError:
                cmap = colormaps.get('viridis')
                colors = [cmap(i / num_groups) for i in range(num_groups)]
        
        for patch, color in zip(data_artists, colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
    
    return data_artists


def _apply_line_styles(bp):
    """Apply consistent styling to medians, whiskers, and caps"""
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)
    
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.2)
    
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.2)

def _generate_pie_chart(ax, theme_name, theme_config, is_scientific, pie_config=None, debug_mode=False):
    """Enhanced pie chart with geometric keypoint calculation."""
    theme = apply_chart_theme(ax, theme_name)
    
    for spine in ['left', 'bottom', 'top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    pie_values, labels_text, pie_metadata = generate_pie_composition(pie_config, debug_mode=debug_mode)
    num_slices = len(pie_values)
    
    pie_cfg = pie_config or {}
    explode = [0.0] * num_slices
    explode_prob = float(pie_cfg.get('explode_prob', 0.4))
    explode_amount = float(pie_cfg.get('explode_amount', 0.10))
    if num_slices > 0 and random.random() < explode_prob:
        explode[random.randint(0, num_slices-1)] = explode_amount
    
    palette = theme.get('palette', 'viridis')
    if isinstance(palette, list) and palette:
        colors = [palette[i % len(palette)] for i in range(num_slices)]
    else:
        cmap = colormaps.get(palette) if isinstance(palette, str) else colormaps.get('viridis')
        colors = [cmap(i / max(1, num_slices)) for i in range(num_slices)]
    
    # --- NOVA LÓGICA DE RÓTULOS ---
    # Chama a nova função ANTES de plotar
    pie_params = apply_pie_label_strategy(pie_values, labels_text)
    clockwise = bool(pie_cfg.get('sorting', {}).get('clockwise', True))
    counterclock = not clockwise
    
    # --- START OF FIX ---
    # Handle variable return values from ax.pie()
    
    autotexts = [] # Initialize autotexts as an empty list
    
    if pie_params.get('autopct') is not None:
        # autopct is provided, so expect 3 return values
        wedges, texts, autotexts = ax.pie(pie_values, explode=explode, colors=colors,
                           startangle=90, counterclock=counterclock,
                           wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
                           **pie_params)
    else:
        # autopct is None, so expect 2 return values
        wedges, texts = ax.pie(pie_values, explode=explode, colors=colors,
                     startangle=90, counterclock=counterclock,
                     wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
                     **pie_params)
    # --- END OF FIX ---
    ax.axis('equal')

    if debug_mode:
        print(f"DEBUG [PIE] Generated {len(wedges)} wedges with {num_slices} slices")
        print(f"DEBUG [PIE] Data values: {[f'{d:.4f}' for d in pie_values]}")

    pie_geometry = calculate_pie_geometry(wedges, ax, debug_mode)
    pie_metadata["explode"] = explode
    pie_metadata["counterclock"] = counterclock

    legend_prob = float(pie_cfg.get('legend_prob', 0.0))
    if labels_text and pie_params.get('labels') is not None and random.random() < legend_prob:
        inside_locs = ['upper right', 'upper left', 'lower left', 'lower right',
                       'center', 'center right', 'center left']
        outside_right = 'center left'
        if num_slices <= 4:
            loc = np.random.choice(inside_locs + [outside_right], p=[0.12] * 7 + [0.16])
        else:
            loc = np.random.choice(inside_locs + [outside_right], p=[0.05] * 7 + [0.65])
        frameon = np.random.choice([True, False], p=[0.4, 0.6])
        if loc == 'center left':
            ax.legend(wedges, labels_text, loc=loc, bbox_to_anchor=(1.04, 0.5), frameon=frameon)
        else:
            ax.legend(wedges, labels_text, loc=loc, frameon=frameon)

    connector_lines = []
    if pie_cfg.get('connector_lines', False):
        labeldistance = pie_params.get('labeldistance', 1.1)
        if labeldistance and labeldistance > 1.0 and texts:
            line_color = pie_cfg.get('connector_line_color', '#333333')
            line_width = float(pie_cfg.get('connector_line_width', 0.8))
            for wedge, label_text in zip(wedges, texts):
                if not label_text.get_text().strip():
                    continue
                theta = np.deg2rad((wedge.theta1 + wedge.theta2) / 2.0)
                sx = wedge.center[0] + wedge.r * np.cos(theta)
                sy = wedge.center[1] + wedge.r * np.sin(theta)
                tx, ty = label_text.get_position()
                line = ax.plot([sx, tx], [sy, ty], color=line_color, linewidth=line_width, zorder=2)[0]
                line.set_gid('pie_connector')
                connector_lines.append(line)

    if debug_mode and pie_geometry:
        print(f"DEBUG [PIE] Pie geometry calculated - center: {pie_geometry.get('center_point')}")
        print(f"DEBUG [PIE] Number of wedges in geometry: {len(pie_geometry.get('wedges', []))}")
    
    apply_typography_variation(ax, domain='scientific' if is_scientific else 'business', theme_config=theme_config)
    
    pie_payload = {"geometry": pie_geometry, "metadata": pie_metadata}

    # The return value correctly handles autotexts being an empty list
    return wedges, autotexts + texts + connector_lines, [], 'vertical', [], [], {'primary_scale_axis': 'none'}, pie_payload


def _ar1_noise(n, phi=0.6, sigma=1.0):
    """AR(1) noise: x[0] = 0, x[t] = phi * x[t-1] + eps[t]."""
    eps = np.random.normal(0.0, sigma, n)
    noise = np.zeros(n)
    if n > 1:
        noise[1:] = lfilter([1.0], [1.0, -phi], eps[1:])
    return noise


def _sample_gmm_1d(n_samples, mean_range, scale_range):
    n_components = random.randint(2, 4)
    means = np.random.uniform(mean_range[0], mean_range[1], n_components)
    scales = np.random.uniform(scale_range[0], scale_range[1], n_components)
    weights = np.random.dirichlet(np.ones(n_components))
    component_choices = np.random.choice(n_components, size=n_samples, p=weights)
    samples = np.empty(n_samples)
    for k in range(n_components):
        mask = component_choices == k
        count = int(np.sum(mask))
        if count > 0:
            samples[mask] = np.random.normal(loc=means[k], scale=scales[k], size=count)
    params = {
        "components": int(n_components),
        "means": means,
        "scales": scales,
        "weights": weights
    }
    return samples, params


def _sample_histogram_distribution(num_samples, is_scientific):
    if is_scientific:
        dist_names = [
            "normal", "gmm", "lognormal", "gamma", "weibull", "student_t",
            "truncnorm", "beta", "chi2", "pareto", "zig", "zip", "zinb"
        ]
        dist_weights = [0.18, 0.14, 0.10, 0.10, 0.08, 0.08, 0.08, 0.06, 0.05, 0.05, 0.04, 0.02, 0.02]
    else:
        dist_names = [
            "normal", "gmm", "lognormal", "pareto", "weibull", "gamma",
            "beta", "student_t", "truncnorm", "zip", "zinb", "zig", "chi2"
        ]
        dist_weights = [0.18, 0.12, 0.12, 0.10, 0.08, 0.08, 0.08, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01]

    dist_name = random.choices(dist_names, weights=dist_weights, k=1)[0]
    value_scale = random.choice([5, 10, 20, 50, 100])
    params = {"value_scale": float(value_scale)}

    if dist_name == "normal":
        loc = np.random.uniform(-0.2, 0.8) * value_scale
        scale = np.random.uniform(0.1, 0.35) * value_scale
        data = np.random.normal(loc=loc, scale=scale, size=num_samples)
        params.update({"loc": float(loc), "scale": float(scale)})
    elif dist_name == "gmm":
        mean_range = (-0.3 * value_scale, 1.0 * value_scale)
        scale_range = (0.08 * value_scale, 0.35 * value_scale)
        data, gmm_params = _sample_gmm_1d(num_samples, mean_range, scale_range)
        params.update(gmm_params)
    elif dist_name == "lognormal":
        mu = np.log(value_scale * np.random.uniform(0.2, 0.8))
        sigma = np.random.uniform(0.3, 0.9)
        data = np.random.lognormal(mean=mu, sigma=sigma, size=num_samples)
        params.update({"mu": float(mu), "sigma": float(sigma)})
    elif dist_name == "pareto":
        shape = np.random.uniform(1.1, 2.6)
        scale = value_scale * np.random.uniform(0.3, 1.0)
        data = stats.pareto.rvs(b=shape, scale=scale, size=num_samples)
        params.update({"shape": float(shape), "scale": float(scale)})
    elif dist_name == "weibull":
        shape_k = np.random.uniform(0.8, 2.5)
        scale_lambda = value_scale * np.random.uniform(0.4, 1.1)
        data = stats.weibull_min.rvs(c=shape_k, scale=scale_lambda, size=num_samples)
        params.update({"shape_k": float(shape_k), "scale_lambda": float(scale_lambda)})
    elif dist_name == "gamma":
        shape = np.random.uniform(1.2, 5.0)
        scale = (value_scale / shape) * np.random.uniform(0.6, 1.2)
        data = stats.gamma.rvs(a=shape, scale=scale, size=num_samples)
        params.update({"shape": float(shape), "scale": float(scale)})
    elif dist_name == "beta":
        a = np.random.uniform(0.6, 5.0)
        b = np.random.uniform(0.6, 5.0)
        data = stats.beta.rvs(a=a, b=b, size=num_samples) * value_scale
        params.update({"a": float(a), "b": float(b)})
    elif dist_name == "student_t":
        df = np.random.uniform(2.5, 10.0)
        scale = value_scale * np.random.uniform(0.12, 0.35)
        loc = np.random.uniform(-0.1, 0.4) * value_scale
        data = stats.t.rvs(df=df, loc=loc, scale=scale, size=num_samples)
        params.update({"df": float(df), "loc": float(loc), "scale": float(scale)})
    elif dist_name == "truncnorm":
        loc = np.random.uniform(0.0, 0.7) * value_scale
        scale = value_scale * np.random.uniform(0.15, 0.35)
        lower = -0.2 * value_scale
        upper = 1.2 * value_scale
        a, b = (lower - loc) / scale, (upper - loc) / scale
        data = stats.truncnorm.rvs(a=a, b=b, loc=loc, scale=scale, size=num_samples)
        params.update({"loc": float(loc), "scale": float(scale), "lower": float(lower), "upper": float(upper)})
    elif dist_name == "chi2":
        df = np.random.uniform(2.0, 10.0)
        scale = value_scale * np.random.uniform(0.2, 0.6)
        data = stats.chi2.rvs(df=df, size=num_samples) * scale
        params.update({"df": float(df), "scale": float(scale)})
    elif dist_name == "zip":
        pi = np.random.uniform(0.2, 0.6)
        lambda_rate = np.random.uniform(2.0, 20.0)
        structural_zeros = stats.bernoulli.rvs(p=pi, size=num_samples)
        poisson_counts = stats.poisson.rvs(mu=lambda_rate, size=num_samples)
        data = np.where(structural_zeros == 1, 0.0, poisson_counts.astype(float))
        params.update({"pi": float(pi), "lambda": float(lambda_rate)})
    elif dist_name == "zinb":
        pi = np.random.uniform(0.2, 0.6)
        n_param = np.random.uniform(1.5, 8.0)
        p_param = np.random.uniform(0.2, 0.8)
        structural_zeros = stats.bernoulli.rvs(p=pi, size=num_samples)
        nbinom_counts = stats.nbinom.rvs(n=n_param, p=p_param, size=num_samples)
        data = np.where(structural_zeros == 1, 0.0, nbinom_counts.astype(float))
        params.update({"pi": float(pi), "n": float(n_param), "p": float(p_param)})
    else:  # zig
        pi = np.random.uniform(0.2, 0.6)
        mu = np.random.uniform(0.2, 0.8) * value_scale
        sigma = value_scale * np.random.uniform(0.1, 0.35)
        structural_zeros = stats.bernoulli.rvs(p=pi, size=num_samples)
        gaussian_continuous = stats.norm.rvs(loc=mu, scale=sigma, size=num_samples)
        data = np.where(structural_zeros == 1, 0.0, gaussian_continuous)
        params.update({"pi": float(pi), "mu": float(mu), "sigma": float(sigma)})

    return data, {"name": dist_name, "params": params}


def _inject_missingness(data):
    mode = random.choice(["mcar", "mar", "mnar"])
    rate = np.random.uniform(0.03, 0.12)
    mask = np.zeros_like(data, dtype=bool)

    if mode == "mcar":
        mask = np.random.random(size=data.size) < rate
    elif mode == "mar":
        denom = np.ptp(data) if np.ptp(data) != 0 else 1.0
        covariate = (data - np.min(data)) / denom
        threshold = np.random.uniform(0.2, 0.6)
        mask = (covariate < threshold) & (np.random.random(size=data.size) < rate)
    else:  # mnar
        threshold = np.quantile(data, 0.7)
        mask = (data > threshold) & (np.random.random(size=data.size) < rate)

    data_missing = data.astype(float)
    data_missing[mask] = np.nan
    return data_missing, {
        "method": mode,
        "rate": float(rate),
        "missing_count": int(np.sum(mask))
    }


def _inject_outliers(data):
    outlier_info = []
    degraded = data.astype(float).copy()
    total_samples = degraded.size

    if total_samples == 0:
        return degraded, outlier_info

    if random.random() < 0.6:
        contamination_rate = np.random.uniform(0.008, 0.02)
        n_corruptions = int(total_samples * contamination_rate)
        if n_corruptions > 0:
            target_indices = np.random.choice(total_samples, size=n_corruptions, replace=False)
            magnitude_range = (5.0, 25.0)
            spike_scalars = np.random.uniform(low=magnitude_range[0], high=magnitude_range[1], size=n_corruptions)
            sign_inversions = np.random.choice([-1.0, 1.0], size=n_corruptions, p=[0.3, 0.7])
            degraded[target_indices] *= (spike_scalars * sign_inversions)
            outlier_info.append({
                "type": "sensor_spike",
                "rate": float(contamination_rate),
                "count": int(n_corruptions)
            })

    if random.random() < 0.5:
        error_rate = np.random.uniform(0.004, 0.015)
        n_errors = int(total_samples * error_rate)
        if n_errors > 0:
            target_indices = np.random.choice(total_samples, size=n_errors, replace=False)
            decimal_shift_magnitudes = np.array([0.01, 0.1, 10.0, 100.0, 1000.0])
            shift_probabilities = np.array([0.05, 0.25, 0.40, 0.25, 0.05])
            applied_shifts = np.random.choice(decimal_shift_magnitudes, size=n_errors, p=shift_probabilities)
            degraded[target_indices] *= applied_shifts
            outlier_info.append({
                "type": "fat_finger",
                "rate": float(error_rate),
                "count": int(n_errors)
            })

    return degraded, outlier_info


def _apply_heteroscedastic_noise(data):
    mode = random.choice(["proportional", "exponential", "cyclic"])
    noisy = data.astype(float).copy()
    base_scale = np.nanstd(noisy) if np.nanstd(noisy) > 0 else 1.0

    if mode == "proportional":
        gamma = np.random.uniform(0.05, 0.15)
        noise_floor = np.random.uniform(0.01, 0.05) * base_scale
        std_dev_array = (gamma * np.abs(noisy)) + noise_floor
        noise = np.random.normal(0.0, std_dev_array)
        noisy += noise
        params = {"gamma": float(gamma), "noise_floor": float(noise_floor)}
    elif mode == "exponential":
        alpha = np.random.uniform(0.02, 0.08)
        noise_floor = np.random.uniform(0.01, 0.04) * base_scale
        std_dev_array = noise_floor * np.exp(alpha * np.abs(noisy))
        max_allowable_sigma = np.nanmax(np.abs(noisy)) * 10.0 if np.nanmax(np.abs(noisy)) > 0 else base_scale
        std_dev_array = np.clip(std_dev_array, a_min=noise_floor, a_max=max_allowable_sigma)
        noise = np.random.normal(0.0, std_dev_array)
        noisy += noise
        params = {"alpha": float(alpha), "noise_floor": float(noise_floor)}
    else:
        amplitude = np.random.uniform(0.2, 0.8) * base_scale
        frequency = np.random.uniform(0.5, 2.0)
        domain_x = np.linspace(0, 2 * np.pi, noisy.size)
        std_dev_array = amplitude * np.abs(np.sin(frequency * domain_x)) + 0.01 * base_scale
        noise = np.random.normal(0.0, std_dev_array)
        noisy += noise
        params = {"amplitude": float(amplitude), "frequency": float(frequency)}

    return noisy, {"type": mode, "params": params}


def _choose_histogram_binning_strategy(data, dist_name):
    n = data.size
    if n < 200:
        candidates = ["sturges", "sqrt"]
    else:
        skew = stats.skew(data) if n > 2 else 0.0
        if not np.isfinite(skew):
            skew = 0.0
        if abs(skew) > 1.0:
            candidates = ["fd", "doane", "rice"]
        else:
            candidates = ["fd", "scott", "auto"]
    if dist_name in ["pareto", "lognormal", "gamma", "weibull"]:
        candidates.extend(["fd", "doane"])
    return random.choice(candidates)


def _compute_histogram_edges(data, strategy, log_binning):
    if log_binning:
        log_data = np.log10(data)
        log_edges = np.histogram_bin_edges(log_data, bins=strategy)
        edges = np.power(10.0, log_edges)
        width = float(np.median(np.diff(log_edges))) if len(log_edges) > 1 else None
        width_type = "log10"
    else:
        edges = np.histogram_bin_edges(data, bins=strategy)
        width = float(np.median(np.diff(edges))) if len(edges) > 1 else None
        width_type = "linear"
    return edges, width, width_type


def _generate_histogram(ax, theme_name, theme_config, is_scientific, debug_mode=False):
    """Generate histogram with realistic data distribution"""
    # Apply general theme settings (background, grid, etc.)
    theme = apply_chart_theme(ax, theme_name)
    hist_cfg = theme_config.get('histogram_config', {}) if isinstance(theme_config, dict) else {}

    num_samples = random.randint(300, 1200)
    data, dist_meta = _sample_histogram_distribution(num_samples, is_scientific)

    degradation_meta = {}
    noise_prob = hist_cfg.get('heteroscedastic_noise_probability', 0.35)
    if dist_meta["name"] not in ["zip", "zinb"] and random.random() < noise_prob:
        data, noise_meta = _apply_heteroscedastic_noise(data)
        degradation_meta["heteroscedastic_noise"] = noise_meta

    autocorr_prob = hist_cfg.get('autocorrelation_probability', 0.20)
    if dist_meta["name"] not in ["zip", "zinb"] and random.random() < autocorr_prob:
        phi = np.random.uniform(0.3, 0.8)
        sigma = np.nanstd(data) * np.random.uniform(0.05, 0.20)
        data = data + _ar1_noise(data.size, phi=phi, sigma=sigma)
        degradation_meta["autocorrelation"] = {"phi": float(phi), "sigma": float(sigma)}

    missing_prob = hist_cfg.get('missingness_probability', 0.25)
    if random.random() < missing_prob:
        data, missing_meta = _inject_missingness(data)
        degradation_meta["missingness"] = missing_meta

    outlier_prob = hist_cfg.get('outlier_probability', 0.25)
    if random.random() < outlier_prob:
        data, outlier_meta = _inject_outliers(data)
        if outlier_meta:
            degradation_meta["outliers"] = outlier_meta

    data = data[np.isfinite(data)]
    if data.size < 20:
        fallback_size = max(50, num_samples // 4)
        data = np.random.normal(loc=0.0, scale=1.0, size=fallback_size)
        dist_meta = {
            "name": "normal_fallback",
            "params": {"loc": 0.0, "scale": 1.0, "fallback": True}
        }
        degradation_meta["fallback"] = "insufficient_finite_samples"

    if dist_meta["name"] in ["zip", "zinb"]:
        data = np.clip(data, 0.0, None)
    
    # Prioritize using theme's color palette
    hist_color = None
    palette = theme.get('palette')
    
    if isinstance(palette, list) and palette:
        # If palette is a list of colors, choose one
        hist_color = random.choice(palette)
    elif isinstance(palette, str):
        # If palette is a colormap name, get a representative color from it
        try:
            # Get a color from the middle of the colormap
            hist_color = colormaps.get(palette)(0.4)
        except (ValueError, AttributeError):
            # If colormap name is invalid, hist_color remains None
            pass
    
    # Fall back to original hardcoded colors if theme palette is not usable
    if hist_color is None:
        hist_color = random.choice(['#4472C4', '#5B9BD5', '#66C2A5'])

    log_binning = False
    data_full = data
    if dist_meta["name"] in ["pareto", "lognormal", "gamma", "weibull"] and random.random() < 0.45:
        log_binning = True
    if log_binning:
        positive_data = data[data > 0]
        if positive_data.size < 20:
            log_binning = False
            data = data_full
        else:
            data = positive_data

    strategy = _choose_histogram_binning_strategy(data, dist_meta["name"])
    edges, width, width_type = _compute_histogram_edges(data, strategy, log_binning)

    n, bins, patches = ax.hist(data, bins=edges, color=hist_color, zorder=2)

    if log_binning:
        try:
            ax.set_xscale('log')
        except Exception:
            pass
    
    data_artists = patches
    
    bar_info_list = []
    for r in data_artists:
        bar_info_list.append({
            'center': r.get_x() + r.get_width()/2, 
            'height': r.get_height(), 
            'width': r.get_width(),
            'bottom': 0,
            'top': r.get_height(),
            'value': r.get_height(),
            'x_value': r.get_x() + r.get_width()/2
        })
    
    from synth.semantics.sampler import sample_axis_pair
    x_label, y_label, domain_cat = sample_axis_pair(
        "histogram", "vertical", is_scientific, semantic_domain=theme_config.get('semantic_domain') if isinstance(theme_config, dict) else None
    )
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    if isinstance(theme_config, dict):
        theme_config['semantic_domain'] = domain_cat
    
    # Histogram data labels typically show frequency/count values on top of bars
    data_label_artists = []
    
    # Add data labels with configurable probability (histograms don't always have labels)
    data_label_prob = hist_cfg.get('data_label_probability', 0.1)
    if random.random() < data_label_prob:
        # Select subset of bars to label (not all bars, typically higher frequency ones)
        # Sort bars by height and label top 30-50% of bars
        sorted_bars = sorted(zip(patches, n), key=lambda x: x[1], reverse=True)
        num_to_label = max(3, int(len(sorted_bars) * random.uniform(0.3, 0.5)))
        bars_to_label = sorted_bars[:num_to_label]
        
        for patch, height in bars_to_label:
            if height > 0:  # Only label non-empty bars
                x_pos = patch.get_x() + patch.get_width() / 2.0
                y_pos = height
                
                # Format label: show integer count
                label_text = f'{int(height)}'
                
                # Add text annotation
                text_artist = ax.text(
                    x_pos, y_pos,
                    label_text,
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    color='black',
                    zorder=3
                )
                data_label_artists.append(text_artist)
    
    apply_typography_variation(ax, domain='scientific' if is_scientific else 'business', theme_config=theme_config)

    active_profile = theme_config.get('profile') if isinstance(theme_config, dict) else None
    if active_profile is None and hasattr(ax, 'figure') and hasattr(ax.figure, '_generator_config'):
        active_profile = ax.figure._generator_config.get('profile')
    if active_profile and active_profile != 'legacy':
        seed = ax.figure._generator_config.get('seed') if hasattr(ax, 'figure') and hasattr(ax.figure, '_generator_config') else 42
        image_idx = getattr(ax.figure, '_generator_image_idx', 0) if hasattr(ax, 'figure') else 0
        from feature_rng import feature_rng
        rng = feature_rng(seed if seed is not None else 42, image_idx, "legend_spawn")
        if rng.random() < 0.70:
            import matplotlib.patches as mpatches
            handle = mpatches.Patch(color=hist_color if isinstance(hist_color, str) else '#4472C4', label='Distribution')
            ax.legend(handles=[handle], loc='upper right')

    skew_value = stats.skew(data) if data.size > 2 else 0.0
    if not np.isfinite(skew_value):
        skew_value = 0.0
    data_stats = {
        "count": int(data.size),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "skew": float(skew_value)
    }
    bin_meta = {
        "strategy": strategy,
        "bin_count": int(len(edges) - 1),
        "bin_width": width,
        "width_type": width_type,
        "edges": edges,
        "log_binning": log_binning,
        "log_base": 10 if log_binning else None
    }
    histogram_metadata = {
        "distribution": dist_meta,
        "degradation": degradation_meta,
        "data_stats": data_stats,
        "binning": bin_meta
    }

    scale_axis_info = {
        'primary_scale_axis': 'y',
        'x_scale': 'log' if log_binning else 'linear'
    }

    # Return 8 values: include data_label_artists in other_artists
    # Data labels are "other_artists" not "data_artists"
    return data_artists, data_label_artists, bar_info_list, 'vertical', [], [], \
           scale_axis_info, histogram_metadata

def _generate_area_chart(ax, theme_name, theme_config, is_scientific, debug_mode=False):
    """
    Generate an area chart with realistic continuous data, supporting stacked,
    overlapping, and percentage stacking modes with accurate boundary annotations.

    Segmentation semantics (area_seg) are stacking-mode dependent:

    - 'overlapping' / 'single': fill_bottom = 0, fill_top = data. Each series'
      polygon is amodal — it covers the full geometric area down to the zero
      baseline regardless of what other series draw on top of it. This is
      intentional even though layers are alpha-blended (alpha=0.55): the
      annotation is the complete underlying shape, not just the visible,
      mutually-exclusive surface a viewer would see.
    - 'stacked': fill_bottom = y_stack_previous (cumulative sum of underlying
      layers), fill_top = y_stack_previous + data. Each polygon is an isolated
      band sitting strictly on top of previous layers, so by construction
      there is no occluded region under it.

    Do not "simplify" the stacked branch to reuse the zero baseline from the
    overlapping branch, and do not clip the overlapping branch's polygon to
    only the visible surface — either change would silently invert the
    intended amodal/band distinction above.
    """
    theme = apply_chart_theme(ax, theme_name)
    area_cfg = theme_config.get('area_chart_config', {}) if isinstance(theme_config, dict) else {}
    num_series = random.randint(1, 4)
    num_points = random.randint(8, 25)
    max_scale = random.choice([50, 100, 500, 1000])
    
    if num_series == 1:
        stacking_mode = 'single'
    else:
        mode_weights = area_cfg.get('stacked_mode_weights', {'stacked': 65, 'overlapping': 35})
        modes = list(mode_weights.keys())
        weights = [mode_weights[m] for m in modes]
        stacking_mode = random.choices(modes, weights=weights, k=1)[0]

    if debug_mode:
        print(f"DEBUG [AREA] Stacking Mode: {stacking_mode}")

    data_artists = []
    other_artists = []
    keypoint_info = []
    x = np.arange(num_points)
    
    palette = theme.get('palette', 'viridis')
    
    if isinstance(palette, list):
        colors = [palette[i % len(palette)] for i in range(num_series)]
    else:
        try:
            cmap = colormaps.get(palette)
            colors = [cmap(i / max(1, num_series - 1)) for i in range(num_series)]
        except (ValueError, KeyError):
            cmap = colormaps.get('viridis')
            colors = [cmap(i / max(1, num_series - 1)) for i in range(num_series)]
    
    if debug_mode:
        print(f"DEBUG AREA: Generated {len(colors)} colors for {num_series} series")
    
    if len(colors) < num_series:
        colors = colors * ((num_series // len(colors)) + 1)
    
    colors = colors[:num_series]
    
    # 1. Generate series data
    all_series_data = []
    
    if stacking_mode == 'stacked':
        series_max = max_scale / max(1, num_series)
    else:  # 'overlapping' or 'single'
        series_max = max_scale

    if debug_mode:
        print(f"DEBUG AREA: Stacking mode={stacking_mode}, Total max_scale={max_scale}")
        print(f"DEBUG AREA: Num series={num_series}, Per-series max={series_max:.2f}")

    use_synthetic = isinstance(theme_config, dict) and theme_config.get('use_synthetic_data_engine', False)
    syn_domain = theme_config.get('synthetic_domain', None) if isinstance(theme_config, dict) else None
    syn_table = None

    if use_synthetic:
        from synth.tabular import sample_multivariate_table
        table_domain = syn_domain or (theme_config.get('semantic_domain') if isinstance(theme_config, dict) else None) or ('biomedical' if is_scientific else 'business')
        syn_table = sample_multivariate_table(num_rows=num_points, num_series=num_series, domain=table_domain)
        if isinstance(theme_config, dict) and syn_table is not None:
            theme_config['semantic_domain'] = syn_table.get('domain')
        for series_idx in range(num_series):
            data = syn_table['data'][:, series_idx]
            all_series_data.append(data.copy())
    else:
        for series_idx in range(num_series):
            data_raw = generate_realistic_data(num_points, series_max, 
                                              allow_negative=False,
                                              domain='scientific' if is_scientific else 'business')
            # Ensure smooth continuous variation without flat ceiling clipping
            data = np.maximum(0.02 * series_max, data_raw)
            all_series_data.append(data.copy())
            
            if debug_mode:
                print(f"DEBUG AREA: Series {series_idx} data range: {np.min(data):.2f} to {np.max(data):.2f}")

    # 2. Plotting and annotation
    y_stack = np.zeros(num_points)

    for series_idx, data in enumerate(all_series_data):
        color = colors[series_idx]
        
        if stacking_mode in ['overlapping', 'single']:
            # Amodal: full geometric area down to the zero baseline, regardless
            # of alpha-blended occlusion from other series drawn on top.
            y_stack_previous = np.zeros(num_points) 
            boundary_y = data 
            alpha = 0.55 if stacking_mode == 'overlapping' else 0.75
        else:
            # Mutually-exclusive band: bounded below by the cumulative sum of
            # underlying layers, so there is no occluded region beneath it.
            # .copy() here is deliberate: y_stack is mutated in-place below
            # (`y_stack += data`), and this array is captured verbatim into
            # keypoint_info['fill_bottom']. Without the copy, this still reads
            # correctly today only because extraction happens before the
            # mutation on the last line of the loop body -- an implicit
            # ordering dependency that a future edit could easily break.
            y_stack_previous = y_stack.copy()
            boundary_y = y_stack + data
            alpha = 0.75 
        
        if debug_mode:
            print(f"DEBUG [AREA] Series {series_idx}: y_stack_previous range [{np.min(y_stack_previous):.2f}, {np.max(y_stack_previous):.2f}]")
            print(f"DEBUG [AREA] Series {series_idx}: boundary_y range [{np.min(boundary_y):.2f}, {np.max(boundary_y):.2f}]")
        
        plotted = [(float(x[i]), float(boundary_y[i]), int(i)) for i in range(len(boundary_y))]

        area = ax.fill_between(x, y_stack_previous, boundary_y, color=color, alpha=alpha,
                               label=f'Series {series_idx+1}', zorder=2)
        data_artists.append(area)
        
        line, = ax.plot(x, boundary_y, color='white', linewidth=1.5, alpha=0.9, zorder=3)
        other_artists.append(line)
        
        inflection_pts = detect_inflection_points(x, boundary_y, threshold=0.1)
        prominence_factor = 0.08 if is_scientific else 0.05
        peaks, valleys = detect_extrema(x, boundary_y, prominence_factor=prominence_factor)
        
        keypoint_info.append({
            'series_idx': series_idx,
            'start': (float(x[0]), float(boundary_y[0]), 0),
            'end': (float(x[-1]), float(boundary_y[-1]), len(x)-1),
            'inflections': [(float(x_val), float(y_val), int(idx)) for x_val, y_val, idx in inflection_pts],
            'peaks': [(float(x_val), float(y_val), int(idx)) for x_val, y_val, idx in peaks],
            'valleys': [(float(x_val), float(y_val), int(idx)) for x_val, y_val, idx in valleys],
            'boundary_points': [(float(x[i]), float(boundary_y[i]), int(i)) for i in range(len(x))],
            'fill_bottom': [(float(x[i]), float(y_stack_previous[i]), int(i)) for i in range(len(x))], 
            'fill_top': [(float(x[i]), float(boundary_y[i]), int(i)) for i in range(len(x))],
            'plotted_points': plotted,
            'stacking_mode': stacking_mode
        })
        
        if stacking_mode != 'overlapping':
            y_stack += data
    
    # 3. Axis configuration
    if use_synthetic and syn_table is not None:
        ax.set_xlabel(syn_table['x_label'])
        if stacking_mode == 'percentage':
            ax.set_ylabel("Percentage (%)")
        else:
            ax.set_ylabel(syn_table['y_label'])
        ax.set_title(syn_table['title'], fontsize=14, pad=15)
    else:
        from synth.semantics.sampler import sample_axis_pair
        x_label, y_label, domain_cat = sample_axis_pair(
            "area", "vertical", is_scientific, semantic_domain=theme_config.get('semantic_domain') if isinstance(theme_config, dict) else None
        )
        ax.set_xlabel(x_label)
        
        if stacking_mode == 'percentage':
            ax.set_ylabel("Percentage (%)")
        else:
            ax.set_ylabel(y_label)
        if isinstance(theme_config, dict):
            theme_config['semantic_domain'] = domain_cat
    
    legend_prob = area_cfg.get('legend_probability', 0.7)
    if num_series > 1 and random.random() < legend_prob:
        legend = apply_legend_variation(ax, num_series)
        other_artists.append(legend)
    
    apply_typography_variation(ax, domain='scientific' if is_scientific else 'business', theme_config=theme_config)
    
    if stacking_mode != 'percentage':
        data_min = float(min(np.min(boundary_y), np.min(y_stack)))
        apply_axis_scaling(ax, data_min=data_min, orientation='vertical')
    
    current_scale = ax.get_yscale()

    if stacking_mode == 'percentage':
        ax.set_ylim(0, 100)
    else:
        bottom_limit = None
        if current_scale != 'log':
            bottom_limit = 0
        
        if stacking_mode == 'stacked':
            actual_max = float(np.max(y_stack))
        else:  # 'overlapping' or 'single'
            actual_max = float(max(np.max(series) for series in all_series_data))
        
        top_limit = actual_max * 1.15 
        
        if not np.isfinite(top_limit) or top_limit <= 0:
            top_limit = max_scale * 1.2
        
        ax.set_ylim(bottom=bottom_limit, top=top_limit)
        
        if debug_mode:
            print(f"DEBUG [AREA_YLIM] Mode={stacking_mode}, Scale={current_scale}, ActualMax={actual_max:.2f}, TopLimit={top_limit:.2f}, BottomLimit={bottom_limit}")
            
    return data_artists, other_artists, [], 'vertical', None, [], {'primary_scale_axis': 'y'}, keypoint_info

def _generate_heatmap_chart(ax, theme_name, theme_config, is_scientific, debug_mode=False):
    """
    Gera um heatmap com estruturas de dados realistas (correlação, cluster, etc.)
    e usa pcolormesh para anotação robusta de células.
    """
    theme = apply_chart_theme(ax, theme_name)
    apply_typography_variation(ax, domain='scientific' if is_scientific else 'business', theme_config=theme_config)

    # 1. Gerar dados estruturados
    data, cmap_type, heatmap_meta = generate_structured_heatmap(debug_mode=debug_mode)
    rows, cols = data.shape
    context_cfg = heatmap_meta.get("context_config") or {}
    context_id = heatmap_meta.get("context_id")
    context_domain = context_cfg.get("domain")

    HEATMAP_CONTEXT_TO_DOMAIN = {
        "genomic_expression_heatmap": "biomedical",
        "pharmacokinetics_heatmap": "biomedical",
        "cohort_retention_heatmap": "business",
        "correlation_matrix": None,
    }

    if context_id in HEATMAP_CONTEXT_TO_DOMAIN:
        mapped_dom = HEATMAP_CONTEXT_TO_DOMAIN[context_id]
        if mapped_dom is not None:
            heatmap_domain = mapped_dom
            is_scientific = (heatmap_domain != "business")
        else:
            sub_weights = theme_config.get('scientific_subdomain_weights', {'biomedical': 0.70, 'engineering': 0.30}) if isinstance(theme_config, dict) else {'biomedical': 0.70, 'engineering': 0.30}
            subdomains = list(sub_weights.keys())
            weights = list(sub_weights.values())
            heatmap_domain = random.choices(subdomains, weights=weights, k=1)[0]
            is_scientific = True
    elif context_domain == "business":
        is_scientific = False
        heatmap_domain = "business"
    elif context_domain == "scientific":
        is_scientific = True
        sub_weights = theme_config.get('scientific_subdomain_weights', {'biomedical': 0.70, 'engineering': 0.30}) if isinstance(theme_config, dict) else {'biomedical': 0.70, 'engineering': 0.30}
        subdomains = list(sub_weights.keys())
        weights = list(sub_weights.values())
        heatmap_domain = random.choices(subdomains, weights=weights, k=1)[0]
    else:
        if isinstance(theme_config, dict) and theme_config.get('semantic_domain'):
            heatmap_domain = theme_config.get('semantic_domain')
            is_scientific = (heatmap_domain != "business")
        else:
            heatmap_domain = "biomedical" if is_scientific else "business"

    if isinstance(theme_config, dict):
        theme_config['effective_is_scientific'] = is_scientific
        theme_config['semantic_domain'] = heatmap_domain

    # 2. Selecionar colormap apropriado com base no tipo de dados
    COLORMAP_CATEGORIES = {
        'sequential': ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                       'Greys', 'Purples', 'Blues', 'Greens', 'Oranges',
                       'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu'],

        'diverging': ['coolwarm', 'bwr', 'seismic', 'RdBu', 'RdGy',
                      'RdYlBu', 'RdYlGn', 'Spectral', 'PiYG', 'PRGn']
    }

    palette = theme.get('palette')
    cmap = None
    cmap_name = None
    context_cmap = context_cfg.get("colormap")

    if context_cmap:
        if isinstance(context_cmap, list):
            cmap = ListedColormap(context_cmap)
        else:
            cmap_name = context_cmap
    elif isinstance(palette, list) and palette:
        cmap = ListedColormap(palette)
    else:
        if isinstance(palette, str) and palette:
            cmap_name = palette
            if cmap_type == 'diverging' and cmap_name not in COLORMAP_CATEGORIES['diverging']:
                cmap_name = None
        if cmap_type == 'diverging':
            if cmap_name is None:
                cmap_name = np.random.choice(COLORMAP_CATEGORIES['diverging'])
        else:
            if cmap_name is None:
                cmap_name = np.random.choice(COLORMAP_CATEGORIES['sequential'])

    if cmap is None:
        try:
            cmap = colormaps.get(cmap_name)
        except Exception:
            fallback = 'coolwarm' if cmap_type == 'diverging' else 'viridis'
            cmap = colormaps.get(fallback)

    # 3. Usar pcolormesh (gera QuadMesh, que generator.py lida bem)
    x = np.arange(cols + 1)
    y = np.arange(rows + 1)

    cmap = cmap.copy()
    cmap.set_bad(color="#E0E0E0")
    masked = np.ma.masked_invalid(data)

    if cmap_type == 'diverging':
        if heatmap_meta.get('is_correlation', False):
            vmin, vmax = -1.0, 1.0
        else:
            abs_max = np.nanmax(np.abs(data)) if np.isfinite(np.nanmax(np.abs(data))) else 1.0
            vmin, vmax = -abs_max, abs_max
    else:
        vmin = np.nanmin(data) if np.isfinite(np.nanmin(data)) else 0.0
        vmax = np.nanmax(data) if np.isfinite(np.nanmax(data)) else 1.0

    mesh = ax.pcolormesh(
        x, y, masked, cmap=cmap, vmin=vmin, vmax=vmax, zorder=2,
        edgecolors='white', linewidth=0.5 if rows * cols < 200 else 0
    )
    data_artists = [mesh]

    # 4. Adicionar colorbar
    colorbar_cfg = theme.get('colorbar', {})
    cbar = ax.figure.colorbar(mesh, ax=ax, **colorbar_cfg)
    # A lógica de anotação encontra isso
    other_artists = [cbar]

    # 5. Adicionar rótulos de dados (texto)
    # A lógica de anotação encontra isso
    text_labels = []
    # Só adiciona rótulos se a grade não for muito densa
    heatmap_cfg = theme_config.get('heatmap_config', {}) if isinstance(theme_config, dict) else {}
    annotate_prob = heatmap_cfg.get('annotate_cells_probability', 0.8)
    should_annotate = rows * cols < 150 and random.random() < annotate_prob
    format_str = None
    if should_annotate:
        if "annotation_format" in context_cfg:
            format_str = context_cfg.get("annotation_format")
        else:
            format_str = random.choice(HEATMAP_ANNOTATION_FORMATS)

        if format_str is not None:
            format_str = str(format_str)

    if should_annotate and format_str is not None:
        for r in range(rows):
            for c in range(cols):
                val = data[r, c]
                if not np.isfinite(val):
                    continue

                if vmax > vmin:
                    norm_val = (val - vmin) / (vmax - vmin)
                else:
                    norm_val = 0.0

                if cmap_type == 'diverging':
                    text_color = 'white' if 0.25 < norm_val < 0.75 else 'black'
                else:
                    text_color = 'white' if norm_val < 0.35 else 'black'

                try:
                    if '%' in format_str and abs(val) > 1.5:
                        text_str = format_str.format(val / 100.0)
                    else:
                        text_str = format_str.format(val)
                except Exception:
                    text_str = f"{val:.2f}"

                txt = ax.text(
                    c + 0.5, r + 0.5, text_str,
                    ha='center', va='center', fontsize=8, color=text_color, zorder=10
                )
                text_labels.append(txt)

    other_artists.extend(text_labels)

    # 6. Configurar eixos (importante para heatmaps) - ENHANCED WITH DIVERSE LABELS
    ax.set_xticks(np.arange(cols) + 0.5)
    ax.set_yticks(np.arange(rows) + 0.5)

    # ENHANCED: Select varied axis labels based on domain and context
    context_xlabel_pool = context_cfg.get("xlabel_pool")
    context_ylabel_pool = context_cfg.get("ylabel_pool")
    context_title_pool = context_cfg.get("title_pool")
    context_cbar_title = context_cfg.get("cbar_label")

    if is_scientific:
        xlabel = random.choice(context_xlabel_pool) if context_xlabel_pool else random.choice(HEATMAP_XLABELS_SCIENTIFIC)
        ylabel = random.choice(context_ylabel_pool) if context_ylabel_pool else random.choice(HEATMAP_YLABELS_SCIENTIFIC)
        colorbar_title = context_cbar_title if context_cbar_title else random.choice(COLORBAR_TITLES_SCIENTIFIC)
        if context_title_pool:
            chart_title = random.choice(context_title_pool)
        else:
            from synth.semantics.sampler import sample_chart_title
            chart_title = sample_chart_title(chart_type="heatmap", is_scientific=True, domain=heatmap_domain)
    else:
        xlabel = random.choice(context_xlabel_pool) if context_xlabel_pool else random.choice(HEATMAP_XLABELS_BUSINESS)
        ylabel = random.choice(context_ylabel_pool) if context_ylabel_pool else random.choice(HEATMAP_YLABELS_BUSINESS)
        colorbar_title = context_cbar_title if context_cbar_title else random.choice(COLORBAR_TITLES_BUSINESS)
        if context_title_pool:
            chart_title = random.choice(context_title_pool)
        else:
            from synth.semantics.sampler import sample_chart_title
            chart_title = sample_chart_title(chart_type="heatmap", is_scientific=False, domain=heatmap_domain)

    # Set labels with variety
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(chart_title)

    # ENHANCED: Generate varied tick labels based on xlabel/ylabel context
    if any(word in xlabel.lower() for word in ['time', 'day', 'week', 'quarter']):
        xticklabels = generate_temporal_labels(cols, xlabel)
    elif any(word in xlabel.lower() for word in ['dose', 'concentration', 'temperature']):
        xticklabels = generate_numeric_labels(cols, xlabel)
    elif any(word in xlabel.lower() for word in ['gene', 'protein', 'metabolite']):
        xticklabels = generate_biological_labels(cols, xlabel)
    elif any(word in xlabel.lower() for word in ['product', 'category', 'region']):
        xticklabels = generate_categorical_labels(cols, xlabel)
    else:
        xticklabels = [f"{xlabel.split()[0][:3]}{i+1}" for i in range(cols)]

    if any(word in ylabel.lower() for word in ['time', 'day', 'week', 'quarter']):
        yticklabels = generate_temporal_labels(rows, ylabel)
    elif any(word in ylabel.lower() for word in ['dose', 'concentration', 'temperature']):
        yticklabels = generate_numeric_labels(rows, ylabel)
    elif any(word in ylabel.lower() for word in ['gene', 'protein', 'metabolite']):
        yticklabels = generate_biological_labels(rows, ylabel)
    elif any(word in ylabel.lower() for word in ['product', 'category', 'region']):
        yticklabels = generate_categorical_labels(rows, ylabel)
    else:
        yticklabels = [f"{ylabel.split()[0][:3]}{i+1}" for i in range(rows)]

    row_order = heatmap_meta.get('row_order')
    col_order = heatmap_meta.get('col_order')
    if row_order is not None and len(row_order) == len(yticklabels):
        yticklabels = [yticklabels[i] for i in list(row_order)]
    if col_order is not None and len(col_order) == len(xticklabels):
        xticklabels = [xticklabels[i] for i in list(col_order)]

    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)

    # Girar rótulos do eixo x se houver muitos
    if cols > 10:
        ax.tick_params(axis='x', labelrotation=90)

    # ENHANCED: Add colorbar with varied title
    cbar.set_label(colorbar_title, rotation=270, labelpad=20)

    # Inverter eixo Y para corresponder ao layout da matriz (linha 0 no topo)
    ax.invert_yaxis()

    # Garantir que nenhuma linha de grade do tema interfira
    ax.grid(False)

    heatmap_payload = {
        'data': data,
        'meta': heatmap_meta
    }

    return data_artists, other_artists, [], 'vertical', [], [], \
           {'primary_scale_axis': 'none'}, heatmap_payload


# Additional helper functions for completeness
def add_jitter_overlay(ax, bar_info, orientation='vertical'):
    """Add jitter points overlaid on bars for scientific visualization"""
    for info in bar_info:
        center_pos, mean_val, width_val = info['center'], info['height'], info['width']
        
        if mean_val == 0: 
            continue
        
        n_points = random.randint(8, 20)
        points = np.random.normal(loc=mean_val, scale=abs(mean_val) * random.uniform(0.1, 0.25), size=n_points)
        jitter = np.random.uniform(-width_val * 0.3, width_val * 0.3, size=n_points)
        
        if orientation == 'vertical':
            ax.scatter(center_pos + jitter, points, c='black', s=8, alpha=0.3, zorder=10)
        else:
            ax.scatter(points, center_pos + jitter, c='black', s=8, alpha=0.3, zorder=10)


def detect_inflection_points(x_data, y_data, threshold=0.1):
    """Detect inflection points using second derivative analysis."""
    if len(x_data) < 3:
        return []
    
    inflection_points = []
    y_range = max(y_data) - min(y_data)
    
    for i in range(1, len(y_data) - 1):
        d2y = y_data[i+1] - 2*y_data[i] + y_data[i-1]
        if abs(d2y) > threshold * y_range:
            inflection_points.append((x_data[i], y_data[i], i))
    
    return inflection_points

def detect_extrema(xdata, ydata, window_size=3, prominence_factor=0.05):
    """
    Detect local maxima (peaks) and minima (valleys) robustly.
    """
    if len(ydata) < 3:
        return [], []
    
    # Smooth with conservative sigma to preserve peak locations
    y_smooth = gaussian_filter(ydata, sigma=0.8)
    
    # Adaptive prominence based on data range
    y_range = np.max(ydata) - np.min(ydata)
    if y_range == 0:
        return [], []
    
    min_prominence = prominence_factor * y_range
    
    # Find peaks using scipy (robust to noise)
    peaks_idx, peak_props = find_peaks(
        y_smooth, 
        distance=max(1, window_size // 2),
        prominence=min_prominence
    )
    
    valleys_idx, valley_props = find_peaks(
        -y_smooth,
        distance=max(1, window_size // 2),
        prominence=min_prominence
    )
    
    # Sort by prominence and take top 2 of each
    if len(peaks_idx) > 2:
        peak_prominences = peak_props['prominences']
        top_peaks = peaks_idx[np.argsort(peak_prominences)[-2:]]
        peaks_idx = sorted(top_peaks)
    
    if len(valleys_idx) > 2:
        valley_prominences = valley_props['prominences']
        top_valleys = valleys_idx[np.argsort(valley_prominences)[-2:]]
        valleys_idx = sorted(top_valleys)
    
    # Return using ORIGINAL ydata for exact coordinates
    peaks = [(float(xdata[i]), float(ydata[i]), int(i)) for i in peaks_idx]
    valleys = [(float(xdata[i]), float(ydata[i]), int(i)) for i in valleys_idx]
    
    return peaks, valleys



HEATMAP_GENERATION_CONFIG = {
    "size_range": {"rows": (8, 30), "cols": (8, 30)},
    "type_weights": {
        "correlation_davies_higham": 14,
        "correlation_lkj": 12,
        "bicluster_checkerboard": 10,
        "bicluster_block": 10,
        "bicluster_additive": 10,
        "bicluster_multiplicative": 8,
        "perlin": 10,
        "fractal": 8,
        "sarima": 6,
        "confusion": 6,
        "clustered_patterns": 6,
        "biclustered_checkerboard": 6,
        "block_diagonal_communities": 6,
        "toeplitz_autoregressive": 5
    },
    "normalization_weights": {
        "none": 20,
        "row_zscore": 25,
        "global_minmax": 20,
        "robust_iqr": 35
    },
    "ordering_weights": {
        "none": 30,
        "olo": 35,
        "fiedler": 35
    },
    "missingness": {
        "mode_weights": {
            "none": 45,
            "mcar": 25,
            "mnar_logistic": 10,
            "mnar_quantile": 10,
            "mnar_llod_uloq": 10
        },
        "missing_rate_range": (0.02, 0.10),
        "logistic_beta0_range": (-2.0, 2.0),
        "logistic_beta1_range": (0.2, 1.2),
        "quantile_range": (0.80, 0.95),
        "drop_prob_range": (0.25, 0.60),
        "llod_quantile_range": (0.02, 0.15),
        "uloq_quantile_range": (0.85, 0.98),
        "llod_uloq_jitter": (0.85, 1.15),
        "llod_uloq_min_span": 1e-6
    },
    "context_weights": {
        "none": 40,
        "genomic_expression_heatmap": 20,
        "cohort_retention_heatmap": 20,
        "correlation_matrix": 10,
        "pharmacokinetics_heatmap": 10
    },
    "noise": {
        "gaussian_prob": 0.25,
        "gaussian_std_range": (0.0, 0.08),
        "heteroscedastic_prob": 0.30,
        "heteroscedastic_mode_weights": {"magnitude_scaled": 60, "parametric": 40},
        "omega_range": (0.02, 0.15),
        "delta_range": (0.05, 0.60),
        "parametric_alpha_range": (0.01, 0.08),
        "parametric_beta_range": (0.02, 0.20),
        "parametric_gamma_range": (0.8, 1.6),
        "pareto_outlier_prob": 0.005,
        "pareto_outlier_scale_range": (6.0, 14.0),
        "pareto_outlier_chance": 0.35,
        "poisson_prob": 0.20,
        "k_dispersion_range": (0.4, 2.0),
        "outlier_prob": 0.25,
        "outlier_fraction_range": (0.01, 0.05),
        "outlier_scale_range": (2.0, 6.0)
    },
    "correlation": {
        "dim_range": (8, 20),
        "strength_weights": {"uniform": 50, "high": 25, "low": 25},
        "lkj_eta_range": (0.6, 3.0),
        "noise_std_range": (0.0, 0.05)
    },
    "bicluster": {
        "checkerboard_noise_range": (0.05, 0.25),
        "block_noise_range": (0.05, 0.35),
        "clusters_range": (2, 5),
        "additive_mu_range": (5.0, 25.0),
        "bicluster_noise_range": (0.5, 2.0)
    },
    "spatial": {
        "res_range": (3, 8),
        "octaves_range": (2, 6),
        "persistence_range": (0.35, 0.65)
    },
    "sarima": {
        "order_choices": [(1, 0, 1), (2, 0, 1), (1, 1, 1)],
        "seasonal_order_choices": [(1, 0, 1, 4), (1, 0, 0, 6), (0, 1, 1, 4)],
        "rho_range": (0.15, 0.6)
    }
}


def _pick_weighted_option(options, default=None):
    if not options:
        return default
    names = list(options.keys())
    weights = [float(options[k]) for k in names]
    return random.choices(names, weights=weights, k=1)[0]


def _resolve_heatmap_size(size, cfg):
    if size is None:
        rows = random.randint(*cfg["size_range"]["rows"])
        cols = random.randint(*cfg["size_range"]["cols"])
        return rows, cols
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return int(size[0]), int(size[1])
    return int(size), int(size)


# --- Generate PSD correlation and covariance matrices for heatmap use ---
def generate_davies_higham(dim, correlation_strength="uniform", random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    if correlation_strength == "high":
        eigs = np.random.dirichlet(np.ones(dim) * 0.1) * dim
    elif correlation_strength == "low":
        eigs = np.random.dirichlet(np.ones(dim) * 10.0) * dim
    else:
        eigs = np.random.dirichlet(np.ones(dim)) * dim

    eigs = eigs * (dim / np.sum(eigs))
    return random_correlation.rvs(eigs)

def generate_vine_lkj(dim, eta, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    beta_param = eta - 1.0 + dim / 2.0
    P = np.zeros((dim, dim))
    S = np.eye(dim)

    for k in range(dim - 1):
        for i in range(k + 1, dim):
            sampled_beta = stats.beta.rvs(a=beta_param, b=beta_param)
            P[k, i] = (sampled_beta - 0.5) * 2.0

            p = P[k, i]
            for l in range(k - 1, -1, -1):
                term1 = np.sqrt((1.0 - P[l, i] ** 2) * (1.0 - P[l, k] ** 2))
                p = p * term1 + P[l, i] * P[l, k]

            S[k, i] = p
            S[i, k] = p

    S = (S + S.T) / 2.0
    np.fill_diagonal(S, 1.0)
    return np.clip(S, -1.0, 1.0)


# --- Generate block, checkerboard, and coherent bicluster structures ---
def generate_spectral_checkerboard(shape, clusters, noise, random_state=None):
    matrix, rows, cols = make_checkerboard(
        shape=shape,
        n_clusters=clusters,
        noise=noise,
        minval=-5.0,
        maxval=5.0,
        shuffle=True,
        random_state=random_state
    )
    return matrix, rows, cols

def generate_block_biclusters(shape, clusters, noise, random_state=None):
    if isinstance(clusters, (list, tuple)):
        n_clusters = int(clusters[0]) if clusters else 2
    else:
        n_clusters = int(clusters)

    matrix, rows, cols = make_biclusters(
        shape=shape,
        n_clusters=n_clusters,
        noise=noise,
        minval=-5.0,
        maxval=5.0,
        shuffle=True,
        random_state=random_state
    )
    return matrix, rows, cols

def inject_additive_coherent_bicluster(matrix, shape_ij, mu, noise):
    n_rows, n_cols = matrix.shape
    b_rows, b_cols = shape_ij

    idx_I = np.random.choice(n_rows, b_rows, replace=False)
    idx_J = np.random.choice(n_cols, b_cols, replace=False)

    alpha_i = np.random.uniform(2.0, 5.0, b_rows)[:, np.newaxis]
    beta_j = np.random.uniform(2.0, 5.0, b_cols)
    bicluster_core = mu + alpha_i + beta_j + np.random.normal(0, noise, shape_ij)

    out_matrix = matrix.copy()
    out_matrix[np.ix_(idx_I, idx_J)] = bicluster_core

    return out_matrix, idx_I, idx_J

def inject_multiplicative_coherent_bicluster(matrix, shape_ij, mu, noise):
    n_rows, n_cols = matrix.shape
    b_rows, b_cols = shape_ij

    idx_I = np.random.choice(n_rows, b_rows, replace=False)
    idx_J = np.random.choice(n_cols, b_cols, replace=False)

    alpha_i = np.random.uniform(1.1, 2.0, b_rows)[:, np.newaxis]
    beta_j = np.random.uniform(1.1, 2.0, b_cols)
    bicluster_core = mu * alpha_i * beta_j + np.random.normal(0, noise, shape_ij)

    out_matrix = matrix.copy()
    out_matrix[np.ix_(idx_I, idx_J)] = bicluster_core

    return out_matrix, idx_I, idx_J


# --- Vectorized 2D Perlin noise and fractal noise ---
def _fade(t):
    return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

def generate_perlin_2d(shape, res, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    shape = (int(shape[0]), int(shape[1]))
    res = (int(res[0]), int(res[1]))

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (int(np.ceil(shape[0] / res[0])), int(np.ceil(shape[1] / res[1])))

    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    grid = grid[:shape[0], :shape[1], :]

    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))

    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)

    g00 = g00[:shape[0], :shape[1]]
    g10 = g10[:shape[0], :shape[1]]
    g01 = g01[:shape[0], :shape[1]]
    g11 = g11[:shape[0], :shape[1]]

    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)

    t = _fade(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11

    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape, dtype=float)
    frequency = 1
    amplitude = 1.0

    for _ in range(int(octaves)):
        res_octave = (frequency * res[0], frequency * res[1])
        noise += amplitude * generate_perlin_2d(shape, res_octave)
        frequency *= 2
        amplitude *= persistence

    return noise


# --- SARIMA temporal rows with SAR spatial dependence across columns ---
def generate_sarima_matrix(n_steps, n_series, order=(1, 0, 0), seasonal_order=(1, 0, 0, 4)):
    matrix = np.zeros((n_steps, n_series))
    for j in range(n_series):
        matrix[:, j] = _ar1_noise(n_steps, phi=0.6)
    return matrix

def apply_sar_spatial_dependence(matrix, rho):
    n_cols = matrix.shape[1]

    indices = np.arange(n_cols)
    dist = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])

    W = 1.0 / (dist + np.eye(n_cols))
    np.fill_diagonal(W, 0)
    W = W / np.sum(W, axis=1, keepdims=True)

    spatial_multiplier = np.linalg.inv(np.eye(n_cols) - rho * W)
    return matrix @ spatial_multiplier.T


# --- Apply MCAR and MNAR missingness patterns ---
def inject_mcar(matrix, missing_rate):
    degraded = matrix.copy()
    mask = np.random.rand(*matrix.shape) < missing_rate
    degraded[mask] = np.nan
    return degraded, mask

def inject_mnar_logistic(matrix, beta_0, beta_1):
    degraded = matrix.copy()
    prob_missing = 1.0 / (1.0 + np.exp(-(beta_0 + beta_1 * matrix)))
    mask = np.random.rand(*matrix.shape) < prob_missing
    degraded[mask] = np.nan
    return degraded, mask

def inject_mnar_quantile_censorship(matrix, q, drop_prob, upper_bound=True):
    degraded = matrix.copy()
    threshold = np.nanquantile(matrix, q)
    if upper_bound:
        condition_mask = matrix > threshold
    else:
        condition_mask = matrix < threshold
    prob_mask = np.random.rand(*matrix.shape) < drop_prob
    mask = condition_mask & prob_mask
    degraded[mask] = np.nan
    return degraded, mask

def inject_mnar_detection_limits(
    matrix,
    llod=None,
    uloq=None,
    llod_quantile=None,
    uloq_quantile=None,
    jitter_range=(0.85, 1.15),
    min_span=1e-6
):
    degraded = matrix.copy()
    finite_vals = degraded[np.isfinite(degraded)]
    if finite_vals.size == 0:
        mask = np.zeros_like(degraded, dtype=bool)
        return degraded, mask, {"llod": None, "uloq": None}

    llod_val = llod
    uloq_val = uloq

    if llod_val is None and llod_quantile is not None:
        llod_val = float(np.nanquantile(finite_vals, llod_quantile))
    if uloq_val is None and uloq_quantile is not None:
        uloq_val = float(np.nanquantile(finite_vals, uloq_quantile))

    if llod_val is None and uloq_val is None:
        llod_val = float(np.nanmin(finite_vals))
        uloq_val = float(np.nanmax(finite_vals))

    jitter_min, jitter_max = jitter_range
    if llod_val is not None:
        llod_val *= random.uniform(jitter_min, jitter_max)
    if uloq_val is not None:
        uloq_val *= random.uniform(jitter_min, jitter_max)

    if llod_val is not None and uloq_val is not None:
        if (uloq_val - llod_val) < min_span:
            mid = 0.5 * (llod_val + uloq_val)
            llod_val = mid - min_span * 0.5
            uloq_val = mid + min_span * 0.5
        if llod_val > uloq_val:
            llod_val, uloq_val = uloq_val, llod_val

    mask = np.zeros_like(degraded, dtype=bool)
    if llod_val is not None:
        mask |= degraded < llod_val
    if uloq_val is not None:
        mask |= degraded > uloq_val

    degraded[mask] = np.nan
    return degraded, mask, {"llod": float(llod_val) if llod_val is not None else None,
                            "uloq": float(uloq_val) if uloq_val is not None else None}


# --- Magnitude-scaled noise and Poisson-like variance ---
def inject_magnitude_scaled_gaussian(matrix, omega, delta):
    local_variance = (omega * np.abs(matrix)) ** 2 + delta ** 2
    local_sigma = np.sqrt(local_variance)
    noise = np.random.normal(loc=0.0, scale=local_sigma)
    noisy = matrix + noise
    noisy[~np.isfinite(matrix)] = np.nan
    return noisy

def inject_poisson_like_variance(matrix, k_dispersion):
    min_val = np.nanmin(matrix)
    shift = 0.0 if min_val >= 0 else np.abs(min_val) + 1e-6
    positive_matrix = matrix + shift

    local_sigma = np.sqrt(k_dispersion * positive_matrix)
    noise = np.random.normal(loc=0.0, scale=local_sigma)
    noisy = matrix + noise
    noisy[~np.isfinite(matrix)] = np.nan
    return noisy

def inject_parametric_heteroscedastic(
    matrix,
    alpha,
    beta,
    gamma=1.0,
    pareto_outliers=False,
    outlier_prob=0.005,
    outlier_scale=10.0
):
    finite_mask = np.isfinite(matrix)
    base = matrix.astype(float).copy()
    if not np.any(finite_mask):
        return base

    local_variance = alpha + beta * np.power(np.abs(base), gamma)
    local_variance = np.where(local_variance > 0, local_variance, alpha)
    local_std = np.sqrt(local_variance)
    noise = np.random.normal(loc=0.0, scale=local_std)
    corrupted = base + noise

    if outlier_prob > 0:
        outlier_mask = (np.random.rand(*base.shape) < outlier_prob) & finite_mask
        if np.any(outlier_mask):
            if pareto_outliers:
                pareto_shape = 3.0
                pareto_noise = np.random.pareto(a=pareto_shape, size=base.shape) * outlier_scale
                direction = np.sign(np.random.uniform(-1, 1, size=base.shape))
                corrupted[outlier_mask] += pareto_noise[outlier_mask] * direction[outlier_mask]
            else:
                extreme_noise = np.random.normal(loc=0.0, scale=local_std * outlier_scale)
                corrupted[outlier_mask] += extreme_noise[outlier_mask]

    corrupted[~finite_mask] = np.nan
    return corrupted


# --- Ordering helpers for heatmaps ---
def seriate_optimal_leaf_ordering(matrix, metric="euclidean", method="ward"):
    row_dist = pdist(matrix, metric=metric)
    row_linkage = linkage(row_dist, method=method)
    row_optimal_Z = optimal_leaf_ordering(row_linkage, row_dist)
    row_order = leaves_list(row_optimal_Z)

    col_dist = pdist(matrix.T, metric=metric)
    col_linkage = linkage(col_dist, method=method)
    col_optimal_Z = optimal_leaf_ordering(col_linkage, col_dist)
    col_order = leaves_list(col_optimal_Z)

    ordered = matrix[row_order, :][:, col_order]
    return ordered, row_order, col_order

def _compute_rbf_kernel(X, gamma):
    pairwise_sq_dists = np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=2)
    return np.exp(-gamma * pairwise_sq_dists)

def seriate_spectral_fiedler(matrix, gamma=1.0):
    def get_fiedler_indices(data):
        A = _compute_rbf_kernel(data, gamma=gamma)
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        eigenvalues, eigenvectors = eigh(L)
        fiedler_vector = eigenvectors[:, 1]
        return np.argsort(fiedler_vector)

    row_order = get_fiedler_indices(matrix)
    col_order = get_fiedler_indices(matrix.T)
    ordered = matrix[row_order, :][:, col_order]
    return ordered, row_order, col_order


# --- Normalization helpers for heatmaps ---
def row_wise_zscore(matrix, epsilon=1e-8):
    out = matrix.astype(float).copy()
    finite_mask = np.isfinite(out)
    if not np.any(finite_mask):
        return out
    row_means = np.nanmean(out, axis=1, keepdims=True)
    row_stds = np.nanstd(out, axis=1, keepdims=True)
    normed = (out - row_means) / (row_stds + epsilon)
    out[finite_mask] = normed[finite_mask]
    return out

def global_minmax(matrix):
    out = matrix.astype(float).copy()
    finite_mask = np.isfinite(out)
    if not np.any(finite_mask):
        return out
    mat_min = np.nanmin(out)
    mat_max = np.nanmax(out)
    if not np.isfinite(mat_min) or not np.isfinite(mat_max):
        return out
    if np.isclose(mat_max, mat_min):
        out[finite_mask] = 0.0
        return out
    normed = (out - mat_min) / (mat_max - mat_min)
    out[finite_mask] = normed[finite_mask]
    return out

def robust_iqr_scaler(matrix, epsilon=1e-8):
    out = matrix.astype(float).copy()
    finite_mask = np.isfinite(out)
    if not np.any(finite_mask):
        return out
    medians = np.nanmedian(out, axis=0, keepdims=True)
    q75 = np.nanpercentile(out, 75, axis=0, keepdims=True)
    q25 = np.nanpercentile(out, 25, axis=0, keepdims=True)
    iqr = q75 - q25
    normed = (out - medians) / (iqr + epsilon)
    out[finite_mask] = normed[finite_mask]
    return out

def fixed_bounds_pm1(matrix):
    out = matrix.astype(float).copy()
    finite_mask = np.isfinite(out)
    out[finite_mask] = np.clip(out[finite_mask], -1.0, 1.0)
    return out

def log10_scale(matrix, epsilon=1e-6):
    out = matrix.astype(float).copy()
    finite_mask = np.isfinite(out)
    if not np.any(finite_mask):
        return out
    sign = np.sign(out)
    scaled = np.log10(np.abs(out) + epsilon)
    out[finite_mask] = (sign * scaled)[finite_mask]
    return out


def _inject_heatmap_outliers(matrix, frac_range, scale_range):
    degraded = matrix.copy()
    total = degraded.size
    if total == 0:
        return degraded

    fraction = random.uniform(*frac_range)
    n_outliers = max(1, int(total * fraction))
    flat_idx = np.random.choice(total, n_outliers, replace=False)
    scale = random.uniform(*scale_range)
    flat = degraded.reshape(-1)
    flat[flat_idx] = flat[flat_idx] * scale
    return degraded


def generate_perlin_heatmap(rows, cols, scale=6.0, octaves=4, persistence=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    res = (max(2, int(scale)), max(2, int(scale)))
    base = generate_fractal_noise_2d(
        (rows, cols), res, octaves=octaves, persistence=persistence
    )
    if np.nanmax(base) == np.nanmin(base):
        return np.zeros_like(base)
    return (base - np.nanmin(base)) / (np.nanmax(base) - np.nanmin(base))


def generate_correlated_blocks(rows, cols, num_blocks=3, block_corr=0.8, noise_level=0.1):
    row_blocks = np.random.randint(0, num_blocks, rows)
    corr_matrix = np.eye(rows)
    for block_id in range(num_blocks):
        block_rows = np.where(row_blocks == block_id)[0]
        for i in block_rows:
            for j in block_rows:
                if i != j:
                    corr_matrix[i, j] = block_corr

    mean = np.zeros(rows)
    samples = np.random.multivariate_normal(mean=mean, cov=corr_matrix, size=cols).T
    data = np.abs(samples) + np.random.normal(0, noise_level, (rows, cols))
    if np.nanmax(data) == np.nanmin(data):
        return np.zeros_like(data)
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))


def generate_clustered_heatmap(rows, cols, num_clusters=5, cluster_spread=0.15, base_intensity=0.2):
    cluster_centers = np.random.rand(num_clusters, 2)
    cluster_centers[:, 0] *= rows
    cluster_centers[:, 1] *= cols

    cluster_intensities = np.random.uniform(0.3, 1.0, num_clusters)
    coords = np.array([[i, j] for i in range(rows) for j in range(cols)])
    distances = cdist(coords, cluster_centers)

    influences = np.exp(-distances ** 2 / (2 * (cluster_spread * min(rows, cols)) ** 2))
    weighted = influences * cluster_intensities
    data = weighted.sum(axis=1).reshape(rows, cols)

    data = base_intensity + data * (1 - base_intensity)
    data += np.random.normal(0, 0.05, (rows, cols))
    return np.clip(data, 0, 1)


def generate_block_diagonal_communities(
    rows,
    cols,
    num_blocks=8,
    block_size_bounds=(15, 60),
    off_diagonal_noise_std=0.15,
    internal_block_density=0.90
):
    size = min(rows, cols)
    min_size, max_size = block_size_bounds
    sizes = []
    remaining = size

    for i in range(num_blocks):
        if remaining <= 0:
            break
        max_allowed = min(max_size, remaining - (num_blocks - i - 1) * min_size)
        if max_allowed < min_size:
            block_size = remaining
        else:
            block_size = random.randint(min_size, max_allowed)
        sizes.append(block_size)
        remaining -= block_size

    if remaining > 0:
        sizes.append(remaining)

    blocks = []
    for block_size in sizes:
        block = np.random.normal(0, 1, (block_size, block_size))
        if internal_block_density < 1.0:
            mask = np.random.rand(block_size, block_size) < internal_block_density
            block = block * mask
        blocks.append(block)

    matrix = block_diag(*blocks)
    if off_diagonal_noise_std > 0:
        matrix = matrix + np.random.normal(0, off_diagonal_noise_std, matrix.shape)

    if matrix.shape[0] != rows or matrix.shape[1] != cols:
        out = np.zeros((rows, cols), dtype=float)
        out[:matrix.shape[0], :matrix.shape[1]] = matrix[:rows, :cols]
        matrix = out

    return matrix


def generate_toeplitz_autoregressive(rows, cols, decay_coeff=0.88, noise_std=0.02):
    size = min(rows, cols)
    decay = float(decay_coeff)
    base = decay ** np.arange(size)
    matrix = toeplitz(base)
    if noise_std > 0:
        matrix = matrix + np.random.normal(0, noise_std, matrix.shape)
    if matrix.shape[0] != rows or matrix.shape[1] != cols:
        out = np.zeros((rows, cols), dtype=float)
        out[:matrix.shape[0], :matrix.shape[1]] = matrix[:rows, :cols]
        matrix = out
    return matrix


def _generate_structural_theme(theme_id, rows, cols):
    theme = STRUCTURAL_THEMES.get(theme_id, {})
    params = theme.get("generation_parameters", {})
    meta = {"theme_id": theme_id, "params": params}

    if theme_id == "biclustered_checkerboard":
        clusters = params.get("n_clusters", (5, 4))
        noise = params.get("noise", 12.5)
        data, _, _ = generate_spectral_checkerboard(
            (rows, cols), clusters, noise
        )
        return data, meta

    if theme_id == "block_diagonal_communities":
        num_blocks = int(params.get("num_blocks", 8))
        bounds = params.get("block_size_bounds", (15, 60))
        off_noise = params.get("off_diagonal_noise_std", 0.15)
        density = params.get("internal_block_density", 0.90)
        data = generate_block_diagonal_communities(
            rows,
            cols,
            num_blocks=num_blocks,
            block_size_bounds=bounds,
            off_diagonal_noise_std=off_noise,
            internal_block_density=density
        )
        return data, meta

    if theme_id == "toeplitz_autoregressive":
        decay = params.get("decay_coefficient", 0.88)
        data = generate_toeplitz_autoregressive(rows, cols, decay_coeff=decay)
        return data, meta

    return generate_clustered_heatmap(rows, cols, num_clusters=5), meta


def generate_structured_heatmap(heatmap_type="auto", size=None, debug_mode=False):
    cfg = HEATMAP_GENERATION_CONFIG

    context_id = None
    context_cfg = None
    context_weights = cfg.get("context_weights", {})
    if context_weights:
        context_id = _pick_weighted_option(context_weights, default="none")
        if context_id and context_id != "none":
            context_cfg = CONTEXT_CONFIGURATIONS.get(context_id)

    if heatmap_type == "auto" and context_cfg and context_cfg.get("heatmap_type"):
        heatmap_type = context_cfg["heatmap_type"]
    if heatmap_type == "auto":
        heatmap_type = _pick_weighted_option(cfg["type_weights"], default="perlin")

    rows, cols = _resolve_heatmap_size(size, cfg)
    meta = {
        "type": heatmap_type,
        "ordering": "none",
        "normalization": "none",
        "missingness": "none",
        "noise": [],
        "noise_params": [],
        "row_order": None,
        "col_order": None,
        "is_correlation": False,
        "context_id": context_id,
        "context_config": context_cfg
    }

    if debug_mode:
        print(f"DEBUG [HEATMAP_GEN] type={heatmap_type} size=({rows},{cols})")

    data = None
    cmap_type = "sequential"

    if heatmap_type in ["correlation_davies_higham", "correlation_lkj"]:
        dim_min, dim_max = cfg["correlation"]["dim_range"]
        if size is not None:
            if isinstance(size, (list, tuple)) and len(size) == 2:
                dim = int(min(size))
            else:
                dim = int(size)
        else:
            dim = random.randint(dim_min, dim_max)
        strength = _pick_weighted_option(cfg["correlation"]["strength_weights"], default="uniform")
        if heatmap_type == "correlation_davies_higham":
            data = generate_davies_higham(dim, strength)
        else:
            eta_min, eta_max = cfg["correlation"]["lkj_eta_range"]
            eta = random.uniform(eta_min, eta_max)
            data = generate_vine_lkj(dim, eta)
        cmap_type = "diverging"
        meta["is_correlation"] = True
        rows, cols = dim, dim

    elif heatmap_type == "bicluster_checkerboard":
        clusters = random.randint(*cfg["bicluster"]["clusters_range"])
        noise = random.uniform(*cfg["bicluster"]["checkerboard_noise_range"])
        data, _, _ = generate_spectral_checkerboard(
            (rows, cols), (clusters, clusters), noise
        )

    elif heatmap_type == "bicluster_block":
        clusters = random.randint(*cfg["bicluster"]["clusters_range"])
        noise = random.uniform(*cfg["bicluster"]["block_noise_range"])
        data, _, _ = generate_block_biclusters(
            (rows, cols), clusters, noise
        )

    elif heatmap_type == "bicluster_additive":
        base = np.random.normal(0, 1, (rows, cols))
        mu = random.uniform(*cfg["bicluster"]["additive_mu_range"])
        noise = random.uniform(*cfg["bicluster"]["bicluster_noise_range"])
        shape_ij = (random.randint(3, max(3, rows // 2)), random.randint(3, max(3, cols // 2)))
        data, _, _ = inject_additive_coherent_bicluster(
            base, shape_ij, mu, noise
        )

    elif heatmap_type == "bicluster_multiplicative":
        base = np.random.normal(1.0, 0.5, (rows, cols))
        mu = random.uniform(*cfg["bicluster"]["additive_mu_range"])
        noise = random.uniform(*cfg["bicluster"]["bicluster_noise_range"])
        shape_ij = (random.randint(3, max(3, rows // 2)), random.randint(3, max(3, cols // 2)))
        data, _, _ = inject_multiplicative_coherent_bicluster(
            base, shape_ij, mu, noise
        )

    elif heatmap_type in STRUCTURAL_THEMES:
        data, structural_meta = _generate_structural_theme(heatmap_type, rows, cols)
        meta["structural_theme"] = heatmap_type
        meta["structural_meta"] = structural_meta

    elif heatmap_type == "perlin":
        res_min, res_max = cfg["spatial"]["res_range"]
        res = random.randint(res_min, res_max)
        data = generate_perlin_2d((rows, cols), (res, res))

    elif heatmap_type == "fractal":
        res_min, res_max = cfg["spatial"]["res_range"]
        res = random.randint(res_min, res_max)
        oct_min, oct_max = cfg["spatial"]["octaves_range"]
        octaves = random.randint(oct_min, oct_max)
        p_min, p_max = cfg["spatial"]["persistence_range"]
        persistence = random.uniform(p_min, p_max)
        data = generate_fractal_noise_2d(
            (rows, cols), (res, res), octaves=octaves, persistence=persistence
        )

    elif heatmap_type == "sarima":
        order = random.choice(cfg["sarima"]["order_choices"])
        seasonal_order = random.choice(cfg["sarima"]["seasonal_order_choices"])
        try:
            base = generate_sarima_matrix(rows, cols, order, seasonal_order)
            rho = random.uniform(*cfg["sarima"]["rho_range"])
            data = apply_sar_spatial_dependence(base, rho)
        except Exception:
            data = np.random.normal(0, 1, (rows, cols))

    elif heatmap_type == "confusion":
        dim = min(rows, cols)
        data = np.random.uniform(0, 10, (dim, dim))
        np.fill_diagonal(data, np.random.uniform(50, 100, dim))
        for i in range(dim - 1):
            data[i, i + 1] += np.random.uniform(10, 30)
            data[i + 1, i] += np.random.uniform(10, 30)
        data = data.astype(float)
        rows, cols = dim, dim

    else:
        data = generate_clustered_heatmap(rows, cols, num_clusters=5)

    data = np.asarray(data, dtype=float)

    if np.nanmin(data) < 0 < np.nanmax(data) and not meta["is_correlation"]:
        cmap_type = "diverging"

    # Apply missingness
    missing_cfg = cfg["missingness"]
    missing_mode = _pick_weighted_option(missing_cfg["mode_weights"], default="none")
    missing_mask = None
    missing_params = None
    if missing_mode != "none":
        if missing_mode == "mcar":
            rate = random.uniform(*missing_cfg["missing_rate_range"])
            data, missing_mask = inject_mcar(data, rate)
            missing_params = {"rate": float(rate)}
        elif missing_mode == "mnar_logistic":
            beta_0 = random.uniform(*missing_cfg["logistic_beta0_range"])
            beta_1 = random.uniform(*missing_cfg["logistic_beta1_range"])
            data, missing_mask = inject_mnar_logistic(data, beta_0, beta_1)
            missing_params = {"beta_0": float(beta_0), "beta_1": float(beta_1)}
        elif missing_mode == "mnar_quantile":
            q = random.uniform(*missing_cfg["quantile_range"])
            drop_prob = random.uniform(*missing_cfg["drop_prob_range"])
            data, missing_mask = inject_mnar_quantile_censorship(
                data, q=q, drop_prob=drop_prob, upper_bound=True
            )
            missing_params = {"quantile": float(q), "drop_prob": float(drop_prob)}
        else:
            llod_q = random.uniform(*missing_cfg["llod_quantile_range"])
            uloq_q = random.uniform(*missing_cfg["uloq_quantile_range"])
            jitter = missing_cfg.get("llod_uloq_jitter", (0.85, 1.15))
            min_span = missing_cfg.get("llod_uloq_min_span", 1e-6)
            data, missing_mask, det_meta = inject_mnar_detection_limits(
                data,
                llod_quantile=llod_q,
                uloq_quantile=uloq_q,
                jitter_range=jitter,
                min_span=min_span
            )
            missing_params = {
                "llod_quantile": float(llod_q),
                "uloq_quantile": float(uloq_q),
                "llod": det_meta.get("llod"),
                "uloq": det_meta.get("uloq")
            }
        meta["missingness"] = missing_mode
        meta["missingness_params"] = missing_params
        if missing_mask is not None:
            meta["missing_count"] = int(np.sum(missing_mask))

    if meta["is_correlation"] and missing_mask is not None:
        tri_mask = np.triu(missing_mask, 1)
        missing_mask = tri_mask | tri_mask.T
        np.fill_diagonal(missing_mask, False)
        data = data.copy()
        data[missing_mask] = np.nan

    # Apply noise
    noise_cfg = cfg["noise"]
    if not meta["is_correlation"] and random.random() < noise_cfg["gaussian_prob"]:
        std = random.uniform(*noise_cfg["gaussian_std_range"])
        data = data + np.random.normal(0, std, size=data.shape)
        meta["noise"].append("gaussian")

    if not meta["is_correlation"] and random.random() < noise_cfg["heteroscedastic_prob"]:
        hetero_mode = _pick_weighted_option(
            noise_cfg.get("heteroscedastic_mode_weights", {"magnitude_scaled": 100}),
            default="magnitude_scaled"
        )
        if hetero_mode == "parametric":
            alpha = random.uniform(*noise_cfg["parametric_alpha_range"])
            beta = random.uniform(*noise_cfg["parametric_beta_range"])
            gamma = random.uniform(*noise_cfg["parametric_gamma_range"])
            use_pareto = random.random() < noise_cfg.get("pareto_outlier_chance", 0.0)
            outlier_prob = noise_cfg.get("pareto_outlier_prob", 0.005)
            outlier_scale = random.uniform(*noise_cfg["pareto_outlier_scale_range"])
            data = inject_parametric_heteroscedastic(
                data,
                alpha,
                beta,
                gamma=gamma,
                pareto_outliers=use_pareto,
                outlier_prob=outlier_prob,
                outlier_scale=outlier_scale
            )
            meta["noise"].append("heteroscedastic_parametric")
            meta["noise_params"].append({
                "type": "parametric",
                "alpha": float(alpha),
                "beta": float(beta),
                "gamma": float(gamma),
                "pareto_outliers": bool(use_pareto),
                "outlier_prob": float(outlier_prob),
                "outlier_scale": float(outlier_scale)
            })
        else:
            omega = random.uniform(*noise_cfg["omega_range"])
            delta = random.uniform(*noise_cfg["delta_range"])
            data = inject_magnitude_scaled_gaussian(data, omega, delta)
            meta["noise"].append("heteroscedastic")
            meta["noise_params"].append({
                "type": "magnitude_scaled",
                "omega": float(omega),
                "delta": float(delta)
            })

    if not meta["is_correlation"] and random.random() < noise_cfg["poisson_prob"]:
        k_dispersion = random.uniform(*noise_cfg["k_dispersion_range"])
        data = inject_poisson_like_variance(data, k_dispersion)
        meta["noise"].append("poisson")

    if not meta["is_correlation"] and random.random() < noise_cfg["outlier_prob"]:
        data = _inject_heatmap_outliers(
            data,
            noise_cfg["outlier_fraction_range"],
            noise_cfg["outlier_scale_range"]
        )
        meta["noise"].append("outliers")

    if meta["is_correlation"]:
        noise_std = random.uniform(*cfg["correlation"]["noise_std_range"])
        if noise_std > 0:
            tri = np.triu(np.random.normal(0, noise_std, size=data.shape), 1)
            data = data + tri + tri.T
            meta["noise"].append("corr_gaussian")
        np.fill_diagonal(data, 1.0)
        data = np.clip(data, -1.0, 1.0)

    # Apply normalization (skip for correlation unless fixed bounds)
    norm_mode = None
    context_norm = context_cfg.get("normalization_strategy") if context_cfg else None
    if context_norm:
        norm_map = {
            "row_z_score": "row_zscore",
            "global_min_max": "global_minmax",
            "robust_iqr": "robust_iqr",
            "fixed_bounds_pm_1": "fixed_bounds_pm1",
            "log10_scale": "log10_scale",
            "none": "none"
        }
        norm_mode = norm_map.get(context_norm, context_norm)
    else:
        norm_mode = _pick_weighted_option(cfg["normalization_weights"], default="none")

    if meta["is_correlation"] and norm_mode not in ("none", "fixed_bounds_pm1"):
        norm_mode = "none"

    if norm_mode == "row_zscore":
        data = row_wise_zscore(data)
    elif norm_mode == "global_minmax":
        data = global_minmax(data)
    elif norm_mode == "robust_iqr":
        data = robust_iqr_scaler(data)
    elif norm_mode == "fixed_bounds_pm1":
        data = fixed_bounds_pm1(data)
    elif norm_mode == "log10_scale":
        data = log10_scale(data)

    meta["normalization"] = norm_mode

    # Apply ordering
    ordering_mode = _pick_weighted_option(cfg["ordering_weights"], default="none")
    fill_val = np.nanmedian(data)
    if not np.isfinite(fill_val):
        fill_val = 0.0
    data_for_order = np.nan_to_num(data, nan=fill_val)
    if ordering_mode == "olo" and data_for_order.shape[0] > 2 and data_for_order.shape[1] > 2:
        try:
            _, row_order, col_order = seriate_optimal_leaf_ordering(data_for_order)
            data = data[row_order, :][:, col_order]
            meta["ordering"] = "olo"
            meta["row_order"] = row_order
            meta["col_order"] = col_order
        except Exception:
            meta["ordering"] = "none"
    elif ordering_mode == "fiedler" and data_for_order.shape[0] > 2 and data_for_order.shape[1] > 2:
        try:
            _, row_order, col_order = seriate_spectral_fiedler(data_for_order)
            data = data[row_order, :][:, col_order]
            meta["ordering"] = "fiedler"
            meta["row_order"] = row_order
            meta["col_order"] = col_order
        except Exception:
            meta["ordering"] = "none"

    if meta["is_correlation"]:
        data = (data + data.T) / 2.0
        np.fill_diagonal(data, 1.0)
        data = np.clip(data, -1.0, 1.0)
    else:
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        if np.isfinite(min_val) and np.isfinite(max_val) and min_val < 0 < max_val:
            cmap_type = "diverging"
        else:
            cmap_type = "sequential"

    return data, cmap_type, meta


def generate_temporal_labels(n, label_type):
    """Generate varied temporal labels"""
    if 'quarter' in label_type.lower():
        return [f"Q{(i % 4) + 1}'{20 + i//4}" for i in range(n)]
    elif 'week' in label_type.lower():
        return [f"W{i+1}" for i in range(n)]
    elif 'day' in label_type.lower():
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        return [days[i % 7] for i in range(n)]
    elif 'hour' in label_type.lower() or '(h)' in label_type:
        return [f"{i}h" if i < 24 else f"{i}h" for i in range(n)]
    else:
        return [f"T{i+1}" for i in range(n)]

def generate_numeric_labels(n, label_type):
    """Generate numeric labels with units"""
    if 'dose' in label_type.lower() or 'concentration' in label_type.lower():
        base = np.logspace(-1, 2, n)
        return [f"{v:.1f}" for v in base]
    elif 'temperature' in label_type.lower():
        temps = np.linspace(20, 40, n)
        return [f"{t:.0f}°" for t in temps]
    else:
        return [f"{i*10}" for i in range(n)]

def generate_biological_labels(n, label_type):
    """Generate biological entity labels"""
    prefixes = ['BRCA', 'TP53', 'EGFR', 'MYC', 'KRAS', 'ALK', 'RET', 'MET']
    if 'gene' in label_type.lower():
        return [f"{random.choice(prefixes)}{random.randint(1,5)}" for _ in range(n)]
    elif 'protein' in label_type.lower():
        return [f"P{random.randint(10000,99999)}" for _ in range(n)]
    else:
        return [f"Bio{i+1}" for i in range(n)]

def generate_categorical_labels(n, label_type):
    """Generate business/categorical labels"""
    if 'product' in label_type.lower():
        return [f"SKU-{random.randint(1000,9999)}" for _ in range(n)]
    elif 'region' in label_type.lower():
        regions = ['North', 'South', 'East', 'West', 'Central']
        return [f"{random.choice(regions)}-{i+1}" for i in range(n)]
    else:
        return [f"Cat{i+1}" for i in range(n)]



def calculate_pie_geometry(wedges, ax, debug_mode=False):
    """
    Calculate geometric keypoints for pie chart.
    CRITICAL FIX: Use wedge.center (displaced center) for arc calculations.
    CRITICAL FIX: True original center is (0,0) in data coordinates.
    """
    if not wedges:
        return None
    
    if debug_mode:
        print(f"DEBUG [PIE_GEOM] Calculating geometry for {len(wedges)} wedges")
    
    # CRITICAL FIX: The true, non-exploded center of a matplotlib pie chart
    # drawn at the origin is (0.0, 0.0) in data coordinates.
    original_centerx = 0.0
    original_centery = 0.0
    
    wedge_geometry = []
    
    for idx, wedge in enumerate(wedges):
        #  Use this wedge's specific center.
        # This is (0,0) for non-exploded wedges and (dx, dy) for exploded ones.
        wedge_cx, wedge_cy = wedge.center 
        
        radius = wedge.r
        theta1 = np.deg2rad(wedge.theta1)
        theta2 = np.deg2rad(wedge.theta2)
        thetamid = (theta1 + theta2) / 2
        
        # Calculate intermediate arc sample points at 1/3 and 2/3 of the arc
        theta_inter_1 = theta1 + (theta2 - theta1) / 3.0
        theta_inter_2 = theta1 + 2 * (theta2 - theta1) / 3.0
        
        angle_span = wedge.theta2 - wedge.theta1
        num_arc_points = max(5, int(angle_span / 15))
        theta_samples = np.linspace(theta1, theta2, num_arc_points)
        
        arc_boundary_points = [
            (float(wedge_cx + radius * np.cos(theta)), 
             float(wedge_cy + radius * np.sin(theta)))
            for theta in theta_samples
        ]
        
        wedge_geometry.append({
            'wedge_idx': idx,
            'center': (float(wedge_cx), float(wedge_cy)),
            'original_center': (original_centerx, original_centery),
            'radius': float(radius),
            'theta1_deg': float(wedge.theta1),
            'theta2_deg': float(wedge.theta2),
            'arc_start': (
                float(wedge_cx + radius * np.cos(theta1)),
                float(wedge_cy + radius * np.sin(theta1))
            ),
            'arc_end': (
                float(wedge_cx + radius * np.cos(theta2)),
                float(wedge_cy + radius * np.sin(theta2))
            ),
            'arc_inter_1': (
                float(wedge_cx + radius * np.cos(theta_inter_1)),
                float(wedge_cy + radius * np.sin(theta_inter_1))
            ),
            'arc_inter_2': (
                float(wedge_cx + radius * np.cos(theta_inter_2)),
                float(wedge_cy + radius * np.sin(theta_inter_2))
            ),
            'arc_mid': (
                float(wedge_cx + radius * np.cos(thetamid)),
                float(wedge_cy + radius * np.sin(thetamid))
            ),
            'arc_boundary': arc_boundary_points,
            'wedge_label_point': (
                float(wedge_cx + radius * 0.7 * np.cos(thetamid)),
                float(wedge_cy + radius * 0.7 * np.sin(thetamid))
            ),
            'angle_span': float(angle_span)
        })
        
        if debug_mode:
            print(f"DEBUG [PIE_GEOM] Wedge {idx}: Center=({wedge_cx:.2f},{wedge_cy:.2f}), R={radius:.2f}")
            print(f"DEBUG [PIE_GEOM] Wedge {idx}: ArcStart=({wedge_geometry[-1]['arc_start'][0]:.2f},{wedge_geometry[-1]['arc_start'][1]:.2f})")
    
    return {
        'center_point': (original_centerx, original_centery), # Return TRUE center
        'wedges': wedge_geometry
    }


def extract_scale_axis_info(ax, chart_type_str):
    """
    Extract information about scale axes (primary and secondary) for the chart.
    """
    # Determine primary and secondary scale axes based on chart type and orientation
    if chart_type_str in ['bar', 'histogram']:
        # For bar charts, primary scale axis is typically the value axis (y-axis for vertical, x-axis for horizontal)
        if hasattr(ax, '_orientation') and ax._orientation == 'horizontal':
            primary_scale_axis = 'x'
            secondary_scale_axis = 'y' if ax.yaxis_inverted() else None
        else:
            primary_scale_axis = 'y'  # Default to y-axis for vertical bars
            secondary_scale_axis = 'x' if ax.xaxis_inverted() else None
    elif chart_type_str in ['line', 'area', 'scatter']:
        # For line charts, primary scale axis is typically y-axis
        primary_scale_axis = 'y'
        secondary_scale_axis = 'x'
    else:
        # Default for other chart types
        primary_scale_axis = 'y'
        secondary_scale_axis = None

    return {
        "primary_scale_axis": primary_scale_axis,
        "secondary_scale_axis": secondary_scale_axis
    }


def extract_bar_info(ax, chart_type_str):
    """
    Extract detailed information about bars in bar charts and histograms.
    """
    if chart_type_str not in ['bar', 'histogram']:
        return []

    if hasattr(ax, '_bar_info_list') and ax._bar_info_list:
        return ax._bar_info_list

    bar_info_list = []

    # Iterate through all patches in the axes to find bar rectangles
    for i, patch in enumerate(ax.patches):
        if hasattr(patch, 'get_xy') and hasattr(patch, 'get_width') and hasattr(patch, 'get_height'):
            x, y = patch.get_xy()
            width = patch.get_width()
            height = patch.get_height()

            # Calculate center based on orientation
            center_x = x + width / 2
            center_y = y + height / 2

            bar_info = {
                "center": float(center_x) if chart_type_str == 'histogram' else (float(center_x) if width > height else float(center_y)),
                "height": float(height),
                "width": float(width),
                "bottom": float(y),
                "top": float(y + height),
                "series_idx": 0,  # Would need to determine series in multi-series charts
                "bar_idx": i,
                "axis": "primary"  # Default to primary axis
            }

            # Determine if it's horizontal or vertical based on dimensions
            if chart_type_str == 'histogram':
                bar_info["orientation"] = "vertical"
                bar_info["x_value"] = float(center_x)
            elif width > height:
                bar_info["orientation"] = "horizontal"
            else:
                bar_info["orientation"] = "vertical"

            bar_info_list.append(bar_info)

    return bar_info_list


def extract_keypoint_info(ax, chart_type_str):
    """
    Extract keypoint information for line, area, and pie charts.
    """
    if chart_type_str not in ['line', 'area', 'pie']:
        return []

    keypoint_info_list = []

    # For line and area charts, extract line data points
    for i, line in enumerate(ax.lines):
        if hasattr(line, 'get_data'):
            x_data, y_data = line.get_data()

            # Get inflection points if possible
            inflection_indices = []
            if len(y_data) > 2:
                # Simple inflection detection (where second derivative changes sign)
                y_diff = np.diff(y_data)
                y_diff2 = np.diff(y_diff)
                inflection_indices = np.where(y_diff2[:-1] * y_diff2[1:] < 0)[0] + 1  # +1 because diff reduces length by 1

            points = []
            for j, (x, y) in enumerate(zip(x_data, y_data)):
                is_inflection = j in inflection_indices
                points.append({
                    "x": float(x),
                    "y": float(y),
                    "is_inflection": bool(is_inflection)
                })

            keypoint_info_list.append({
                "series_idx": i,
                "points": points
            })

    return keypoint_info_list


def extract_boxplot_metadata(ax, chart_type_str):
    """
    Extract metadata for boxplot charts.
    """
    if chart_type_str != 'box':
        return {}

    # Look for boxplot elements in the axes
    box_metadata = {
        "num_groups": 0,
        "box_width": 0.0,
        "orientation": "vertical",  # Default
        "medians": []
    }

    # Find boxplot elements by looking for specific artists
    median_lines = []
    for line in ax.lines:
        # Matplotlib boxplots typically have specific line styles for median lines
        if hasattr(line, 'get_color'):
            # Check if this might be a median line by its properties
            x_data, y_data = line.get_data()
            if len(x_data) == 2 and len(y_data) == 2:
                # A median line is typically a horizontal line segment
                if abs(y_data[0] - y_data[1]) < 0.01:  # Almost same y values
                    median_lines.append(line)

    medians = []
    for i, line in enumerate(median_lines):
        x_data, y_data = line.get_data()
        median_x = np.mean(x_data)
        median_y = np.mean(y_data)

        medians.append({
            "group_index": i,
            "group_label": f"Group_{i}",
            "median_value": float(median_y),
            "lower_left": {"x": float(min(x_data)), "y": float(min(y_data))},
            "upper_right": {"x": float(max(x_data)), "y": float(max(y_data))},
            "center_x": float(median_x),
            "center_y": float(median_y),
            "line_length": float(abs(x_data[1] - x_data[0]))
        })

    box_metadata["num_groups"] = len(medians)
    box_metadata["medians"] = medians

    return box_metadata


def extract_pie_geometry(ax, chart_type_str):
    """
    Extract geometric information for pie charts.
    """
    if chart_type_str != 'pie':
        return {}

    # Look for wedge patches which represent pie slices
    wedges = []
    for i, patch in enumerate(ax.patches):
        if hasattr(patch, 'theta1') and hasattr(patch, 'theta2'):
            # This is likely a wedge/pie slice
            center_x, center_y = patch.center
            radius = patch.r
            start_angle = patch.theta1
            end_angle = patch.theta2
            mid_angle = (start_angle + end_angle) / 2

            wedges.append({
                "wedge_index": i,
                "start_angle": float(start_angle),
                "end_angle": float(end_angle),
                "mid_angle": float(mid_angle),
                "percentage": float((end_angle - start_angle) / 360.0 * 100)
            })

    # For pie charts, the center is typically (0, 0) unless offset
    center_point = {"x": 0.0, "y": 0.0}
    if ax.patches:
        # Use the center of the first patch as the pie center
        first_patch = ax.patches[0]
        if hasattr(first_patch, 'center'):
            center_x, center_y = first_patch.center
            center_point = {"x": float(center_x), "y": float(center_y)}

    return {
        "center_point": center_point,
        "radius": float(radius if 'radius' in locals() else 0.5),
        "wedges": wedges
    }


