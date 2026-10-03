"""
themes.py

Visual styling profiles, publication presets, and typography configurations.
Provides aesthetic configurations (colors, spines, grids, fonts) for chart generation.
"""

THEMES = {
    'default': {
        'facecolor': 'white',
        'grid_color': '#CCCCCC',
        'grid_style': '--',
        'grid_linewidth': 0.8,
        'font': 'DejaVu Sans',
        'font_size': 10,
        'palette': 'viridis',
        'line_width': 1.5,
        'marker_size': 6,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'axis_color': '#333333',
        'legend_frame': False
    },

    'excel': {
        'facecolor': '#F2F2F2',
        'grid_color': 'white',
        'grid_style': '-',
        'grid_linewidth': 1.5,
        'font': 'Calibri',
        'font_size': 11,
        'palette': ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#70AD47'],
        'line_width': 2.0,
        'marker_size': 7,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'axis_color': '#1F4E79',
        'legend_frame': True,
        'legend_facecolor': 'white'
    },

    'ggplot': {
        'facecolor': '#EBEBEB',
        'grid_color': 'white',
        'grid_style': '-',
        'grid_linewidth': 1.0,
        'font': 'Arial',
        'font_size': 10,
        'palette': ['#F8766D', '#7CAE00', '#00BFC4', '#C77CFF'],
        'line_width': 1.25,
        'marker_size': 5,
        'spines': {'top': False, 'right': False, 'left': False, 'bottom': False},
        'axis_color': '#4D4D4D',
        'legend_frame': False
    },

    'prism': {
        'facecolor': 'white',
        'grid_color': '#E0E0E0',
        'grid_style': ':',
        'grid_linewidth': 1,
        'font': 'Arial',
        'font_size': 9,
        'palette': ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3'],
        'line_width': 1.0,
        'marker_size': 6,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'tick_direction': 'out',
        'legend_frame': False
    },
        'Rainbow': {
        'facecolor': 'white',
        'grid_color': '#E0E0E0',
        'grid_style': ':',
        'grid_linewidth': 1,
        'font': 'Arial',
        'font_size': 9,
        'palette': 'rainbow',
        'line_width': 1.0,
        'marker_size': 6,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'tick_direction': 'out',
        'legend_frame': False
    },

    'powerpoint': {
        'facecolor': 'white',
        'grid_color': '#D9D9D9',
        'grid_style': '-',
        'font': 'Calibri',
        'font_size': 12,
        'palette': ['#5A9BD5', '#ED7D31', '#70AD47', '#FFC000', '#4472C4', '#9E480E'],
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'add_shadows': True,
        'line_width': 2.0,
        'marker_size': 8,
        'legend_frame': True
    },

    'minimal': {
        'facecolor': 'white',
        'grid_color': None,
        'grid_style': 'none',
        'font': 'Helvetica',
        'font_size': 10,
        'palette': ['#222222'],
        'line_width': 1.2,
        'marker_size': 5,
        'spines': {'top': False, 'right': False, 'left': False, 'bottom': True},
        'axis_color': '#222222',
        'legend_frame': False
    },
    
    'pastel': {
        'facecolor': 'white',
        'grid_color': '#F4F4F4',
        'grid_style': '--',
        'font': 'Calibri',
        'font_size': 11,
        'palette': ['#A3C4DC', '#F6C4C9', '#FFD8A9', '#C7EFCF', '#D5C6E0'],
        'line_width': 1.0,
        'marker_size': 6,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'legend_frame': False
    },

    'colorblind_friendly': {
        'facecolor': 'white',
        'grid_color': '#EDEDED',
        'grid_style': '--',
        'font': 'Arial',
        'font_size': 10,
        'palette': ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7'],  # Okabe-Ito / Tol-like
        'line_width': 1.6,
        'marker_size': 7,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'legend_frame': False
    },

    'retro': {
        'facecolor': '#FFF8E7',
        'grid_color': '#E6D6B5',
        'grid_style': ':',
        'font': 'Georgia',
        'font_size': 11,
        'palette': ['#E4572E', '#17BEBB', '#FFC914', '#2E4057'],
        'line_width': 1.8,
        'marker_size': 6,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'axis_color': '#3A3A3A',
        'legend_frame': True,
        'legend_facecolor': '#FFF8E7'
    },

    'seaborn_like': {
        'facecolor': '#F5F5F5',
        'grid_color': '#EDEDED',
        'grid_style': '-',
        'font': 'Helvetica',
        'font_size': 10,
        'palette': 'tab10',
        'line_width': 1.4,
        'marker_size': 6,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'axis_color': '#333333',
        'legend_frame': False
    },

    'presentation_bold': {
        'facecolor': 'white',
        'grid_color': '#DDDDDD',
        'grid_style': '-',
        'font': 'Montserrat',
        'font_size': 14,
        'palette': ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600'],
        'line_width': 3.0,
        'marker_size': 10,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'legend_frame': True
    },

    'clinical': {
        'facecolor': 'white',
        'grid_color': '#F0F0F0',
        'grid_style': '--',
        'font': 'Arial',
        'font_size': 9,
        'palette': ['#1f77b4', '#ff7f0e', '#2ca02c'],
        'line_width': 1.2,
        'marker_size': 5,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'errorbar_capsize': 3,
        'legend_frame': False
    },

    'heatmap': {
        'facecolor': 'white',
        'grid_color': None,
        'grid_style': 'none',
        'font': 'DejaVu Sans',
        'font_size': 9,
        'palette': 'inferno',
        'line_width': 0.6,
        'spines': {'top': False, 'right': False, 'left': False, 'bottom': False},
        'colorbar': {'orientation': 'vertical', 'shrink': 0.8},
        'legend_frame': False
    },

    'corporate': {
        'facecolor': '#FAFAFA',
        'grid_color': '#E8E8E8',
        'grid_style': '-',
        'font': 'Roboto',
        'font_size': 11,
        'palette': ['#0D47A1', '#1976D2', '#42A5F5', '#90CAF9'],
        'line_width': 1.5,
        'marker_size': 6,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'legend_frame': True,
        'legend_facecolor': '#FFFFFF'
    },

    'map_atlas': {
        'facecolor': '#F8F8F8',
        'grid_color': '#EDEDED',
        'grid_style': '--',
        'font': 'Liberation Sans',
        'font_size': 9,
        'palette': ['#2b8cbe', '#7bccc4', '#edf8b1'],
        'line_width': 0.8,
        'spines': {'top': False, 'right': False, 'left': False, 'bottom': False},
        'legend_frame': False
    },

    'accessible_compact': {
        'facecolor': 'white',
        'grid_color': '#EAEAEA',
        'grid_style': '--',
        'font': 'Arial',
        'font_size': 12,
        'palette': ['#000000', '#1B9E77', '#D95F02', '#7570B3', '#E7298A'],
        'line_width': 2.0,
        'marker_size': 8,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'axis_color': "#00000032",
        'legend_frame': True,
        'contrast_enhanced': True
    },
    'high_contrast_qualitative': {
        'facecolor': 'white',
        'grid_color': '#E0E0E0',
        'grid_style': '-',
        'font': 'Arial',
        'font_size': 11,
        'palette': 'Set1',
        'line_width': 1.8,
        'marker_size': 7,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'legend_frame': False
    },

        'pastel_tones1': {
        'facecolor': '#FEFEFE',
        'grid_color': '#F0F0F0',
        'grid_style': '--',
        'font': 'Calibri',
        'font_size': 11,
        'palette': 'Pastel1',
        'line_width': 1.5,
        'marker_size': 6,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'legend_frame': False
    },

    'pastel_tones2': {
        'facecolor': '#FEFEFE',
        'grid_color': '#F0F0F0',
        'grid_style': '--',
        'font': 'Calibri',
        'font_size': 11,
        'palette': 'Pastel2',
        'line_width': 1.5,
        'marker_size': 6,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'legend_frame': False
    },

    'scientific_diverging': {
        'facecolor': 'white',
        'grid_color': '#DDDDDD',
        'grid_style': ':',
        'font': 'Helvetica',
        'font_size': 10,
        'palette': 'coolwarm',
        'line_width': 1.5,
        'marker_size': 6,
        'spines': {'top': False, 'right': False, 'left': True, 'bottom': True},
        'legend_frame': False
    },
}

PUBLICATION_THEMES = {
    'nature': {
        'font': 'Arial',
        'font_size': 7,
        'title_size': 8,
        'label_size': 7,
        'line_width': 0.5,
        'marker_size': 3,
        'palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
        'grid_style': 'none',
        'spine_width': 0.5,
        'figure_size': (3.3, 2.4),  # inches (single column)
        'dpi': 300,
        'legend_frame': False
    },

    'science': {
        'font': 'Helvetica',
        'font_size': 6,
        'title_size': 7,
        'label_size': 6,
        'line_width': 0.75,
        'marker_size': 3,
        'palette': ['#E31A1C', '#1F78B4', '#33A02C'],
        'grid_style': 'none',
        'spine_width': 0.75,
        'figure_size': (3.5, 2.5),
        'dpi': 300,
        'legend_frame': False
    },

    'cell': {
        'font': 'Helvetica',
        'font_size': 8,
        'title_size': 9,
        'label_size': 8,
        'line_width': 0.6,
        'marker_size': 3,
        'palette': ['#0072B2', '#D55E00', '#009E73'],
        'grid_style': 'none',
        'spine_width': 0.6,
        'figure_size': (6.5, 4.0),
        'dpi': 300,
        'legend_frame': True
    },

    'lancet': {
        'font': 'Times New Roman',
        'font_size': 8,
        'title_size': 9,
        'label_size': 8,
        'line_width': 0.8,
        'marker_size': 3,
        'palette': ['#000000', '#666666'],
        'grid_style': 'none',
        'spine_width': 0.8,
        'figure_size': (7.0, 4.5),
        'dpi': 300,
        'legend_frame': False
    },

    'pnas': {
        'font': 'Times New Roman',
        'font_size': 8,
        'title_size': 9,
        'label_size': 8,
        'line_width': 0.7,
        'marker_size': 3,
        'palette': ['#1f77b4', '#ff7f0e', '#2ca02c'],
        'grid_style': 'none',
        'spine_width': 0.7,
        'figure_size': (5.0, 3.5),
        'dpi': 300,
        'legend_frame': False
    },

    'bmj': {
        'font': 'Arial',
        'font_size': 8,
        'title_size': 9,
        'label_size': 8,
        'line_width': 0.8,
        'marker_size': 3,
        'palette': ['#2C7FB8', '#7FCDBB', '#EDF8B1'],
        'grid_style': 'none',
        'spine_width': 0.8,
        'figure_size': (6.0, 4.0),
        'dpi': 300,
        'legend_frame': False
    },

    'nanotech_short': {
        'font': 'Helvetica',
        'font_size': 7,
        'title_size': 8,
        'label_size': 7,
        'line_width': 0.6,
        'marker_size': 2.5,
        'palette': ['#4E79A7', '#F28E2B', '#E15759'],
        'grid_style': 'none',
        'spine_width': 0.5,
        'figure_size': (3.2, 2.4),
        'dpi': 600,
        'legend_frame': False
    },

    'open_access_poster': {
        'font': 'Arial',
        'font_size': 10,
        'title_size': 12,
        'label_size': 10,
        'line_width': 1.0,
        'marker_size': 4,
        'palette': ['#1b9e77', '#d95f02', '#7570b3'],
        'grid_style': '--',
        'spine_width': 0.7,
        'figure_size': (8.0, 6.0),
        'dpi': 300,
        'legend_frame': True
    },

    'print_high_contrast': {
        'font': 'Times New Roman',
        'font_size': 8,
        'title_size': 9,
        'label_size': 8,
        'line_width': 1.0,
        'marker_size': 3,
        'palette': ['#000000', '#666666', '#AAAAAA'],
        'grid_style': 'none',
        'spine_width': 1.0,
        'figure_size': (6.0, 4.0),
        'dpi': 600,
        'legend_frame': False
    }

}

# Adicionado com base na análise para Diversidade de Tipografia
FONT_FAMILIES = {
    'sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'serif': ['Times New Roman', 'Georgia', 'DejaVu Serif'],
    'monospace': ['Courier New', 'DejaVu Sans Mono'], # Para rótulos de dados
}
