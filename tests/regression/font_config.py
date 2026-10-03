"""Font determinism configuration for regression testing.

Pins DejaVu Sans and disables hinting to guarantee reproducible text rendering.
"""
import contextlib
import matplotlib as mpl
import matplotlib.font_manager as fm

# Matplotlib bundles DejaVu Sans by default
DEJAVU_SANS_PATH = fm.findfont('DejaVu Sans')


def configure_deterministic_fonts():
    """Pin DejaVu Sans and disable font hinting globally."""
    try:
        fm.fontManager.addfont(DEJAVU_SANS_PATH)
    except Exception:
        pass
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
    mpl.rcParams['text.hinting'] = 'none'


@contextlib.contextmanager
def deterministic_font_context():
    """Context manager for deterministic font rendering."""
    with mpl.rc_context({
        'font.family': 'DejaVu Sans',
        'font.sans-serif': ['DejaVu Sans'],
        'text.hinting': 'none',
    }):
        yield
