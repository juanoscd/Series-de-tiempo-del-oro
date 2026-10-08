"""Paleta de colores y estilo común para todas las figuras."""

from pathlib import Path

import matplotlib.pyplot as plt

BLACK = '#000000'   # fondo
PURPLE = '#A855F7'  # color principal
LILAC = '#C4A0FF'   # serie histórica
CYAN = '#22D3EE'    # pronóstico
RED = '#FF3B5C'     # volatilidad / riesgo
GREEN = '#22C55E'   # alza
GOLD = '#E8B44A'    # líneas de referencia
WHITE = '#FFFFFF'
GRID = '#2A1B4A'

CARPETA_FIGURAS = Path('figuras')


def aplicar_estilo():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': BLACK,
        'axes.facecolor': BLACK,
        'savefig.facecolor': BLACK,
        'axes.edgecolor': GRID,
        'axes.labelcolor': WHITE,
        'axes.titlecolor': WHITE,
        'xtick.color': WHITE,
        'ytick.color': WHITE,
        'text.color': WHITE,
        'grid.color': GRID,
        'grid.alpha': 0.6,
        'legend.facecolor': BLACK,
        'legend.edgecolor': GRID,
        'font.family': 'DejaVu Sans',
        'axes.titlesize': 14,
        'axes.labelsize': 11,
    })


def guardar(fig, nombre):
    """Guarda la figura en figuras/<nombre>.png (si nombre es None no hace nada)."""
    if not nombre:
        return
    CARPETA_FIGURAS.mkdir(exist_ok=True)
    fig.savefig(CARPETA_FIGURAS / f'{nombre}.png', dpi=150, bbox_inches='tight')
