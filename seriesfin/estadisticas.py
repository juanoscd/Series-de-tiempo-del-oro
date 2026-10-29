"""Estadísticas descriptivas de retornos, riesgo de cola y pruebas de normalidad."""

import numpy as np
import pandas as pd
from scipy import stats

from . import SEED


def calcular_stats(r):
    """r: retornos logarítmicos diarios (en decimal)."""
    mu, sigma = r.mean(), r.std()
    var = r.quantile(0.05)
    return {
        'media': mu,
        'vol_d': sigma,
        'vol_a': sigma * np.sqrt(252),
        'ret_acum': np.expm1(r.sum()),
        'skew': r.skew(),
        'kurt': r.kurtosis(),
        'sharpe': mu / sigma * np.sqrt(252),   # sin tasa libre de riesgo
        'var': var,
        'cvar': r[r <= var].mean(),
    }


# Lecturas rápidas de cada métrica

def lee_vol(v):
    if v < 0.15:
        return 'Baja'
    if v < 0.30:
        return 'Moderada (típica de índices y oro)'
    if v < 0.50:
        return 'Alta (similar a una acción volátil)'
    return 'Muy alta (especulativa)'


def lee_skew(s):
    if s < -0.5:
        return 'Negativa marcada: pesan más las caídas bruscas'
    if s < -0.1:
        return 'Ligeramente negativa'
    if s <= 0.1:
        return 'Aproximadamente simétrica'
    if s <= 0.5:
        return 'Ligeramente positiva'
    return 'Positiva marcada: pesan más los saltos al alza'


def lee_kurt(k):
    if k > 1:
        return 'Colas pesadas: más días extremos que en una normal'
    if k < -1:
        return 'Colas livianas'
    return 'Colas parecidas a la normal'


def lee_sharpe(s):
    if s < 0:
        return 'Negativo: el retorno medio fue negativo'
    if s < 0.5:
        return 'Bajo: poco retorno por unidad de riesgo'
    if s < 1:
        return 'Aceptable'
    if s < 2:
        return 'Bueno'
    return 'Muy alto (puede ser un periodo atípico)'


def lee_cvar(ratio):
    if ratio > 1.5:
        return f'Cola muy pesada (CVaR/VaR = {ratio:.2f})'
    if ratio > 1.3:
        return f'Cola moderadamente pesada (CVaR/VaR = {ratio:.2f})'
    return f'Cola parecida a la normal (CVaR/VaR = {ratio:.2f})'


def tabla_resumen(s, precio, multiplicador=1):
    """Filas [estadística, valor, interpretación] para la consola y el dashboard."""
    exposicion = precio * multiplicador
    perdida_var = abs(np.expm1(s['var'])) * exposicion
    perdida_cvar = abs(np.expm1(s['cvar'])) * exposicion
    signo = 'Ganó' if s['media'] > 0 else 'Perdió'

    return [
        ['Retorno medio diario', f"{s['media']:.4%}",
         f"{signo} valor en promedio (~{s['media'] * 252:.1%} anual)"],
        ['Volatilidad diaria', f"{s['vol_d']:.4%}",
         f"Un día típico se mueve ±{s['vol_d']:.2%}"],
        ['Volatilidad anualizada', f"{s['vol_a']:.2%}", lee_vol(s['vol_a'])],
        ['Retorno acumulado', f"{s['ret_acum']:.2%}",
         f"1 dólar al inicio → {1 + s['ret_acum']:.2f} al final"],
        ['Asimetría', f"{s['skew']:.4f}", lee_skew(s['skew'])],
        ['Curtosis (exceso)', f"{s['kurt']:.4f}", lee_kurt(s['kurt'])],
        ['Sharpe (aprox.)', f"{s['sharpe']:.4f}", lee_sharpe(s['sharpe'])],
        ['VaR diario 95%', f"{s['var']:.4%}",
         f"1 de cada 20 días se pierde {abs(s['var']):.2%} o más "
         f"(~${perdida_var:,.0f} por contrato)"],
        ['CVaR diario 95%', f"{s['cvar']:.4%}",
         f"{lee_cvar(s['cvar'] / s['var'])}; pérdida media ~${perdida_cvar:,.0f}"],
    ]


def pruebas_normalidad(r):
    """
    Compara la normal contra una t de Student: tests de normalidad, AIC,
    conteo de días extremos y VaR con cada distribución.
    """
    r = r.dropna()
    n = len(r)

    jb_stat, jb_p = stats.jarque_bera(r)
    muestra = r if n <= 5000 else r.sample(5000, random_state=SEED)  # Shapiro admite hasta 5000
    sh_stat, sh_p = stats.shapiro(muestra)

    mu, sigma = stats.norm.fit(r)
    gl, loc, escala = stats.t.fit(r)

    ll_normal = stats.norm.logpdf(r, mu, sigma).sum()
    ll_t = stats.t.logpdf(r, gl, loc, escala).sum()

    # días a más de 3 desviaciones de la media
    observados = int((np.abs(r - mu) > 3 * sigma).sum())
    esperados_normal = n * 2 * stats.norm.sf(3)
    esperados_t = n * (stats.t.cdf(mu - 3 * sigma, gl, loc, escala)
                       + stats.t.sf(mu + 3 * sigma, gl, loc, escala))

    tabla_var = pd.DataFrame(
        [{'Confianza': f'{1 - a:.0%}',
          'Histórico': r.quantile(a),
          'Normal': stats.norm.ppf(a, mu, sigma),
          't de Student': stats.t.ppf(a, gl, loc, escala)} for a in (0.05, 0.01)]
    ).set_index('Confianza')

    return {
        'n': n,
        'jb': (jb_stat, jb_p),
        'shapiro': (sh_stat, sh_p),
        'normal': (mu, sigma),
        't': (gl, loc, escala),
        'aic_normal': 2 * 2 - 2 * ll_normal,
        'aic_t': 2 * 3 - 2 * ll_t,
        'extremos': {'observados': observados,
                     'esperados_normal': esperados_normal,
                     'esperados_t': esperados_t},
        'tabla_var': tabla_var,
    }


def reporte_normalidad(res):
    jb_stat, jb_p = res['jb']
    sh_stat, sh_p = res['shapiro']
    gl = res['t'][0]
    ext = res['extremos']

    print(f"Jarque-Bera:  {jb_stat:,.2f} (p = {jb_p:.3g})")
    print(f"Shapiro-Wilk: {sh_stat:.4f} (p = {sh_p:.3g})")
    if min(jb_p, sh_p) < 0.05:
        print('Se rechaza normalidad: con la normal se subestima el riesgo de cola.')
    else:
        print('No se rechaza normalidad con esta muestra.')

    if gl < 5:
        lectura = 'colas muy pesadas'
    elif gl < 30:
        lectura = 'colas más pesadas que la normal'
    else:
        lectura = 'prácticamente normal'
    print(f"\nt de Student: {gl:.2f} grados de libertad ({lectura})")

    gana = 't' if res['aic_t'] < res['aic_normal'] else 'normal'
    print(f"AIC normal: {res['aic_normal']:,.1f} | AIC t: {res['aic_t']:,.1f} → mejor ajuste: {gana}")

    print(f"\nDías a más de 3σ: {ext['observados']} observados, "
          f"{ext['esperados_normal']:.1f} esperados con la normal, "
          f"{ext['esperados_t']:.1f} con la t")
