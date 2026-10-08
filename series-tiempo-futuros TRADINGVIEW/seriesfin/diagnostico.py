"""Estacionariedad, autocorrelación y efecto ARCH."""

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf


def probar_estacionariedad(x, nombre=''):
    """
    ADF  (H0: raíz unitaria)  → p < 0.05 apoya estacionariedad
    KPSS (H0: estacionaria)   → p > 0.05 apoya estacionariedad
    """
    x = pd.Series(x).dropna()
    p_adf = adfuller(x, autolag='AIC')[1]
    with warnings.catch_warnings():
        # KPSS avisa cuando el p-valor queda fuera de su tabla (se reporta 0.01 o 0.1)
        warnings.simplefilter('ignore')
        p_kpss = kpss(x, regression='c', nlags='auto')[1]

    adf_ok, kpss_ok = p_adf < 0.05, p_kpss > 0.05
    if adf_ok and kpss_ok:
        veredicto = 'Estacionaria'
    elif not adf_ok and not kpss_ok:
        veredicto = 'No estacionaria'
    else:
        veredicto = 'Ambigua'

    return {'Serie': nombre, 'ADF p': round(p_adf, 4),
            'KPSS p': round(p_kpss, 4), 'Veredicto': veredicto}


def tabla_estacionariedad(precio, retornos, nombre):
    filas = [
        probar_estacionariedad(precio, f'{nombre} precio'),
        probar_estacionariedad(precio.diff(), f'{nombre} Δ precio'),
        probar_estacionariedad(retornos, f'{nombre} retorno log'),
        probar_estacionariedad(retornos ** 2, f'{nombre} retorno²'),
    ]
    return pd.DataFrame(filas).set_index('Serie')


def autocorrelaciones(r, nlags=30):
    """ACF y PACF de los retornos y ACF de los retornos al cuadrado."""
    r = r.dropna()
    return {
        'acf': acf(r, nlags=nlags),
        'pacf': pacf(r, nlags=nlags),
        'acf_cuadrado': acf(r ** 2, nlags=nlags),
        'banda': 1.96 / np.sqrt(len(r)),
        'nlags': nlags,
    }


def rezagos_significativos(ac):
    banda = ac['banda']
    return {
        'retornos': int((np.abs(ac['acf'][1:]) > banda).sum()),
        'retornos²': int((np.abs(ac['acf_cuadrado'][1:]) > banda).sum()),
    }


def prueba_arch(r, nlags=10):
    """ARCH-LM de Engle. H0: no hay heterocedasticidad condicional."""
    lm, p_lm, _, _ = het_arch(r.dropna(), nlags=nlags)
    return lm, p_lm
