"""VAR sobre los retornos de varios activos: rezagos, Granger, impulso-respuesta."""

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


def _var(retornos):
    # índice entero para evitar los avisos de frecuencia de statsmodels
    return VAR(retornos.reset_index(drop=True))


def seleccionar_rezagos(retornos, maxlags=10):
    sel = _var(retornos).select_order(maxlags=maxlags)
    tabla = pd.Series(sel.selected_orders, name='Rezagos').to_frame()
    tabla.index = tabla.index.str.upper()
    return tabla, max(1, int(sel.aic))


def ajustar(retornos, rezagos):
    return _var(retornos).fit(rezagos)


def matriz_granger(modelo):
    """p-valores. H0: la columna no causa en el sentido de Granger a la fila."""
    nombres = list(modelo.names)
    m = pd.DataFrame(np.nan, index=nombres, columns=nombres)
    for destino in nombres:
        for causa in nombres:
            if destino != causa:
                m.loc[destino, causa] = modelo.test_causality(destino, [causa], kind='f').pvalue
    return m.round(4)


def impulso_respuesta(retornos, rezagos, shock, pasos=15):
    """
    IRF ortogonalizada (shock de 1 desviación estándar en `shock`).

    Con Cholesky el resultado depende del orden de las variables: la primera
    afecta a las demás el mismo día y no al revés. Por eso se calcula también
    con el orden invertido; si las dos curvas se parecen, el orden no importa mucho.
    """
    tickers = list(retornos.columns)
    salida = {}
    for etiqueta, orden in (('original', tickers), ('invertido', tickers[::-1])):
        irf = ajustar(retornos[orden], rezagos).irf(pasos).orth_irfs
        j = orden.index(shock)
        salida[etiqueta] = pd.DataFrame({t: irf[:, orden.index(t), j] for t in tickers})
    return salida


def correlacion_residuos(modelo):
    return float(np.asarray(modelo.resid_corr)[0, 1])


def pronosticar(modelo, retornos, pasos=10):
    y0 = retornos.values[-modelo.k_ar:]
    pron = modelo.forecast(y0, steps=pasos)
    return pd.DataFrame(pron, columns=retornos.columns,
                        index=pd.RangeIndex(1, pasos + 1, name='Día'))
