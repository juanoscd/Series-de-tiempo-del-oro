"""ARIMA sobre el precio: selección de orden, diagnóstico y pronóstico."""

import warnings

import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA


def dividir(serie, fecha_corte):
    corte = pd.Timestamp(fecha_corte)
    return serie.loc[serie.index < corte], serie.loc[serie.index >= corte]


def _ajustar(y, orden):
    # Se ajusta con índice entero: con fechas de días hábiles sin frecuencia
    # statsmodels no puede pronosticar fuera de muestra.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ARIMA(y.reset_index(drop=True), order=orden).fit()


def seleccionar_orden(train, p_max=3, q_max=3, d=1):
    """Rejilla (p, d, q) ordenada por AIC. Devuelve (tabla, mejor_orden)."""
    filas = []
    for p in range(p_max + 1):
        for q in range(q_max + 1):
            if p == 0 and q == 0:
                continue
            try:
                m = _ajustar(train, (p, d, q))
            except Exception:
                continue
            filas.append({'p': p, 'd': d, 'q': q, 'AIC': m.aic, 'BIC': m.bic})

    tabla = pd.DataFrame(filas).sort_values('AIC').reset_index(drop=True)
    mejor = tuple(int(tabla.loc[0, c]) for c in ('p', 'd', 'q'))
    return tabla.round(2), mejor


def ajustar(train, orden):
    return _ajustar(train, orden)


def coeficientes(modelo):
    return pd.DataFrame({'coef': modelo.params, 'p-valor': modelo.pvalues}).round(4)


def residuos(modelo, train, orden):
    """Residuos con fechas. Con d=1 el primero es el precio inicial completo, se descarta."""
    d = orden[1]
    resid = modelo.resid.iloc[d:].copy()
    resid.index = train.index[d:]
    return resid


def ljung_box(resid, orden, lags=(5, 10, 20)):
    """Ljung-Box con los grados de libertad corregidos por los p + q parámetros ARMA."""
    model_df = orden[0] + orden[2]
    lags = [l for l in lags if l > model_df]
    return acorr_ljungbox(resid, lags=lags, model_df=model_df)


def pronosticar(modelo, test, h_fut, niveles=(80, 95, 99)):
    """
    Pronóstico dinámico sobre todo el periodo de prueba más `h_fut` días hábiles
    después del último dato. Devuelve un DataFrame con la media y las bandas.
    """
    h = len(test) + h_fut
    fc = modelo.get_forecast(steps=h)

    fechas_fut = pd.bdate_range(test.index[-1] + pd.offsets.BDay(1), periods=h_fut)
    fechas = test.index.append(fechas_fut)

    pron = pd.DataFrame({'media': fc.predicted_mean.values}, index=fechas)
    for nivel in niveles:
        ic = fc.conf_int(alpha=1 - nivel / 100)
        pron[f'inf_{nivel}'] = ic.iloc[:, 0].values
        pron[f'sup_{nivel}'] = ic.iloc[:, 1].values
    return pron
