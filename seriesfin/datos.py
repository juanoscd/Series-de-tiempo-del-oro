"""Descarga desde Yahoo Finance y limpieza básica."""

import numpy as np
import pandas as pd
import yfinance as yf

# Dólares por punto de precio de cada contrato (para pasar el VaR a USD).
# Para acciones o ETF el multiplicador es 1.
MULTIPLICADORES = {
    'MGC=F': 10,   # micro oro: 10 onzas
    'GC=F': 100,   # oro: 100 onzas
    'MNQ=F': 2,    # micro Nasdaq 100
    'NQ=F': 20,
    'MES=F': 5,    # micro S&P 500
    'ES=F': 50,
}


def descargar(tickers, inicio, fin=None):
    """OHLCV ajustado. Con varios tickers las columnas quedan como (campo, ticker)."""
    df = yf.download(tickers, start=inicio, end=fin, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f'Yahoo no devolvió datos para {tickers}')
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def limpiar(df, z_umbral=6.0):
    """
    Limpia el OHLCV de un solo ticker y agrega retornos logarítmicos.

    - quita fechas duplicadas (se queda con la última)
    - rellena nulos solo hacia adelante, para no usar información futura
    - marca como outlier los retornos con z robusto (MAD) mayor a `z_umbral`.
      No se eliminan: en futuros continuos suelen ser saltos por roll.

    Devuelve (df_limpio, reporte).
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    reporte = {'filas_originales': len(df)}

    df = df.apply(pd.to_numeric, errors='coerce').sort_index()
    reporte['duplicados'] = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep='last')]

    reporte['nulos'] = int(df.isna().sum().sum())
    n = len(df)
    df = df.ffill().dropna()
    reporte['filas_descartadas'] = n - len(df)

    df['log_ret'] = np.log(df['Close']).diff()
    mediana = df['log_ret'].median()
    mad = (df['log_ret'] - mediana).abs().median()
    z = 0.6745 * (df['log_ret'] - mediana) / mad
    df['outlier_ret'] = z.abs() > z_umbral

    reporte['outliers_marcados'] = int(df['outlier_ret'].sum())
    reporte['filas_finales'] = len(df)
    reporte['desde'] = df.index[0].date()
    reporte['hasta'] = df.index[-1].date()
    return df, reporte


def cargar_sistema(tickers, principal, inicio, fin=None):
    """
    Precios de cierre de varios tickers en fechas comunes, OHLC del principal
    y retornos logarítmicos en % (la escala en % ayuda a que GARCH converja).
    """
    raw = descargar(tickers, inicio, fin)

    precios = raw['Close'][tickers].dropna()
    precios = precios[~precios.index.duplicated(keep='last')]

    ohlc = pd.DataFrame({c: raw[c][principal] for c in ['Open', 'High', 'Low', 'Close']})
    ohlc = ohlc.dropna()

    retornos = np.log(precios).diff().dropna() * 100
    return precios, ohlc, retornos


def resumen_precios(precios, retornos):
    filas = []
    for t in precios.columns:
        p0, p1 = precios[t].iloc[0], precios[t].iloc[-1]
        filas.append({
            'Ticker': t,
            'Inicial': round(p0, 2),
            'Final': round(p1, 2),
            'Retorno (%)': round((p1 / p0 - 1) * 100, 1),
            'Vol. anual (%)': round(retornos[t].std() * np.sqrt(252), 1),
        })
    return pd.DataFrame(filas).set_index('Ticker')
