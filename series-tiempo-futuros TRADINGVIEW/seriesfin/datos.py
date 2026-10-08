"""Descarga desde TradingView (tvDatafeed) y limpieza básica."""

import os
import re
import time
import warnings

import numpy as np
import pandas as pd
from tvDatafeed import Interval, TvDatafeed

# Dólares por punto de precio de cada contrato (para pasar el VaR a USD).
# La clave es la raíz del símbolo: COMEX:MGC1! → MGC. Acciones y ETF usan 1.
MULTIPLICADORES = {
    'MGC': 10,    # micro oro: 10 onzas
    'GC': 100,    # oro: 100 onzas
    'MNQ': 2,     # micro Nasdaq 100
    'NQ': 20,
    'MES': 5,     # micro S&P 500
    'ES': 50,
}

MAX_BARRAS = 5000   # tope de TradingView por consulta
_cliente = None


def _conectar():
    """Una sola sesión por kernel. Con TV_USUARIO y TV_CLAVE en el entorno se inicia sesión."""
    global _cliente
    if _cliente is None:
        usuario, clave = os.getenv('TV_USUARIO'), os.getenv('TV_CLAVE')
        _cliente = TvDatafeed(usuario, clave) if usuario and clave else TvDatafeed()
    return _cliente


def _separar(ticker):
    """'COMEX:MGC1!' → ('COMEX', 'MGC1!')."""
    if ':' not in ticker:
        raise ValueError(f"Usa el formato BOLSA:SÍMBOLO de TradingView (p. ej. 'COMEX:MGC1!'), "
                         f"no '{ticker}'")
    bolsa, simbolo = ticker.split(':', 1)
    return bolsa, simbolo


def multiplicador(ticker):
    raiz = re.sub(r'\d+!$', '', ticker.split(':')[-1])
    return MULTIPLICADORES.get(raiz, 1)


def _fecha_sesion(indice):
    """
    tvDatafeed entrega la hora de apertura de la barra en la hora local del equipo.
    En futuros la sesión diaria abre la tarde anterior (17:00 CT), así que la fecha
    sin corregir queda un día antes. Se pasa a hora de Nueva York y, si la barra
    abre después de las 15:00, se asigna al día siguiente.
    """
    utc = pd.to_datetime([time.mktime(t.timetuple()) for t in indice], unit='s', utc=True)
    ny = utc.tz_convert('America/New_York').tz_localize(None)
    fechas = ny.normalize() + pd.to_timedelta(np.where(ny.hour >= 15, 1, 0), unit='D')
    return pd.DatetimeIndex(fechas, name='Date')


def _historico(ticker, n_barras, intentos=3):
    bolsa, simbolo = _separar(ticker)
    tv = _conectar()
    for _ in range(intentos):
        df = tv.get_hist(symbol=simbolo, exchange=bolsa, interval=Interval.in_daily,
                         n_bars=n_barras)
        if df is not None and not df.empty:
            return df
        time.sleep(1)
    raise ValueError(f'TradingView no devolvió datos para {ticker}')


def _descargar_uno(ticker, inicio, fin=None):
    inicio = pd.Timestamp(inicio)
    n_barras = min(len(pd.bdate_range(inicio, pd.Timestamp.today())) + 50, MAX_BARRAS)

    df = _historico(ticker, n_barras)
    df = df.drop(columns='symbol', errors='ignore')
    df.columns = [c.capitalize() for c in df.columns]   # open → Open, etc.
    df.index = _fecha_sesion(df.index)
    df = df[~df.index.duplicated(keep='last')].sort_index()

    if df.index[0] > inicio + pd.Timedelta(days=7):
        warnings.warn(f'{ticker}: TradingView solo entregó datos desde {df.index[0].date()} '
                      f'(máximo {MAX_BARRAS} barras por consulta)')

    df = df.loc[df.index >= inicio]
    if fin is not None:
        df = df.loc[df.index < pd.Timestamp(fin)]   # fin excluido
    return df


def descargar(tickers, inicio, fin=None):
    """
    OHLCV diario. Los tickers van como BOLSA:SÍMBOLO ('COMEX:MGC1!', 'AMEX:GLD').
    Con un solo ticker (str) las columnas son Open, High, Low, Close, Volume;
    con una lista quedan como (campo, ticker) y las fechas se alinean por unión.
    """
    if isinstance(tickers, str):
        return _descargar_uno(tickers, inicio, fin)

    datos = {t: _descargar_uno(t, inicio, fin) for t in tickers}
    df = pd.concat(datos, axis=1).swaplevel(axis=1).sort_index(axis=1)
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
    raw = descargar(list(tickers), inicio, fin)

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
