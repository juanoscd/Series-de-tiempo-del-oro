"""
Pronóstico ARIMA animado en Streamlit.

    streamlit run app_arima_vivo.py

Se corre desde la carpeta del proyecto para que encuentre `seriesfin`.
"""

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from seriesfin import SEED, animado, arima, datos

st.set_page_config(page_title='ARIMA en vivo', layout='wide')

with st.sidebar:
    st.header('Parámetros')
    ticker = st.text_input('Ticker (BOLSA:SÍMBOLO)', 'OANDA:XAUUSD')
    inicio = st.date_input('Inicio de los datos', pd.Timestamp('2017-01-01'))
    corte = st.date_input('Inicio del test', pd.Timestamp.today() - pd.offsets.BDay(7))
    h_fut = st.slider('Días hábiles a proyectar después de hoy', 1, 30, 10)
    n_caminos = st.slider('Caminos simulados', 0, 400, 300, step=50)
    dias_antes = st.slider('Días de velas antes del corte', 20, 250, 60, step=10)
    auto = st.toggle('Elegir (p, d, q) por AIC', value=True)
    if not auto:
        p = st.number_input('p', 0, 5, 2)
        q = st.number_input('q', 0, 5, 2)


@st.cache_data(ttl=3600, show_spinner='Descargando de TradingView…')
def cargar(ticker, inicio):
    df = datos.descargar(ticker, str(inicio))
    return df[['Open', 'High', 'Low', 'Close']].dropna()


@st.cache_data(show_spinner='Ajustando ARIMA y simulando caminos…')
def modelar(ohlc, corte, h_fut, n_caminos, orden=None):
    train, test = arima.dividir(ohlc['Close'], str(corte))
    if orden is None:
        _, orden = arima.seleccionar_orden(train)
    modelo = arima.ajustar(train, orden)
    pron = arima.pronosticar(modelo, test, h_fut)
    caminos = animado.simular_caminos(modelo, len(pron), n=n_caminos, semilla=SEED) if n_caminos else None
    return pron, caminos, orden, len(train), len(test)


ohlc = cargar(ticker, inicio)
if pd.Timestamp(corte) > ohlc.index[-1]:
    st.error(f'El corte debe ser como máximo {ohlc.index[-1].date()}, el último dato disponible.')
    st.stop()

pron, caminos, orden, n_train, n_test = modelar(ohlc, corte, h_fut, n_caminos,
                                                None if auto else (int(p), 1, int(q)))

html = animado.pronostico_arima_vivo(ohlc, pron, corte, orden, ticker, caminos,
                                     dias_antes=dias_antes, mostrar=False)
components.html(html, height=640)

st.caption(f'ARIMA{orden} · entrenamiento: {n_train} días · prueba: {n_test} días · '
           f'los caminos salen del mismo modelo que las bandas.')
