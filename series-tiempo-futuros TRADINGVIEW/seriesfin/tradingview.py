"""
Cliente mínimo para bajar velas históricas de TradingView por su websocket.

Reemplaza a tvDatafeed, que corta la conexión a los 5 segundos y falla con
"Connection timed out" cuando se piden muchas barras sin cuenta.
Funciona sin usuario ni clave (sesión anónima), suficiente para datos diarios.

    df = historico('OANDA:XAUUSD', n_barras=2000)
"""

import json
import random
import re
import string
import time

import pandas as pd

URL_WS = 'wss://data.tradingview.com/socket.io/websocket'
ORIGEN = 'https://data.tradingview.com'
MAX_BARRAS = 5000

INTERVALOS = {'1D': '1D', '1W': '1W', '1M': '1M', '1h': '60', '4h': '240',
              '15m': '15', '5m': '5', '1m': '1'}

_SEPARADOR = re.compile(r'~m~\d+~m~')
_ERRORES = {'symbol_error', 'series_error', 'critical_error', 'protocol_error'}


def _sesion(prefijo):
    return prefijo + ''.join(random.choices(string.ascii_lowercase, k=12))


def _empaquetar(funcion, parametros):
    cuerpo = json.dumps({'m': funcion, 'p': parametros}, separators=(',', ':'))
    return f'~m~{len(cuerpo)}~m~{cuerpo}'


def historico(simbolo, intervalo='1D', n_barras=5000, ajuste_roll=False, timeout=60):
    """
    Velas OHLCV de un símbolo de TradingView ('BOLSA:SIMBOLO', p. ej. 'COMEX:MGC1!').

    ajuste_roll: en futuros continuos pide la serie ajustada por vencimientos
                 (back-adjustment), sin los saltos de cada roll.

    Devuelve un DataFrame con índice de fechas en UTC y columnas
    Open, High, Low, Close, Volume.
    """
    from websocket import WebSocketTimeoutException, create_connection

    if ':' not in simbolo:
        raise ValueError(f"El símbolo debe ser 'BOLSA:SIMBOLO', no {simbolo!r}")
    n_barras = min(int(n_barras), MAX_BARRAS)
    tf = INTERVALOS.get(intervalo, intervalo)

    ws = create_connection(URL_WS, origin=ORIGEN, header=['User-Agent: Mozilla/5.0'],
                           timeout=timeout)
    try:
        cs = _sesion('cs_')
        config = {'symbol': simbolo, 'adjustment': 'splits', 'session': 'regular'}
        if ajuste_roll:
            config['backadjustment'] = 'default'

        for funcion, params in [
            ('set_auth_token', ['unauthorized_user_token']),   # sin cuenta
            ('chart_create_session', [cs, '']),
            ('resolve_symbol', [cs, 'sym_1', '=' + json.dumps(config, separators=(',', ':'))]),
            ('create_series', [cs, 's1', 's1', 'sym_1', tf, n_barras]),
        ]:
            ws.send(_empaquetar(funcion, params))

        velas = []
        limite = time.time() + timeout
        while True:
            if time.time() > limite:
                raise TimeoutError(f'TradingView no terminó de enviar {simbolo} en {timeout} s')
            try:
                crudo = ws.recv()
            except WebSocketTimeoutException:
                raise TimeoutError(f'TradingView no respondió para {simbolo} en {timeout} s')

            terminado = False
            for msg in filter(None, _SEPARADOR.split(crudo)):
                if msg.startswith('~h~'):          # latido: hay que devolverlo
                    ws.send(f'~m~{len(msg)}~m~{msg}')
                    continue
                try:
                    data = json.loads(msg)
                except ValueError:
                    continue
                tipo = data.get('m')
                if tipo in _ERRORES:
                    raise ValueError(f'TradingView ({tipo}) para {simbolo}: {data.get("p")}')
                if tipo in ('timescale_update', 'du'):
                    serie = data['p'][1].get('s1', {})
                    velas.extend(b['v'] for b in serie.get('s', []))
                elif tipo == 'series_completed':
                    terminado = True
            if terminado:
                break
    finally:
        ws.close()

    if not velas:
        raise ValueError(f'TradingView no devolvió velas para {simbolo}')

    df = pd.DataFrame([v[:6] + [0.0] * (6 - len(v[:6])) for v in velas],
                      columns=['ts', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df.index = pd.to_datetime(df.pop('ts'), unit='s', utc=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df.index.name = 'Date'
    return df.astype(float)


def a_fecha_diaria(indice_utc):
    """
    Lleva la hora de cada vela diaria a su fecha de negociación.

    TradingView fecha la vela diaria con la hora de apertura de la sesión: 09:30
    de Nueva York en acciones, pero 17:00-18:00 del día anterior en futuros y
    forex. Pasando a hora de Nueva York y sumando 7 horas, las dos caen en el día
    correcto (también sirve para bolsas de Europa y Asia).
    """
    ny = indice_utc.tz_convert('America/New_York') + pd.Timedelta(hours=7)
    return ny.normalize().tz_localize(None)
