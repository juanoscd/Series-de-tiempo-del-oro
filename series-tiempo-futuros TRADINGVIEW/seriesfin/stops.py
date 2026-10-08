"""
Plan de stop loss y tamaño de posición para mañana a partir del GARCH.

La idea: el stop se pone fuera del ruido normal del mercado (medido con la
volatilidad pronosticada) y el número de contratos se calcula a partir de ese
stop y del riesgo máximo por operación, no al revés.
"""

import textwrap

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .estilo import CYAN, GOLD, GREEN, PURPLE, RED, WHITE, guardar as _guardar

DIAS = 252
HORAS_SESION = 23          # el oro en CME negocia ~23 horas al día

CONTRATOS = {
    'MGC': {'multiplicador': 10, 'tick': 0.10, 'nombre': 'Micro oro (10 oz)'},
    'GC': {'multiplicador': 100, 'tick': 0.10, 'nombre': 'Oro (100 oz)'},
}

# rango esperado (máximo − mínimo) de un movimiento browniano: 2·√(2/π)·σ ≈ 1.6σ
FACTOR_RANGO = 2 * np.sqrt(2 / np.pi)


def atr(ohlc, n=14):
    """Average True Range de Wilder, en puntos de precio."""
    cierre_prev = ohlc['Close'].shift(1)
    tr = pd.concat([ohlc['High'] - ohlc['Low'],
                    (ohlc['High'] - cierre_prev).abs(),
                    (ohlc['Low'] - cierre_prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _redondear(x, tick):
    return np.ceil(x / tick) * tick


def _cuantil(garch, alfa):
    """Cuantil estandarizado de las innovaciones del modelo (t, t asimétrica o normal)."""
    try:
        dist = garch.model.distribution
        nombres = dist.parameter_names()
        params = garch.params[nombres].values if nombres else None
        return float(np.asarray(dist.ppf(alfa, params)).ravel()[0])
    except Exception:
        return stats.norm.ppf(alfa)


def plan_stop(garch, v, ohlc, entrada=None, cuenta=50_000, riesgo_pct=0.01, contrato='MGC',
              horas=None, gvz=None, k_stops=(1.0, 1.5, 2.0, 2.5), drawdown_max=None):
    """
    Calcula rangos, stops y contratos para la próxima sesión.

    garch, v     : resultado de volatilidad.buscar_modelo y volatilidad.resumir
    ohlc         : OHLC diario del contrato (para el ATR y el gráfico)
    entrada      : precio de entrada; None = último cierre
    cuenta       : tamaño de la cuenta en USD
    riesgo_pct   : fracción de la cuenta que se acepta perder si salta el stop
    contrato     : 'MGC' o 'GC' (el principal del reporte; se muestran ambos)
    horas        : horas que se piensa mantener la posición; None = un día completo
    gvz          : volatilidad implícita del oro (índice GVZ) para comparar, opcional
    k_stops      : distancias de stop en múltiplos de σ
    drawdown_max : drawdown máximo permitido (cuentas de fondeo), opcional
    """
    ultimo_garch = v['vol_cond'].index[-1]
    if hasattr(ultimo_garch, 'date') and ultimo_garch < ohlc.index[-1]:
        print(f'Aviso: el GARCH termina el {ultimo_garch.date()} y el último dato es del '
              f'{ohlc.index[-1].date()}. Pon GARCH_SOLO_TRAIN = False para pronosticar desde hoy.\n')

    spec = CONTRATOS[contrato]
    tick = spec['tick']
    ref = float(entrada) if entrada is not None else float(ohlc['Close'].iloc[-1])
    riesgo_usd = cuenta * riesgo_pct

    # volatilidad de mañana según el GARCH (resumir la entrega anualizada en %)
    sig_dia_pct = v['vol_pron'][0] / np.sqrt(DIAS)
    sig_dia_pts = ref * sig_dia_pct / 100
    horizonte_h = horas if horas else HORAS_SESION
    escala = np.sqrt(min(horizonte_h, HORAS_SESION) / HORAS_SESION)
    sig = sig_dia_pts * escala                      # σ en puntos para el horizonte elegido

    atr14 = float(atr(ohlc).iloc[-1])
    rango_garch = FACTOR_RANGO * sig_dia_pts
    sig_gvz = ref * gvz / 100 / np.sqrt(DIAS) if gvz else None

    mu = garch.params.get('mu', 0.0)
    var99_pct = -(mu + sig_dia_pct * _cuantil(garch, 0.01))
    var99_pts = ref * var99_pct / 100

    filas = []
    for k in k_stops:
        d = _redondear(k * sig, tick)
        fila = {
            'Stop (σ)': k,
            'Distancia (pts)': d,
            'Stop largo': ref - d,
            'Stop corto': ref + d,
            'Prob. de tocarlo': 2 * stats.norm.sf(k),
            'Objetivo 2R largo': ref + 2 * d,
            'Objetivo 2R corto': ref - 2 * d,
        }
        for nombre, c in CONTRATOS.items():
            usd = d * c['multiplicador']
            fila[f'USD por {nombre}'] = usd
            fila[f'Contratos {nombre}'] = int(riesgo_usd // usd)
        filas.append(fila)
    tabla = pd.DataFrame(filas).set_index('Stop (σ)')

    plan = {
        'ref': ref, 'contrato': contrato, 'cuenta': cuenta, 'riesgo_usd': riesgo_usd,
        'horas': horizonte_h, 'sig_dia_pct': sig_dia_pct, 'sig_dia_pts': sig_dia_pts,
        'sig': sig, 'atr14': atr14, 'rango_garch': rango_garch, 'sig_gvz': sig_gvz,
        'var99_pts': var99_pts, 'tabla': tabla, 'drawdown_max': drawdown_max,
        'vol_hoy': v['vol_cond'].iloc[-1], 'vol_lp': v['vol_lp'], 'modelo': v['nombre'],
    }
    _imprimir(plan, spec)
    return plan


def _imprimir(p, spec):
    mult = spec['multiplicador']
    t = p['tabla']
    ref, sig = p['ref'], p['sig']
    horizonte = 'la próxima sesión' if p['horas'] >= HORAS_SESION else f"las próximas {p['horas']:g} horas"

    print(f"Plan de stop para {horizonte} | {p['modelo']}")
    print('─' * 72)
    print(f"  Precio de referencia             {ref:>12,.2f}")
    print(f"  Cuenta / riesgo por operación    {p['cuenta']:>12,.0f}   "
          f"{p['riesgo_usd']:,.0f} USD ({p['riesgo_usd'] / p['cuenta']:.1%})")
    print(f"  Volatilidad GARCH hoy / largo p. {p['vol_hoy']:>11.1f}%   {p['vol_lp']:.1f}% (anualizadas)")
    print(f"  σ de mañana                      {p['sig_dia_pct']:>11.2f}%   {p['sig_dia_pts']:,.1f} pts")
    if p['horas'] < HORAS_SESION:
        print(f"  σ para {p['horas']:g} horas                  {sig:>12,.1f} pts")
    print(f"  Rango esperado del día (1.6σ)    {p['rango_garch']:>12,.1f} pts")
    print(f"  ATR(14)                          {p['atr14']:>12,.1f} pts   "
          f"ATR / rango GARCH = {p['atr14'] / p['rango_garch']:.2f}")
    if p['sig_gvz']:
        print(f"  σ implícita (GVZ)                {p['sig_gvz']:>12,.1f} pts   "
              f"GVZ / GARCH = {p['sig_gvz'] / p['sig_dia_pts']:.2f}")
    print(f"  VaR 99% de mañana                {p['var99_pts']:>12,.1f} pts   "
          f"{p['var99_pts'] * mult:,.0f} USD por {p['contrato']}")

    print(f'\n  Rangos para {horizonte}')
    for k in (1, 2):
        print(f"    ±{k}σ: {ref - k * sig:,.1f} – {ref + k * sig:,.1f}")

    print('\n  Stops posibles')
    cols = ['Distancia (pts)', 'Stop largo', 'Stop corto', 'Prob. de tocarlo',
            'USD por MGC', 'Contratos MGC', 'USD por GC', 'Contratos GC']
    vista = t[cols].copy()
    vista['Prob. de tocarlo'] = (vista['Prob. de tocarlo'] * 100).round(1).astype(str) + '%'
    with pd.option_context('display.float_format', '{:,.1f}'.format, 'display.width', 140):
        print(vista.to_string())

    # lectura
    print()
    c = f"Contratos {p['contrato']}"
    usd = f"USD por {p['contrato']}"
    base = t.loc[1.5] if 1.5 in t.index else t.iloc[len(t) // 2]
    max_pts_1 = p['riesgo_usd'] / mult
    lineas = []

    k0 = t.index[0]
    lineas.append(f"La probabilidad de tocarlo supone precio sin tendencia y distribución normal; con colas "
                  f"pesadas es algo mayor. Un stop a {k0:g}σ ({t.iloc[0]['Distancia (pts)']:,.1f} pts) se toca en "
                  f"{t.iloc[0]['Prob. de tocarlo']:.0%} de los casos solo por ruido.")
    if base[c] >= 1:
        lineas.append(f"Con stop a {base.name}σ ({base['Distancia (pts)']:,.1f} pts) cada {p['contrato']} arriesga "
                      f"{base[usd]:,.0f} USD: el tamaño máximo es {int(base[c])} contrato(s), "
                      f"stop largo en {base['Stop largo']:,.1f} y corto en {base['Stop corto']:,.1f}.")
    else:
        k_base = base.name
        horas_max = HORAS_SESION * (max_pts_1 / (k_base * p['sig_dia_pts'])) ** 2
        riesgo_min = base[usd]
        opciones = [f"subir el riesgo a {riesgo_min:,.0f} USD ({riesgo_min / p['cuenta']:.1%} de la cuenta)"]
        if horas_max >= 0.5:
            opciones.append(f"mantener la posición como máximo {horas_max:.1f} horas (parámetro horas)")
        if p['contrato'] == 'GC':
            opciones.append('usar MGC')
        opciones.append('esperar a que baje la volatilidad')
        lineas.append(f"Con {p['riesgo_usd']:,.0f} USD de riesgo, un {p['contrato']} admite como máximo "
                      f"{max_pts_1:,.1f} pts de stop, que es {max_pts_1 / sig:.2f}σ: queda dentro del ruido. "
                      f"Para tener 1 contrato con stop a {k_base:g}σ hay que " + ', '.join(opciones[:-1])
                      + f" o {opciones[-1]}.")
    if p['atr14'] / p['rango_garch'] > 1.2:
        lineas.append('El ATR está por encima del rango que espera el GARCH: los últimos 14 días fueron más '
                      'movidos que lo que el modelo proyecta para mañana. Si tu stop se basa en ATR quedará '
                      'más holgado.')
    elif p['atr14'] / p['rango_garch'] < 0.8:
        lineas.append('El ATR está por debajo del rango GARCH: el modelo espera más movimiento del que hubo en '
                      'las últimas dos semanas (todavía pesa un shock reciente o la volatilidad de largo plazo).')
    if p['sig_gvz'] and p['sig_gvz'] / p['sig_dia_pts'] > 1.2:
        lineas.append('El mercado de opciones (GVZ) espera más volatilidad que el GARCH: suele pasar antes de '
                      'eventos macro. Toma el mayor de los dos para el stop.')
    if p['drawdown_max']:
        uso = base['USD por MGC'] / p['drawdown_max']
        lineas.append(f"Un MGC con stop a {base.name}σ consume {uso:.0%} del drawdown máximo de "
                      f"{p['drawdown_max']:,.0f} USD.")
    lineas.append('En días con datos macro (CPI, empleo, Fed) el rango real suele superar al pronóstico; '
                  'reduce tamaño o amplía el stop esos días.')

    for linea in lineas:
        print(textwrap.fill(linea, width=96, initial_indent='  → ', subsequent_indent='    '))


def grafico_stop(ohlc, plan, dias=40, k_stop=1.5, guardar=None):
    """Velas recientes con los rangos ±1σ/±2σ y los stops de la próxima sesión."""
    velas = ohlc.iloc[-dias:]
    ref, sig = plan['ref'], plan['sig']
    d = plan['tabla'].loc[k_stop, 'Distancia (pts)'] if k_stop in plan['tabla'].index else k_stop * sig
    manana = velas.index[-1] + pd.offsets.BDay(1)

    fig, ax = plt.subplots(figsize=(14, 6.5))
    xs = mdates.date2num(velas.index.to_pydatetime())
    for x, (o, h, l, c) in zip(xs, velas[['Open', 'High', 'Low', 'Close']].values):
        color = GREEN if c >= o else RED
        ax.vlines(x, l, h, color=color, lw=0.9, zorder=3)
        ax.add_patch(plt.Rectangle((x - 0.3, min(o, c)), 0.6, max(abs(c - o), 1e-9),
                                   facecolor=color, edgecolor=color, zorder=4))
    ax.xaxis_date()

    x0, x1 = mdates.date2num(manana - pd.Timedelta(hours=10)), mdates.date2num(manana + pd.Timedelta(hours=10))
    ax.fill_between([x0, x1], ref - 2 * sig, ref + 2 * sig, color=PURPLE, alpha=0.18, label='±2σ')
    ax.fill_between([x0, x1], ref - sig, ref + sig, color=PURPLE, alpha=0.38, label='±1σ')
    ax.axhline(ref, color=WHITE, lw=1, ls='--', alpha=0.7)
    ax.axhline(ref - d, color=RED, lw=1.3, ls=':', label=f'stop largo {k_stop}σ: {ref - d:,.1f}')
    ax.axhline(ref + d, color=CYAN, lw=1.3, ls=':', label=f'stop corto {k_stop}σ: {ref + d:,.1f}')

    ax.text(x1, ref, f'  {ref:,.1f}', color=WHITE, va='center', fontsize=9)
    for k in (1, 2):
        ax.text(x1, ref + k * sig, f'  +{k}σ {ref + k * sig:,.0f}', color=GOLD, va='center', fontsize=9)
        ax.text(x1, ref - k * sig, f'  −{k}σ {ref - k * sig:,.0f}', color=GOLD, va='center', fontsize=9)

    ax.set_xlim(xs[0] - 1, x1 + 4)
    horizonte = 'próxima sesión' if plan['horas'] >= HORAS_SESION else f"{plan['horas']:g} horas"
    ax.set_title(f"{plan['contrato']} — rango y stops para la {horizonte} ({plan['modelo']})", fontweight='bold')
    ax.set_ylabel('Precio (USD)')
    ax.legend(loc='upper left', framealpha=0.3, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _guardar(fig, guardar)
    plt.show()
