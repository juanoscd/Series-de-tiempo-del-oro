"""
Versión interactiva (Plotly) de las figuras de trading.

Mismas firmas que `graficos.py`, así el notebook cambia de una a otra con una
variable. Los diagnósticos estadísticos (QQ, ACF, residuos) se quedan en
matplotlib: ahí la interactividad no agrega información.

Cada función muestra la figura, la guarda en figuras/<nombre>.html si se pide
(y en .png si está instalado kaleido) y la devuelve por si se quiere retocar.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import stats

from .estilo import BLACK, CARPETA_FIGURAS, CYAN, GOLD, GREEN, GRID, LILAC, PURPLE, RED, WHITE

TINTA = '#E6E1F0'        # texto principal
TINTA_SUAVE = '#A39BB5'  # texto secundario, ejes
SUPERFICIE = '#140C24'   # tooltips y encabezados

pio.templates['seriesfin'] = go.layout.Template(layout=dict(
    paper_bgcolor=BLACK,
    plot_bgcolor=BLACK,
    font=dict(family='Inter, DejaVu Sans, sans-serif', color=TINTA, size=12),
    title=dict(font=dict(size=17, color=WHITE), x=0.01, xanchor='left'),
    colorway=[PURPLE, CYAN, GOLD],
    hovermode='x unified',
    hoverlabel=dict(bgcolor=SUPERFICIE, bordercolor=GRID, font=dict(color=TINTA, size=12)),
    legend=dict(orientation='h', x=0, y=1.0, yanchor='bottom', bgcolor='rgba(0,0,0,0)',
                font=dict(color=TINTA_SUAVE)),
    margin=dict(l=64, r=110, t=90, b=40),
    xaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False, tickfont=dict(color=TINTA_SUAVE),
               showspikes=True, spikemode='across', spikesnap='cursor', spikethickness=1,
               spikedash='dot', spikecolor='#6B5A8E'),
    yaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False, tickfont=dict(color=TINTA_SUAVE),
               showspikes=True, spikemode='across', spikesnap='cursor', spikethickness=1,
               spikedash='dot', spikecolor='#6B5A8E'),
))
TEMA = 'plotly_dark+seriesfin'

CONFIG = {'displaylogo': False, 'scrollZoom': True,
          'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d']}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _rgba(hex_color, alpha):
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'


def _mostrar(fig, guardar, ancho=1400, alto=None):
    if guardar:
        CARPETA_FIGURAS.mkdir(exist_ok=True)
        fig.write_html(CARPETA_FIGURAS / f'{guardar}.html', include_plotlyjs='cdn', config=CONFIG)
        try:   # el PNG necesita kaleido; si no está, solo se guarda el HTML
            fig.write_image(CARPETA_FIGURAS / f'{guardar}.png', width=ancho,
                            height=alto or fig.layout.height or 700, scale=2)
        except Exception:
            pass
    fig.show(config=CONFIG)
    return fig


def _sin_huecos(index):
    """Oculta fines de semana y festivos sin datos para que las velas queden pegadas."""
    idx = pd.DatetimeIndex(index).normalize()
    faltan = pd.bdate_range(idx.min(), idx.max()).difference(idx)
    return [dict(bounds=['sat', 'mon']), dict(values=faltan.strftime('%Y-%m-%d').tolist())]


def _velas(ohlc, nombre='Precio'):
    """Velas con alcistas huecas y bajistas rellenas: la dirección no depende solo del color."""
    rango = (ohlc['High'] - ohlc['Low']).values
    return go.Candlestick(
        x=ohlc.index, open=ohlc['Open'], high=ohlc['High'], low=ohlc['Low'], close=ohlc['Close'],
        name=nombre, whiskerwidth=0, showlegend=False,
        increasing=dict(line=dict(color=GREEN, width=1.2), fillcolor=BLACK),
        decreasing=dict(line=dict(color=RED, width=1.2), fillcolor=RED),
        text=[f'Rango del día: {x:,.1f}' for x in rango],
    )


def _linea_v(fig, x, color, texto, row=None, col=None, abajo=False):
    """Línea vertical con etiqueta (add_vline falla con anotaciones en ejes de fecha)."""
    ref = {} if row is None else dict(row=row, col=col)
    fig.add_vline(x=x, line=dict(color=color, width=1, dash='dash'), opacity=0.8, **ref)
    xref = 'x' if row in (None, 1) else f'x{row}'
    yref = 'y domain' if row in (None, 1) else f'y{row} domain'
    fig.add_annotation(x=x, xref=xref, y=0.02 if abajo else 0.98, yref=yref, text=f' {texto}',
                       showarrow=False, xanchor='left', font=dict(color=TINTA, size=11))


def _ejes_fecha(fig, index, filas=1):
    for fila in range(1, filas + 1):
        fig.update_xaxes(rangebreaks=_sin_huecos(index), rangeslider_visible=False, row=fila, col=1)


# ---------------------------------------------------------------------------
# Plan de stop
# ---------------------------------------------------------------------------

def grafico_stop(ohlc, plan, dias=60, k_stop=1.5, guardar=None):
    """
    Velas recientes, rango de la próxima sesión (±1σ, ±2σ), stops largo y corto,
    y abajo el rango diario real contra el rango que espera el GARCH.
    """
    velas = ohlc.iloc[-dias:]
    ref, sig, sig_d = plan['ref'], plan['sig'], plan['sig_dia_pts']
    tabla = plan['tabla']
    d = tabla.loc[k_stop, 'Distancia (pts)'] if k_stop in tabla.index else k_stop * sig
    fila = tabla.loc[k_stop] if k_stop in tabla.index else None
    contrato = plan['contrato']
    manana = velas.index[-1] + pd.offsets.BDay(1)
    medio_dia = pd.Timedelta(hours=14)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24],
                        vertical_spacing=0.03)
    fig.add_trace(_velas(velas, contrato), row=1, col=1)

    # bandas de la próxima sesión
    for k, alpha in ((2, 0.16), (1, 0.38)):
        fig.add_shape(type='rect', x0=manana - medio_dia, x1=manana + medio_dia,
                      y0=ref - k * sig, y1=ref + k * sig, fillcolor=_rgba(PURPLE, alpha),
                      line=dict(width=0), layer='below', row=1, col=1)

    niveles = [(-2, ref - 2 * sig), (-1, ref - sig), (0, ref), (1, ref + sig), (2, ref + 2 * sig)]
    etiquetas = [f'{"ref" if k == 0 else f"{k:+d}σ"}  {y:,.0f}' for k, y in niveles]
    fig.add_trace(go.Scatter(
        x=[manana] * 5, y=[y for _, y in niveles], mode='markers+text', name='Rango próxima sesión',
        marker=dict(symbol='line-ew', size=30, line=dict(width=2, color=GOLD)),
        text=etiquetas, textposition='middle right', textfont=dict(color=TINTA, size=11),
        customdata=[[k] for k, _ in niveles],
        hovertemplate='<b>%{y:,.1f}</b>  (%{customdata[0]:+d}σ)<extra>Rango próxima sesión</extra>',
        showlegend=False), row=1, col=1)

    # entrada y stops
    fig.add_hline(y=ref, line=dict(color=WHITE, width=1, dash='dash'), opacity=0.6, row=1, col=1)
    for y, color, texto, pos in (
        (ref - d, RED, f'Stop largo {k_stop:g}σ  {ref - d:,.1f}', 'bottom left'),
        (ref + d, CYAN, f'Stop corto {k_stop:g}σ  {ref + d:,.1f}', 'top left'),
    ):
        fig.add_hline(y=y, line=dict(color=color, width=1.6, dash='dot'), row=1, col=1,
                      annotation_text=texto, annotation_position=pos,
                      annotation_font=dict(color=TINTA, size=11))

    # rango diario real vs esperado
    rango = velas['High'] - velas['Low']
    fig.add_trace(go.Bar(x=velas.index, y=rango, name='Rango diario (máx − mín)',
                         marker=dict(color=_rgba(LILAC, 0.55), line=dict(width=0)),
                         hovertemplate='%{y:,.1f} pts<extra>Rango diario</extra>'), row=2, col=1)
    fig.add_trace(go.Scatter(x=[velas.index[0], manana], y=[plan['rango_garch']] * 2, mode='lines',
                             name=f"Rango esperado GARCH ({plan['rango_garch']:,.0f} pts)",
                             line=dict(color=GOLD, width=2, dash='dash'),
                             hovertemplate='%{y:,.1f} pts<extra>Esperado GARCH</extra>'), row=2, col=1)
    fig.add_trace(go.Scatter(x=[velas.index[-1]], y=[plan['atr14']], mode='markers',
                             name=f"ATR(14) {plan['atr14']:,.0f} pts",
                             marker=dict(color=CYAN, size=9, line=dict(color=BLACK, width=2)),
                             hovertemplate='%{y:,.1f} pts<extra>ATR(14)</extra>'), row=2, col=1)

    # título con los números clave
    horizonte = 'próxima sesión' if plan['horas'] >= 23 else f"próximas {plan['horas']:g} horas"
    detalle = (f"σ {sig:,.1f} pts · stop {k_stop:g}σ = {d:,.1f} pts")
    if fila is not None:
        detalle += (f" · {fila[f'USD por {contrato}']:,.0f} USD por {contrato}"
                    f" · {int(fila[f'Contratos {contrato}'])} contrato(s) con {plan['riesgo_usd']:,.0f} USD de riesgo")
    fig.update_layout(
        template=TEMA, height=760, bargap=0.25,
        title=dict(text=f"{contrato} — rango y stops para la {horizonte}"
                        f"<br><sup><span style='color:{TINTA_SUAVE}'>{detalle} · {plan['modelo']}</span></sup>"),
        legend=dict(y=-0.06, yanchor='top'),
    )
    _ejes_fecha(fig, velas.index, filas=2)
    fig.update_xaxes(range=[velas.index[0] - pd.Timedelta(days=1), manana + pd.Timedelta(days=2)])
    fig.update_yaxes(tickformat=',.0f', title_text='Precio', row=1, col=1)
    fig.update_yaxes(tickformat=',.0f', title_text='Puntos', row=2, col=1)
    return _mostrar(fig, guardar, alto=760)


# ---------------------------------------------------------------------------
# Retornos y riesgo
# ---------------------------------------------------------------------------

def dashboard(df, r, s, filas, ticker, guardar=None):
    """Distribución, retornos, precio con medias móviles, volatilidad rodante y tabla."""
    periodo = f'{df.index[0].year}-{df.index[-1].year}'
    ma50, ma200 = df['Close'].rolling(50).mean(), df['Close'].rolling(200).mean()
    vol_30 = r.rolling(30).std() * np.sqrt(252) * 100

    fig = make_subplots(
        rows=3, cols=2, row_heights=[0.3, 0.3, 0.4], vertical_spacing=0.08, horizontal_spacing=0.07,
        specs=[[{}, {}], [{}, {}], [{'type': 'table', 'colspan': 2}, None]],
        subplot_titles=('Distribución de retornos', 'Retornos diarios',
                        'Precio y medias móviles', 'Volatilidad anualizada (30 días)'))

    fig.add_trace(go.Histogram(x=r, nbinsx=80, histnorm='probability density', name='Retornos',
                               marker=dict(color=_rgba(PURPLE, 0.85), line=dict(color=BLACK, width=0.5)),
                               hovertemplate='%{x:.2%}: %{y:.1f}<extra></extra>', showlegend=False),
                  row=1, col=1)
    x = np.linspace(r.min(), r.max(), 200)
    fig.add_trace(go.Scatter(x=x, y=stats.norm.pdf(x, r.mean(), r.std()), name='Normal',
                             line=dict(color=WHITE, dash='dash', width=1.5), hoverinfo='skip'),
                  row=1, col=1)
    fig.add_vline(x=s['var'], line=dict(color=GOLD, dash='dash', width=2), row=1, col=1,
                  annotation_text=f"VaR 95% {s['var']:.2%}", annotation_font=dict(color=TINTA, size=10))
    fig.add_vline(x=s['cvar'], line=dict(color=RED, width=2), row=1, col=1,
                  annotation_text=f"CVaR {s['cvar']:.2%}", annotation_position='top left',
                  annotation_font=dict(color=TINTA, size=10))

    colores = np.where(r >= 0, _rgba(GREEN, 0.7), _rgba(RED, 0.7))
    fig.add_trace(go.Bar(x=r.index, y=r, marker=dict(color=colores, line=dict(width=0)),
                         name='Retorno', showlegend=False,
                         hovertemplate='%{x|%d %b %Y}: <b>%{y:.2%}</b><extra></extra>'), row=1, col=2)

    for serie, nombre, color, ancho in ((df['Close'], 'Cierre', LILAC, 1.2), (ma50, 'MA50', CYAN, 1.8),
                                        (ma200, 'MA200', GOLD, 1.8)):
        fig.add_trace(go.Scatter(x=serie.index, y=serie, name=nombre, line=dict(color=color, width=ancho),
                                 hovertemplate='%{y:,.2f}'), row=2, col=1)

    fig.add_trace(go.Scatter(x=vol_30.index, y=vol_30, name='Vol. 30 días', fill='tozeroy',
                             line=dict(color=RED, width=1.4), fillcolor=_rgba(RED, 0.22),
                             hovertemplate='%{y:.1f}%', showlegend=False), row=2, col=2)
    fig.add_hline(y=s['vol_a'] * 100, line=dict(color=WHITE, dash='dash', width=1), row=2, col=2,
                  annotation_text=f"promedio {s['vol_a']:.1%}", annotation_font=dict(color=TINTA, size=10))

    columnas = list(zip(*filas))
    n = len(filas)
    fig.add_trace(go.Table(
        columnwidth=[0.2, 0.13, 0.67],
        header=dict(values=['<b>Estadística</b>', '<b>Valor</b>', '<b>Interpretación</b>'],
                    fill_color=SUPERFICIE, line_color=GRID, align='left', font=dict(color=WHITE, size=12),
                    height=30),
        cells=dict(values=columnas, align='left', line_color=GRID, height=28,
                   fill_color=[['#0D0D0D' if i % 2 else '#161616' for i in range(n)]] * 3,
                   font=dict(color=[TINTA, WHITE, TINTA_SUAVE], size=12))), row=3, col=1)

    fig.update_layout(template=TEMA, height=1250, hovermode='closest', bargap=0.05,
                      title=dict(text=f'{ticker} — retornos y riesgo ({periodo})'),
                      legend=dict(y=1.02))
    fig.update_xaxes(tickformat='.1%', row=1, col=1)
    fig.update_yaxes(tickformat='.1%', row=1, col=2)
    fig.update_yaxes(tickformat=',.0f', row=2, col=1)
    fig.update_annotations(font=dict(color=TINTA))
    return _mostrar(fig, guardar, alto=1250)


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------

def series(precios, ret_principal, principal, guardar=None):
    base100 = precios / precios.iloc[0] * 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.66, 0.34], vertical_spacing=0.05)
    for t, color in zip(precios.columns, [PURPLE, CYAN, GOLD]):
        fig.add_trace(go.Scatter(x=base100.index, y=base100[t], name=t, line=dict(color=color, width=1.8),
                                 customdata=precios[t], hovertemplate='%{y:.1f}  (precio %{customdata:,.2f})'),
                      row=1, col=1)
    fig.add_hline(y=100, line=dict(color=GRID, width=1), row=1, col=1)
    fig.add_trace(go.Scatter(x=ret_principal.index, y=ret_principal, name=f'Retorno {principal}',
                             line=dict(color=LILAC, width=0.8), hovertemplate='%{y:.2f}%', showlegend=False),
                  row=2, col=1)
    fig.update_layout(template=TEMA, height=720,
                      title=dict(text=f'{" vs ".join(precios.columns)} (base 100)'
                                      f'<br><sup><span style="color:{TINTA_SUAVE}">abajo: retornos diarios de '
                                      f'{principal} (%)</span></sup>'))
    fig.update_yaxes(title_text='Base 100', row=1, col=1)
    fig.update_yaxes(title_text='Retorno (%)', row=2, col=1)
    return _mostrar(fig, guardar, alto=720)


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------

def pronostico_arima(ohlc, pron, corte, orden, ticker, dias_antes=60, guardar=None):
    """Velas desde `dias_antes` días antes del corte, pronóstico y bandas al 80/95/99%."""
    corte = pd.Timestamp(corte)
    velas = ohlc.loc[ohlc.index >= corte - pd.Timedelta(days=dias_antes)]
    ultimo = velas.index[-1]

    fig = go.Figure()
    for nivel, alpha in ((99, 0.10), (95, 0.18), (80, 0.32)):
        fig.add_trace(go.Scatter(x=pron.index, y=pron[f'sup_{nivel}'], mode='lines', line=dict(width=0),
                                 legendgroup=str(nivel), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=pron.index, y=pron[f'inf_{nivel}'], mode='lines', line=dict(width=0),
                                 fill='tonexty', fillcolor=_rgba(PURPLE, alpha), name=f'IC {nivel}%',
                                 legendgroup=str(nivel),
                                 customdata=pron[f'sup_{nivel}'],
                                 hovertemplate=f'IC {nivel}%: ' + '%{y:,.0f} – %{customdata:,.0f}<extra></extra>'))
    fig.add_trace(_velas(velas, ticker))
    fig.add_trace(go.Scatter(x=pron.index, y=pron['media'], name=f'ARIMA{orden}',
                             line=dict(color=CYAN, width=2, dash='dot'),
                             hovertemplate='<b>%{y:,.2f}</b><extra>ARIMA</extra>'))

    _linea_v(fig, corte, GOLD, 'inicio test')
    _linea_v(fig, ultimo, WHITE, 'último dato', abajo=True)

    y_min = min(velas['Low'].min(), pron['inf_95'].min())
    y_max = max(velas['High'].max(), pron['sup_95'].max())
    margen = (y_max - y_min) * 0.04
    fig.update_layout(template=TEMA, height=680,
                      title=dict(text=f'{ticker} — pronóstico ARIMA{orden} a {len(pron)} días hábiles'))
    fig.update_xaxes(rangebreaks=_sin_huecos(velas.index.append(pron.index)), rangeslider_visible=False)
    fig.update_yaxes(range=[y_min - margen, y_max + margen], tickformat=',.0f', title_text='Precio')
    return _mostrar(fig, guardar, alto=680)


# ---------------------------------------------------------------------------
# Volatilidad
# ---------------------------------------------------------------------------

def clustering(serie, ret, ticker, ventana=21, guardar=None):
    vol = ret.rolling(ventana).std() * np.sqrt(252)
    umbral = float(vol.quantile(0.90))
    alta = vol.where(vol > umbral)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.58, 0.42], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=serie.index, y=serie, name='Precio', line=dict(color=PURPLE, width=1.3),
                             hovertemplate='%{y:,.2f}'), row=1, col=1)
    fig.add_trace(go.Scatter(x=vol.index, y=vol, name=f'Vol. realizada {ventana} días', fill='tozeroy',
                             line=dict(color=RED, width=1.2), fillcolor=_rgba(RED, 0.18),
                             hovertemplate='%{y:.1f}%'), row=2, col=1)
    fig.add_trace(go.Scatter(x=alta.index, y=alta, name='Sobre el percentil 90', mode='lines',
                             line=dict(color=RED, width=2.4), hoverinfo='skip'), row=2, col=1)
    fig.add_hline(y=umbral, line=dict(color=GOLD, dash='dash', width=1.2), row=2, col=1,
                  annotation_text=f'percentil 90 = {umbral:.1f}%', annotation_font=dict(color=TINTA, size=11))
    fig.update_layout(template=TEMA, height=720,
                      title=dict(text=f'{ticker}: precio y volatilidad realizada ({ventana} días)'))
    fig.update_yaxes(title_text='Precio', tickformat=',.0f', row=1, col=1)
    fig.update_yaxes(title_text='Vol. anualizada (%)', row=2, col=1)
    _mostrar(fig, guardar, alto=720)
    return int((vol > umbral).sum())


def volatilidad(v, res, ticker, guardar=None):
    """Volatilidad condicional con su pronóstico, curva del pronóstico y residuos estandarizados."""
    vol_cond, vol_lp, vol_pron = v['vol_cond'], v['vol_lp'], np.asarray(v['vol_pron'])
    h = len(vol_pron)
    fechas_pron = pd.bdate_range(vol_cond.index[-1] + pd.offsets.BDay(1), periods=h)

    fig = make_subplots(rows=2, cols=2, row_heights=[0.6, 0.4], vertical_spacing=0.12,
                        horizontal_spacing=0.08, specs=[[{'colspan': 2}, None], [{}, {}]],
                        subplot_titles=('', f'Pronóstico a {h} días', 'Residuos estandarizados'))

    fig.add_trace(go.Scatter(x=vol_cond.index, y=vol_cond, name='Volatilidad condicional', fill='tozeroy',
                             line=dict(color=RED, width=1.2), fillcolor=_rgba(RED, 0.16),
                             hovertemplate='%{y:.1f}%'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[vol_cond.index[-1], *fechas_pron], y=[vol_cond.iloc[-1], *vol_pron],
                             name='Pronóstico', mode='lines+markers', line=dict(color=CYAN, width=2.2),
                             marker=dict(size=8, line=dict(color=BLACK, width=2)),
                             hovertemplate='<b>%{y:.1f}%</b><extra>Pronóstico</extra>'), row=1, col=1)
    fig.add_hline(y=vol_lp, line=dict(color=GOLD, dash='dash', width=1.4), row=1, col=1,
                  annotation_text=f'largo plazo {vol_lp:.1f}%', annotation_font=dict(color=TINTA, size=11))

    fig.add_trace(go.Scatter(x=np.arange(1, h + 1), y=vol_pron, mode='lines+markers', showlegend=False,
                             line=dict(color=CYAN, width=2.2), marker=dict(size=8, line=dict(color=BLACK, width=2)),
                             hovertemplate='día %{x}: <b>%{y:.2f}%</b><extra></extra>'), row=2, col=1)
    fig.add_hline(y=vol_lp, line=dict(color=GOLD, dash='dash', width=1.2), row=2, col=1)

    z = res.std_resid.dropna()
    fig.add_trace(go.Histogram(x=z, nbinsx=60, histnorm='probability density', showlegend=False,
                               marker=dict(color=_rgba(PURPLE, 0.85), line=dict(color=BLACK, width=0.5)),
                               hovertemplate='%{x:.2f}: %{y:.3f}<extra></extra>'), row=2, col=2)
    x = np.linspace(z.min(), z.max(), 300)
    fig.add_trace(go.Scatter(x=x, y=stats.norm.pdf(x), name='N(0, 1)', line=dict(color=GOLD, width=1.8),
                             hoverinfo='skip'), row=2, col=2)

    fig.update_layout(template=TEMA, height=820, bargap=0.05,
                      title=dict(text=f"{ticker} — {v['nombre']}"
                                      f"<br><sup><span style='color:{TINTA_SUAVE}'>hoy {vol_cond.iloc[-1]:.1f}% · "
                                      f"mañana {vol_pron[0]:.1f}% · largo plazo {vol_lp:.1f}% "
                                      f"(anualizadas)</span></sup>"))
    fig.update_xaxes(range=[vol_cond.index[0], fechas_pron[-1] + pd.Timedelta(days=5)], row=1, col=1)
    fig.update_yaxes(title_text='Vol. anualizada (%)', row=1, col=1)
    fig.update_xaxes(title_text='Días hábiles', dtick=1, row=2, col=1)
    fig.update_xaxes(showspikes=False, row=2, col=2)
    fig.update_annotations(font=dict(color=TINTA))
    return _mostrar(fig, guardar, alto=820)


# ---------------------------------------------------------------------------
# VAR
# ---------------------------------------------------------------------------

def var_sistema(irf, pron_var, shock, k_ar, guardar=None):
    tickers = list(pron_var.columns)
    colores = dict(zip(tickers, [PURPLE, CYAN, GOLD]))

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.14,
                        subplot_titles=(f'Impulso-respuesta a un shock de 1σ en {shock}',
                                        f'Pronóstico VAR({k_ar}) de retornos diarios (%)'))
    for t in tickers:
        fig.add_trace(go.Scatter(x=irf['original'].index, y=irf['original'][t], name=t, legendgroup=t,
                                 mode='lines+markers', line=dict(color=colores[t], width=2.2),
                                 marker=dict(size=8, line=dict(color=BLACK, width=2)),
                                 hovertemplate='<b>%{y:+.3f}</b>'), row=1, col=1)
        fig.add_trace(go.Scatter(x=irf['invertido'].index, y=irf['invertido'][t], name=f'{t} (orden invertido)',
                                 legendgroup=t, mode='lines', line=dict(color=colores[t], width=1.4, dash='dash'),
                                 hovertemplate='%{y:+.3f}'), row=1, col=1)
        fig.add_trace(go.Bar(x=pron_var.index, y=pron_var[t], name=t, legendgroup=t, showlegend=False,
                             marker=dict(color=colores[t], line=dict(color=BLACK, width=2)),
                             hovertemplate='<b>%{y:+.4f}%</b>'), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=WHITE, width=0.8), row=1, col=1)
    fig.add_hline(y=0, line=dict(color=WHITE, width=0.8), row=2, col=1)
    fig.update_layout(template=TEMA, height=800, barmode='group', bargap=0.3,
                      title=dict(text='Sistema VAR'))
    fig.update_xaxes(title_text='Días después del shock', row=1, col=1)
    fig.update_xaxes(title_text='Días hacia adelante', dtick=1, row=2, col=1)
    fig.update_yaxes(title_text='Respuesta (%)', row=1, col=1)
    fig.update_yaxes(title_text='Retorno esperado (%)', row=2, col=1)
    fig.update_annotations(font=dict(color=TINTA))
    return _mostrar(fig, guardar, alto=800)
