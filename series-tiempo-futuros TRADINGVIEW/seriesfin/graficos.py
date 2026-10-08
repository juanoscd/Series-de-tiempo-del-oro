"""Figuras del análisis. Cada función dibuja, guarda (si se pide) y muestra."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy import stats
from statsmodels.tsa.stattools import acf

from .estilo import (BLACK, CYAN, GOLD, GREEN, GRID, LILAC, PURPLE, RED, WHITE,
                     guardar as _guardar)


def _cerrar(fig, guardar):
    _guardar(fig, guardar)
    plt.show()


def _eje_anual(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))


# Retornos y distribución

def dashboard(df, r, s, filas, ticker, guardar=None):
    """Distribución, retornos, precio con medias móviles, volatilidad rodante y tabla."""
    periodo = f'{df.index[0].year}-{df.index[-1].year}'
    ma50 = df['Close'].rolling(50).mean()
    ma200 = df['Close'].rolling(200).mean()
    vol_30 = r.rolling(30).std() * np.sqrt(252) * 100

    fig = plt.figure(figsize=(14, 15))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1.05], figure=fig)
    ax_hist, ax_ret = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_px, ax_vol = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    ax_tab = fig.add_subplot(gs[2, :])

    ax_hist.hist(r, bins=80, density=True, color=PURPLE, edgecolor=BLACK, alpha=0.85)
    x = np.linspace(r.min(), r.max(), 200)
    ax_hist.plot(x, stats.norm.pdf(x, r.mean(), r.std()), color=WHITE, ls='--', lw=1.5,
                 label='Normal')
    ax_hist.axvline(s['var'], color=GOLD, ls='--', lw=2, label=f"VaR 95%: {s['var']:.2%}")
    ax_hist.axvline(s['cvar'], color=RED, lw=2, label=f"CVaR 95%: {s['cvar']:.2%}")
    ax_hist.set_title('Distribución de retornos logarítmicos', fontweight='bold')
    ax_hist.set_xlabel('Retorno diario')
    ax_hist.legend(fontsize=8)

    ax_ret.fill_between(r.index, r, where=r < 0, color=RED, alpha=0.5)
    ax_ret.fill_between(r.index, r, where=r >= 0, color=GREEN, alpha=0.5)
    ax_ret.axhline(0, color=WHITE, lw=0.5)
    ax_ret.set_title('Retornos diarios', fontweight='bold')
    ax_ret.set_ylabel('Retorno logarítmico')

    ax_px.plot(df.index, df['Close'], color=LILAC, lw=1, label='Cierre')
    ax_px.plot(df.index, ma50, color=CYAN, lw=1.5, label='MA50')
    ax_px.plot(df.index, ma200, color=GOLD, lw=1.5, label='MA200')
    ax_px.set_title('Precio de cierre y medias móviles', fontweight='bold')
    ax_px.set_ylabel('Precio (USD)')
    ax_px.legend(fontsize=8)

    ax_vol.fill_between(vol_30.index, vol_30, color=RED, alpha=0.5)
    ax_vol.axhline(s['vol_a'] * 100, color=WHITE, ls='--', lw=1,
                   label=f"Promedio del periodo: {s['vol_a']:.1%}")
    ax_vol.set_title('Volatilidad anualizada rodante (30 días)', fontweight='bold')
    ax_vol.set_ylabel('Volatilidad (%)')
    ax_vol.legend(fontsize=8)

    for ax in (ax_ret, ax_px, ax_vol):
        _eje_anual(ax)
    for ax in (ax_hist, ax_ret, ax_px, ax_vol):
        ax.grid(True, alpha=0.3)

    ax_tab.axis('off')
    ax_tab.set_title('Estadísticas de retornos', fontweight='bold')
    tabla = ax_tab.table(cellText=filas, colLabels=['Estadística', 'Valor', 'Interpretación'],
                         colWidths=[0.20, 0.13, 0.67], cellLoc='left', loc='center')
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 1.9)
    for (fila, col), celda in tabla.get_celld().items():
        celda.set_edgecolor(GRID)
        celda.PAD = 0.02
        if fila == 0:
            celda.set_facecolor('#1A1030')
            celda.set_text_props(fontweight='bold', color=WHITE)
        else:
            celda.set_facecolor('#0D0D0D' if fila % 2 else '#161616')
            if col == 1:
                celda.set_text_props(fontweight='bold')

    fig.suptitle(f'{ticker} — retornos y riesgo ({periodo})', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _cerrar(fig, guardar)


def distribucion(r, ajuste, ticker, guardar=None):
    """Densidad en escala log y QQ-plots contra la normal y la t ajustada."""
    r = r.dropna()
    mu, sigma = ajuste['normal']
    gl, loc, escala = ajuste['t']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.hist(r, bins=80, density=True, color=PURPLE, alpha=0.75, edgecolor=BLACK)
    x = np.linspace(r.min(), r.max(), 400)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color=WHITE, ls='--', lw=1.8, label='Normal')
    ax.plot(x, stats.t.pdf(x, gl, loc, escala), color=GOLD, lw=2, label=f't (gl = {gl:.1f})')
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-2)
    ax.set_title('Densidad en escala log', fontweight='bold')
    ax.legend()

    for ax, dist, params, titulo in (
        (axes[1], 'norm', (), 'QQ-plot contra la normal'),
        (axes[2], stats.t, (gl,), f'QQ-plot contra t (gl = {gl:.1f})'),
    ):
        (teo, obs), (pend, inter, _) = stats.probplot(r, dist=dist, sparams=params)
        ax.scatter(teo, obs, s=6, color=LILAC, alpha=0.7)
        ax.plot(teo, pend * teo + inter, color=GOLD, lw=1.5)
        ax.set_title(titulo, fontweight='bold')
        ax.set_xlabel('Cuantiles teóricos')
        ax.set_ylabel('Cuantiles observados')

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'Forma de la distribución — {ticker}', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _cerrar(fig, guardar)


# Series y diagnóstico

def series(precios, ret_principal, principal, guardar=None):
    base100 = precios / precios.iloc[0] * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.12})
    for t, c in zip(precios.columns, [PURPLE, CYAN, GOLD, GREEN]):
        ax1.plot(base100.index, base100[t], color=c, lw=1.6, label=t)
    ax1.axhline(100, color=GRID, lw=1)
    ax1.set_title(f'{" vs ".join(precios.columns)} (base 100)', fontweight='bold')
    ax1.set_ylabel('Índice (base 100)')
    ax1.legend(loc='upper left', framealpha=0.3)

    ax2.plot(ret_principal.index, ret_principal, color=LILAC, lw=0.6)
    ax2.axhline(0, color=GRID, lw=1)
    ax2.set_title(f'Retornos logarítmicos diarios de {principal} (%)', fontsize=11)
    ax2.set_ylabel('Retorno (%)')

    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    _cerrar(fig, guardar)


def nivel_vs_retornos(serie, ret, principal, guardar=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    axes[0].plot(serie.index, serie, color=LILAC, lw=1.2)
    axes[0].plot(serie.index, serie.rolling(252).mean(), color=RED, lw=1.6,
                 label='Media móvil 252 días')
    axes[0].set_title(f'{principal} en nivel: la media cambia', fontsize=12)
    axes[0].set_ylabel('Precio (USD)')

    axes[1].plot(ret.index, ret, color=PURPLE, lw=0.6)
    axes[1].plot(ret.index, ret.rolling(252).mean(), color=CYAN, lw=1.6,
                 label='Media móvil 252 días')
    axes[1].axhline(0, color=GRID, lw=1)
    axes[1].set_title(f'{principal} en retornos: media estable cerca de 0', fontsize=12)
    axes[1].set_ylabel('Retorno (%)')

    for ax in axes:
        ax.legend(framealpha=0.3, fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _cerrar(fig, guardar)


def autocorrelaciones(ac, guardar=None):
    nlags, banda = ac['nlags'], ac['banda']
    datos = [ac['acf'][1:], ac['pacf'][1:], ac['acf_cuadrado'][1:]]
    tope = max(0.25, float(np.max(np.abs(np.concatenate(datos)))) * 1.15)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titulos = ['ACF de retornos', 'PACF de retornos', 'ACF de retornos²']
    lags = np.arange(1, nlags + 1)
    for ax, titulo, valores, color in zip(axes, titulos, datos, [PURPLE, CYAN, RED]):
        ax.bar(lags, valores, color=color, width=0.7, alpha=0.9)
        ax.axhline(0, color=WHITE, lw=0.8)
        for b in (banda, -banda):
            ax.axhline(b, color=GOLD, ls='--', lw=1)
        ax.set_title(titulo, fontsize=12)
        ax.set_xlabel('Rezago')
        ax.set_ylim(-tope, tope)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel('Autocorrelación')
    fig.tight_layout()
    _cerrar(fig, guardar)


# ARIMA

def residuos_arima(resid, nlags=25, guardar=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(resid.index, resid, color=PURPLE, lw=0.7)
    axes[0].axhline(0, color=GRID, lw=1)
    axes[0].set_title('Residuos en el tiempo', fontsize=12)

    axes[1].hist(resid, bins=60, color=PURPLE, alpha=0.85, edgecolor=BLACK)
    axes[1].set_title('Distribución de residuos', fontsize=12)

    valores = acf(resid, nlags=nlags)[1:]
    banda = 1.96 / np.sqrt(len(resid))
    axes[2].bar(np.arange(1, nlags + 1), valores, color=CYAN, width=0.7)
    for b in (banda, -banda):
        axes[2].axhline(b, color=GOLD, ls='--', lw=1)
    axes[2].axhline(0, color=WHITE, lw=0.8)
    axes[2].set_title('ACF de residuos', fontsize=12)
    axes[2].set_xlabel('Rezago')

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _cerrar(fig, guardar)


def _velas(ax, ohlc, ancho=0.6):
    xs = mdates.date2num(ohlc.index.to_pydatetime())
    for x, (o, h, l, c) in zip(xs, ohlc[['Open', 'High', 'Low', 'Close']].values):
        color = GREEN if c >= o else RED
        ax.vlines(x, l, h, color=color, lw=0.9, zorder=3)
        ax.add_patch(Rectangle((x - ancho / 2, min(o, c)), ancho, max(abs(c - o), 1e-9),
                               facecolor=color, edgecolor=color, zorder=4))
    ax.xaxis_date()


def pronostico_arima(ohlc, pron, corte, orden, ticker, dias_antes=60, guardar=None):
    """Velas desde `dias_antes` días antes del corte, pronóstico y bandas de confianza."""
    corte = pd.Timestamp(corte)
    inicio = corte - pd.Timedelta(days=dias_antes)
    velas = ohlc.loc[ohlc.index >= inicio]
    ultimo = velas.index[-1]

    fig, ax = plt.subplots(figsize=(20, 6.5))
    for nivel, alpha in ((99, 0.10), (95, 0.18), (80, 0.32)):
        ax.fill_between(pron.index, pron[f'inf_{nivel}'], pron[f'sup_{nivel}'],
                        color=PURPLE, alpha=alpha, label=f'IC {nivel}%')

    _velas(ax, velas)
    ax.plot(pron.index, pron['media'], color=CYAN, lw=2, ls=':', label=f'ARIMA{orden}')

    ax.axvline(corte, color=GOLD, ls='--', lw=1, alpha=0.7)
    ax.axvline(ultimo, color=WHITE, ls='--', lw=1.2, alpha=0.7)

    # el eje Y se ajusta a las velas y al IC 95% para que el 99% no las aplaste
    y_min = min(velas['Low'].min(), pron['inf_95'].min())
    y_max = max(velas['High'].max(), pron['sup_95'].max())
    margen = (y_max - y_min) * 0.04
    ax.set_ylim(y_min - margen, y_max + margen)
    ax.set_xlim(inicio, pron.index[-1] + pd.Timedelta(days=3))

    ax.text(corte, y_max + margen * 0.5, ' inicio test', color=GOLD, fontsize=10, va='top')
    ax.text(ultimo, y_min - margen * 0.5, ' último dato', color=WHITE, fontsize=10,
            va='bottom')

    ax.set_title(f'{ticker} — pronóstico ARIMA{orden} a {len(pron)} días hábiles',
                 fontweight='bold')
    ax.set_ylabel('Precio (USD)')
    ax.legend(loc='upper left', framealpha=0.3, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _cerrar(fig, guardar)


def error_horizonte(tabla_h, test, pred_1paso, pred_multi, guardar=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    ax1.plot(tabla_h.index, tabla_h['RMSE'], color=CYAN, lw=2.2, marker='o', markersize=6)
    ax1.set_title('El error crece con el horizonte', fontsize=12)
    ax1.set_xlabel('Horizonte (días hábiles)')
    ax1.set_ylabel('RMSE (USD)')

    ax2.plot(test.index, test, color=WHITE, lw=1.4, label='Real')
    ax2.plot(pred_1paso.index, pred_1paso, color=CYAN, lw=1.2, ls=':', label='ARIMA a 1 paso')
    multi = pred_multi.loc[test.index]
    ax2.plot(multi.index, multi, color=RED, lw=1.8, ls='--', label='ARIMA multi-paso')
    ax2.set_title('Pronóstico a 1 paso vs multi-paso', fontsize=12)
    ax2.set_ylabel('Precio (USD)')
    ax2.legend(framealpha=0.3, fontsize=9)

    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _cerrar(fig, guardar)


# Volatilidad

def clustering(serie, ret, ticker, ventana=21, guardar=None):
    vol = ret.rolling(ventana).std() * np.sqrt(252)
    umbral = float(vol.quantile(0.90))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [1.4, 1], 'hspace': 0.1})
    ax1.plot(serie.index, serie, color=PURPLE, lw=1.1)
    ax1.set_ylabel('Precio (USD)')
    ax1.set_title(f'{ticker}: precio y volatilidad realizada ({ventana} días)',
                  fontweight='bold')

    ax2.plot(vol.index, vol, color=RED, lw=1.1)
    ax2.fill_between(vol.index, 0, vol, color=RED, alpha=0.25)
    ax2.axhline(umbral, color=GOLD, ls='--', lw=1.2, label=f'Percentil 90 = {umbral:.1f}%')
    ax2.set_ylabel('Volatilidad anualizada (%)')
    ax2.legend(framealpha=0.3, fontsize=9)

    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    _cerrar(fig, guardar)
    return int((vol > umbral).sum())


def volatilidad(v, res, ticker, guardar=None):
    """Volatilidad condicional, pronóstico y residuos estandarizados de un modelo GARCH."""
    vol_cond, vol_lp, vol_pron = v['vol_cond'], v['vol_lp'], v['vol_pron']
    horizonte = len(vol_pron)

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, height_ratios=[1.5, 1], hspace=0.32, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(vol_cond.index, vol_cond, color=RED, lw=1.1, label='Volatilidad condicional')
    ax1.fill_between(vol_cond.index, 0, vol_cond, color=RED, alpha=0.18)
    ax1.axhline(vol_lp, color=GOLD, ls='--', lw=1.4, label=f'Largo plazo = {vol_lp:.1f}%')
    ax1.set_title(f"{ticker} — {v['nombre']}", fontweight='bold')
    ax1.set_ylabel('Volatilidad anualizada (%)')
    ax1.legend(framealpha=0.3, fontsize=9)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(range(1, horizonte + 1), vol_pron, color=CYAN, lw=2.2, marker='o', markersize=3)
    ax2.axhline(vol_lp, color=GOLD, ls='--', lw=1.2, label='Largo plazo')
    ax2.set_title(f'Pronóstico de volatilidad a {horizonte} días', fontsize=12)
    ax2.set_xlabel('Horizonte (días hábiles)')
    ax2.set_ylabel('Vol. anualizada (%)')
    ax2.legend(framealpha=0.3, fontsize=9)

    ax3 = fig.add_subplot(gs[1, 1])
    z = res.std_resid.dropna()
    ax3.hist(z, bins=60, density=True, color=PURPLE, alpha=0.85, edgecolor=BLACK,
             label='Residuos estandarizados')
    x = np.linspace(z.min(), z.max(), 300)
    ax3.plot(x, stats.norm.pdf(x), color=GOLD, lw=1.8, label='N(0, 1)')
    ax3.set_title('Residuos estandarizados', fontsize=12)
    ax3.legend(framealpha=0.3, fontsize=8)

    for ax in (ax1, ax2, ax3):
        ax.grid(True, alpha=0.3)
    _cerrar(fig, guardar)


# VAR

def var_sistema(irf, pron_var, shock, k_ar, guardar=None):
    tickers = list(pron_var.columns)
    colores = dict(zip(tickers, [PURPLE, CYAN, GOLD, GREEN]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'hspace': 0.35})

    for t in tickers:
        ax1.plot(irf['original'].index, irf['original'][t], color=colores[t], lw=2,
                 marker='o', markersize=3, label=t)
        ax1.plot(irf['invertido'].index, irf['invertido'][t], color=colores[t], lw=1.2,
                 ls='--', alpha=0.8, label=f'{t} (orden invertido)')
    ax1.axhline(0, color=WHITE, lw=0.9)
    ax1.set_title(f'Impulso-respuesta a un shock de 1 desviación estándar en {shock}',
                  fontweight='bold')
    ax1.set_xlabel('Días después del shock')
    ax1.set_ylabel('Respuesta (%)')
    ax1.legend(framealpha=0.3, fontsize=8)

    n = len(tickers)
    ancho = 0.8 / n
    for k, t in enumerate(tickers):
        ax2.bar(pron_var.index + (k - (n - 1) / 2) * ancho, pron_var[t], width=ancho,
                color=colores[t], label=t, alpha=0.9)
    ax2.axhline(0, color=WHITE, lw=0.9)
    ax2.set_title(f'Pronóstico VAR({k_ar}) de retornos diarios (%)', fontsize=12)
    ax2.set_xlabel('Días hacia adelante')
    ax2.set_ylabel('Retorno esperado (%)')
    ax2.set_xticks(pron_var.index)
    ax2.legend(framealpha=0.3, fontsize=9)

    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3, axis='y')
    _cerrar(fig, guardar)
