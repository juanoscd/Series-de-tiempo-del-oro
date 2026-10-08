"""
Lecturas para el analista: métricas complementarias y su interpretación.

Cada función imprime un bloque con números y conclusiones que cambian según
los resultados, y devuelve un diccionario con los hallazgos clave. Esos
diccionarios se juntan al final en `resumen` para armar el tablero del analista.

Solo depende de numpy, pandas y scipy; los objetos de statsmodels y arch se
reciben ya ajustados desde el notebook.
"""

import textwrap

import numpy as np
import pandas as pd
from scipy import stats

DIAS = 252
ANCHO = 98
TICKERS_TASA = ('US02Y', 'US05Y', 'US10Y', 'US20Y', 'US30Y', 'TNX', 'TYX', 'FVX', 'IRX')


# ---------------------------------------------------------------------------
# Formato
# ---------------------------------------------------------------------------

def _titulo(texto):
    print(f'\n{texto}\n{"─" * len(texto)}')


def _fila(etiqueta, valor, nota=''):
    print(f'  {etiqueta:<46}{valor:>16}   {nota}'.rstrip())


def _lee(texto):
    print(textwrap.fill(texto, width=ANCHO, initial_indent='  → ', subsequent_indent='    '))


def _p(p):
    return f'{p:.3f}' if p >= 0.001 else '<0.001'


def _sig(p, alfa=0.05):
    return 'significativo' if p < alfa else 'no significativo'


def _usd(x):
    return f'${x:,.0f}'


def _es_oro(ticker):
    return any(k in ticker.upper() for k in ('XAU', 'MGC', 'GC1', 'GC=', 'GLD', 'IAU', 'GOLD'))


def _es_tasa(ticker):
    return any(k in ticker.upper() for k in TICKERS_TASA)


def _corto(ticker):
    """'OANDA:XAUUSD' → 'XAUUSD'."""
    return ticker.split(':')[-1]


# ---------------------------------------------------------------------------
# Pruebas auxiliares
# ---------------------------------------------------------------------------

def ljung_box(x, lags=10):
    """Q de Ljung-Box y su p-valor (H0: sin autocorrelación hasta `lags`)."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    x = x - x.mean()
    n = len(x)
    den = (x ** 2).sum()
    rho = np.array([(x[k:] * x[:-k]).sum() / den for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lags + 1)))
    return q, stats.chi2.sf(q, lags)


def razon_varianzas(r, q):
    """
    Variance ratio de Lo-MacKinlay con z robusto a heterocedasticidad.
    VR = 1 en un paseo aleatorio; > 1 indica persistencia (momentum), < 1 reversión.
    """
    x = np.asarray(pd.Series(r).dropna(), dtype=float)
    t = len(x)
    mu = x.mean()
    e = x - mu
    var1 = (e ** 2).sum() / (t - 1)
    sumas = np.convolve(x, np.ones(q), mode='valid')
    m = q * (t - q + 1) * (1 - q / t)
    varq = ((sumas - q * mu) ** 2).sum() / m
    vr = varq / var1

    e2 = e ** 2
    den = e2.sum() ** 2
    theta = sum((2 * (q - j) / q) ** 2 * t * (e2[j:] * e2[:-j]).sum() / den for j in range(1, q))
    z = (vr - 1) / np.sqrt(theta / t)
    return vr, z, 2 * stats.norm.sf(abs(z))


def kupiec(excepciones, n, p):
    """Prueba POF de Kupiec. H0: la frecuencia de excepciones del VaR es la nominal."""
    x = excepciones
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    elif x == n:
        lr = -2 * n * np.log(p)
    else:
        pi = x / n
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)
                   - (n - x) * np.log(1 - pi) - x * np.log(pi))
    return lr, stats.chi2.sf(lr, 1)


def diebold_mariano(e_modelo, e_bench, h=1):
    """DM con pérdida cuadrática y corrección de Harvey-Leybourne-Newbold."""
    d = np.asarray(e_modelo) ** 2 - np.asarray(e_bench) ** 2
    n = len(d)
    if n < 3 or np.var(d) == 0:
        return np.nan, np.nan
    gamma = [np.mean((d[k:] - d.mean()) * (d[:n - k] - d.mean())) for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    dm = d.mean() / np.sqrt(var_d)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return dm, 2 * stats.t.sf(abs(dm), n - 1)


def _drawdowns(r):
    equity = np.exp(r.cumsum())
    pico = equity.cummax()
    dd = equity / pico - 1
    bajo = dd < 0
    # racha más larga bajo el máximo previo
    grupos = (~bajo).cumsum()
    rachas = bajo.groupby(grupos).sum()
    return dd, int(rachas.max()) if len(rachas) else 0


# ---------------------------------------------------------------------------
# 1. Datos
# ---------------------------------------------------------------------------

def limpieza(df, reporte, ticker):
    r = df['log_ret'].dropna()
    outlier = df.loc[r.index, 'outlier_ret'].astype(bool)
    n = len(r)

    r2 = r ** 2
    peso_outliers = r2[outlier].sum() / r2.sum()
    k = max(1, int(round(0.01 * n)))
    peso_top1 = r2.sort_values(ascending=False).iloc[:k].sum() / r2.sum()
    z99 = stats.norm.ppf(0.995)
    peso_top1_normal = 2 * (z99 * stats.norm.pdf(z99) + stats.norm.sf(z99))

    sigma_rob = 1.4826 * (r - r.median()).abs().median()
    extremos = (r.abs() / sigma_rob).sort_values(ascending=False).head(3)
    huecos = int((r.index.to_series().diff().dt.days > 4).sum())

    _titulo(f'Lectura de los datos: {_corto(ticker)}')
    _fila('Observaciones', f'{n:,}', f"{reporte.get('desde')} a {reporte.get('hasta')}")
    _fila('Retornos marcados como atípicos', f'{outlier.sum()}', f'{outlier.mean():.2%} de los días')
    _fila('Varianza explicada por atípicos', f'{peso_outliers:.1%}')
    _fila('Varianza del 1% de días más extremos', f'{peso_top1:.1%}',
          f'en una normal sería ~{peso_top1_normal:.1%}')
    _fila('Huecos de más de 4 días sin datos', f'{huecos}')
    print('\n  Movimientos más grandes (en desviaciones robustas):')
    for fecha, z in extremos.items():
        print(f'    {fecha.date()}   {r.loc[fecha]:+.2%}   ({z:.1f} σ)')
    print()

    if peso_top1 > 2 * peso_top1_normal:
        _lee(f'El riesgo está concentrado: el 1% de los días explica {peso_top1:.0%} de la varianza '
             f'total, más del doble de lo que pasaría con retornos normales. La volatilidad "promedio" '
             'esconde que unos pocos días hacen casi todo el daño; cualquier métrica basada solo en la '
             'desviación estándar subestima ese riesgo.')
    else:
        _lee(f'La varianza está razonablemente repartida: el 1% de días más extremos explica '
             f'{peso_top1:.0%} del total, cerca de lo que daría una normal.')

    if outlier.sum() > 0:
        if '1!' in ticker or ticker.endswith('=F'):
            _lee('Es un futuro continuo sin ajuste por roll: revisa si las fechas de los atípicos '
                 'coinciden con vencimientos. Si es así, son saltos artificiales que inflan la '
                 'volatilidad, el VaR y la curtosis.')
        else:
            _lee('La serie no tiene rolls, así que los atípicos son movimientos reales del mercado '
                 '(noticias, datos macro, eventos de liquidez) y deben quedarse en el análisis de riesgo.')
    if huecos > 0:
        _lee(f'Hay {huecos} huecos largos en las fechas; conviene verificar que no falten sesiones, '
             'porque un retorno que abarca varios días se ve como un shock de un solo día.')

    return {'n_datos': n, 'peso_top1': peso_top1, 'outliers': int(outlier.sum())}


# ---------------------------------------------------------------------------
# 2. Riesgo
# ---------------------------------------------------------------------------

def riesgo(r, s, precio, multiplicador, ticker, riesgo_usd=500):
    r = r.dropna()
    exposicion = precio * multiplicador
    a_usd = lambda q: abs(np.expm1(q)) * exposicion

    cagr = np.expm1(r.mean() * DIAS)
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(DIAS)
    sortino = r.mean() * DIAS / downside if downside > 0 else np.nan

    dd, dias_bajo = _drawdowns(r)
    mdd = dd.min()
    fecha_valle = dd.idxmin()
    fecha_pico = np.exp(r.cumsum()).loc[:fecha_valle].idxmax()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan

    var95, var99 = r.quantile(0.05), r.quantile(0.01)
    cvar95, cvar99 = r[r <= var95].mean(), r[r <= var99].mean()
    z95 = stats.norm.ppf(0.05)
    ratio_normal = stats.norm.pdf(z95) / 0.05 / abs(z95)
    ratio_cola = cvar95 / var95
    escala_99_95 = var99 / var95
    escala_normal = stats.norm.ppf(0.01) / z95

    ganancias, perdidas = r[r > 0], r[r < 0]
    profit_factor = ganancias.sum() / abs(perdidas.sum())

    vol21 = r.rolling(21).std().dropna() * np.sqrt(DIAS)
    vol_actual = vol21.iloc[-1]
    pct_vol = stats.percentileofscore(vol21, vol_actual)

    contratos = int(riesgo_usd // a_usd(var99)) if a_usd(var99) > 0 else 0

    _titulo(f'Lectura de riesgo: {_corto(ticker)}')
    _fila('Retorno anual compuesto (CAGR)', f'{cagr:.2%}')
    _fila('Volatilidad anual', f"{s['vol_a']:.2%}")
    _fila('Sharpe (rf = 0)', f"{s['sharpe']:.2f}")
    _fila('Sortino (rf = 0)', f'{sortino:.2f}')
    _fila('Máximo drawdown', f'{mdd:.2%}', f'{fecha_pico.date()} → {fecha_valle.date()}')
    _fila('Drawdown actual', f'{dd.iloc[-1]:.2%}')
    _fila('Racha más larga bajo el máximo', f'{dias_bajo} días', f'~{dias_bajo / DIAS:.1f} años')
    _fila('Calmar (CAGR / |MDD|)', f'{calmar:.2f}')
    _fila('Días positivos', f'{(r > 0).mean():.1%}')
    _fila('Ganancia media / pérdida media', f'{ganancias.mean() / abs(perdidas.mean()):.2f}')
    _fila('Profit factor (comprar y mantener)', f'{profit_factor:.2f}')
    print()
    _fila('Exposición por contrato', _usd(exposicion), f'precio {precio:,.2f} × {multiplicador}')
    _fila('VaR 95% / CVaR 95% (histórico)', f'{var95:.2%} / {cvar95:.2%}',
          f'{_usd(a_usd(var95))} / {_usd(a_usd(cvar95))} por contrato')
    _fila('VaR 99% / CVaR 99% (histórico)', f'{var99:.2%} / {cvar99:.2%}',
          f'{_usd(a_usd(var99))} / {_usd(a_usd(cvar99))} por contrato')
    _fila('CVaR95 / VaR95', f'{ratio_cola:.2f}', f'normal: {ratio_normal:.2f}')
    _fila('VaR99 / VaR95', f'{escala_99_95:.2f}', f'normal: {escala_normal:.2f}')
    _fila('Vol. realizada últimos 21 días', f'{vol_actual:.2%}', f'percentil {pct_vol:.0f} de la historia')
    _fila(f'Contratos con riesgo de {_usd(riesgo_usd)}/día', f'{contratos}', 'usando el VaR 99% histórico')
    print()

    if pct_vol >= 80:
        regimen = 'alto'
        _lee(f'Régimen de volatilidad ALTO: el último mes ({vol_actual:.1%} anual) está en el percentil '
             f'{pct_vol:.0f} de su propia historia. Los rangos diarios, el VaR y los stops calculados con '
             'el promedio de toda la muestra se quedan cortos hoy; hay que dimensionar con la volatilidad '
             'reciente o con la del GARCH.')
    elif pct_vol <= 20:
        regimen = 'bajo'
        _lee(f'Régimen de volatilidad BAJO (percentil {pct_vol:.0f}). Las métricas de toda la muestra '
             'exageran el riesgo de hoy, pero la volatilidad revierte a la media: los periodos tranquilos '
             'suelen terminar con un salto, así que no conviene apalancarse por la calma.')
    else:
        regimen = 'normal'
        _lee(f'Régimen de volatilidad normal (percentil {pct_vol:.0f}): las métricas históricas son una '
             'referencia razonable para el riesgo actual.')

    if s['sharpe'] > 0:
        # con retornos simétricos la semidesviación es σ/√2, así que Sortino ≈ √2 · Sharpe
        cociente = sortino / s['sharpe']
        if cociente > 1.1 * np.sqrt(2):
            _lee(f'Sortino / Sharpe = {cociente:.2f}, por encima de √2 ≈ 1.41 (lo que daría una distribución '
                 'simétrica): la volatilidad a la baja es menor que la de alza. Para quien está largo, la '
                 'desviación estándar exagera el riesgo "malo".')
        elif cociente < 0.9 * np.sqrt(2):
            _lee(f'Sortino / Sharpe = {cociente:.2f}, por debajo de √2 ≈ 1.41: la volatilidad se concentra en '
                 'las caídas. La desviación estándar subestima el riesgo de una posición larga.')
        else:
            _lee(f'Sortino / Sharpe = {cociente:.2f}, cerca de √2 ≈ 1.41: la volatilidad a la baja y al alza '
                 'pesan parecido.')

    _lee(f'Comprar y mantener habría exigido aguantar una caída de {abs(mdd):.1%} y hasta {dias_bajo} días '
         f'hábiles sin recuperar el máximo. El Calmar de {calmar:.2f} resume cuánto retorno anual hubo '
         'por cada punto de drawdown máximo' + (': un valor por encima de 1 es sólido.' if calmar > 1
                                                else '; por debajo de 1 la recompensa no compensa bien el peor tramo.'))

    if ratio_cola > ratio_normal * 1.08 or escala_99_95 > escala_normal * 1.08:
        _lee(f'Las colas son más pesadas que las de una normal: el CVaR95 es {ratio_cola:.2f} veces el VaR95 '
             f'(normal {ratio_normal:.2f}) y el VaR99 es {escala_99_95:.2f} veces el VaR95 (normal '
             f'{escala_normal:.2f}). Cuando se rompe el VaR, la pérdida promedio es mayor de lo que sugiere '
             'una campana; el CVaR es la cifra que debe guiar el capital mínimo por contrato.')

    if s['skew'] < -0.2:
        _lee(f'Asimetría negativa ({s["skew"]:.2f}): las caídas fuertes son más frecuentes que las subidas '
             'fuertes. Los stops pueden ejecutarse con deslizamiento en esos días.')
    elif s['skew'] > 0.2:
        _lee(f'Asimetría positiva ({s["skew"]:.2f}): los saltos grandes tienden a ser al alza, algo típico '
             'de activos refugio en episodios de estrés.' if _es_oro(ticker) else
             f'Asimetría positiva ({s["skew"]:.2f}): los saltos grandes tienden a ser al alza.')

    if contratos == 0:
        _lee(f'Con {_usd(riesgo_usd)} de riesgo diario no cabe ni un contrato: un día malo al 1% cuesta '
             f'{_usd(a_usd(var99))}. Hace falta subir el presupuesto de riesgo o usar un instrumento más chico.')
    else:
        _lee(f'Con un presupuesto de {_usd(riesgo_usd)} de pérdida diaria al 99%, el tamaño máximo es '
             f'{contratos} contrato(s). Ese número cambia con el régimen: en la sección 5 se recalcula con la '
             'volatilidad condicional del GARCH, que es la más relevante para mañana.')

    return {'cagr': cagr, 'sharpe': s['sharpe'], 'sortino': sortino, 'mdd': mdd,
            'var99_usd': a_usd(var99), 'cvar99_usd': a_usd(cvar99), 'regimen_hist': regimen,
            'pct_vol21': pct_vol, 'vol21': vol_actual, 'contratos_hist': contratos}


def distribucion(r, ajuste):
    r = r.dropna()
    n = len(r)
    mu, sigma = ajuste['normal']
    gl, loc, escala = ajuste['t']
    jb_p = ajuste['jb'][1]
    ext = ajuste['extremos']
    tv = ajuste['tabla_var']

    curt_t = 6 / (gl - 4) if gl > 4 else np.inf
    filas_bt = []
    for conf, alfa in (('95%', 0.05), ('99%', 0.01)):
        for metodo in ('Normal', 't de Student', 'Histórico'):
            q = tv.loc[conf, metodo]
            x = int((r < q).sum())
            _, p = kupiec(x, n, alfa)
            filas_bt.append((conf, metodo, q, x, n * alfa, p))

    # frecuencia de días a -3σ y -4σ: normal vs datos
    frec = []
    for k in (3, 4):
        obs = int((r < mu - k * sigma).sum())
        p_norm = stats.norm.cdf(-k)
        anios_norm = 1 / (p_norm * DIAS)
        anios_obs = n / obs / DIAS if obs else np.inf
        frec.append((k, obs, n * p_norm, anios_norm, anios_obs))

    sub99 = (tv.loc['99%', 'Normal'] - tv.loc['99%', 'Histórico']) / abs(tv.loc['99%', 'Histórico'])

    _titulo('Lectura de la distribución')
    _fila('Grados de libertad de la t', f'{gl:.2f}')
    _fila('Curtosis (exceso) implícita en la t', 'no existe (gl ≤ 4)' if np.isinf(curt_t) else f'{curt_t:.2f}')
    _fila('Curtosis (exceso) muestral', f'{r.kurtosis():.2f}')
    _fila('Días a ±3σ: observados / normal', f"{ext['observados']} / {ext['esperados_normal']:.1f}",
          f"{ext['observados'] / ext['esperados_normal']:.1f} veces lo esperado")
    print('\n  Caídas extremas: cada cuánto ocurren')
    for k, obs, esp, an_n, an_o in frec:
        texto_obs = f'cada {an_o:.1f} años' if np.isfinite(an_o) else 'nunca en la muestra'
        print(f'    -{k}σ: {obs} días (normal esperaría {esp:.1f}). '
              f'Normal: una vez cada {an_n:,.0f} años | datos: {texto_obs}')

    print('\n  Backtest del VaR dentro de muestra (Kupiec, H0: frecuencia correcta)')
    print(f'    {"Nivel":<6}{"Método":<14}{"VaR":>9}{"Excepciones":>13}{"Esperadas":>11}{"p":>8}')
    for conf, metodo, q, x, esp, p in filas_bt:
        marca = '' if p >= 0.05 else '  ← rechaza'
        print(f'    {conf:<6}{metodo:<14}{q:>9.2%}{x:>13}{esp:>11.1f}{_p(p):>8}{marca}')
    print()

    if gl < 4:
        _lee(f'Con {gl:.1f} grados de libertad la cuarta potencia de los retornos no tiene media finita: '
             'la curtosis muestral no converge y cambia mucho al agregar o quitar un solo día extremo. '
             'No conviene usarla como métrica de riesgo; el CVaR y la t son más estables.')
    elif gl < 8:
        _lee(f'{gl:.1f} grados de libertad es la zona típica de activos financieros líquidos: colas claramente '
             'más pesadas que la normal, pero con varianza y curtosis finitas.')
    else:
        _lee(f'{gl:.1f} grados de libertad: colas solo moderadamente más pesadas que la normal.')

    rechaza_normal = [f for f in filas_bt if f[1] == 'Normal' and f[5] < 0.05]
    if rechaza_normal or sub99 > 0.05:
        _lee(f'La normal falla donde importa: al 99% el VaR normal es {abs(tv.loc["99%", "Normal"]):.2%} contra '
             f'{abs(tv.loc["99%", "Histórico"]):.2%} histórico ({abs(sub99):.0%} '
             f'{"menos" if sub99 > 0 else "más"} exigente). Un desk que use VaR normal tendrá más excepciones '
             'de las prometidas, justo en los días en que más duele.')
    _lee('Ojo con la lectura del backtest: es dentro de muestra (el VaR se estimó con los mismos datos) y '
         'no condiciona por volatilidad. Un VaR incondicional puede pasar Kupiec en el total y aun así '
         'concentrar las excepciones en los periodos agitados; por eso el VaR del GARCH (sección 5) es el '
         'que sirve para el día a día.')

    return {'gl_t': gl, 'normal_subestima': bool(rechaza_normal or sub99 > 0.05)}


# ---------------------------------------------------------------------------
# 3. Estructura temporal
# ---------------------------------------------------------------------------

def revisar_configuracion(precios, sistema, principal, corte):
    """Advertencias sobre la configuración antes de modelar."""
    _titulo('Chequeo de la configuración')
    n_test = int((precios.index >= pd.Timestamp(corte)).sum())
    avisos = 0

    if n_test < 30:
        avisos += 1
        _lee(f'El periodo de prueba tiene {n_test} días. Con tan pocas observaciones ninguna métrica '
             'fuera de muestra (RMSE, U de Theil, Diebold-Mariano, cobertura de bandas) tiene potencia '
             'estadística: un par de días buenos o malos cambia la conclusión. Para evaluar en serio, '
             'mueve CORTE de modo que el test tenga al menos 60 días, idealmente 250.')
    tasas = [t for t in sistema if _es_tasa(t)]
    for t in tasas:
        avisos += 1
        _lee(f'{_corto(t)} es un rendimiento, no un precio. Su "retorno logarítmico" es el cambio relativo de la '
             'tasa (4.00% → 4.04% = +1%), no el retorno de un bono, y tiene el signo contrario: si la tasa '
             'sube, el bono cae. Las conclusiones de signo del VAR siguen valiendo, pero la magnitud se lee '
             'mejor en puntos básicos (diferencias simples × 100) o con un instrumento negociable como ZN1! o TLT.')
    if _es_oro(principal) and tasas:
        _lee('Relación esperada a priori: el oro no paga cupón, así que su costo de oportunidad sube con '
             'las tasas reales. Lo habitual es una correlación diaria negativa entre oro y rendimientos, '
             'más fuerte en periodos dominados por la Fed que en periodos de compras de bancos centrales '
             'o de estrés geopolítico.')
    if not avisos:
        print('  Sin observaciones.')
    return {'n_test': n_test}


def estacionariedad(tabla_est, ret, principal):
    vr = [(q, *razon_varianzas(ret, q)) for q in (2, 5, 10, 20)]

    _titulo('Lectura de estacionariedad y eficiencia')
    for nombre, fila in tabla_est.iterrows():
        _fila(nombre, fila['Veredicto'], f"ADF p = {fila['ADF p']}, KPSS p = {fila['KPSS p']}")
    print('\n  Razón de varianzas de Lo-MacKinlay (paseo aleatorio: VR = 1)')
    print(f'    {"q (días)":<10}{"VR":>8}{"z robusto":>12}{"p":>9}   lectura')
    for q, v, z, p in vr:
        if p >= 0.05:
            txt = 'consistente con paseo aleatorio'
        else:
            txt = 'persistencia (momentum)' if v > 1 else 'reversión a la media'
        print(f'    {q:<10}{v:>8.3f}{z:>12.2f}{_p(p):>9}   {txt}')
    print()

    filas = list(tabla_est.index)
    precio_ok = tabla_est.loc[filas[0], 'Veredicto'] != 'Estacionaria'
    _lee('El precio tiene raíz unitaria: no regresa a ningún nivel "justo" por sí solo. Cualquier '
         'estrategia de reversión sobre el precio en nivel (comprar porque "está barato") no tiene respaldo '
         'estadístico; la reversión solo tiene sentido sobre spreads cointegrados o sobre la volatilidad.'
         if precio_ok else
         'Las pruebas no confirman raíz unitaria en el precio, algo raro en un activo financiero. Suele '
         'pasar con muestras cortas o con un rango lateral largo; no conviene apostar a esa reversión sin '
         'un argumento económico.')

    fila_r2 = [f for f in filas if '²' in f]
    if fila_r2 and tabla_est.loc[fila_r2[0], 'Veredicto'] != 'Estacionaria':
        _lee('Los retornos al cuadrado salen no estacionarios o ambiguos. No es que la varianza explote: es '
             'la firma de una volatilidad muy persistente (cercana a IGARCH), que las pruebas confunden con '
             'raíz unitaria. Se confirma en la persistencia del GARCH.')

    sig = [(q, v) for q, v, z, p in vr if p < 0.05]
    if not sig:
        _lee('Ningún horizonte de 2 a 20 días muestra desviación significativa del paseo aleatorio. No hay '
             'evidencia de tendencia ni de reversión sistemática en los retornos acumulados: las señales '
             'direccionales deben venir de otra información (flujos, macro, posicionamiento), no del '
             'patrón de precios pasado.')
    else:
        q, v = sig[-1]
        tipo = 'momentum (los movimientos tienden a extenderse)' if v > 1 else 'reversión (los movimientos tienden a corregirse)'
        _lee(f'A {q} días hay {tipo}, con VR = {v:.2f}. Antes de operarlo: el efecto debe sobrevivir a costos, '
             'mantenerse en submuestras y no depender de un par de episodios extremos.')
    return {'vr_sig': bool(sig), 'precio_raiz_unitaria': precio_ok}


def autocorrelacion(ret, ac, principal):
    r = ret.dropna()
    n = len(r)
    banda = 1.96 / np.sqrt(n)
    rho1 = r.autocorr(1)
    rho1_2 = (r ** 2).autocorr(1)
    lb_r = ljung_box(r, 10)
    lb_r2 = ljung_box(r ** 2, 10)
    lb_abs = ljung_box(r.abs(), 10)

    fuera = np.where(np.abs(ac['acf_cuadrado'][1:]) < ac['banda'])[0]
    memoria = int(fuera[0] + 1) if len(fuera) else ac['nlags']

    # efecto apalancamiento: ¿el signo de hoy anticipa la magnitud de mañana?
    lev = np.corrcoef(r.iloc[:-1].values, (r ** 2).iloc[1:].values)[0, 1]
    lev_abs = np.corrcoef(r.iloc[:-1].values, r.abs().iloc[1:].values)[0, 1]

    _titulo('Lectura de autocorrelación')
    _fila('ρ(1) de los retornos', f'{rho1:+.4f}', f'banda ±{banda:.4f}')
    _fila('R² implícito del día siguiente', f'{rho1 ** 2:.3%}')
    _fila('Ljung-Box(10) retornos', f'p = {_p(lb_r[1])}')
    _fila('ρ(1) de los retornos²', f'{rho1_2:+.4f}')
    _fila('Ljung-Box(10) retornos²', f'p = {_p(lb_r2[1])}')
    _fila('Ljung-Box(10) |retornos|', f'p = {_p(lb_abs[1])}')
    _fila('Primer rezago con ACF(r²) no significativa', f'{memoria}')
    _fila('corr(r hoy, r² mañana)', f'{lev:+.4f}', f'banda ±{banda:.4f}')
    _fila('corr(r hoy, |r| mañana)', f'{lev_abs:+.4f}')
    print()

    if abs(rho1) > banda:
        tipo = 'reversión de un día al otro' if rho1 < 0 else 'continuación de un día al otro'
        _lee(f'ρ(1) = {rho1:+.3f} es estadísticamente significativo ({tipo}), pero explica apenas '
             f'{rho1 ** 2:.2%} de la varianza del día siguiente. Con muestras grandes cualquier efecto '
             'diminuto sale significativo; después de spread y comisiones esto rara vez deja dinero.')
    else:
        _lee('La autocorrelación de los retornos es indistinguible de cero: lo que pasó ayer no dice nada '
             'útil sobre la dirección de hoy. Es lo esperable en un mercado líquido y eficiente.')

    if lb_r2[1] < 0.05:
        _lee(f'La magnitud sí tiene memoria (Ljung-Box de r² con p {_p(lb_r2[1])}); la ACF de r² se mantiene '
             f'significativa hasta el rezago {memoria}. Un día agitado anuncia más días agitados: es la base '
             'para ajustar el tamaño de la posición y la distancia de los stops según la volatilidad reciente.')

    if abs(lev) > banda:
        if lev < 0:
            _lee(f'corr(r, r² siguiente) = {lev:+.3f}: las caídas anticipan más volatilidad que las subidas '
                 '(efecto apalancamiento clásico, típico de índices accionarios).')
        else:
            extra = (' En el oro esto es coherente con su papel de refugio: los repuntes fuertes llegan con '
                     'miedo en el mercado y vienen acompañados de más movimiento.' if _es_oro(principal) else '')
            _lee(f'corr(r, r² siguiente) = {lev:+.3f}: las subidas anticipan más volatilidad que las caídas '
                 f'(efecto apalancamiento invertido).{extra}')
    else:
        _lee('No se detecta asimetría simple entre el signo de hoy y la volatilidad de mañana; el GJR o el '
             'EGARCH dirán si aparece una vez controlada la persistencia.')

    return {'rho1': rho1, 'memoria_vol': memoria, 'arch_lb_p': lb_r2[1], 'apalancamiento': lev}


# ---------------------------------------------------------------------------
# 4. ARIMA
# ---------------------------------------------------------------------------

def arima(modelo, orden, resid, tabla_aic, train, test):
    params, pvalues = modelo.params, modelo.pvalues
    arma = [k for k in params.index if k.startswith(('ar.', 'ma.'))]
    sig = [k for k in arma if pvalues[k] < 0.05]

    delta_2 = tabla_aic['AIC'].iloc[1] - tabla_aic['AIC'].iloc[0] if len(tabla_aic) > 1 else np.nan
    simples = tabla_aic[(tabla_aic['p'] + tabla_aic['q']) == 1]
    delta_simple = simples['AIC'].min() - tabla_aic['AIC'].iloc[0] if len(simples) else np.nan

    sigma = np.sqrt(params.get('sigma2', np.nan))
    lb_res = ljung_box(resid, 10)
    lb_res2 = ljung_box(resid ** 2, 10)
    jb_p = stats.jarque_bera(resid.dropna())[1]

    cancelan = False
    if 'ar.L1' in params and 'ma.L1' in params:
        cancelan = abs(params['ar.L1']) > 0.3 and abs(params['ar.L1'] + params['ma.L1']) < 0.15

    _titulo(f'Lectura del ARIMA{orden}')
    _fila('Coeficientes ARMA significativos', f'{len(sig)} de {len(arma)}', ', '.join(sig))
    _fila('ΔAIC frente al segundo mejor', f'{delta_2:.2f}')
    _fila('ΔAIC frente al mejor modelo de 1 parámetro', f'{delta_simple:.2f}')
    _fila('Desv. estándar del error (1 día)', f'{sigma:,.2f}', f'{sigma / train.iloc[-1]:.2%} del último precio')
    _fila('Ljung-Box(10) residuos', f'p = {_p(lb_res[1])}')
    _fila('Ljung-Box(10) residuos²', f'p = {_p(lb_res2[1])}')
    _fila('Jarque-Bera residuos', f'p = {_p(jb_p)}')
    _fila('Observaciones train / test', f'{len(train):,} / {len(test):,}')
    print()

    if delta_simple < 2:
        _lee(f'El orden elegido mejora el AIC en solo {delta_simple:.1f} puntos frente a un modelo de un '
             'parámetro. Diferencias menores a 2 no distinguen modelos: por parsimonia, el modelo más simple '
             'es igual de válido.')
    elif delta_2 < 2:
        _lee(f'El segundo mejor modelo queda a {delta_2:.1f} puntos de AIC: la elección del orden es frágil y '
             'puede cambiar con unos días más de datos. No conviene darle una interpretación económica al orden.')
    if cancelan:
        _lee(f'ar.L1 = {params["ar.L1"]:.3f} y ma.L1 = {params["ma.L1"]:.3f} casi se cancelan: las raíces AR y '
             'MA se anulan entre sí y el modelo se comporta como un paseo aleatorio con parámetros de adorno. '
             'Es una señal típica de sobreajuste.')
    if not sig:
        _lee('Ningún coeficiente AR o MA es significativo: en la práctica el modelo es un paseo aleatorio.')

    if lb_res[1] >= 0.05:
        _lee('Los residuos no tienen autocorrelación: el ARIMA extrajo toda la estructura lineal que había '
             '(que era poca).')
    else:
        _lee(f'Quedó autocorrelación en los residuos (p {_p(lb_res[1])}). Puede faltar un rezago, pero en precios '
             'de este tipo suele deberse a cambios de régimen más que a dinámica lineal explotable.')
    if lb_res2[1] < 0.05:
        _lee('Los residuos al cuadrado sí están autocorrelacionados: el error del ARIMA no tiene varianza '
             'constante. Las bandas del pronóstico suponen varianza fija, así que son demasiado anchas en '
             'épocas calmas y demasiado estrechas en épocas agitadas. Para intervalos honestos hay que '
             'combinar la media con la volatilidad del GARCH.')

    return {'orden': orden, 'arima_coef_sig': len(sig), 'arima_resid_arch': lb_res2[1] < 0.05}


def evaluacion(serie, train, test, pred_1paso, pron, multiplicador=1):
    real = test
    ayer = serie.shift(1).loc[test.index]
    pred = pd.Series(pred_1paso, index=test.index)
    e_m, e_rw = real - pred, real - ayer
    n = len(real)

    rmse_m, rmse_rw = np.sqrt((e_m ** 2).mean()), np.sqrt((e_rw ** 2).mean())
    dm, p_dm = diebold_mariano(e_m.values, e_rw.values)

    dir_pred = np.sign(pred - ayer)
    dir_real = np.sign(real - ayer)
    validos = dir_pred != 0
    aciertos = int((dir_pred[validos] == dir_real[validos]).sum())
    n_dir = int(validos.sum())
    p_dir = stats.binomtest(aciertos, n_dir, 0.5).pvalue if n_dir > 0 else np.nan

    media = pron['media'].reindex(test.index)
    sesgo = (real - media).mean()
    cobertura = {}
    for nivel in (80, 95, 99):
        inf, sup = f'inf_{nivel}', f'sup_{nivel}'
        if inf in pron and sup in pron:
            li, ls = pron[inf].reindex(test.index), pron[sup].reindex(test.index)
            cobertura[nivel] = float(((real >= li) & (real <= ls)).mean())

    _titulo('Lectura de la evaluación fuera de muestra')
    _fila('Días de prueba', f'{n}')
    _fila('RMSE ARIMA a 1 paso', f'{rmse_m:,.2f}', f'{_usd(rmse_m * multiplicador)} por contrato')
    _fila('RMSE paseo aleatorio', f'{rmse_rw:,.2f}', f'{_usd(rmse_rw * multiplicador)} por contrato')
    _fila('U de Theil a 1 paso', f'{rmse_m / rmse_rw:.3f}')
    _fila('Diebold-Mariano (H0: igual precisión)', f'{dm:+.2f}', f'p = {_p(p_dm)}' if np.isfinite(p_dm) else '')
    _fila('Acierto de dirección a 1 paso', f'{aciertos}/{n_dir}' if n_dir else 'n/a',
          f'{aciertos / n_dir:.1%}, binomial p = {_p(p_dir)}' if n_dir else '')
    _fila('Sesgo medio del multi-paso (real − pron.)', f'{sesgo:+,.2f}')
    for nivel, c in cobertura.items():
        _fila(f'Cobertura banda {nivel}%', f'{c:.0%}', f'nominal {nivel}%')
    print()

    if n < 30:
        _lee(f'Con {n} días de prueba estas cifras son anecdóticas: el intervalo de confianza de una tasa de '
             f'acierto con n = {n} va de aproximadamente ±{1.96 * np.sqrt(0.25 / max(n, 1)):.0%} alrededor de '
             'la estimación. Sirven para ver que el código funciona, no para concluir.')

    if np.isfinite(p_dm) and p_dm < 0.05:
        quien = 'el ARIMA' if dm < 0 else 'el paseo aleatorio'
        _lee(f'Diebold-Mariano rechaza igual precisión: {quien} pronostica mejor en este test. Si gana el '
             'ARIMA, hay que confirmarlo con re-estimación móvil (walk-forward) antes de creerlo.')
    else:
        _lee(f'Diebold-Mariano no distingue al ARIMA del paseo aleatorio (U = {rmse_m / rmse_rw:.3f}). El '
             'modelo no agrega información sobre el precio de mañana más allá del precio de hoy: la '
             'hipótesis de mercado eficiente en forma débil no se rechaza.')

    if n_dir:
        if p_dir < 0.05 and aciertos / n_dir > 0.5:
            _lee(f'El acierto direccional ({aciertos / n_dir:.0%}) supera al azar con significancia. Es lo único '
                 'que importaría para operar; validar en otra ventana y con costos.')
        else:
            _lee(f'Acierto direccional de {aciertos / n_dir:.0%}: estadísticamente igual a lanzar una moneda.')

    err_multi = (real - media).dropna()
    con_tendencia = len(err_multi) >= 10 and abs(sesgo) > 1.5 * err_multi.std()
    if con_tendencia:
        sentido = 'al alza' if sesgo > 0 else 'a la baja'
        _lee(f'El error multi-paso tiene un sesgo de {sesgo:+,.0f} puntos: el precio se fue {sentido} durante el '
             'test y el ARIMA, sin constante, proyecta casi plano desde el último dato del entrenamiento. El '
             'error refleja la tendencia del periodo, que ningún modelo de precio sin información externa podía '
             'anticipar.')

    for nivel, c in cobertura.items():
        if nivel == 95 and n >= 10:
            if c < 0.85 and con_tendencia:
                _lee(f'La banda del 95% contuvo solo {c:.0%} de los precios. La causa principal es la tendencia '
                     'del test, no solo la volatilidad: las bandas se abren con √h pero la media no se mueve.')
            elif c < 0.85:
                _lee(f'La banda del 95% contuvo solo {c:.0%} de los precios: las bandas son demasiado estrechas '
                     'para el régimen actual. Coherente con residuos heterocedásticos: el ARIMA usa una varianza '
                     'promedio de toda la muestra.')
            elif c == 1.0 and nivel == 95:
                _lee('Todos los precios cayeron dentro de la banda del 95%. Si el test es corto o tranquilo, '
                     'las bandas pueden estar exageradas por la volatilidad de episodios pasados.')

    return {'u_theil': rmse_m / rmse_rw, 'dm_p': p_dm,
            'acierto_dir': aciertos / n_dir if n_dir else np.nan, 'cobertura_95': cobertura.get(95)}


# ---------------------------------------------------------------------------
# 5. Volatilidad
# ---------------------------------------------------------------------------

def efecto_arch(lm, p_lm, dias_altos, n):
    _titulo('Lectura del efecto ARCH')
    _fila('ARCH-LM(10)', f'{lm:,.1f}', f'p = {_p(p_lm)}')
    _fila('Días sobre el percentil 90 de vol.', f'{dias_altos}', f'{dias_altos / n:.1%} de la muestra')
    print()
    if p_lm < 0.05:
        _lee('Hay heterocedasticidad condicional clara: la varianza de hoy depende de los shocks recientes. '
             'Usar una sola volatilidad histórica para todo el periodo mezcla regímenes muy distintos. '
             'Modelar la varianza condicional está justificado.')
    else:
        _lee('No se detecta efecto ARCH con 10 rezagos; un GARCH aportaría poco sobre la volatilidad histórica.')
    return {'arch_p': p_lm}


def _var_condicional(res, sigma_d, mu, alfa):
    """VaR a 1 día en % con la distribución de las innovaciones del modelo."""
    try:
        dist = res.model.distribution
        nombres = dist.parameter_names()
        params = res.params[nombres].values if nombres else None
        q = float(np.asarray(dist.ppf(alfa, params)).ravel()[0])
    except Exception:
        q = stats.norm.ppf(alfa)
    return mu + sigma_d * q


def _impacto(v, z=2.0):
    """Cuánto más sube la varianza tras un shock de −zσ que tras uno de +zσ."""
    a, g = v['alpha'], v['gamma']
    if v['es_egarch']:
        c = np.sqrt(2 / np.pi)
        neg, pos = a * (z - c) - g * z, a * (z - c) + g * z
        return np.exp(neg - pos)
    if a <= 0:
        return np.nan
    return max(a + g, 1e-6) / a


def volatilidad(garch, v, ret, precio, multiplicador, horizonte, riesgo_usd=500,
                tabla_vol=None, principal=''):
    raiz = np.sqrt(DIAS)
    params, pvalues = garch.params, garch.pvalues
    mu = params.get('mu', 0.0)

    vol_cond = v['vol_cond']
    vol_hoy_a = vol_cond.iloc[-1]
    vol_man_a = v['vol_pron'][0]
    vol_fin_a = v['vol_pron'][-1]
    vol_lp_a = v['vol_lp']
    pct = stats.percentileofscore(vol_cond, vol_hoy_a)

    sig_man = vol_man_a / raiz                          # % diario
    sig_h = np.sqrt(np.sum((np.asarray(v['vol_pron']) / raiz) ** 2))   # % acumulado en el horizonte

    var95 = _var_condicional(garch, sig_man, mu, 0.05)
    var99 = _var_condicional(garch, sig_man, mu, 0.01)
    var99_hist = ret.quantile(0.01)
    exposicion = precio * multiplicador
    usd = lambda pct_: abs(np.expm1(pct_ / 100)) * exposicion
    contratos = int(riesgo_usd // usd(var99)) if usd(var99) > 0 else 0
    stop_pts = precio * 2 * sig_man / 100

    rango = lambda s, k: (precio * np.exp(-k * s / 100), precio * np.exp(k * s / 100))
    r1_man, r2_man = rango(sig_man, 1), rango(sig_man, 2)
    r1_h, r2_h = rango(sig_h, 1), rango(sig_h, 2)

    z = garch.std_resid.dropna()
    lb_z2 = ljung_box(z ** 2, 10)
    lb_z = ljung_box(z, 10)
    r2 = ((ret - mu) ** 2).reindex(garch.conditional_volatility.index)
    mz_r2 = np.corrcoef(r2.dropna(), (garch.conditional_volatility ** 2).loc[r2.dropna().index])[0, 1] ** 2

    impacto = _impacto(v) if v['asimetria'] else np.nan
    no_sig = [k for k in params.index if pvalues[k] >= 0.05 and k != 'mu']

    _titulo(f'Lectura de la volatilidad: {v["nombre"]}')
    if tabla_vol is not None and len(tabla_vol) > 1:
        _fila('ΔAIC frente al segundo modelo', f'{tabla_vol["AIC"].iloc[1] - tabla_vol["AIC"].iloc[0]:.2f}',
              tabla_vol.index[1])
        base = [i for i in tabla_vol.index if i.startswith('GARCH(1,1)')]
        if base:
            _fila('ΔAIC frente a GARCH(1,1)', f'{tabla_vol.loc[base[0], "AIC"] - tabla_vol["AIC"].iloc[0]:.2f}', base[0])
    _fila('Parámetros no significativos', f'{len(no_sig)}', ', '.join(no_sig))
    _fila('Persistencia', f'{v["persistencia"]:.4f}', f'vida media {v["vida_media"]:.1f} días')
    if v['asimetria']:
        _fila('Impacto de −2σ vs +2σ en la varianza', f'{impacto:.2f}x')
    print()
    _fila('Volatilidad hoy (anual)', f'{vol_hoy_a:.1f}%', f'percentil {pct:.0f} del historial del modelo')
    _fila('Volatilidad de largo plazo (anual)', f'{vol_lp_a:.1f}%', f'hoy / largo plazo = {vol_hoy_a / vol_lp_a:.2f}')
    _fila('Pronóstico mañana (diaria)', f'{sig_man:.2f}%', f'{vol_man_a:.1f}% anual')
    _fila(f'Pronóstico día {horizonte} (anual)', f'{vol_fin_a:.1f}%')
    _fila(f'Volatilidad acumulada a {horizonte} días', f'{sig_h:.2f}%')
    print()
    _fila('Precio de referencia', f'{precio:,.2f}')
    _fila('Rango ±1σ mañana (~68%)', f'{r1_man[0]:,.0f} – {r1_man[1]:,.0f}')
    _fila('Rango ±2σ mañana (~95%)', f'{r2_man[0]:,.0f} – {r2_man[1]:,.0f}')
    _fila(f'Rango ±1σ a {horizonte} días', f'{r1_h[0]:,.0f} – {r1_h[1]:,.0f}')
    _fila(f'Rango ±2σ a {horizonte} días', f'{r2_h[0]:,.0f} – {r2_h[1]:,.0f}')
    print()
    _fila('VaR condicional 95% mañana', f'{var95:.2f}%', f'{_usd(usd(var95))} por contrato')
    _fila('VaR condicional 99% mañana', f'{var99:.2f}%', f'{_usd(usd(var99))} por contrato')
    _fila('VaR 99% histórico (incondicional)', f'{var99_hist:.2f}%', f'condicional / histórico = {var99 / var99_hist:.2f}')
    _fila(f'Contratos con {_usd(riesgo_usd)} de riesgo', f'{contratos}', 'según VaR 99% condicional')
    _fila('Stop a 2σ diarios', f'{stop_pts:,.2f} pts', f'{_usd(stop_pts * multiplicador)} por contrato')
    print()
    _fila('Ljung-Box(10) residuos estandarizados²', f'p = {_p(lb_z2[1])}')
    _fila('Ljung-Box(10) residuos estandarizados', f'p = {_p(lb_z[1])}')
    _fila('Curtosis (exceso) residuos estandarizados', f'{z.kurtosis():.2f}', f'retornos crudos: {ret.kurtosis():.2f}')
    _fila('R² de r² sobre σ² (dentro de muestra)', f'{mz_r2:.1%}')
    print()

    # interpretación
    if pct >= 80:
        regimen = 'alto'
        _lee(f'Régimen de volatilidad ALTO: la volatilidad condicional ({vol_hoy_a:.1f}% anual) está en el '
             f'percentil {pct:.0f} de su historia y {vol_hoy_a / vol_lp_a:.1f} veces su nivel de largo plazo.')
    elif pct <= 20:
        regimen = 'bajo'
        _lee(f'Régimen de volatilidad BAJO: percentil {pct:.0f}, {vol_hoy_a / vol_lp_a:.2f} veces el largo plazo.')
    else:
        regimen = 'normal'
        _lee(f'Régimen de volatilidad normal: percentil {pct:.0f}, {vol_hoy_a / vol_lp_a:.2f} veces el largo plazo.')

    direccion = 'bajando hacia' if vol_fin_a < vol_hoy_a else 'subiendo hacia'
    _lee(f'El modelo proyecta la volatilidad {direccion} su nivel de largo plazo: de {vol_hoy_a:.1f}% hoy a '
         f'{vol_fin_a:.1f}% en {horizonte} días. Con una vida media de {v["vida_media"]:.0f} días, la mitad de '
         'la distancia actual al largo plazo se cierra en ese plazo.')

    if v['persistencia'] > 0.99:
        _lee(f'Persistencia de {v["persistencia"]:.3f}: el proceso está muy cerca de IGARCH. Los shocks casi no '
             'se disipan, la volatilidad de largo plazo queda mal estimada y los pronósticos a horizontes '
             'largos dependen mucho de ella. Es común cuando la muestra mezcla regímenes distintos; un modelo '
             'con quiebres o una ventana más corta puede bajar la persistencia.')

    if v['asimetria'] and np.isfinite(impacto):
        p_gamma = min((pvalues[k] for k in params.index if k.startswith('gamma[')), default=1.0)
        if impacto > 1:
            _lee(f'Un shock de −2σ sube la varianza de mañana {impacto:.1f} veces más que uno de +2σ '
                 f'(γ {_sig(p_gamma)}, p {_p(p_gamma)}): las caídas generan más miedo que las subidas.')
        else:
            extra = (' En el oro es la firma del activo refugio: los repuntes vienen con estrés en otros mercados.'
                     if _es_oro(principal) else '')
            _lee(f'Un shock de +2σ sube la varianza de mañana {1 / impacto:.1f} veces más que uno de −2σ '
                 f'(γ {_sig(p_gamma)}, p {_p(p_gamma)}): asimetría invertida.{extra} Para quien está corto, '
                 'el riesgo de cola es mayor de lo que dice la volatilidad simétrica.')

    if 'lambda' in params:
        lam, p_lam = params['lambda'], pvalues['lambda']
        lado = 'izquierda (caídas)' if lam < 0 else 'derecha (subidas)'
        _lee(f'La t asimétrica tiene λ = {lam:.3f} ({_sig(p_lam)}): incluso con la volatilidad controlada, la '
             f'cola más larga está a la {lado}.')

    _lee(f'Para operar: mañana un movimiento normal (±1σ) va de {r1_man[0]:,.0f} a {r1_man[1]:,.0f}. Un stop '
         f'a 1σ ({precio * sig_man / 100:,.1f} pts) queda dentro del ruido: un cierre así en contra pasa '
         f'más o menos 1 de cada 6 días aunque la idea sea buena. A 2σ ({stop_pts:,.1f} pts) baja a cerca de '
         '1 de cada 40 cierres (más seguido intradía y con colas pesadas). Con '
         f'{_usd(riesgo_usd)} de riesgo y el VaR 99% condicional, el tamaño máximo es {contratos} contrato(s).')

    ratio_var = var99 / var99_hist
    if ratio_var > 1.15:
        _lee(f'El VaR condicional es {ratio_var:.2f} veces el histórico: hoy el riesgo es mayor que el promedio, '
             'y un VaR fijo estaría subestimando la pérdida probable de mañana.')
    elif ratio_var < 0.85:
        _lee(f'El VaR condicional es {ratio_var:.2f} veces el histórico: hoy el riesgo es menor que el promedio; '
             'un VaR fijo inmovilizaría capital de más.')

    if lb_z2[1] >= 0.05:
        _lee('Los residuos estandarizados al cuadrado ya no tienen autocorrelación: el modelo capturó el '
             'clustering de volatilidad.')
    else:
        _lee(f'Queda estructura en los residuos estandarizados al cuadrado (p {_p(lb_z2[1])}): el modelo no '
             'captura toda la dinámica de la varianza. Pueden faltar rezagos, un componente de largo plazo '
             '(GARCH de dos componentes) o variables exógenas como el VIX o el calendario macro.')

    _lee(f'El R² de {mz_r2:.0%} parece bajo, pero es normal: r² es un proxy muy ruidoso de la varianza real '
         '(un día puede moverse poco aunque el riesgo sea alto). Con volatilidad intradía como referencia el '
         'ajuste sería mucho mejor.')

    return {'modelo_vol': v['nombre'], 'regimen_garch': regimen, 'pct_vol_garch': pct,
            'vol_hoy': vol_hoy_a, 'vol_lp': vol_lp_a, 'vol_fin': vol_fin_a, 'sig_man': sig_man,
            'sig_h': sig_h, 'rango1_man': r1_man, 'rango2_man': r2_man, 'rango1_h': r1_h,
            'var99_cond': var99, 'var99_cond_usd': usd(var99), 'contratos_garch': contratos,
            'stop_pts': stop_pts, 'stop_usd': stop_pts * multiplicador, 'persistencia': v['persistencia'],
            'vida_media': v['vida_media'], 'impacto_asim': impacto, 'horizonte': horizonte,
            'precio': precio}


# ---------------------------------------------------------------------------
# 6. Sistema
# ---------------------------------------------------------------------------

def sistema(modelo_var, retornos, principal, irf=None, ventana=63, h_fevd=10):
    nombres = list(modelo_var.names)
    otros = [t for t in nombres if t != principal]
    i = nombres.index(principal)

    granger = {}
    for destino in nombres:
        for causa in nombres:
            if destino != causa:
                granger[(causa, destino)] = modelo_var.test_causality(destino, [causa], kind='f').pvalue

    try:
        fevd = modelo_var.fevd(h_fevd).decomp
        part_otros = {t: float(fevd[i, -1, nombres.index(t)]) for t in otros}
    except Exception:
        part_otros = {}
    try:
        p_blanco = modelo_var.test_whiteness(nlags=max(modelo_var.k_ar + 5, 10)).pvalue
    except Exception:
        p_blanco = np.nan
    try:
        estable = bool(modelo_var.is_stable())
    except Exception:
        estable = None
    corr_res = float(np.asarray(modelo_var.resid_corr)[0, 1])

    _titulo('Lectura del sistema')
    _fila('Rezagos del VAR', f'{modelo_var.k_ar}')
    _fila('Estable (raíces dentro del círculo)', 'sí' if estable else 'no' if estable is not None else 'n/a')
    _fila('Blancura de residuos (Portmanteau)', f'p = {_p(p_blanco)}' if np.isfinite(p_blanco) else 'n/a')
    _fila('Correlación contemporánea de residuos', f'{corr_res:+.3f}')
    print('\n  Causalidad de Granger (H0: la causa no ayuda a predecir al destino)')
    for (causa, destino), p in granger.items():
        print(f'    {_corto(causa):>10} → {_corto(destino):<10} p = {_p(p):>7}   {_sig(p)}')
    if part_otros:
        print(f'\n  Descomposición de varianza del error de {_corto(principal)} a {h_fevd} días')
        for t, sh in part_otros.items():
            print(f'    explicada por shocks de {_corto(t)}: {sh:.1%}')

    resultados = {'granger': granger, 'corr_res': corr_res, 'fevd': part_otros}

    for otro in otros:
        x, y = retornos[principal], retornos[otro]
        corr_total = x.corr(y)
        corr_mov = x.rolling(ventana).corr(y).dropna()
        beta_total = x.cov(y) / y.var()
        rec = retornos.iloc[-ventana:]
        beta_rec = rec[principal].cov(rec[otro]) / rec[otro].var()
        print(f'\n  {_corto(principal)} vs {_corto(otro)}')
        _fila('Correlación toda la muestra', f'{corr_total:+.3f}')
        _fila(f'Correlación últimos {ventana} días', f'{corr_mov.iloc[-1]:+.3f}',
              f'rango histórico {corr_mov.min():+.2f} a {corr_mov.max():+.2f}')
        _fila(f'Beta de {_corto(principal)} sobre {_corto(otro)} (total)', f'{beta_total:+.3f}')
        _fila(f'Beta últimos {ventana} días', f'{beta_rec:+.3f}')
        _fila('% del tiempo con correlación negativa', f'{(corr_mov < 0).mean():.0%}')
        resultados.update({'corr_total': corr_total, 'corr_reciente': corr_mov.iloc[-1],
                           'beta_total': beta_total, 'beta_reciente': beta_rec, 'otro': otro})

        if irf is not None:
            orig, inv = irf['original'][otro], irf['invertido'][otro]
            print(f'\n  Respuesta de {_corto(otro)} a un shock de 1σ en {_corto(principal)}')
            _fila('Día 0 (orden original / invertido)', f'{orig.iloc[0]:+.3f} / {inv.iloc[0]:+.3f}')
            _fila('Días 1 a 10 acumulado', f'{orig.iloc[1:11].sum():+.3f} / {inv.iloc[1:11].sum():+.3f}')
            resultados['irf_dia0'] = orig.iloc[0]
            resultados['irf_acum'] = orig.iloc[1:11].sum()
    print()

    # interpretación
    for otro in otros:
        p_hacia = granger[(otro, principal)]
        p_desde = granger[(principal, otro)]
        if p_hacia < 0.05 and p_desde < 0.05:
            _lee(f'Hay retroalimentación: {_corto(otro)} y {_corto(principal)} se anticipan mutuamente en el '
                 'sentido de Granger.')
        elif p_hacia < 0.05:
            _lee(f'{_corto(otro)} anticipa a {_corto(principal)} en el sentido de Granger (p {_p(p_hacia)}), no al '
                 'revés. Antes de usarlo como señal: con miles de datos, efectos minúsculos salen significativos, '
                 'y la descomposición de varianza dice cuánto pesa en realidad.')
        elif p_desde < 0.05:
            _lee(f'{_corto(principal)} anticipa a {_corto(otro)} (p {_p(p_desde)}), no al revés.')
        else:
            _lee(f'Ninguna serie anticipa a la otra con datos diarios: la información se transmite el mismo día. '
                 'Esa relación contemporánea es la que importa para coberturas, no para señales.')

        if part_otros.get(otro) is not None:
            sh = part_otros[otro]
            peso = 'muy poco' if sh < 0.05 else 'una parte relevante' if sh < 0.25 else 'una parte importante'
            _lee(f'A {h_fevd} días, los shocks de {_corto(otro)} explican {sh:.1%} de la varianza del error de '
                 f'{_corto(principal)}: {peso} del riesgo de {_corto(principal)} viene del otro activo. Ojo: con '
                 'Cholesky esta cifra incluye el efecto del mismo día y depende del orden de las variables.')

    if 'corr_total' in resultados:
        ct, cr = resultados['corr_total'], resultados['corr_reciente']
        otro = resultados['otro']
        if abs(ct) < 0.15:
            _lee(f'La correlación de largo plazo es baja ({ct:+.2f}): como diversificador {_corto(otro)} aporta, '
                 'pero como cobertura directa sirve poco.')
        else:
            _lee(f'Correlación de largo plazo {ct:+.2f}. Para cubrir 1 unidad de {_corto(principal)} con '
                 f'{_corto(otro)} el beta histórico es {resultados["beta_total"]:+.3f} y el reciente '
                 f'{resultados["beta_reciente"]:+.3f}.')
        if abs(cr - ct) > 0.2 or np.sign(cr) != np.sign(ct):
            _lee(f'La correlación reciente ({cr:+.2f}) se alejó de la histórica ({ct:+.2f}). La relación no es '
                 'estable: una cobertura calibrada con toda la muestra puede fallar justo ahora. Conviene '
                 'recalibrar con ventana móvil o con un modelo DCC.')
        if _es_oro(principal) and _es_tasa(otro):
            if ct < 0:
                debil = ', aunque a nivel diario la relación es débil' if abs(ct) < 0.15 else ''
                _lee('El signo negativo confirma el canal de tasas: cuando suben los rendimientos, el oro tiende '
                     f'a caer{debil}. Para un operador de oro, el calendario de datos de inflación, empleo y '
                     'decisiones de la Fed es un factor de riesgo directo.')
            else:
                _lee('La correlación con los rendimientos no es negativa: el oro está siendo dominado por otros '
                     'factores (compras de bancos centrales, riesgo geopolítico, dólar). En esos regímenes las '
                     'tasas pierden poder explicativo.')

    if abs(corr_res) < 0.2:
        _lee(f'Correlación de residuos {corr_res:+.2f}: el orden de Cholesky casi no cambia la impulso-respuesta.')
    else:
        _lee(f'Correlación de residuos {corr_res:+.2f}: buena parte del vínculo es simultáneo y la IRF depende del '
             'orden. Compara las curvas original e invertida antes de contar una historia causal.')

    if np.isfinite(p_blanco) and p_blanco < 0.05:
        _lee('Los residuos del VAR no son ruido blanco: queda dinámica sin capturar (probablemente en la '
             'varianza, que el VAR no modela). Los p-valores de Granger son aproximados.')

    return resultados


# ---------------------------------------------------------------------------
# 7. Resumen
# ---------------------------------------------------------------------------

def resumen(h, principal, ticker_riesgo=None, riesgo_usd=500):
    """Tablero final para el analista a partir de los diccionarios de cada sección."""
    p = _corto(principal)
    _titulo(f'Tablero del analista: {p}')

    if 'precio' in h:
        print('\n  Dónde estamos')
        _fila('Último precio', f"{h['precio']:,.2f}")
        _fila('Régimen de volatilidad (GARCH)', h['regimen_garch'].upper(),
              f"percentil {h['pct_vol_garch']:.0f}, {h['vol_hoy']:.1f}% anual")
        _fila('Tendencia de la volatilidad', 'a la baja' if h['vol_fin'] < h['vol_hoy'] else 'al alza',
              f"{h['vol_hoy']:.1f}% → {h['vol_fin']:.1f}% en {h['horizonte']} días")
        _fila('Rango probable mañana (±1σ)', f"{h['rango1_man'][0]:,.0f} – {h['rango1_man'][1]:,.0f}")
        _fila('Rango extremo mañana (±2σ)', f"{h['rango2_man'][0]:,.0f} – {h['rango2_man'][1]:,.0f}")
        _fila(f"Rango probable a {h['horizonte']} días (±1σ)", f"{h['rango1_h'][0]:,.0f} – {h['rango1_h'][1]:,.0f}")

        print('\n  Gestión del riesgo')
        _fila('VaR 99% de mañana por contrato', _usd(h['var99_cond_usd']))
        _fila(f'Contratos con {_usd(riesgo_usd)} de riesgo', f"{h['contratos_garch']}")
        _fila('Stop mínimo sugerido (2σ)', f"{h['stop_pts']:,.1f} pts", f"{_usd(h['stop_usd'])} por contrato")
        if 'cvar99_usd' in h:
            _fila('Pérdida media en el peor 1% de días', _usd(h['cvar99_usd']), f'{_corto(ticker_riesgo or principal)}, histórico')
        if 'mdd' in h:
            _fila('Máximo drawdown histórico', f"{h['mdd']:.1%}")

    print('\n  Qué es predecible')
    if 'u_theil' in h:
        gana = np.isfinite(h.get('dm_p', np.nan)) and h['dm_p'] < 0.05 and h['u_theil'] < 1
        _fila('Dirección del precio', 'con ventaja' if gana else 'sin ventaja',
              f"U de Theil {h['u_theil']:.3f}, acierto {h['acierto_dir']:.0%}" if np.isfinite(h.get('acierto_dir', np.nan)) else '')
    if 'vr_sig' in h:
        _fila('Tendencia / reversión (2 a 20 días)', 'detectada' if h['vr_sig'] else 'no detectada')
    if 'persistencia' in h:
        _fila('Volatilidad', 'sí', f"persistencia {h['persistencia']:.3f}, vida media {h['vida_media']:.0f} días")
    if 'granger' in h and 'otro' in h:
        o = h['otro']
        p_g = h['granger'].get((o, principal), np.nan)
        _fila(f'{_corto(o)} anticipa a {p}', 'sí' if p_g < 0.05 else 'no', f'Granger p = {_p(p_g)}')
        _fila(f'Correlación con {_corto(o)} (total / reciente)', f"{h['corr_total']:+.2f} / {h['corr_reciente']:+.2f}")

    print('\n  Conclusiones')
    conclusiones = []
    if h.get('normal_subestima'):
        conclusiones.append(f"Las colas son pesadas (t con {h.get('gl_t', np.nan):.1f} gl): el VaR normal "
                            'subestima las pérdidas extremas. Usar VaR histórico, t o condicional.')
    if 'u_theil' in h:
        conclusiones.append('El precio se comporta como un paseo aleatorio: ni el ARIMA ni los patrones de '
                            'autocorrelación dan ventaja direccional. La entrada en una operación tiene que '
                            'justificarse con información que no está en el precio pasado.')
    if 'persistencia' in h:
        conclusiones.append(f"La volatilidad sí es predecible: el régimen actual es {h['regimen_garch']} y los "
                            f"shocks tardan unos {h['vida_media']:.0f} días en disiparse a la mitad. Es el "
                            'insumo útil del análisis: tamaño de posición, stops y VaR deben moverse con ella.')
    if 'impacto_asim' in h and np.isfinite(h['impacto_asim']):
        if h['impacto_asim'] < 1:
            conclusiones.append('Las subidas fuertes generan más volatilidad que las caídas: el riesgo de cola '
                                'es mayor para posiciones cortas.')
        elif h['impacto_asim'] > 1:
            conclusiones.append('Las caídas generan más volatilidad que las subidas: el riesgo de cola es mayor '
                                'para posiciones largas.')
    if 'corr_total' in h:
        estable = abs(h['corr_reciente'] - h['corr_total']) <= 0.2 and np.sign(h['corr_reciente']) == np.sign(h['corr_total'])
        o = _corto(h['otro'])
        texto_est = 'estable' if estable else 'inestable en el tiempo'
        p_g = h.get('granger', {}).get((h['otro'], principal), 1.0)
        fevd_o = h.get('fevd', {}).get(h['otro'])
        if p_g < 0.05:
            peso = f', pero solo explica {fevd_o:.0%} de su varianza a 10 días' if fevd_o is not None else ''
            conclusiones.append(f'{o} anticipa a {p} en el sentido de Granger{peso}. La correlación diaria es '
                                f"{h['corr_total']:+.2f} y {texto_est}. Útil como filtro de contexto, no como "
                                'señal por sí sola.')
        else:
            conclusiones.append(f"La relación con {o} es contemporánea (correlación {h['corr_total']:+.2f}) y "
                                f'{texto_est}: sirve para entender de dónde viene el riesgo, no para anticipar '
                                'movimientos.')
    if h.get('n_test', 999) < 30:
        conclusiones.append(f"La evaluación fuera de muestra usa solo {h['n_test']} días: ampliar el test antes "
                            'de sacar conclusiones sobre el ARIMA.')
    for c in conclusiones:
        _lee(c)

    print('\n  Siguientes pasos sugeridos')
    pasos = ['Re-estimar GARCH y ARIMA en ventana móvil (walk-forward) y backtestear el VaR condicional '
             'con Kupiec y Christoffersen.',
             'Probar volatilidad realizada intradía (Parkinson o Garman-Klass con el OHLC) como mejor proxy '
             'que r² para evaluar el GARCH.']
    if 'corr_total' in h:
        pasos.append('Modelar la correlación variable en el tiempo (DCC-GARCH) si el sistema se usa para coberturas.')
    if h.get('n_test', 999) < 60:
        pasos.append('Mover CORTE para que el test tenga al menos 60 días hábiles.')
    for i, paso in enumerate(pasos, 1):
        print(textwrap.fill(paso, width=ANCHO, initial_indent=f'  {i}. ', subsequent_indent='     '))
