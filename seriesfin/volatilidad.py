"""Modelos de volatilidad condicional (GARCH, GJR-GARCH, EGARCH)."""

import itertools
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

from . import SEED

NOMBRES_DIST = {'normal': 'normal', 't': 't', 'skewt': 't asimétrica'}


def nombre_modelo(vol, p, o, q, dist):
    if vol == 'EGARCH':
        base = f'EGARCH({p},{o},{q})'
    elif o > 0:
        base = f'GJR-GARCH({p},{o},{q})'
    else:
        base = f'GARCH({p},{q})'
    return f'{base} {NOMBRES_DIST.get(dist, dist)}'


def configuracion(vol='Garch', p=1, o=0, q=1, dist='t'):
    return {'vol': vol, 'p': p, 'o': o, 'q': q, 'dist': dist,
            'nombre': nombre_modelo(vol, p, o, q, dist)}


def ajustar(r, config):
    """r en %, media constante."""
    c = config
    modelo = arch_model(r.dropna(), mean='Constant', vol=c['vol'],
                        p=c['p'], o=c['o'], q=c['q'], dist=c['dist'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return modelo.fit(disp='off', show_warning=False)


def buscar_modelo(r, vols=('Garch', 'EGARCH'), ps=(1, 2), os_=(0, 1), qs=(1, 2),
                  dists=('t', 'skewt')):
    """
    Ajusta todas las combinaciones y las ordena por AIC.
    Descarta las que no convergen. Devuelve (tabla, mejor_resultado, mejor_config).
    """
    filas, ajustes = [], {}
    for vol, p, o, q, dist in itertools.product(vols, ps, os_, qs, dists):
        config = configuracion(vol, p, o, q, dist)
        try:
            res = ajustar(r, config)
        except Exception:
            continue
        if res.convergence_flag != 0:
            continue
        ajustes[config['nombre']] = (res, config)
        filas.append({'Modelo': config['nombre'], 'AIC': res.aic, 'BIC': res.bic,
                      'Log-verosimilitud': res.loglikelihood})

    tabla = pd.DataFrame(filas).sort_values('AIC').set_index('Modelo').round(2)
    res, config = ajustes[tabla.index[0]]
    return tabla, res, config


def resumir(res, config, horizonte=10, semilla=SEED):
    """
    Persistencia, vida media de un shock, volatilidad de largo plazo y pronóstico.
    Las volatilidades salen anualizadas y en %.

    Persistencia:
      GARCH / GJR: Σα + Σβ + ½·Σγ  (½ porque γ solo actúa en días negativos)
      EGARCH:      Σβ              (α y γ miden el impacto del shock, no su duración)
    """
    params = res.params
    suma = lambda prefijo: sum(v for k, v in params.items() if k.startswith(prefijo))
    alpha, beta, gamma = suma('alpha['), suma('beta['), suma('gamma[')

    es_egarch = config['vol'] == 'EGARCH'
    persistencia = beta if es_egarch else alpha + beta + 0.5 * gamma
    vida_media = np.log(0.5) / np.log(persistencia) if 0 < persistencia < 1 else np.inf

    vol_cond = res.conditional_volatility * np.sqrt(252)
    if not es_egarch and persistencia < 1:
        vol_lp = np.sqrt(params['omega'] / (1 - persistencia)) * np.sqrt(252)
    else:
        # sin fórmula cerrada simple: se usa la varianza condicional media
        vol_lp = np.sqrt((res.conditional_volatility ** 2).mean()) * np.sqrt(252)

    if es_egarch:
        # EGARCH no tiene pronóstico analítico a más de un paso
        try:
            pron = res.forecast(horizon=horizonte, reindex=False, method='simulation',
                                simulations=2000,
                                random_state=np.random.RandomState(semilla))
        except TypeError:
            pron = res.forecast(horizon=horizonte, reindex=False, method='simulation',
                                simulations=2000)
    else:
        pron = res.forecast(horizon=horizonte, reindex=False)
    vol_pron = np.sqrt(pron.variance.values[-1]) * np.sqrt(252)

    return {
        'nombre': config['nombre'],
        'es_egarch': es_egarch,
        'asimetria': config['o'] > 0,
        'alpha': alpha, 'beta': beta, 'gamma': gamma,
        'persistencia': persistencia,
        'vida_media': vida_media,
        'nu': params.get('nu', params.get('eta', np.nan)),
        'vol_cond': vol_cond,
        'vol_lp': vol_lp,
        'vol_pron': vol_pron,
    }


def interpretar(v):
    """Lectura en texto de los parámetros de `resumir`."""
    lineas = []

    if v['es_egarch']:
        lineas.append(f"Reacción a shocks (α = {v['alpha']:.4f}): en EGARCH α actúa sobre "
                      "log(σ²), no se compara directamente con el de un GARCH.")
    else:
        nivel = 'alta' if v['alpha'] > 0.1 else 'moderada'
        lineas.append(f"Reacción a shocks (α = {v['alpha']:.4f}): sensibilidad {nivel} "
                      "a las noticias del día.")

    memoria = 'la volatilidad tarda en calmarse' if v['beta'] > 0.85 else 'la volatilidad se disipa rápido'
    lineas.append(f"Memoria (β = {v['beta']:.4f}): {memoria}.")

    lineas.append(f"Persistencia = {v['persistencia']:.4f}; un shock tarda "
                  f"{v['vida_media']:.1f} días hábiles en perder la mitad de su efecto.")

    if v['asimetria']:
        g = v['gamma']
        if (v['es_egarch'] and g < 0) or (not v['es_egarch'] and g > 0):
            texto = 'las caídas suben la volatilidad más que las subidas (efecto apalancamiento).'
        else:
            texto = ('las subidas suben la volatilidad más que las caídas; '
                     'es común en el oro por su papel de refugio.')
        lineas.append(f"Asimetría (γ = {g:.4f}): {texto}")
    else:
        lineas.append('Sin término de asimetría en el modelo elegido.')

    if np.isfinite(v['nu']):
        colas = 'muy pesadas' if v['nu'] < 5 else 'moderadas'
        lineas.append(f"Grados de libertad = {v['nu']:.2f}: colas {colas}.")

    return lineas
