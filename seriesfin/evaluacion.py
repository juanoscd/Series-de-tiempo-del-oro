"""Evaluación fuera de muestra del ARIMA contra pronósticos ingenuos."""

import numpy as np
import pandas as pd


def metricas(real, pred, benchmark=None, nombre='Modelo'):
    """
    RMSE, MAE, MAPE, R² y U de Theil (RMSE del modelo / RMSE del benchmark).
    U < 1 significa que el modelo le gana al benchmark.
    """
    real = pd.Series(real).dropna()
    pred = pd.Series(pred).reindex(real.index).dropna()
    real = real.reindex(pred.index)

    error = real - pred
    rmse = float(np.sqrt((error ** 2).mean()))
    ss_tot = float(((real - real.mean()) ** 2).sum())

    theil = np.nan
    if benchmark is not None:
        bench = pd.Series(benchmark).reindex(real.index)
        rmse_b = float(np.sqrt(((real - bench) ** 2).mean()))
        if rmse_b > 0:
            theil = rmse / rmse_b

    return {
        'Modelo': nombre,
        'RMSE': rmse,
        'MAE': float(error.abs().mean()),
        'MAPE (%)': float((error.abs() / real.abs()).mean() * 100),
        'R²': 1 - float((error ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan,
        'U de Theil': theil,
    }


def pronostico_un_paso(modelo, serie, train, test):
    """
    Predicción a 1 paso sobre el test sin re-estimar parámetros: cada día usa
    el precio real del día anterior.
    """
    extendido = modelo.apply(serie.reset_index(drop=True))
    pred = extendido.get_prediction(start=len(train), end=len(serie) - 1,
                                    dynamic=False).predicted_mean
    return pd.Series(pred.values, index=test.index)


def evaluar(test, train, serie, pred_1paso, pred_multi, orden):
    """
    Cada pronóstico se compara con su benchmark justo:
    a 1 paso contra el precio de ayer, multi-paso contra el último precio del train.
    """
    naive_1paso = serie.shift(1).loc[test.index]
    naive_multi = pd.Series(train.iloc[-1], index=test.index)

    filas = [
        metricas(test, pred_1paso, naive_1paso, f'ARIMA{orden} a 1 paso'),
        metricas(test, naive_1paso, naive_1paso, 'Ingenuo: precio de ayer'),
        metricas(test, pred_multi, naive_multi, f'ARIMA{orden} a {len(test)} pasos'),
        metricas(test, naive_multi, naive_multi, 'Ingenuo: último precio del train'),
    ]
    return pd.DataFrame(filas).set_index('Modelo').round(4)


def error_por_horizonte(pred_multi, test, train, horizontes=(1, 5, 10, 21, 63, 126)):
    """Cómo crece el error del pronóstico multi-paso a medida que se aleja el horizonte."""
    filas = []
    for h in horizontes:
        if h > len(test):
            continue
        real = test.iloc[:h]
        naive = pd.Series(train.iloc[-1], index=real.index)
        m = metricas(real, pred_multi, naive)
        filas.append({'Horizonte': h, 'RMSE': m['RMSE'], 'MAE': m['MAE'],
                      'U de Theil': m['U de Theil']})
    return pd.DataFrame(filas).set_index('Horizonte').round(4)
