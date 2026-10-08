"""
Pronóstico ARIMA animado: velas, caminos simulados del modelo, bandas y
comparación día a día con el periodo de prueba.

Python calcula todo (pronóstico, caminos) y el navegador solo dibuja, así la
animación corre fluida en un notebook, en Streamlit o como HTML suelto.

    caminos = animado.simular_caminos(modelo_arima, len(pron), n=300)
    animado.pronostico_arima_vivo(ohlc, pron, CORTE, orden, PRINCIPAL, caminos)
"""

import html as _html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .estilo import BLACK, CARPETA_FIGURAS, CYAN, GOLD, GREEN, GRID, LILAC, PURPLE, RED, WHITE

TINTA = '#E6E1F0'
TINTA_SUAVE = '#A39BB5'
SUPERFICIE = '#140C24'


def simular_caminos(modelo, h, n=300, semilla=42):
    """
    `n` trayectorias de precio a `h` pasos desde el final del entrenamiento.
    Salen del mismo modelo que las bandas, así que deberían caer dentro de ellas
    en la proporción del nivel de confianza.
    """
    sim = modelo.simulate(nsimulations=h, repetitions=n, anchor='end', random_state=semilla)
    caminos = np.asarray(sim, dtype=float).reshape(h, n).T

    # Con d > 0 statsmodels devuelve niveles; si viniera en diferencias se integra.
    ultimo = float(np.asarray(modelo.model.endog).ravel()[-1])
    if abs(np.median(caminos[:, 0]) - ultimo) > 20 * np.std(caminos[:, 0]) + 1e-9:
        caminos = ultimo + np.cumsum(caminos, axis=1)
    return caminos


def _redondear(valores, dec=2):
    return [None if pd.isna(v) else round(float(v), dec) for v in valores]


def _datos(ohlc, pron, corte, orden, ticker, caminos, dias_antes, max_caminos):
    corte = pd.Timestamp(corte)
    velas = ohlc.loc[ohlc.index >= corte - pd.Timedelta(days=dias_antes)]
    train = velas.loc[velas.index < corte]
    if train.empty:
        raise ValueError('No hay velas antes del corte: sube `dias_antes`.')

    # Eje x en días de negociación: velas + días futuros del pronóstico.
    fechas = velas.index.append(pron.index.difference(velas.index)).sort_values()
    pos = {f: i for i, f in enumerate(fechas)}

    if caminos is not None:
        caminos = np.asarray(caminos, dtype=float)[:max_caminos]
        caminos = [_redondear(c) for c in caminos]

    return {
        'ticker': ticker,
        'orden': str(tuple(orden)),
        'fechas': [f.strftime('%Y-%m-%d') for f in fechas],
        'n_train': len(train),
        'n_velas': len(velas),
        'velas': {c[0].lower(): _redondear(velas[c]) for c in ['Open', 'High', 'Low', 'Close']},
        'x_pron': [pos[f] for f in pron.index],
        'media': _redondear(pron['media']),
        'bandas': {n: {'inf': _redondear(pron[f'inf_{n}']), 'sup': _redondear(pron[f'sup_{n}'])}
                   for n in (99, 95, 80) if f'inf_{n}' in pron},
        'caminos': caminos or [],
        'colores': {'fondo': BLACK, 'grid': GRID, 'morado': PURPLE, 'lila': LILAC, 'cian': CYAN,
                    'rojo': RED, 'verde': GREEN, 'oro': GOLD, 'blanco': WHITE, 'tinta': TINTA,
                    'suave': TINTA_SUAVE, 'superficie': SUPERFICIE},
    }


def pronostico_arima_vivo(ohlc, pron, corte, orden, ticker, caminos=None, dias_antes=60,
                          max_caminos=400, alto=640, guardar=None, mostrar=True):
    """
    Devuelve el HTML de la animación. En un notebook además la muestra; en
    Streamlit se usa con `components.html(html, height=alto)`.
    """
    datos = _datos(ohlc, pron, corte, orden, ticker, caminos, dias_antes, max_caminos)
    pagina = _PLANTILLA.replace('__DATOS__', json.dumps(datos, separators=(',', ':')))

    if guardar:
        CARPETA_FIGURAS.mkdir(exist_ok=True)
        Path(CARPETA_FIGURAS / f'{guardar}.html').write_text(pagina, encoding='utf-8')

    if mostrar:
        try:
            from IPython import get_ipython
            from IPython.display import HTML, display
            if get_ipython() is not None:
                marco = (f'<iframe srcdoc="{_html.escape(pagina, quote=True)}" '
                         f'style="width:100%;height:{alto}px;border:0;background:{BLACK}"></iframe>')
                display(HTML(marco))
        except ImportError:
            pass
    return pagina


_PLANTILLA = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARIMA en vivo</title>
<style>
  :root { --fondo:#000; --tinta:#E6E1F0; --suave:#A39BB5; --grid:#2A1B4A; --sup:#140C24;
          --morado:#A855F7; --cian:#22D3EE; --oro:#E8B44A; --rojo:#FF3B5C; --verde:#22C55E; }
  * { box-sizing:border-box; }
  html, body { margin:0; height:100%; background:var(--fondo); color:var(--tinta);
               font-family:Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }
  .marco { display:flex; flex-direction:column; height:100%; min-height:520px; padding:14px 16px 10px; gap:10px; }
  .cabeza { display:flex; flex-wrap:wrap; justify-content:space-between; align-items:flex-start; gap:10px; }
  h1 { margin:0; font-size:17px; font-weight:600; color:#fff; }
  .fase { margin-top:4px; font-size:12px; color:var(--suave); font-family:ui-monospace, Menlo, monospace; min-height:16px; }
  .fase b { color:var(--cian); font-weight:500; }
  .controles { display:flex; align-items:center; gap:8px; flex-wrap:wrap; font-size:12px; color:var(--suave); }
  button, select { background:var(--sup); color:var(--tinta); border:1px solid var(--grid); border-radius:6px;
                   padding:5px 10px; font:inherit; cursor:pointer; }
  button:hover, select:hover { border-color:var(--morado); }
  label { display:flex; align-items:center; gap:6px; cursor:pointer; }
  input[type=range] { accent-color:var(--morado); width:90px; }
  input[type=checkbox] { accent-color:var(--morado); }
  .metricas { display:grid; grid-template-columns:repeat(auto-fit, minmax(118px, 1fr)); gap:8px; }
  .met { background:var(--sup); border:1px solid var(--grid); border-radius:8px; padding:7px 10px; }
  .met span { display:block; font-size:10.5px; color:var(--suave); text-transform:uppercase; letter-spacing:.04em; }
  .met strong { display:block; margin-top:2px; font-size:16px; font-weight:600; font-variant-numeric:tabular-nums; color:#fff; }
  .lienzo { position:relative; flex:1; min-height:300px; }
  canvas { position:absolute; inset:0; width:100%; height:100%; display:block; }
  .tip { position:absolute; pointer-events:none; background:var(--sup); border:1px solid var(--grid); border-radius:6px;
         padding:7px 9px; font-size:11.5px; line-height:1.5; font-variant-numeric:tabular-nums; display:none;
         white-space:nowrap; z-index:2; }
  .tip .t { color:var(--suave); margin-bottom:2px; }
</style>
</head>
<body>
<div class="marco">
  <div class="cabeza">
    <div>
      <h1 id="titulo"></h1>
      <div class="fase" id="fase"></div>
    </div>
    <div class="controles">
      <label>Caminos <input type="range" id="nCaminos" min="0" step="10"> <span id="nCaminosTxt"></span></label>
      <select id="velocidad" title="Velocidad">
        <option value="0.5">0.5×</option><option value="1" selected>1×</option>
        <option value="2">2×</option><option value="4">4×</option>
      </select>
      <label><input type="checkbox" id="bucle" checked> Repetir</label>
      <button id="reiniciar">↻ Reiniciar</button>
    </div>
  </div>
  <div class="metricas">
    <div class="met"><span>Caminos simulados</span><strong id="mCaminos">0</strong></div>
    <div class="met"><span>Días de prueba</span><strong id="mDias">–</strong></div>
    <div class="met"><span>Dentro IC 80%</span><strong id="m80">–</strong></div>
    <div class="met"><span>Dentro IC 95%</span><strong id="m95">–</strong></div>
    <div class="met"><span>Error medio abs.</span><strong id="mMae">–</strong></div>
    <div class="met"><span>Último vs pronóstico</span><strong id="mUlt">–</strong></div>
  </div>
  <div class="lienzo" id="lienzo">
    <canvas id="cv"></canvas>
    <div class="tip" id="tip"></div>
  </div>
</div>
<script>
(function () {
  var D = __DATOS__;
  var C = D.colores;
  var cv = document.getElementById('cv'), g = cv.getContext('2d');
  var caja = document.getElementById('lienzo'), tip = document.getElementById('tip');
  var $ = function (id) { return document.getElementById(id); };
  var reducir = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  $('titulo').textContent = D.ticker + ' — ARIMA' + D.orden + ' en vivo';

  var V = D.velas, NV = D.n_velas, NT = D.n_train, NX = D.fechas.length;
  var XP = D.x_pron, H = XP.length;
  var ultimoTrain = V.c[NT - 1];
  var nTest = NV - NT;   // velas reales después del corte

  // Control de caminos
  var NC_MAX = D.caminos.length;
  var rango = $('nCaminos');
  rango.max = NC_MAX; rango.value = Math.min(NC_MAX, 300);
  if (!NC_MAX) rango.parentNode.style.display = 'none';
  var NC = +rango.value;
  $('nCaminosTxt').textContent = NC;

  // Rango vertical: velas + banda más ancha
  var lo = Infinity, hi = -Infinity;
  for (var i = 0; i < NV; i++) { lo = Math.min(lo, V.l[i]); hi = Math.max(hi, V.h[i]); }
  var bandaAncha = D.bandas['99'] || D.bandas['95'] || D.bandas['80'];
  if (bandaAncha) for (i = 0; i < H; i++) { lo = Math.min(lo, bandaAncha.inf[i]); hi = Math.max(hi, bandaAncha.sup[i]); }
  var pad = (hi - lo) * 0.05; lo -= pad; hi += pad;

  // Geometría
  var W, Ht, dpr, M = { l: 62, r: 64, t: 14, b: 30 };
  var capa = document.createElement('canvas'), gc = capa.getContext('2d');
  function medir() {
    dpr = window.devicePixelRatio || 1;
    W = caja.clientWidth; Ht = caja.clientHeight;
    cv.width = capa.width = Math.round(W * dpr);
    cv.height = capa.height = Math.round(Ht * dpr);
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    gc.setTransform(dpr, 0, 0, dpr, 0, 0);
    reconstruirCapa();
  }
  function X(i) { return M.l + (i + 0.5) * (W - M.l - M.r) / NX; }
  function Y(p) { return M.t + (hi - p) / (hi - lo) * (Ht - M.t - M.b); }
  function paso() { return (W - M.l - M.r) / NX; }

  // Formatos
  var fmt = function (v, d) { return v.toLocaleString('en-US', { minimumFractionDigits: d || 0, maximumFractionDigits: d || 0 }); };
  var MESES = ['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic'];
  function fecha(s, larga) { var p = s.split('-'); return +p[2] + ' ' + MESES[+p[1] - 1] + (larga ? ' ' + p[0] : ''); }
  function rgba(hex, a) { var n = parseInt(hex.slice(1), 16); return 'rgba(' + (n >> 16) + ',' + (n >> 8 & 255) + ',' + (n & 255) + ',' + a + ')'; }
  function suave(t) { t = Math.max(0, Math.min(1, t)); return t * t * (3 - 2 * t); }

  // ── Línea de tiempo (segundos a 1×) ─────────────────────────────────────
  // Igual que un Monte Carlo en vivo: los primeros caminos van despacio y el
  // ritmo acelera; luego se revela el test una vela a la vez.
  var T_VELAS = 2.2, T_CORTE = 0.6;
  function duracion(k) { return k === 0 ? 2.2 : Math.max(0.35, 1.3 * Math.exp(-k / 14)); }
  function intervalo(k) { return k === 0 ? 2.4 : 0.55 * Math.exp(-k / 16) + 0.012; }
  var nacimientos = [];
  function programar() {
    nacimientos = []; var t = 0;
    for (var k = 0; k < NC; k++) { nacimientos.push(t); t += intervalo(k); }
  }
  function tCaminos() { return NC ? nacimientos[NC - 1] + duracion(NC - 1) + 0.3 : 0.4; }
  var T_BANDAS = 1.0, T_VELA_TEST = 0.75, T_ESPERA = 5;

  var reloj = 0, ultimoT = null, terminados = 0, finCiclo, fundido = 0;
  function reiniciar() {
    NC = +rango.value; $('nCaminosTxt').textContent = NC;
    programar(); reloj = reducir ? 1e6 : 0; terminados = 0; fundido = 0;
    gc.clearRect(0, 0, W, Ht);
    finCiclo = T_VELAS + T_CORTE + tCaminos() + T_BANDAS + nTest * T_VELA_TEST + 0.5;
  }

  function trazarCamino(ctx, c, hasta) {
    ctx.beginPath();
    ctx.moveTo(X(NT - 1), Y(ultimoTrain));
    var n = Math.floor(hasta), f = hasta - n;
    for (var t = 0; t < Math.min(n, H); t++) ctx.lineTo(X(XP[t]), Y(c[t]));
    if (n < H && f > 0) {
      var x0 = n ? X(XP[n - 1]) : X(NT - 1), y0 = n ? Y(c[n - 1]) : Y(ultimoTrain);
      ctx.lineTo(x0 + (X(XP[n]) - x0) * f, y0 + (Y(c[n]) - y0) * f);
    }
    ctx.stroke();
  }
  function caminoACapa(k) {
    gc.save(); rectPlot(gc); gc.clip();
    gc.strokeStyle = rgba(C.cian, NC > 150 ? 0.07 : 0.12); gc.lineWidth = 1;
    trazarCamino(gc, D.caminos[k], H);
    gc.restore();
  }
  function reconstruirCapa() {
    if (!gc || !W) return;
    gc.clearRect(0, 0, W, Ht);
    for (var k = 0; k < terminados; k++) caminoACapa(k);
  }
  function rectPlot(ctx) { ctx.beginPath(); ctx.rect(M.l, M.t, W - M.l - M.r, Ht - M.t - M.b); }

  // Histograma de precios finales simulados, pegado al borde derecho
  var NB = 28;
  function histograma(k) {
    var cubos = new Array(NB).fill(0), max = 0;
    for (var j = 0; j < k; j++) {
      var v = D.caminos[j][H - 1], b = Math.floor((hi - v) / (hi - lo) * NB);
      if (b >= 0 && b < NB) { cubos[b]++; max = Math.max(max, cubos[b]); }
    }
    return { cubos: cubos, max: max };
  }

  // ── Dibujo ─────────────────────────────────────────────────────────────
  function ejes() {
    g.font = '11px Inter, system-ui, sans-serif'; g.textBaseline = 'middle';
    var bruto = (hi - lo) / 6, mag = Math.pow(10, Math.floor(Math.log10(bruto)));
    var salto = [1, 2, 2.5, 5, 10].map(function (m) { return m * mag; }).find(function (s) { return s >= bruto; });
    for (var p = Math.ceil(lo / salto) * salto; p <= hi; p += salto) {
      var y = Math.round(Y(p)) + 0.5;
      g.strokeStyle = C.grid; g.lineWidth = 1; g.beginPath(); g.moveTo(M.l, y); g.lineTo(W - M.r, y); g.stroke();
      g.fillStyle = C.suave; g.textAlign = 'right'; g.fillText(fmt(p), M.l - 8, y);
    }
    // Una etiqueta de fecha cada ~90 px
    g.textAlign = 'center'; g.textBaseline = 'top';
    var cada = Math.max(1, Math.round(90 / paso()));
    for (var i = 0; i < NX; i += cada) g.fillText(fecha(D.fechas[i]), X(i), Ht - M.b + 8);
  }

  function vela(i, a) {
    var o = V.o[i], c = V.c[i], sube = c >= o, col = sube ? C.verde : C.rojo;
    var x = X(i), w = Math.max(2, paso() * 0.62);
    g.globalAlpha = a; g.strokeStyle = col; g.lineWidth = 1.2;
    g.beginPath(); g.moveTo(x, Y(V.h[i])); g.lineTo(x, Y(V.l[i])); g.stroke();
    var y1 = Y(Math.max(o, c)), alto = Math.max(1, Y(Math.min(o, c)) - y1);
    if (sube) { g.fillStyle = C.fondo; g.fillRect(x - w / 2, y1, w, alto); g.strokeRect(x - w / 2, y1, w, alto); }
    else { g.fillStyle = col; g.fillRect(x - w / 2, y1, w, alto); }
    g.globalAlpha = 1;
  }

  function lineaV(i, col, texto, abajo, a, centro) {
    if (a <= 0) return;
    var x = Math.round(X(i) - (centro ? 0 : paso() / 2)) + 0.5;
    g.globalAlpha = a; g.strokeStyle = col; g.setLineDash([5, 4]); g.lineWidth = 1;
    g.beginPath(); g.moveTo(x, M.t); g.lineTo(x, Ht - M.b); g.stroke(); g.setLineDash([]);
    g.fillStyle = C.tinta; g.font = '11px Inter, system-ui, sans-serif'; g.textAlign = 'left';
    g.textBaseline = abajo ? 'bottom' : 'top';
    g.fillText(texto, x + 5, abajo ? Ht - M.b - 6 : M.t + 4);
    g.globalAlpha = 1;
  }

  function banda(b, a, alfa) {
    g.beginPath();
    g.moveTo(X(NT - 1), Y(ultimoTrain));
    for (var t = 0; t < H; t++) g.lineTo(X(XP[t]), Y(b.sup[t]));
    for (t = H - 1; t >= 0; t--) g.lineTo(X(XP[t]), Y(b.inf[t]));
    g.closePath(); g.fillStyle = rgba(C.morado, alfa * a); g.fill();
  }

  function dentro(i, n) { var b = D.bandas[n], t = i - NT; return b && V.c[i] >= b.inf[t] && V.c[i] <= b.sup[t]; }

  function dibujar(dt) {
    g.clearRect(0, 0, W, Ht);
    g.fillStyle = C.fondo; g.fillRect(0, 0, W, Ht);
    ejes();

    var t = reloj, t0;
    // 1) Velas de entrenamiento
    var nVis = reducir ? NT : Math.min(NT, t / T_VELAS * NT);
    for (var i = 0; i < Math.ceil(nVis); i++) vela(i, Math.min(1, nVis - i));

    // 2) Corte
    t0 = T_VELAS; var aCorte = suave((t - t0) / T_CORTE);
    lineaV(NT, C.oro, 'inicio test', false, aCorte);

    // 3) Caminos simulados
    t0 = T_VELAS + T_CORTE; var tc = t - t0, activos = 0;
    g.save(); rectPlot(g); g.clip();
    if (tc > 0) {
      while (terminados < NC && tc >= nacimientos[terminados] + duracion(terminados)) caminoACapa(terminados++);
      g.drawImage(capa, 0, 0, W, Ht);
      g.lineWidth = 1.4;
      for (var k = terminados; k < NC && nacimientos[k] <= tc; k++) {
        var prog = (tc - nacimientos[k]) / duracion(k) * H;
        g.strokeStyle = rgba(C.cian, 0.9); trazarCamino(g, D.caminos[k], prog);
        activos++;
      }
    }
    g.restore();
    var visibles = Math.min(NC, terminados + activos);

    // Histograma de cierres simulados
    if (terminados > 3) {
      var hg = histograma(terminados), alto = (Ht - M.t - M.b) / NB, x0 = W - M.r + 6, ancho = M.r - 12;
      for (var b = 0; b < NB; b++) if (hg.cubos[b]) {
        g.fillStyle = rgba(C.cian, 0.25 + 0.45 * hg.cubos[b] / hg.max);
        g.fillRect(x0, M.t + b * alto + 1, ancho * hg.cubos[b] / hg.max, alto - 2);
      }
    }

    // 4) Bandas y media
    t0 += tCaminos(); var aB = suave((t - t0 + (NC ? 0.6 : 0)) / T_BANDAS);
    if (aB > 0 && mostrarBandas) {
      if (D.bandas['99']) banda(D.bandas['99'], aB, 0.10);
      if (D.bandas['95']) banda(D.bandas['95'], aB, 0.16);
      if (D.bandas['80']) banda(D.bandas['80'], aB, 0.28);
    }
    if (aB > 0) {
      var nm = Math.ceil(H * aB);
      g.strokeStyle = C.cian; g.lineWidth = 2; g.setLineDash([3, 4]);
      g.beginPath(); g.moveTo(X(NT - 1), Y(ultimoTrain));
      for (var s = 0; s < nm; s++) g.lineTo(X(XP[s]), Y(D.media[s]));
      g.stroke(); g.setLineDash([]);
      g.fillStyle = C.cian;
      for (s = 0; s < nm; s++) { g.beginPath(); g.arc(X(XP[s]), Y(D.media[s]), 2.4, 0, 7); g.fill(); }
    }

    // 5) Velas de prueba, una a una, con marca según caigan en las bandas
    t0 += T_BANDAS; var nTv = reducir ? nTest : Math.max(0, Math.min(nTest, (t - t0) / T_VELA_TEST));
    var den80 = 0, den95 = 0, errAbs = 0, n = Math.floor(nTv);
    for (i = NT; i < NT + Math.ceil(nTv); i++) {
      var a = Math.min(1, NT + nTv - i);
      vela(i, a);
      var d80 = dentro(i, 80), d95 = dentro(i, 95);
      var col = d80 ? C.verde : (d95 ? C.oro : C.rojo);
      var r = 4 + 8 * (1 - a);
      g.globalAlpha = a; g.strokeStyle = col; g.lineWidth = 1.6;
      g.beginPath(); g.arc(X(i), Y(V.c[i]), r, 0, 7); g.stroke(); g.globalAlpha = 1;
      if (i < NT + n) { den80 += d80; den95 += d95; errAbs += Math.abs(V.c[i] - D.media[i - NT]); }
    }
    if (nTv > 0) lineaV(NV - 1, C.blanco, 'último dato', true, suave((nTv - nTest + 0.5) * 2), true);

    // Métricas
    $('mCaminos').textContent = fmt(visibles);
    if (n > 0) {
      $('mDias').textContent = n + ' / ' + nTest;
      $('m80').textContent = Math.round(100 * den80 / n) + '%';
      $('m95').textContent = Math.round(100 * den95 / n) + '%';
      $('mMae').textContent = '$' + fmt(errAbs / n);
      var u = NT + n - 1, dif = V.c[u] - D.media[u - NT];
      $('mUlt').textContent = (dif >= 0 ? '+' : '−') + '$' + fmt(Math.abs(dif));
      $('mUlt').style.color = Math.abs(dif) < 1e-9 ? '#fff' : (dif > 0 ? C.verde : C.rojo);
    } else {
      ['mDias', 'm80', 'm95', 'mMae', 'mUlt'].forEach(function (id) { $(id).textContent = '–'; $(id).style.color = ''; });
    }

    // Texto de fase
    var fase;
    if (t < T_VELAS) fase = 'Entrenamiento: ' + NT + ' velas hasta ' + fecha(D.fechas[NT - 1], true);
    else if (visibles < NC || (NC && terminados < NC)) fase = 'Simulando el modelo: <b>' + visibles + '</b> de ' + NC + ' futuros posibles';
    else if (nTv < nTest) fase = 'Comparando con la realidad: día <b>' + Math.max(1, Math.ceil(nTv)) + '</b> de ' + nTest;
    else fase = 'Pronóstico a ' + H + ' días hábiles · verde: dentro del IC 80% · oro: dentro del 95% · rojo: fuera';
    if (faseActual !== fase) { $('fase').innerHTML = fase; faseActual = fase; }

    // Fundido para volver a empezar
    if (fundido > 0) { g.fillStyle = 'rgba(0,0,0,' + fundido + ')'; g.fillRect(0, 0, W, Ht); }
    dibujarCursor();
  }
  var faseActual = '', mostrarBandas = true;

  // ── Cursor y tooltip ───────────────────────────────────────────────────
  var cursor = null;
  caja.addEventListener('mousemove', function (e) {
    var r = caja.getBoundingClientRect(); cursor = { x: e.clientX - r.left, y: e.clientY - r.top };
  });
  caja.addEventListener('mouseleave', function () { cursor = null; tip.style.display = 'none'; });
  function dibujarCursor() {
    if (!cursor || cursor.x < M.l || cursor.x > W - M.r) { tip.style.display = 'none'; return; }
    var i = Math.max(0, Math.min(NX - 1, Math.floor((cursor.x - M.l) / paso())));
    var x = Math.round(X(i)) + 0.5;
    g.strokeStyle = '#6B5A8E'; g.setLineDash([2, 3]); g.lineWidth = 1;
    g.beginPath(); g.moveTo(x, M.t); g.lineTo(x, Ht - M.b); g.stroke(); g.setLineDash([]);
    var filas = [];
    if (i < NV) filas.push('A ' + fmt(V.o[i], 2) + ' · M ' + fmt(V.h[i], 2) + ' · m ' + fmt(V.l[i], 2) + ' · C <b>' + fmt(V.c[i], 2) + '</b>');
    var t = XP.indexOf(i);
    if (t >= 0) {
      filas.push('<span style="color:' + C.cian + '">ARIMA ' + fmt(D.media[t], 2) + '</span>');
      [80, 95].forEach(function (n) { var b = D.bandas[n]; if (b) filas.push('IC ' + n + '%: ' + fmt(b.inf[t]) + ' – ' + fmt(b.sup[t])); });
    }
    if (!filas.length) { tip.style.display = 'none'; return; }
    tip.innerHTML = '<div class="t">' + fecha(D.fechas[i], true) + '</div>' + filas.join('<br>');
    tip.style.display = 'block';
    var tw = tip.offsetWidth, th = tip.offsetHeight;
    tip.style.left = (cursor.x + 14 + tw > W ? cursor.x - tw - 14 : cursor.x + 14) + 'px';
    tip.style.top = Math.min(Ht - th - 4, Math.max(4, cursor.y - th / 2)) + 'px';
  }

  // ── Bucle ──────────────────────────────────────────────────────────────
  function cuadro(ahora) {
    var dt = ultimoT === null ? 0 : Math.min(0.1, (ahora - ultimoT) / 1000);
    ultimoT = ahora;
    if (visible) {
      reloj += dt * +$('velocidad').value;
      if ($('bucle').checked && !reducir && reloj > finCiclo + T_ESPERA) {
        fundido += dt * 1.5;
        if (fundido >= 1) reiniciar();
      }
      dibujar(dt);
    }
    requestAnimationFrame(cuadro);
  }

  // Solo anima cuando está en pantalla
  var visible = true;
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (e) { visible = e[0].isIntersecting; }).observe(caja);
  }
  rango.addEventListener('input', function () { $('nCaminosTxt').textContent = rango.value; });
  rango.addEventListener('change', reiniciar);
  $('reiniciar').addEventListener('click', reiniciar);
  new ResizeObserver(medir).observe(caja);

  medir(); reiniciar(); requestAnimationFrame(cuadro);
})();
</script>
</body>
</html>
"""
