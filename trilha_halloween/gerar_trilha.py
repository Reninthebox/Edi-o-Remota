"""Gera uma trilha original de Halloween (60 s, 120 BPM, Ré menor harmônico).
Estrutura pensada para clipe de evento: entrada calma, subida, pico, fecho.
Uso: python3 gerar_trilha.py  -> gera trilha_halloween.wav
"""
import numpy as np, wave, sys

SR = 44100
BPM = 120
BEAT = 60 / BPM            # 0,5 s
BAR = 4 * BEAT             # 2 s
DUR = 60.0
N = int(SR * DUR)
rng = np.random.default_rng(31)
L = np.zeros(N); R = np.zeros(N)

def hz(m): return 440 * 2 ** ((m - 69) / 12)
def idx(t): return int(t * SR)

def env(n, a=0.01, d=0.1, s=0.7, r=0.1):
    t = np.arange(n) / SR; total = n / SR
    e = np.ones(n) * s
    e[t < a] = t[t < a] / a
    m = (t >= a) & (t < a + d); e[m] = 1 - (1 - s) * (t[m] - a) / d
    rel = t > total - r; e[rel] *= np.clip((total - t[rel]) / r, 0, 1)
    return e

def add(sig, t, pan=0.0, gain=1.0):
    i = idx(t); j = min(N, i + len(sig))
    if i >= N: return
    s = sig[: j - i] * gain
    L[i:j] += s * np.sqrt((1 - pan) / 2); R[i:j] += s * np.sqrt((1 + pan) / 2)

def osc(f, n, kind="sine", detune=0.0):
    t = np.arange(n) / SR
    ph = 2 * np.pi * f * (1 + detune) * t
    if kind == "sine": return np.sin(ph)
    if kind == "saw": return 2 * ((f * (1 + detune) * t) % 1) - 1
    if kind == "square": return np.sign(np.sin(ph))
    if kind == "tri": return 2 * np.abs(2 * ((f * t) % 1) - 1) - 1

def lowpass(x, cutoff):
    a = np.exp(-2 * np.pi * cutoff / SR); y = np.empty_like(x); z = 0.0
    for k in range(len(x)): z = (1 - a) * x[k] + a * z; y[k] = z
    return y

def lp_fast(x, cutoff):  # filtro de 1 polo vetorizado em blocos (aprox. via FFT)
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR)
    X /= np.sqrt(1 + (f / cutoff) ** 4)
    return np.fft.irfft(X, len(x))

# ---------- instrumentos ----------
def organ(m, dur):
    n = idx(dur); f = hz(m)
    s = sum(osc(f * h, n) * g for h, g in [(0.5, .5), (1, 1), (2, .5), (3, .25), (4, .2)])
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * np.arange(n) / SR)
    return s * vib * env(n, .25, .2, .8, .6) * 0.08

def musicbox(m, dur=1.2):
    n = idx(dur); f = hz(m); t = np.arange(n) / SR
    s = osc(f, n) + .3 * osc(f * 3.01, n) + .15 * osc(f * 5.2, n)
    return s * np.exp(-t * 3.5) * 0.18

def bell(m, dur=6.0):
    n = idx(dur); f = hz(m); t = np.arange(n) / SR
    parts = [(0.5, 1, 1.2), (1, .8, 1.6), (1.19, .5, 2.2), (1.56, .4, 2.8), (2.0, .35, 3.2), (2.74, .2, 4.5)]
    return sum(g * np.sin(2 * np.pi * f * r * t) * np.exp(-t * d / 2) for r, g, d in parts) * 0.18

def kick():
    n = idx(.45); t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7) * 0.9

def snare(gain=1.0):
    n = idx(.25); t = np.arange(n) / SR
    s = rng.standard_normal(n) * np.exp(-t * 18) * .5 + np.sin(2 * np.pi * 190 * t) * np.exp(-t * 25) * .4
    return s * 0.45 * gain

def clap():
    n = idx(.3); t = np.arange(n) / SR; s = rng.standard_normal(n)
    e = sum(np.exp(-np.clip(t - o, 0, None) * 60) * (t >= o) for o in (0, .012, .024)) + np.exp(-t * 12) * .5
    return lp_fast(s * e, 3000) * 0.35

def hat(open_=False):
    n = idx(.25 if open_ else .06); t = np.arange(n) / SR
    s = rng.standard_normal(n); s = s - lp_fast(s, 7000)
    return s * np.exp(-t * (12 if open_ else 70)) * 0.18

def bass(m, dur):
    n = idx(dur); f = hz(m)
    s = osc(f, n, "saw") + osc(f, n, "saw", .006) + osc(f / 2, n) * 1.2
    return lp_fast(s, 420) * env(n, .005, .08, .6, .03) * 0.22

def lead(m, dur):
    n = idx(dur); f = hz(m)
    s = osc(f, n, "saw", -.008) + osc(f, n, "saw", .008) + .5 * osc(f * 2, n, "square")
    vib = np.sin(2 * np.pi * 6 * np.arange(n) / SR) * .02
    s = s * (1 + vib)
    return lp_fast(s, 2600) * env(n, .01, .15, .55, .08) * 0.07

def theremin(m_from, m_to, dur):
    n = idx(dur); t = np.arange(n) / SR
    mm = np.linspace(m_from, m_to, n) + .35 * np.sin(2 * np.pi * 5 * t)
    f = 440 * 2 ** ((mm - 69) / 12)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, .6, .3, .8, 1.2) * 0.09

def wind(dur, gain=1.0):
    n = idx(dur); t = np.arange(n) / SR
    s = lp_fast(rng.standard_normal(n), 600)
    mod = .5 + .5 * np.sin(2 * np.pi * .23 * t) * np.sin(2 * np.pi * .07 * t + 1)
    return s / (np.abs(s).max() + 1e-9) * mod * env(n, 2, .1, 1, 2) * 0.12 * gain

def riser(dur):
    n = idx(dur); t = np.arange(n) / SR; x = t / dur
    noise = rng.standard_normal(n); noise = noise - lp_fast(noise, 1500)
    sweep = np.sin(2 * np.pi * np.cumsum(200 + 1800 * x ** 2) / SR)
    return (noise * .25 + sweep * .2) * x ** 2 * 0.6

def impact():
    n = idx(3); t = np.arange(n) / SR
    boom = np.sin(2 * np.pi * np.cumsum(30 + 80 * np.exp(-t * 6)) / SR) * np.exp(-t * 1.5)
    return boom * 0.8 + lp_fast(rng.standard_normal(n), 900) * np.exp(-t * 3) * .25

# ---------- harmonia ----------
# Dm  -  Bb  -  Gm  -  A   (Ré menor harmônico)
CHORDS = [[50, 53, 57, 62], [46, 50, 53, 58], [43, 50, 55, 58], [45, 49, 52, 57]]
ROOTS = [38, 34, 31, 33]
# Melodia (8 colcheias por compasso; None = pausa)
MEL = [
    [74, None, 77, 76, 74, None, 73, 74],
    [70, None, 74, 72, 70, None, 69, 70],
    [67, None, 70, 69, 67, 69, 70, 74],
    [73, None, 76, 73, 69, None, 73, None],
]

# ---------- arranjo ----------
#  0– 8 s  Entrada   (compassos 0–3)  : vento, sino, órgão, caixinha de música
#  8–24 s  Subida    (4–11)           : + baixo pulsando, chimbal, caixa em 2/4, theremin
# 24–28 s  Virada    (12–13)          : riser + rufo de caixa, corte seco
# 28–52 s  Pico      (14–25)          : bumbo 4x4, palmas, baixo, lead, órgão
# 52–60 s  Fecho     (26–29)          : 2 compassos de pico com filtro, depois sino final
add(wind(DUR), 0, 0, 0.6)
for b in range(30):
    t0 = b * BAR; c = b % 4
    # órgão (quase o tempo todo)
    if b < 12 or 14 <= b < 28:
        for k, m in enumerate(CHORDS[c]):
            add(organ(m, BAR + .3), t0, pan=(k - 1.5) * .4, gain=(0.7 if 14 <= b < 26 else 0.45 if b < 4 else 0.75))
    # sino de igreja no início de cada frase da entrada
    if b in (0, 2, 4, 8, 14, 22):
        add(bell(38 if b != 14 else 26), t0, 0, 0.6 if b < 4 else 0.8 if b != 14 else 1.3)
    # caixinha de música: entrada e subida
    if b < 12:
        for k, m in enumerate(MEL[c]):
            if m is not None: add(musicbox(m + 12), t0 + k * BEAT / 2, pan=.3 * np.sin(k), gain=.9)
    # theremin na subida
    if b in (6, 10):
        add(theremin(69, 74, 2 * BAR), t0, -.3)
    # baixo pulsando em colcheias
    if 4 <= b < 12:
        for k in range(8):
            add(bass(ROOTS[c] + 12, BEAT / 2 * .9), t0 + k * BEAT / 2, 0, .6 + .05 * (b - 4))
    if 14 <= b < 28:
        for k in range(8):
            if k % 2 == 1:   # contratempo
                add(bass(ROOTS[c] + 12, BEAT / 2 * .9), t0 + k * BEAT / 2, 0, 1.0)
    # bateria
    if 4 <= b < 12:
        for k in range(8): add(hat(), t0 + k * BEAT / 2, .4, .7 if k % 2 else .4)
        if b >= 8:
            for k in (1, 3): add(snare(.7), t0 + k * BEAT, -.1)
        if b >= 8: add(kick(), t0, 0, .7); add(kick(), t0 + 2 * BEAT, 0, .7)
    if 14 <= b < 28:
        for k in range(4): add(kick(), t0 + k * BEAT)
        for k in (1, 3): add(clap(), t0 + k * BEAT, 0)
        for k in range(8): add(hat(open_=(k % 2 == 1)), t0 + k * BEAT / 2, .35, .6)
    # lead fantasmagórico no pico
    if 14 <= b < 26:
        for k, m in enumerate(MEL[c]):
            if m is not None: add(lead(m, BEAT / 2 * .95), t0 + k * BEAT / 2, -.2)
    if 18 <= b < 26:   # dobra a melodia 1 oitava acima na 2ª metade do pico
        for k, m in enumerate(MEL[c]):
            if m is not None: add(musicbox(m + 12, .6), t0 + k * BEAT / 2, .4, .7)

# virada (24–28 s)
add(riser(2 * BAR), 12 * BAR, 0, 1.6)
for k, frac in enumerate(np.linspace(0, 1, 24, endpoint=False)):
    add(snare(.3 + .7 * frac), 13 * BAR + frac * BAR, 0)
for k in range(4): add(kick(), 12 * BAR + k * BEAT, 0, .8)
add(theremin(62, 86, 2 * BAR), 12 * BAR, .3, .8)
add(impact(), 14 * BAR, 0, 1.0)
add(impact(), 26 * BAR, 0, .7)
add(theremin(81, 50, 3 * BAR), 26 * BAR, 0, .7)

mix = np.stack([L, R], 1)
# fecho: filtra o pico nos compassos 26–27 e fade final
i26, i28 = idx(26 * BAR), idx(28 * BAR)
for ch in range(2):
    seg = mix[i26:i28, ch]; mix[i26:i28, ch] = lp_fast(seg, 900) * 0.85
add_bell = bell(26, 4.0) * 1.4
mix[idx(28 * BAR):, 0] += add_bell[: N - idx(28 * BAR)]
mix[idx(28 * BAR):, 1] += add_bell[: N - idx(28 * BAR)]
fade = np.ones(N); fn = idx(1.5); fade[-fn:] = np.linspace(1, 0, fn)
mix *= fade[:, None]

# reverb simples (eco múltiplo) para ar de igreja/caverna
wet = np.zeros_like(mix)
for d, g in [(.043, .35), (.071, .3), (.113, .25), (.167, .2), (.251, .15), (.379, .1)]:
    k = idx(d); wet[k:, 0] += mix[:-k, 1] * g; wet[k:, 1] += mix[:-k, 0] * g
mix = mix + wet * .6
mix = np.tanh(mix * 1.4) / np.tanh(1.4)       # saturação suave
mix /= np.abs(mix).max() / 0.89               # pico em ~ -1 dBFS

out = sys.argv[1] if len(sys.argv) > 1 else "trilha_halloween.wav"
with wave.open(out, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype("<i2").tobytes())
print("ok", out)
