# Trilha de Halloween (original)

Trilha composta para o clipe com trilha de Halloween. **60 s, 120 BPM, Ré menor**, 44,1 kHz estéreo.
Música original gerada por código (`gerar_trilha.py`): sem direitos autorais de terceiros.

- `trilha_halloween.wav`: use esta no Premiere (qualidade máxima)
- `trilha_halloween.mp3`: para ouvir no celular
- `marcadores.csv`: pontos-chave para os marcadores da trilha

## Curva de energia
| Trecho | Ato sugerido | Energia | O que toca |
|---|---|---|---|
| 0–8 s | Entrada | baixa | vento, sino, órgão, caixinha de música |
| 8–24 s | Gente | subindo | baixo pulsando, chimbal, bateria, theremin |
| 24–28 s | Virada | tensão | riser, rufo de caixa |
| 28–52 s | Ápice | máxima | bumbo 4x4, palmas, baixo, lead |
| 52–60 s | Fecho | descendo | música abafada, sino final, fade |

Tempo: 1 batida = 0,5 s, 1 compasso = 2 s. Todo corte em múltiplo de 0,5 s cai na batida.

Para mudar duração, andamento ou instrumentos: edite `gerar_trilha.py` e rode `python3 gerar_trilha.py` (precisa de Python + numpy).
