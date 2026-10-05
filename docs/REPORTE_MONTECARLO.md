# Informe Monte Carlo: estimadores de CATE y de DiD

## 1. Resumen ejecutivo

Este informe consolida tres simulaciones Monte Carlo sobre datos sintéticos con efecto verdadero conocido: la estabilidad de los estimadores de CATE en un RCT (30 semillas), el sesgo y la varianza de los estimadores de diferencias en diferencias con adopción escalonada (500 paneles) y cómo converge el estimador group-time al aumentar el número de sitios.

**Conclusiones**

- **RCT.** Causal Forest lidera el ranking por valor de política en 3 de 3 presupuestos y supera a la regla de riesgo en la mayoría de las semillas; la ventaja se estrecha al ampliar el presupuesto.
- **DiD.** El TWFE ingenuo tiene un sesgo sistemático (+7.1%, subestima el efecto), pero su dispersión es baja (8.0%). El group-time no tiene sesgo (-0.5%) pero con 32 sitios su dispersión es 15.6%. Resultado: el TWFE tiene menor RMSE que el group-time en este diseño (10.7% frente a 15.6%).
- **Cobertura.** Los intervalos bootstrap por sitios cubren menos que el 95% nominal (91.8%); el intervalo analítico del TWFE cubre 84.6%.
- **Escala.** El error del group-time baja como 1/√N; con 16 sitios la cobertura es claramente insuficiente.

### Estimadores CATE (RCT, 30 semillas; media ± desv. estándar)

Valor de política como porcentaje del oráculo (el que conoce el efecto verdadero de cada camión).

| Estimador | % del oráculo, presupuesto 10% | % del oráculo, presupuesto 30% | % del oráculo, presupuesto 60% | Correlación con el CATE verdadero | Qini medio |
|---|---|---|---|---|---|
| Causal Forest | 91.9 ± 7.6 | 94.1 ± 3.6 | 95.8 ± 3.0 | 0.82 | 1097 |
| DR-Learner | 83.3 ± 9.3 | 88.6 ± 6.2 | 93.4 ± 3.5 | 0.70 | 1037 |
| S-Learner | 79.2 ± 7.3 | 82.5 ± 4.9 | 87.1 ± 3.3 | 0.57 | 696 |
| X-Learner | 73.2 ± 5.5 | 76.4 ± 4.0 | 82.3 ± 3.0 | 0.39 | 470 |
| T-Learner | 69.6 ± 5.7 | 73.6 ± 3.6 | 80.5 ± 2.8 | 0.32 | 342 |
| Regla de riesgo | 84.3 ± 3.6 | 88.0 ± 2.2 | 92.1 ± 1.6 | - | - |
| Aleatoria (esperada) | 39.9 ± 1.0 | 56.0 ± 0.6 | 74.4 ± 0.3 | - | - |

### Estimadores DiD (500 paneles, 32 sitios x 36 meses)

Error = (estimación − ATT verdadero) / |ATT verdadero|; positivo significa que subestima el efecto.

| Estimador | Error medio (% del ATT verdadero) | Desv. estándar | RMSE | Cobertura IC 95% |
|---|---|---|---|---|
| TWFE ingenuo | +7.13 | 7.97 | 10.69 | 84.6% |
| Group-time, control never-treated | -0.50 | 15.63 | 15.63 | 91.8% |
| Group-time, control not-yet-treated | -0.49 | 14.50 | 14.49 | 91.8% |
| Group-time, never-treated, base 1 mes, ponderado por cohorte | -1.78 | 24.76 | 24.80 | 92.8% |

## 2. Estudio de estabilidad RCT

Se repite el experimento con 30 semillas distintas. En cada una se entrenan los cinco estimadores, se elige a los camiones con mayor CATE predicho dentro de un presupuesto (fracción de la flota tratada) y se mide el valor de esa política con el efecto verdadero, como porcentaje del oráculo.

### 2.1 Ranking de estimadores por presupuesto

**Presupuesto 10%** (120 camiones). Concordancia entre semillas: Kendall W = 0.78, ranking idéntico al consenso en 63% de las semillas.

| Posición | Estimador | Rango medio | Primero en (semillas) | % del oráculo | Supera a la regla de riesgo |
|---|---|---|---|---|---|
| 1 | Causal Forest | 1.13 | 28 | 91.9 ± 7.6 | 27/30 |
| 2 | DR-Learner | 2.47 | 1 | 83.3 ± 9.3 | 15/30 |
| 3 | S-Learner | 2.73 | 1 | 79.2 ± 7.3 | 6/30 |
| 4 | X-Learner | 3.90 | 0 | 73.2 ± 5.5 | 0/30 |
| 5 | T-Learner | 4.77 | 0 | 69.6 ± 5.7 | 0/30 |

**Presupuesto 30%** (360 camiones). Concordancia entre semillas: Kendall W = 0.92, ranking idéntico al consenso en 77% de las semillas.

| Posición | Estimador | Rango medio | Primero en (semillas) | % del oráculo | Supera a la regla de riesgo |
|---|---|---|---|---|---|
| 1 | Causal Forest | 1.07 | 28 | 94.1 ± 3.6 | 28/30 |
| 2 | DR-Learner | 2.10 | 2 | 88.6 ± 6.2 | 20/30 |
| 3 | S-Learner | 2.90 | 0 | 82.5 ± 4.9 | 2/30 |
| 4 | X-Learner | 4.03 | 0 | 76.4 ± 4.0 | 0/30 |
| 5 | T-Learner | 4.90 | 0 | 73.6 ± 3.6 | 0/30 |

**Presupuesto 60%** (720 camiones). Concordancia entre semillas: Kendall W = 0.95, ranking idéntico al consenso en 80% de las semillas.

| Posición | Estimador | Rango medio | Primero en (semillas) | % del oráculo | Supera a la regla de riesgo |
|---|---|---|---|---|---|
| 1 | Causal Forest | 1.10 | 27 | 95.8 ± 3.0 | 26/30 |
| 2 | DR-Learner | 1.93 | 3 | 93.4 ± 3.5 | 22/30 |
| 3 | S-Learner | 3.00 | 0 | 87.1 ± 3.3 | 0/30 |
| 4 | X-Learner | 4.07 | 0 | 82.3 ± 3.0 | 0/30 |
| 5 | T-Learner | 4.90 | 0 | 80.5 ± 2.8 | 0/30 |

Acuerdo por pares (presupuesto 30%): Causal Forest > DR-Learner en 93% de las semillas, DR-Learner > S-Learner en 90%.

### 2.2 Qini frente a la regla de riesgo

En la práctica el efecto verdadero no se observa, así que el estimador se elige con el área Qini. La pregunta es si esa elección vence a una regla simple de riesgo (tratar a los camiones con mayor riesgo base).

| Presupuesto | Elegido por Qini (% oráculo) | Regla de riesgo (% oráculo) | Diferencia media [IC 95%] | Semillas donde Qini gana | Qini elige al mejor real |
|---|---|---|---|---|---|
| 10% | 88.2 ± 9.5 | 84.3 ± 3.6 | +3.92 pp [+1.08, +6.60] | 22/30 | 18/30 |
| 30% | 91.4 ± 5.5 | 88.0 ± 2.2 | +3.48 pp [+1.52, +5.29] | 23/30 | 19/30 |
| 60% | 94.3 ± 3.8 | 92.1 ± 1.6 | +2.29 pp [+0.91, +3.55] | 24/30 | 18/30 |

Frecuencia con la que el Qini elige cada estimador (30 semillas): Causal Forest 18, DR-Learner 9, X-Learner 2, S-Learner 1. El Qini por semilla es ruidoso (desviación estándar de 545 para el Causal Forest sobre una media de 1097), por lo que a veces elige un estimador que no es el mejor.

### 2.3 Pérdida relativa ex-post

Pérdida de valor de política por haber elegido con Qini en lugar del mejor estimador, medida después con el efecto verdadero (porcentaje relativo al valor del mejor).

| Presupuesto | Pérdida media | Mediana | Percentil 95 | Máxima | Semillas con elección errada |
|---|---|---|---|---|---|
| 10% | 4.81% | 0.00% | 22.52% | 25.21% | 40% |
| 30% | 2.84% | 0.00% | 15.47% | 20.01% | 37% |
| 60% | 1.80% | 0.00% | 9.71% | 15.25% | 40% |

La pérdida típica es nula (mediana 0.0%) porque Qini acierta en la mayoría de las semillas; el costo se concentra en las semillas donde falla (percentil 95 de 15.5% con presupuesto 30%).

Control de coherencia con la semilla 42 publicada: Causal Forest 97.68% (results.json: 97.7%), regla de riesgo 89.72% (results.json: 89.7%), diferencia maxima de Qini 0.00e+00. Coherente: si.

## 3. Evaluación de insesgadez vs. varianza en DiD

Se simulan 500 paneles independientes con el mismo proceso generador del panel publicado (32 sitios x 36 meses, cuatro cohortes de adopción, efecto verdadero dinámico y conocido). El estimando es el ATT verdadero promedio sobre los sitio-mes tratados.

### 3.1 TWFE frente a group-time

| Estimador | Sesgo medio (%) | Desv. estándar | Mediana | Percentiles 5-95 | Error absoluto medio | RMSE |
|---|---|---|---|---|---|---|
| TWFE ingenuo | +7.13 | 7.97 | +7.41 | -6.2 a +20.2 | 8.87 | 10.69 |
| Group-time, control never-treated | -0.50 | 15.63 | +0.29 | -26.6 a +24.0 | 12.51 | 15.63 |
| Group-time, control not-yet-treated | -0.49 | 14.50 | -0.50 | -24.3 a +22.8 | 11.58 | 14.49 |
| Group-time, never-treated, base 1 mes, ponderado por cohorte | -1.78 | 24.76 | -1.87 | -42.4 a +40.0 | 19.83 | 24.80 |

- **El TWFE está sesgado de forma sistemática.** Subestima el efecto en promedio 7.1%, y el sesgo es estable entre paneles (desviación 8.0%). Es coherente con el mecanismo de Goodman-Bacon (sitios ya tratados, con efecto todavía creciendo, usados como control de los que adoptan después), aunque esta simulación mide el sesgo y no aísla su causa.
- **El group-time no tiene sesgo apreciable (-0.50%) pero es mucho más ruidoso** (15.6% de desviación). Con solo 8 sitios de control, el contraste de cada cohorte hereda el ruido de ese grupo.
- **Por RMSE, el TWFE gana en este diseño** (10.7% frente a 15.6%): el group-time queda más cerca de la verdad que el TWFE solo en 38.6% de los paneles. La corrección elimina el sesgo a costa de varianza; con pocos sitios, eso no compensa.
- **El caso publicado no es representativo.** En la semilla 42 el TWFE da +6.6% y el group-time -1.5%. El primero es un valor típico (42.2% de los paneles tiene un error menor o igual); el segundo es una muestra favorable: solo el 7.4% de los paneles logra un error absoluto igual o menor que 1.5%.

### 3.2 Cobertura empírica del intervalo de confianza

Intervalo nominal del 95%. Para el group-time se usa bootstrap por sitios (percentiles, 300 remuestreos por panel); para el TWFE, el intervalo analítico con errores robustos por sitio.

| Estimador | Cobertura | Error MC | Verdad bajo el IC | Verdad sobre el IC | Ancho medio (h) |
|---|---|---|---|---|---|
| TWFE ingenuo | 84.6% | ±1.6 | 15.2% | 0.2% | 2.83 |
| Group-time, control never-treated | 91.8% | ±1.2 | 3.4% | 4.8% | 5.09 |
| Group-time, control not-yet-treated | 91.8% | ±1.2 | 3.8% | 4.4% | 4.77 |
| Group-time, never-treated, base 1 mes, ponderado por cohorte | 92.8% | ±1.2 | 3.2% | 4.0% | 8.56 |

- **Group-time:** la cobertura real (91.8%) queda por debajo del 95% nominal, pero sin sesgo direccional marcado (la verdad queda 3.4% por debajo y 4.8% por encima). Es lo esperable de un bootstrap por clusters con 32 sitios.
- **TWFE:** el intervalo es demasiado angosto (ancho medio 2.8 h frente a 5.1 h) y centrado en un valor sesgado: la verdad queda fuera por debajo en 15.2% de los paneles, y por encima casi nunca.

### 3.3 Impacto del grupo de control not-yet-treated

Usar como control todos los sitios aún no tratados (además de los never-treated) agranda el grupo de comparación sin reintroducir comparaciones prohibidas, porque un sitio nunca se usa como control en un período en que ya está tratado.

| Métrica | Control never-treated | Control not-yet-treated | Cambio |
|---|---|---|---|
| Desviación estándar del error (%) | 15.63 | 14.50 | -7.3% |
| RMSE (%) | 15.63 | 14.49 | -7.3% |
| Sesgo medio (%) | -0.50 | -0.49 | - |
| Cobertura IC 95% | 91.8% | 91.8% | +0.0 pp |
| Ancho medio del IC (h) | 5.09 | 4.77 | -6.4% |

La mejora es real pero modesta: reduce la dispersión en torno a un 7% y deja la cobertura prácticamente igual. Queda más cerca de la verdad que el TWFE en 41.6% de los paneles (contra 38.6% con control never-treated), por lo que no cambia la conclusión de la sección 3.1. A cambio exige un supuesto adicional: tendencias paralelas también entre las cohortes usadas como control.

La variante con base de un mes y ponderación por cohorte (la configuración del bootstrap del repositorio para `mpdta`) es la menos precisa: desviación de 24.8% y IC de 8.6 h de ancho medio.

## 4. Análisis de convergencia en N

Se varía el número de sitios N entre 16, 32, 64, 128 (cuatro cohortes iguales: temprana, intermedia, tardía y never-treated) y se simulan 100 paneles por valor de N. Estimador: group-time, control never-treated, base de 3 meses, media sin ponderar de las celdas post-tratamiento. Intervalo: bootstrap por sitios, 300 remuestreos.

| N (sitios) | σ del error (h) | σ (% del ATT) | σ·√N | Cobertura IC 95% | Ancho medio IC (h) | Sesgo (h) |
|---|---|---|---|---|---|---|
| 16 | 2.135 | 23.4% | 8.54 | 83% ±3.8 | 7.02 | +0.015 |
| 32 | 1.251 | 13.7% | 7.07 | 92% ±2.7 | 5.23 | -0.239 |
| 64 | 0.992 | 10.9% | 7.93 | 91% ±2.9 | 3.70 | +0.123 |
| 128 | 0.705 | 7.7% | 7.97 | 93% ±2.6 | 2.66 | +0.047 |

**Constatación σ·√N ≈ 8.** El producto σ·√N queda entre 7.07 y 8.54 (media 7.88): 8.54 para N = 16, 7.07 para N = 32, 7.93 para N = 64, 7.97 para N = 128. Con N = 64 y N = 128 el valor es 7.93 y 7.97, prácticamente 8. Las desviaciones de N = 16 y N = 32 respecto de la media (+8% y -10%) son compatibles con el error de muestreo de estimar una desviación estándar con 100 paneles (error relativo ≈ 7%, doble ≈ 14%), así que los datos son coherentes con una tasa 1/√N sin evidencia de desvío.

**Cobertura.** Con N = 16 el bootstrap cubre 83%, claramente por debajo del 95%; desde N = 32 queda entre 91% y 93%, a menos de 1.4 errores de simulación del nominal (con 100 paneles, ±2.6 a ±2.9 pp). No hay evidencia suficiente para distinguir esa cobertura del 95%, pero tampoco para afirmar que lo alcanza.

**Implicación para el diseño.** El panel publicado (32 sitios) está justo en el umbral: el error típico es de 14% del efecto, y bajarlo a menos de 9% requiere del orden de 128 sitios.

## Reproducibilidad

- Entradas: `tmp_agent_a/rct_results_30_seeds.json`, `tmp_agent_b/did_results_500_panels.json`, `tmp_agent_b/sensitivity_results.json`.
- RCT: 30 semillas; DiD: 500 paneles (semillas 1000+), 300 remuestreos bootstrap por panel; convergencia: 100 paneles por N (semilla base 5000).
- Todos los datos son sintéticos: el efecto verdadero se conoce porque el simulador lo define; ninguna cifra de este informe proviene de datos reales.
- Este archivo se genera con `python scripts/generate_summary_report.py`; no se edita a mano.
