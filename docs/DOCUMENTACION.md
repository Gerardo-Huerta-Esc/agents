# Documentación exhaustiva: Agentes de IA y mejoras implementadas

**Versión:** 1.4  
**Proyecto:** Tu primer agente de IA — Extensión con mejoras profesionales  
**Audiencia:** Desarrolladores y lectores con formación en física/matemáticas que desean una comprensión formal y profunda de los agentes de IA

---

## Índice general

1. [Introducción](#1-introducción)
2. [Parte 0: Fundamentos matemáticos](#parte-0--fundamentos-matemáticos) — incluye [§0.5B Embeddings](#05b-el-salto-de-la-probabilidad-a-la-geometría-representaciones-vectoriales-y-embeddings), [§0.5C Atención](#05c-mecánica-estadística-de-la-atención-transformers), [§0.5D Control y estabilidad](#05d-teoría-de-control-y-estabilidad-del-bucle-del-agente)
3. [Capítulo 1: ¿Qué es un agente de IA?](#capítulo-1-qué-es-un-agente-de-ia)
4. [Capítulo 2: Fundamentos técnicos](#capítulo-2-fundamentos-técnicos)
5. [Capítulo 3: Tool-calling](#capítulo-3-tool-calling-herramientas-para-el-agente)
6. [Capítulo 4: El patrón ReAct](#capítulo-4-el-patrón-react)
7. [Capítulo 5: Sistemas de memoria en agentes](#capítulo-5-sistemas-de-memoria-en-agentes)
8. [Capítulo 6: Seguridad y guardrails](#capítulo-6-seguridad-y-guardrails)
9. [Capítulo 7: Persistencia y estado](#capítulo-7-persistencia-y-estado)
10. [Capítulo 8: Observabilidad y trazas](#capítulo-8-observabilidad-y-trazas)
11. [Capítulo 9: Implementación detallada](#capítulo-9-implementación-detallada)
12. [Capítulo 10: Uso, personalización y extensión](#capítulo-10-uso-personalización-y-extensión)
13. [Capítulo 11: Comparación original vs mejorado](#capítulo-11-comparación-agente-original-vs-mejorado)
14. [Capítulo 12: Preguntas frecuentes](#capítulo-12-preguntas-frecuentes)
15. [Capítulo 13: Evaluación rigurosa de agentes](#capítulo-13-evaluación-rigurosa-de-agentes)
16. [Capítulo 14: Arquitectura de software — El agente en producción](#capítulo-14-arquitectura-de-software--el-agente-en-producción)
17. [Glosario](#glosario)
18. [Anexo A: Temas matemáticos avanzados](#anexo-a-temas-matemáticos-avanzados)
19. [Referencias](#referencias)

---

## 1. Introducción

Esta documentación acompaña al proyecto de **agente de IA mejorado** (`agent_enhancements.py`). Su objetivo es triple:

1. **Formar** — Explicar desde cero qué son los agentes de IA y cómo funcionan.
2. **Documentar** — Detallar cada pieza del código que hemos implementado.
3. **Guiar** — Permitir que un novato pueda entender, modificar y extender el sistema.

Si nunca has programado un agente, empieza por el Capítulo 1. Si tienes formación en física o matemáticas, la **Parte 0** ofrece el formalismo que subyace a los LLMs y agentes. Si ya conoces los conceptos básicos, puedes ir directo a los capítulos de implementación (9 y 10).

### Prerrequisitos recomendados

- Conocimientos básicos de Python (variables, funciones, clases).
- Haber ejecutado al menos una vez el agente original (`main.py`).
- Cuenta en OpenAI con API key configurada en `.env`.

### Recorrido sugerido para principiantes

1. **Día 1:** Capítulos 1 y 2 (conceptos de agentes y LLMs).
2. **Día 2:** Capítulo 3 (tool-calling) + ejecutar `main.py` y observar el flujo.
3. **Día 3:** Capítulos 4 y 5 (ReAct y memoria).
4. **Día 4:** Capítulos 6, 7 y 8 (seguridad, persistencia, trazas).
5. **Día 5:** Capítulos 9 y 10 (implementación y uso).
6. **Día 6 (opcional):** Capítulo 13 (evaluación rigurosa) — para quienes quieran medir el agente de forma objetiva.
7. **Producción:** Capítulo 14 (arquitectura) — para llevar el agente a un entorno real (inyección, errores, async).

---

# PARTE 0 — FUNDAMENTOS MATEMÁTICOS

*Esta sección está dirigida a lectores con formación sólida en física y matemáticas (cálculo, álgebra lineal, probabilidad, mecánica estadística) que desean una comprensión rigurosa de los fundamentos matemáticos de los agentes de IA. Se asume que el lector es iniciante absoluto en aprendizaje automático y LLMs, pero domina el lenguaje matemático formal. Las ecuaciones usan LaTeX.*

---

## Prólogo: Estructura y prerrequisitos

La Parte 0 está organizada en bloques conceptuales:

1. **Probabilidad y cadenas** (§0.1–0.2): Regla de la cadena, condicionales, factorización de secuencias.
2. **Teoría de la información** (§0.3–0.5): Entropía, información mutua, divergencia KL, compresión.
3. **El salto a la geometría** (§0.5B): Embeddings, espacios vectoriales de alta dimensión, métricas de similitud, hipótesis del colector.
4. **Mecánica estadística de la atención** (§0.5C): Scaled dot-product attention, Q/K/V, analogía Boltzmann, coste $O(L^2)$.
5. **Procesos estocásticos** (§0.6–0.8): Cadenas de Markov, MDPs, políticas, Bellman.
6. **Teoría de control y estabilidad** (§0.5D): Bucle como feedback, convergencia, oscilaciones, puntos de inestabilidad.
7. **Decisión y optimización** (§0.9–0.11): Utilidad, temperatura, exploración/explotación.
8. **Aplicación al agente** (§0.12–0.18): Estado, transiciones, guardrails, coste, ejemplo, resumen, referencias cruzadas.

**Notación:** Variables aleatorias en mayúscula ($X$, $Y$); realizaciones en minúscula ($x$, $y$). $\mathbb{E}[\cdot]$ denota esperanza; $\log$ es logaritmo natural salvo que se indique $\log_2$.

---

## 0.1 Regla de la cadena y factorización de secuencias

### 0.1.1 Regla de la cadena (chain rule)

Para variables aleatorias $X_1, \ldots, X_n$, la **regla de la cadena** de la probabilidad establece que la distribución conjunta se factoriza en producto de condicionales:

$$
P(X_1, X_2, \ldots, X_n) = P(X_1) \cdot P(X_2 \mid X_1) \cdot P(X_3 \mid X_1, X_2) \cdots P(X_n \mid X_1, \ldots, X_{n-1})
$$

En forma compacta:

$$
P(X_{1:n}) = \prod_{t=1}^{n} P(X_t \mid X_{1:t-1})
$$

donde $X_{1:t-1}$ abrevia $(X_1, \ldots, X_{t-1})$. Esta identidad es **exacta**; no hay aproximación. La clave es que modelar $P(X_t \mid X_{1:t-1})$ para cada $t$ es equivalente a modelar la conjunta, pero descompone el problema en predicciones **autoregresivas**: en cada paso se predice el siguiente elemento dados todos los anteriores.

### 0.1.2 Aplicación al texto: tokens

En lenguaje natural, el texto es una secuencia de **tokens** (unidades discretas: palabras, sub-palabras o caracteres). Denotemos $\mathcal{V}$ el **vocabulario** (conjunto finito de tokens, típicamente $|\mathcal{V}| \sim 10^4$–$10^5$). Una secuencia de longitud $n$ es $x_1, x_2, \ldots, x_n$ con $x_t \in \mathcal{V}$.

La probabilidad de la secuencia completa es:

$$
P(x_1, \ldots, x_n) = \prod_{t=1}^{n} P(x_t \mid x_1, \ldots, x_{t-1})
$$

El **modelo de lenguaje** es un artificio que aproxima cada factor $P(x_t \mid x_1, \ldots, x_{t-1})$ mediante una función parametrizada (por ejemplo, una red neuronal con parámetros $\theta$):

$$
P(x_t \mid x_1, \ldots, x_{t-1}) \approx p_\theta(x_t \mid x_1, \ldots, x_{t-1})
$$

### 0.1.3 Condicionales y teorema de Bayes

La definición de probabilidad condicional es $P(A \mid B) = P(A \cap B) / P(B)$. De ahí:

$$
P(X_t \mid X_{1:t-1}) = \frac{P(X_{1:t})}{P(X_{1:t-1})}
$$

El **teorema de Bayes** relaciona $P(A|B)$ con $P(B|A)$:

$$
P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}
$$

En el contexto de modelos de lenguaje, el modelo aprende $P(\text{siguiente token} \mid \text{contexto})$. La inferencia "inversa" (dado un texto, qué contexto lo generó) no se usa explícitamente en la generación, pero subyace en métodos de interpretabilidad y análisis.

### 0.1.4 Ventana de contexto: truncamiento del pasado

En la práctica, la dependencia en todo el pasado $x_1, \ldots, x_{t-1}$ es inviable: la longitud puede crecer sin límite y la arquitectura (p. ej. transformers) impone una longitud máxima $L$. Se **trunca** el contexto a los últimos $L$ tokens:

$$
P(x_t \mid x_1, \ldots, x_{t-1}) \approx P(x_t \mid x_{\max(1,t-L)}, \ldots, x_{t-1})
$$

Para $t \leq L$ se usa todo el pasado; para $t > L$, solo los últimos $L$ tokens. Esto introduce una **pérdida de información**: el modelo no "ve" los tokens anteriores a la ventana. En agentes con conversaciones largas, este es el motivo de usar **memoria episódica** (resúmenes) para preservar información antigua.

---

## 0.2 Generación: muestreo y decodificación

Una vez tenemos $p_\theta(x_t \mid \text{contexto})$, la **generación** consiste en producir tokens secuencialmente. Hay dos estrategias principales:

### 0.2.1 Decodificación greedy (argmax)

Se elige en cada paso el token de mayor probabilidad:

$$
\hat{x}_t = \arg\max_{x \in \mathcal{V}} p_\theta(x \mid x_{1:t-1})
$$

Es **determinista**: dado el mismo contexto, siempre se obtiene el mismo siguiente token. Útil cuando se busca consistencia (p. ej. agentes que ejecutan tareas críticas).

### 0.2.2 Muestreo (sampling)

Se muestrea según la distribución:

$$
x_t \sim p_\theta(\cdot \mid x_{1:t-1})
$$

Esto introduce **aleatoriedad** y variedad. En la práctica se suele modificar la distribución con **temperatura** (véase §0.10).

### 0.2.3 Logits y softmax: de salida de red a probabilidades

Las redes neuronales (incluyendo los LLMs) no producen probabilidades directamente; producen **logits** $z_y \in \mathbb{R}$ para cada token $y \in \mathcal{V}$. La conversión a distribución de probabilidad se hace mediante la **función softmax**:

$$
p(y) = \frac{e^{z_y}}{\sum_{y' \in \mathcal{V}} e^{z_{y'}}} = \frac{e^{z_y}}{Z}
$$

donde $Z = \sum_{y'} e^{z_{y'}}$ es la función de partición. Propiedades:
- $p(y) \in (0, 1)$ y $\sum_y p(y) = 1$
- El argmax de $p$ coincide con el argmax de $z$
- La temperatura $\tau$ se aplica como $p_\tau(y) = \mathrm{softmax}(z/\tau)$

**Gradientes:** Si $\mathcal{L} = -\log p(y^*)$ (cross-entropy para el token verdadero $y^*$), entonces $\frac{\partial \mathcal{L}}{\partial z_y} = p(y) - \mathbb{1}[y = y^*]$, que es la base del backpropagation en clasificación.

### 0.2.4 Muestreo top-k y top-p (nucleus)

Además de la temperatura, se usan heurísticas para truncar la distribución antes de muestrear:
- **Top-k:** Mantener solo los $k$ tokens de mayor probabilidad y renormalizar.
- **Top-p (nucleus):** Mantener el conjunto más pequeño cuyo total de probabilidad sea $\geq p$, luego renormalizar.

Ambos reducen la probabilidad de muestrear tokens de cola (raros, a menudo incoherentes).

### 0.2.5 Generación autoregresiva completa

El proceso de generación es un bucle:

1. Inicializar con el prompt (contexto) $c = (c_1, \ldots, c_m)$.
2. Para $t = 1, 2, \ldots$ hasta parada:
   - Calcular $p_\theta(\cdot \mid c, x_1, \ldots, x_{t-1})$.
   - Elegir $x_t$ (greedy o sampling).
   - Si $x_t$ es token de parada (EOS), terminar.

La secuencia generada es $x_1, x_2, \ldots, x_T$.

---

## 0.3 Entropía de Shannon

### 0.3.1 Definición

Para una variable aleatoria discreta $X$ con distribución $p(x) = P(X=x)$, la **entropía de Shannon** es:

$$
H(X) = -\sum_{x} p(x) \log p(x) = \mathbb{E}_{X \sim p}\left[ -\log p(X) \right]
$$

Por convención, $0 \log 0 = 0$. La entropía mide la **incertidumbre** o **sorpresa media** asociada a $X$.

### 0.3.2 Propiedades (breve repaso)

1. **No negatividad:** $H(X) \geq 0$, con igualdad si y solo si $X$ es determinista (una masa de probabilidad en un solo valor).
2. **Máximo para distribución uniforme:** Para $\mathcal{X}$ finito con $|\mathcal{X}| = N$:
   $$
   H(X) \leq \log N
   $$
   con igualdad si y solo si $p(x) = 1/N$ para todo $x$.
3. **Aditividad para variables independientes:** Si $X \perp Y$:
   $$
   H(X, Y) = H(X) + H(Y)
   $$
4. **Subaditividad general:**
   $$
   H(X, Y) \leq H(X) + H(Y)
   $$
   con igualdad si y solo si $X \perp Y$.

### 0.3.3 Entropía condicional

La **entropía condicional** de $X$ dado $Y$ es:

$$
H(X \mid Y) = -\sum_{x,y} p(x,y) \log p(x|y) = \mathbb{E}_{X,Y}\left[ -\log p(X|Y) \right]
$$

Interpretación: incertidumbre restante sobre $X$ una vez conocido $Y$. Se cumple la **regla de la cadena para entropía**:

$$
H(X, Y) = H(Y) + H(X \mid Y)
$$

Iterando: $H(X_1, \ldots, X_n) = H(X_1) + \sum_{t=2}^{n} H(X_t \mid X_{1:t-1})$.

### 0.3.4 Desigualdad de Jensen y concavidad de la entropía

La función $f(x) = -x \log x$ es **cóncava** en $(0, \infty)$. Por la **desigualdad de Jensen** (para una función cóncava $f$ y una variable aleatoria $X$: $\mathbb{E}[f(X)] \leq f(\mathbb{E}[X])$):

$$
H(X) = \mathbb{E}[-\log p(X)] \geq -\log \mathbb{E}[p(X)]
$$

Más relevante: la entropía $H(p)$ es **cóncava** como función de $p$. Es decir, para $\lambda \in [0,1]$ y distribuciones $p$, $q$:
$$
H(\lambda p + (1-\lambda) q) \geq \lambda H(p) + (1-\lambda) H(q)
$$

Esto implica que "mezclar" distribuciones aumenta la entropía (mayor incertidumbre).

### 0.3.5 Entropía en el modelo de lenguaje

En cada paso $t$, el modelo produce una distribución $p_t(x) = p_\theta(x \mid x_{1:t-1})$ sobre $\mathcal{V}$. La **entropía de la predicción** en ese paso es:

$$
H_t = -\sum_{x \in \mathcal{V}} p_t(x) \log p_t(x)
$$

- $H_t$ alta: el modelo está muy inseguro (distribución dispersa).
- $H_t$ baja: el modelo está seguro (pico pronunciado en uno o pocos tokens).

---

## 0.4 Perplejidad y longitud normalizada

### 0.4.1 Definición de perplejidad

Dada una secuencia observada $\mathbf{x} = (x_1, \ldots, x_n)$ y un modelo que asigna probabilidades $p(x_t \mid x_{1:t-1})$, la **perplejidad** es:

$$
\text{PP}(\mathbf{x}) = \exp\left( -\frac{1}{n} \sum_{t=1}^{n} \log p(x_t \mid x_{1:t-1}) \right)
$$

El término interno $-\frac{1}{n}\sum_t \log p(x_t \mid \ldots)$ es la **log-probabilidad media** (en nats por token). La perplejidad es su exponencial.

### 0.4.2 Interpretación

- $\text{PP} = 1$: el modelo asigna probabilidad 1 a cada token (ideal, inalcanzable en práctica).
- $\text{PP} = |\mathcal{V}|$: equivalente a una distribución uniforme (modelo que no ha aprendido nada).
- Cuanto **menor** la perplejidad, **mejor** el modelo predice la secuencia.

La perplejidad puede verse como el "número efectivo de opciones" que el modelo considera equiprobables para el siguiente token; es la exponencial de la entropía cruzada (véase siguiente sección).

### 0.4.3 Relación con la entropía

Si la secuencia fuera generada por una distribución $q$ y el modelo predice $p$, la log-probabilidad media bajo el modelo es $-\frac{1}{n}\sum_t \log p(x_t \mid \ldots)$. Cuando $q = p$ (el modelo es la distribución verdadera), esto coincide con la **entropía por símbolo** del proceso. Así, $\log \text{PP}$ aproxima la entropía del lenguaje (en nats por token).

---

## 0.5 Información mutua y divergencia de Kullback-Leibler

### 0.5.1 Información mutua

La **información mutua** entre $X$ e $Y$ es:

$$
I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X) = H(X) + H(Y) - H(X,Y)
$$

Mide cuánto reducir la incertidumbre sobre $X$ al conocer $Y$ (o viceversa). Propiedades:
- $I(X;Y) \geq 0$
- $I(X;Y) = 0$ si y solo si $X \perp Y$

### 0.5.2 Divergencia de Kullback-Leibler

Para distribuciones $p$ y $q$ sobre el mismo espacio, la **divergencia KL** de $p$ respecto a $q$ es:

$$
D_{\text{KL}}(p \,\|\, q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_{X \sim p}\left[ \log \frac{p(X)}{q(X)} \right]
$$

No es una distancia (no es simétrica ni cumple la desigualdad triangular). Interpretación: "sorpresa extra" al usar $q$ cuando la verdad es $p$.

### 0.5.3 Entropía cruzada

La **entropía cruzada** entre $p$ (verdad) y $q$ (modelo) es:

$$
H(p, q) = -\sum_x p(x) \log q(x)
$$

Se cumple $H(p, q) = H(p) + D_{\text{KL}}(p \| q)$. En entrenamiento de modelos, se minimiza $H(p_{\text{data}}, p_\theta)$, equivalente a minimizar la KL entre la distribución empírica y el modelo.

### 0.5.4 Teorema de codificación de fuentes (Shannon)

El **teorema de codificación sin ruido** establece que para comprimir una fuente con entropía $H(X)$ bits por símbolo, no podemos usar en promedio menos de $H(X)$ bits por símbolo sin pérdida. La compresión óptima alcanza $H(X)$ usando codificación de Huffman o aritmética. Esto da un límite fundamental al tamaño del resumen de memoria.

### 0.5.5 Rate-distortion y compresión con pérdida

En **compresión con pérdida**, permitimos una distorsión $D$ entre el original y la reconstrucción. La **función rate-distortion** $R(D)$ es el mínimo de bits por símbolo necesario para lograr distorsión $\leq D$. Para una distorsión cuadrática $d(x, \hat{x}) = (x - \hat{x})^2$ bajo Gaussianas, $R(D) = \frac{1}{2}\log(\sigma^2/D)$ para $D \leq \sigma^2$.

En memoria episódica, la "distorsión" es la pérdida de detalle; el "rate" es la longitud del resumen. Una buena compresión maximiza la información mutua $I(M; \tilde{M})$ dado el presupuesto de bits (rate).

---

# 0.5B El salto de la probabilidad a la geometría: representaciones vectoriales y embeddings

*Para un físico o matemático, la magia de los LLMs no reside únicamente en la probabilidad autoregresiva. Bajo la capa de predicción de tokens late un **espacio latente** de representaciones vectoriales que codifica significado. Este capítulo desarrolla la geometría de los embeddings: el puente entre el mundo discreto de tokens y el mundo continuo de vectores, donde la **similitud semántica** se traduce en **proximidad geométrica**.*

---

## 0.5B.1 Motivación: de símbolos a geometría

Hasta ahora hemos trabajado con:
- **Tokens** (símbolos discretos en $\mathcal{V}$)
- **Probabilidades** $P(x_t \mid x_{1:t-1})$ sobre el vocabulario
- **Entropía** y **información** sobre distribuciones discretas

Pero los modelos de lenguaje modernos (y en particular los usados para agentes, RAG y memoria semántica) construyen internamente **representaciones continuas**. Cada token, cada secuencia, se asigna a un punto en $\mathbb{R}^d$ (con $d$ típicamente 768, 1536, 3072 o más). Esta proyección $\mathcal{V} \to \mathbb{R}^d$ (o $\mathcal{V}^* \to \mathbb{R}^d$ para secuencias) es un **embedding**.

**Pregunta clave:** ¿Por qué geometría? Porque permite definir **similitud** entre textos que nunca comparten palabras exactas. "El gato duerme" y "Un felino descansa" son semánticamente cercanos; en el espacio de embeddings, sus vectores estarán próximos. La probabilidad sola no ofrece eso: los tokens son distintos. La geometría del espacio latente codifica **significado**.

---

## 0.5B.2 Definición formal de embedding

### 0.5B.2.1 Embedding como función

Un **embedding** es una función (parametrizada por el modelo $\theta$):

$$
\mathbf{e}_\theta: \mathcal{X} \to \mathbb{R}^d
$$

donde $\mathcal{X}$ puede ser:
- Un token único: $\mathcal{X} = \mathcal{V}$ (embedding de tipo *lookup*)
- Una secuencia de tokens: $\mathcal{X} = \mathcal{V}^* := \bigcup_{n \geq 0} \mathcal{V}^n$
- Un documento o mensaje completo (cadena de caracteres o lista de tokens)

### 0.5B.2.2 Embeddings estáticos vs contextuales

| Tipo | Descripción | Ejemplo | Semántica |
|------|-------------|---------|-----------|
| **Estático** | Un token $x$ tiene un único vector $\mathbf{e}(x)$ independiente del contexto | Word2Vec, GloVe | "banco" (asiento) y "banco" (financiero) comparten vector |
| **Contextual** | El vector depende de toda la secuencia: $\mathbf{e}(x_1, \ldots, x_n)$ | BERT, GPT, sentence-transformers | "banco" tiene vectores distintos según contexto |

Para RAG y memoria semántica se usan **contextuales**: el embedding de una oración captura su significado global, no solo la suma de las palabras.

### 0.5B.2.3 Obtención práctica

En modelos tipo transformer, $\mathbf{e}$ se obtiene como:
- **Estado oculto** de la última capa (antes de la proyección a logits)
- Para secuencias: **mean pooling** $\mathbf{e} = \frac{1}{n}\sum_{t=1}^n \mathbf{h}_t$, o el estado del **último token** $\mathbf{h}_n$, o un token especial `[CLS]` (BERT)

Formalmente, si $f_\theta$ es el transformer y $\mathbf{H} = (\mathbf{h}_1, \ldots, \mathbf{h}_n)$ los estados ocultos:
$$
\mathbf{e}(x_1, \ldots, x_n) = \text{pool}(\mathbf{H}) \in \mathbb{R}^d
$$

### 0.5B.2.4 Propiedad deseada: monotonía de similitud

La propiedad clave es: la **similitud semántica** debe ser **monótona** con la proximidad geométrica. Es decir, existe una métrica $d$ en $\mathbb{R}^d$ tal que:
$$
\text{semántica}(x, x') \approx f\big( d(\mathbf{e}(x), \mathbf{e}(x')) \big)
$$
con $f$ decreciente. El entrenamiento por predicción de tokens (y, en modelos recientes, por contraste entre pares similares/disimilares) induce implícitamente esta estructura.

---

## 0.5B.3 Espacios vectoriales de alta dimensión

Los embeddings viven en $\mathbb{R}^d$ con $d$ grande (768–3072). Esto introduce fenómenos contraintuitivos que un físico reconocerá de la mecánica estadística y la teoría de la medida.

### 0.5B.3.1 Concentración de la medida: la cáscara domina

El volumen de la bola de radio $R$ en $\mathbb{R}^d$ es:
$$
V_d(R) = \frac{\pi^{d/2}}{\Gamma(d/2 + 1)} R^d
$$

La fracción del volumen en la **capa** $[R-\varepsilon, R]$ (con $0 < \varepsilon < R$) es:
$$
\frac{V_d(R) - V_d(R-\varepsilon)}{V_d(R)} = 1 - \left(1 - \frac{\varepsilon}{R}\right)^d
$$

Cuando $d \to \infty$, para $\varepsilon/R$ fijo en $(0,1)$:
$$
\left(1 - \frac{\varepsilon}{R}\right)^d \to 0 \quad \Rightarrow \quad \text{Fracción en la capa} \to 1
$$

**Consecuencia:** Casi todo el volumen (y, para una distribución uniforme en la bola, casi toda la masa de probabilidad) está en una capa arbitrariamente delgada cerca de la superficie. El "interior" de la bola es una fracción negligible. Este es un ejemplo de **concentración de la medida**: en alta dimensión, las distribuciones se concentran en regiones de volumen relativamente pequeño.

### 0.5B.3.2 Distancia entre vectores aleatorios: derivación

Sean $\mathbf{u}, \mathbf{v} \in S^{d-1}$ (vectores unitarios) muestreados de forma **isótropa** (uniformemente en la esfera). El producto escalar $\mathbf{u} \cdot \mathbf{v}$ tiene media $\mathbb{E}[\mathbf{u} \cdot \mathbf{v}] = 0$ por simetría.

La distancia euclidiana al cuadrado es:
$$
\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\mathbf{u} \cdot \mathbf{v} = 2 - 2(\mathbf{u} \cdot \mathbf{v})
$$

Para vectores isótropos, $\mathbb{E}[\|\mathbf{u}-\mathbf{v}\|^2] = 2$. La varianza de $\mathbf{u} \cdot \mathbf{v}$ escala como $1/d$; por tanto, para $d$ grande, $\mathbf{u} \cdot \mathbf{v} \approx 0$ con alta probabilidad, y $\|\mathbf{u}-\mathbf{v}\| \approx \sqrt{2}$.

**Implicación:** Casi todos los pares de vectores aleatorios unitarios están aproximadamente a distancia $\sqrt{2}$; son "equidistantes". La **estructura** de los embeddings (textos similares cerca, disimilares lejos) no surge del azar: surge del entrenamiento, que coloca vectores en regiones no uniformes del espacio.

### 0.5B.3.3 Dimensión intrínseca vs extrínseca

Aunque los embeddings viven en $\mathbb{R}^d$ (dimensión **extrínseca**), los datos suelen concentrarse cerca de una superficie de dimensión **intrínseca** $k \ll d$. Esto es la **hipótesis del colector** (§0.5B.6).

---

## 0.5B.4 Métricas de similitud en $\mathbb{R}^d$

Dados dos vectores $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$, ¿cómo medir si representan contenido "similar"?

### 0.5B.4.1 Producto escalar (dot product)

$$
s_{\text{dot}}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{d} u_i v_i = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta
$$

(última igualdad: **ley del coseno** en $\mathbb{R}^d$). Por **Cauchy-Schwarz**:
$$
|\mathbf{u} \cdot \mathbf{v}| \leq \|\mathbf{u}\| \|\mathbf{v}\|
$$
con igualdad si y solo si $\mathbf{u}$ y $\mathbf{v}$ son linealmente dependientes (paralelos).

- **Rango (sin normalizar):** $(-\infty, +\infty)$
- **Interpretación:** Proyección de $\mathbf{v}$ sobre la dirección de $\mathbf{u}$, escalada por $\|\mathbf{u}\|$

### 0.5B.4.2 Similitud coseno (cosine similarity)

$$
\text{sim}_{\text{cos}}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \cos \theta
$$

Por Cauchy-Schwarz, $\text{sim}_{\text{cos}} \in [-1, 1]$.

- $\text{sim}_{\text{cos}} = 1$: mismos dirección y sentido
- $\text{sim}_{\text{cos}} = 0$: ortogonales
- $\text{sim}_{\text{cos}} = -1$: opuestos

**Invariancia:** $\text{sim}_{\text{cos}}(\alpha \mathbf{u}, \beta \mathbf{v}) = \text{sim}_{\text{cos}}(\mathbf{u}, \mathbf{v})$ para $\alpha, \beta > 0$. Solo importa la **dirección**.

### 0.5B.4.3 Distancia euclidiana y ley del coseno

$$
\|\mathbf{u} - \mathbf{v}\|^2 = (\mathbf{u}-\mathbf{v}) \cdot (\mathbf{u}-\mathbf{v}) = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\mathbf{u} \cdot \mathbf{v}
$$

Para vectores **unitarios** ($\|\mathbf{u}\| = \|\mathbf{v}\| = 1$):
$$
\|\mathbf{u} - \mathbf{v}\|^2 = 2 - 2\cos\theta = 2(1 - \text{sim}_{\text{cos}})
\quad \Rightarrow \quad
d_{\text{Euc}} = \sqrt{2(1 - \text{sim}_{\text{cos}})}
$$

Por tanto, **argmin** de $d_{\text{Euc}}$ = **argmax** de $\text{sim}_{\text{cos}}$: el orden de similitud es idéntico.

### 0.5B.4.4 Implementación en Python

```python
import numpy as np

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Similitud coseno: u·v / (||u|| ||v||). Rango [-1, 1]."""
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Distancia euclidiana."""
    return np.linalg.norm(u - v)

def normalized_euclidean(u: np.ndarray, v: np.ndarray) -> float:
    """Para vectores unitarios: sqrt(2(1 - cos_sim)). Equivalente a ordenar por coseno."""
    u_n, v_n = u / np.linalg.norm(u), v / np.linalg.norm(v)
    return np.sqrt(2 * (1 - np.dot(u_n, v_n)))

# Ejemplo: dos vectores en R^4
u = np.array([1.0, 0.5, 0.0, 0.2])
v = np.array([0.9, 0.6, 0.1, 0.0])
print(f"Cosine sim: {cosine_similarity(u, v):.4f}")  # ~0.99 (muy similares)
print(f"Euclidean:  {euclidean_distance(u, v):.4f}")
```

### 0.5B.4.5 ¿Cuándo usar cuál?

| Métrica | Uso típico | Ventaja |
|---------|------------|---------|
| Producto escalar | Embeddings no normalizados donde la **norma** codifica relevancia | Captura intensidad |
| Similitud coseno | Búsqueda semántica, RAG, clustering | Invariante a norma; $[-1,1]$ interpretable |
| Distancia euclidiana | k-NN, FAISS, Annoy | Fácil de indexar; equiv. a coseno si L2-norm=1 |

### 0.5B.4.6 Geometría en la esfera unitaria $S^{d-1}$

Si $\|\mathbf{u}\| = \|\mathbf{v}\| = 1$, los vectores viven en $S^{d-1}$. La **distancia geodésica** (arco sobre la esfera) es:
$$
d_{\text{geo}}(\mathbf{u}, \mathbf{v}) = \arccos(\mathbf{u} \cdot \mathbf{v}) = \theta
$$

La **distancia euclidiana en $\mathbb{R}^d$** entre dos puntos de la esfera es la cuerda:
$$
\|\mathbf{u} - \mathbf{v}\| = 2\sin\frac{\theta}{2}
$$

Ambas son **funciones crecientes de $\theta$**, así que el orden relativo de "más similar" se preserva. Los índices ANN suelen usar la distancia euclidiana (más barata de calcular que $\arccos$).

---

## 0.5B.5 Búsqueda de vecinos y recuperación

El uso principal de los embeddings en agentes es la **recuperación**: dado un mensaje o consulta $q$, encontrar los $k$ documentos más similares en una base $\mathcal{D} = \{\mathbf{e}(d_1), \ldots, \mathbf{e}(d_N)\}$.

### 0.5B.5.1 Búsqueda exhaustiva (exacta)

**Algoritmo:** Para cada $i \in \{1, \ldots, N\}$, calcular $s_i = \text{sim}_{\text{cos}}(\mathbf{q}, \mathbf{e}(d_i))$, ordenar por $s_i$ descendente y tomar los índices $\{i_1, \ldots, i_k\}$ correspondientes a los $k$ mayores.

**Coste:** $O(N \cdot d)$ en tiempo; $O(N \cdot d)$ en espacio para la matriz de embeddings. Para $N \lesssim 10^5$ suele ser viable; para millones de documentos, prohibitivo.

### 0.5B.5.2 Implementación naive en Python

```python
import numpy as np
from typing import List, Tuple

def top_k_cosine(
    query: np.ndarray,      # shape (d,)
    corpus: np.ndarray,     # shape (N, d)
    k: int = 5
) -> List[Tuple[int, float]]:
    """
    Retorna los k índices de mayor similitud coseno con la consulta.
    corpus[i] es el embedding del documento i.
    """
    # Normalizar para usar producto escalar = coseno
    q_norm = query / np.linalg.norm(query)
    corpus_norm = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    scores = np.dot(corpus_norm, q_norm)  # (N,)
    top_indices = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_indices]

# Ejemplo
d = 4
query = np.random.randn(d)
corpus = np.random.randn(100, d)
results = top_k_cosine(query, corpus, k=3)
print(results)  # [(i1, s1), (i2, s2), (i3, s3)]
```

### 0.5B.5.3 Búsqueda aproximada (ANN)

Para $N \sim 10^6$–$10^9$, la búsqueda exacta es ineficiente. Los **índices aproximados de vecinos más próximos (ANN)** trade-off precisión por velocidad:

- **HNSW** (Hierarchical Navigable Small World): grafos con búsqueda greedy; $O(\log N)$ queries
- **IVF** (Inverted File): clustering + búsqueda solo en el cluster más cercano
- **LSH** (Locality-Sensitive Hashing): hash a buckets; vectores similares caen en el mismo bucket con alta probabilidad

El **recall@k** mide la fracción de los $k$ verdaderos vecinos más cercanos que el ANN recupera. Un recall de 0.95 con 10× menos comparaciones es habitual.

---

## 0.5B.6 La hipótesis del colector (Manifold Hypothesis)

### 0.5B.6.1 Definición de variedad (colector)

Una **variedad diferenciable** $\mathcal{M}$ de dimensión $k$ es un espacio topológico tal que cada punto tiene un entorno **homeomorfo** a un abierto de $\mathbb{R}^k$. Intuitivamente: $\mathcal{M}$ es una "superficie" suave de $k$ dimensiones embebida en $\mathbb{R}^d$ (con $k \leq d$). Ejemplos: la esfera $S^2$ (dimensión 2), el toro $T^2$, una curva en $\mathbb{R}^3$.

### 0.5B.6.2 Enunciado de la hipótesis

Los datos de alta dimensión (imágenes, embeddings de texto) **no** llenan $\mathbb{R}^d$ de forma uniforme. Se concentran cerca de una variedad $\mathcal{M} \subset \mathbb{R}^d$ de dimensión intrínseca $k \ll d$.

Formalmente: existe $\mathcal{M}$ variedad $k$-dimensional y $\varepsilon > 0$ pequeño tal que:
$$
\forall x \in \mathcal{D}, \quad \text{dist}(\mathbf{e}(x), \mathcal{M}) := \inf_{\mathbf{m} \in \mathcal{M}} \|\mathbf{e}(x) - \mathbf{m}\| \leq \varepsilon
$$

### 0.5B.6.3 Motivación para un físico

En mecánica: el estado $(q, p)$ de un sistema con $n$ grados de libertad vive en $\mathbb{R}^{2n}$, pero las **leyes de conservación** (energía, momento, etc.) restringen la trayectoria a una **superficie** de dimensión menor. En ML, el "lenguaje natural" es un subconjunto diminuto del espacio de todas las secuencias: solo aquellas que son gramaticales, coherentes y significativas. Ese subconjunto tiene estructura geométrica de baja dimensión.

### 0.5B.6.4 Consecuencias

1. **Redundancia:** Los $d$ componentes del embedding están correlacionados. PCA encuentra $k$ direcciones que capturan la mayor varianza; el resto puede descartarse con poca pérdida.

2. **Interpolación:** Si $\mathbf{u}, \mathbf{v}$ son embeddings de textos similares y están en (o cerca de) $\mathcal{M}$, el punto $\mathbf{w} = \alpha \mathbf{u} + (1-\alpha)\mathbf{v}$ puede corresponder a un texto "intermedio" (p. ej. mezcla de conceptos). Esto justifica operaciones algebraicas en el espacio latente.

3. **Regularización:** Vectores aleatorios en $\mathbb{R}^d$ típicamente están **fuera** de $\mathcal{M}$; no corresponden a textos naturales. El entrenamiento "aprisiona" los embeddings en regiones estructuradas.

### 0.5B.6.5 Estimación de la dimensión intrínseca

La dimensión $k$ puede estimarse con:
- **PCA:** El número de autovalores "grandes" de la matriz de covarianza.
- **Correlation dimension:** $\lim_{r \to 0} \frac{\log C(r)}{\log r}$ donde $C(r)$ cuenta pares a distancia $< r$.
- **k-NN:** En un punto, el volumen de la bola que contiene $k$ vecinos escala como $r^k$; ajustar $k$ vs $r$ da una estimación.

```python
# Estimación burda por PCA: cuántos componentes explican p.ej. 95% de varianza
def intrinsic_dimension_pca(X: np.ndarray, variance_threshold: float = 0.95) -> int:
    """X shape (n_samples, d). Retorna k tal que k componentes explican >= variance_threshold."""
    X_centered = X - X.mean(axis=0)
    cov = X_centered.T @ X_centered / (X.shape[0] - 1)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    cumvar = np.cumsum(eigenvalues) / eigenvalues.sum()
    return np.searchsorted(cumvar, variance_threshold) + 1
```

Típicamente, para embeddings de texto en $d=768$ o $1536$, estimaciones de $k$ dan valores del orden de decenas a pocas centenas.

---

## 0.5B.7 Memoria semántica y RAG: el puente a la aplicación

### 0.5B.7.1 Memoria episódica vs semántica

| Tipo | Contenido | Ejemplo |
|------|-----------|---------|
| **Episódica** | Secuencia de eventos (qué pasó, cuándo) | "El usuario preguntó X, llamé a read_file, obtuve Y" |
| **Semántica** | Hechos abstractos desligados del episodio | "El usuario prefiere Python" |

La memoria semántica se implementa con embeddings: almacenar hechos como vectores; recuperar por similitud. Consulta "¿Qué prefiere el usuario?" → $\mathbf{e}(\text{consulta})$ → top-$k$ en base de hechos → inyectar en contexto.

### 0.5B.7.2 RAG: pipeline formal

**RAG (Retrieval-Augmented Generation)** es el bucle:

$$
\begin{aligned}
\mathbf{q} &= \mathbf{e}(\text{consulta}) \\
\mathcal{R} &= \text{Retrieve}(\mathbf{q}, \mathcal{D}, k) = \{d_{i_1}, \ldots, d_{i_k}\} \quad \text{por similitud coseno} \\
\text{contexto} &= \text{concat}(\text{consulta}, \mathcal{R}) \\
\text{respuesta} &= \text{LLM}(\text{contexto})
\end{aligned}
$$

La **augmentación** es inyectar $\mathcal{R}$ en el contexto. Sin embeddings, no hay forma eficiente de recuperar documentos "relevantes" para consultas parafraseadas (no hay coincidencia exacta de tokens).

### 0.5B.7.3 Pseudocódigo RAG (Python)

```python
def rag_retrieve_and_generate(
    query: str,
    documents: list[str],           # Base de documentos
    embeddings: np.ndarray,         # shape (N, d) - precalculados
    embed_fn,                       # query -> vector
    llm_fn,                         # (context) -> respuesta
    k: int = 5
) -> str:
    q_vec = embed_fn(query)  # Embed de la consulta
    scores = np.dot(embeddings, q_vec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_vec)
    )
    top_idx = np.argsort(scores)[::-1][:k]
    retrieved = [documents[i] for i in top_idx]
    context = f"Consulta: {query}\n\nDocumentos relevantes:\n" + "\n\n".join(retrieved)
    return llm_fn(context)
```

### 0.5B.7.4 Agentes y RAG

Agentes con acceso a bases grandes usan RAG como **herramienta**: la búsqueda no es por keywords sino por similitud de embeddings. El agente invoca "buscar en la wiki" con la consulta; el módulo RAG embebe, recupera y devuelve fragmentos; el LLM los integra en su razonamiento y genera la acción o respuesta final.

---

## 0.5B.8 Implementaciones prácticas (nota)

Para usar embeddings en un proyecto real:

- **OpenAI API:** `openai.embeddings.create(model="text-embedding-3-small", input=texto)` retorna vectores en $\mathbb{R}^{1536}$ (configurable).
- **sentence-transformers** (Python): modelos locales como `all-MiniLM-L6-v2` ($d=384$), `paraphrase-multilingual` para español.
- **Hugging Face:** `model.encode(textos)` con modelos como `sentence-transformers/all-MiniLM-L6-v2`.

Todos normalizan opcionalmente (L2-norm=1) para que producto escalar = similitud coseno.

---

## 0.5B.9 Resumen geométrico

| Concepto | Formalización |
|----------|---------------|
| Embedding | $\mathbf{e}: \mathcal{X} \to \mathbb{R}^d$ |
| Cauchy-Schwarz | $|\mathbf{u} \cdot \mathbf{v}| \leq \|\mathbf{u}\| \|\mathbf{v}\|$ |
| Similitud coseno | $\text{sim}_{\text{cos}} = \frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|} = \cos\theta \in [-1,1]$ |
| Dist. euclidiana (unitarios) | $\|\mathbf{u}-\mathbf{v}\| = \sqrt{2(1-\text{sim}_{\text{cos}})}$ |
| Volumen en la cáscara | $1 - (1-\varepsilon/R)^d \to 1$ cuando $d \to \infty$ |
| Dist. vectores aleatorios (unit.) | $\mathbb{E}[\|\mathbf{u}-\mathbf{v}\|^2] = 2$ |
| Hipótesis del colector | $\text{dist}(\mathbf{e}(x), \mathcal{M}) \leq \varepsilon$, $\dim \mathcal{M} = k \ll d$ |
| RAG | $\mathbf{q} = \mathbf{e}(q)$, $\mathcal{R} = \text{top-k}(\mathbf{q}, \mathcal{D})$, $\text{LLM}(\text{concat}(q, \mathcal{R}))$ |

---

## 0.5B.10 Referencias cruzadas

- §0.1–0.2: Los tokens y la predicción autoregresiva producen internamente las representaciones que se usan como embeddings.
- §0.5.4–0.5.5: Rate-distortion y compresión conectan con la dimensión intrínseca del colector.
- Cap. 5 (memoria): La memoria episódica actual es buffer FIFO; la extensión semántica usaría embeddings y búsqueda por similitud.
- Anexo A.3: Nota breve sobre embeddings; esta sección desarrolla el formalismo completo.

---

# 0.5C Mecánica estadística de la atención (Transformers)

*Los LLMs modernos son **transformers**: arquitecturas cuyo núcleo es el **mecanismo de atención**. Para un físico, la atención no es solo una operación algebraica: es un sistema de pesos dinámicos donde la "relevancia" de un token sobre otro se expresa como una **distribución de Boltzmann** sobre "energías de interacción". Esta sección desarrolla el formalismo del Scaled Dot-Product Attention y su interpretación en términos de mecánica estadística.*

---

## 0.5C.0 Diferenciación: dos usos del softmax en el transformer

El softmax aparece en **dos lugares** distintos del transformer; conviene no confundirlos:

| Contexto | Dominio | Variable | Interpretación |
|----------|---------|----------|----------------|
| **Atención** (§0.5C) | Posiciones $\{1, \ldots, L\}$ | $A_{ij} = e^{\tilde{S}_{ij}}/Z_i$ | Peso de la posición $j$ sobre la posición $i$ |
| **Salida / logits** (§0.2.3) | Vocabulario $\mathcal{V}$ | $p(y) = e^{z_y}/Z$ | Probabilidad del siguiente token |

En ambos casos $Z$ es una función de partición; la estructura matemática es idéntica (Boltzmann). La diferencia es **qué se normaliza**: en la atención, una distribución sobre **posiciones**; en la salida, una distribución sobre **tokens**.

---

## 0.5C.1 El problema que resuelve la atención

En un modelo autoregresivo, para predecir el token en la posición $t$ el modelo debe considerar **todos** los tokens anteriores $x_1, \ldots, x_{t-1}$. Las redes recurrentes (RNN, LSTM) procesan secuencialmente, lo que limita el paralelismo. La **atención** permite que cada posición "consulte" directamente a las demás, con pesos que se calculan dinámicamente según la **relevancia** entre pares de posiciones.

**Trade-off:** RNN: $O(1)$ por paso pero $O(L)$ pasos secuenciales (no paralelizable). Atención: $O(L^2)$ por capa pero **totalmente paralelizable** en las $L$ posiciones.

**Idea clave:** En lugar de un estado recurrente que comprime el pasado, cada token "pregunta" a los demás cuánto debe influir en su representación. Los pesos de atención son esas influencias.

---

## 0.5C.2 Query, Key, Value (Q, K, V)

Cada posición $i$ tiene un vector de estado $\mathbf{h}_i \in \mathbb{R}^d$. Se proyectan en tres espacios mediante matrices lineales (aprendibles):

$$
\mathbf{q}_i = \mathbf{h}_i W_Q \in \mathbb{R}^{d_k}, \quad
\mathbf{k}_i = \mathbf{h}_i W_K \in \mathbb{R}^{d_k}, \quad
\mathbf{v}_i = \mathbf{h}_i W_V \in \mathbb{R}^{d_v}
$$

- **Query** $\mathbf{q}_i$: "¿Qué busco?" — La posición $i$ como **consultante**.
- **Key** $\mathbf{k}_j$: "¿Qué ofrezco?" — La posición $j$ como **consultada**.
- **Value** $\mathbf{v}_j$: "¿Qué entrego?" — El contenido que $j$ aporta a quien la atiende.

En forma matricial, para una secuencia de longitud $L$:
$$
Q = H W_Q \in \mathbb{R}^{L \times d_k}, \quad
K = H W_K \in \mathbb{R}^{L \times d_k}, \quad
V = H W_V \in \mathbb{R}^{L \times d_v}
$$
con $H = [\mathbf{h}_1, \ldots, \mathbf{h}_L]^T$.

**Flujo desde los embeddings (§0.5B):** Los $\mathbf{h}_i$ provienen de la capa de embedding (o de la capa anterior del transformer). En la primera capa, $H$ es la suma del embedding de tokens más el **posicional encoding** (véase §0.5C.2b).

### 0.5C.2b Posicional encoding (breve)

El transformer no tiene noción inherente de **orden**: sin información de posición, "el gato persiguió al ratón" y "el ratón persiguió al gato" serían indistinguibles. Se añade un **posicional encoding** $\mathbf{p}_i \in \mathbb{R}^d$ a cada posición:
$$
\tilde{\mathbf{h}}_i = \mathbf{h}_i + \mathbf{p}_i
$$

Puede ser **sinusoidal** (fijo): $p_{i,2k} = \sin(i / 10000^{2k/d})$, $p_{i,2k+1} = \cos(i / 10000^{2k/d})$, o **aprendible** (embedding de posiciones). Los $\tilde{\mathbf{h}}_i$ son los que se proyectan en $Q, K, V$. Sin este paso, la arquitectura sería **permutación-invariante**.

---

## 0.5C.3 Scaled Dot-Product Attention: la fórmula

La **relevancia** (score) entre la posición $i$ (query) y la posición $j$ (key) se mide por el **producto escalar** $\mathbf{q}_i \cdot \mathbf{k}_j$. Cuanto mayor el producto escalar, más "alineados" están query y key, más relevante es $j$ para $i$.

**Por qué producto escalar:** En §0.5B vimos que el producto escalar (o similitud coseno, si se normaliza) mide alineación entre vectores. Aquí, $\mathbf{q}_i$ codifica "qué busca" la posición $i$, y $\mathbf{k}_j$ codifica "qué ofrece" la posición $j$. Su producto escalar mide **compatibilidad**: si lo que $i$ busca coincide con lo que $j$ ofrece, el score es alto.

Se forma la **matriz de scores** (logits):
$$
S = Q K^T \in \mathbb{R}^{L \times L}, \quad S_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j
$$

Para evitar que los scores crezcan con $\sqrt{d_k}$ (y saturen el softmax), se **escala** por $\sqrt{d_k}$:
$$
\tilde{S} = \frac{Q K^T}{\sqrt{d_k}}
$$

La **matriz de atención** se obtiene aplicando softmax por **filas** (cada fila $i$ es una distribución sobre las $L$ posiciones):
$$
A = \mathrm{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right), \quad A_{ij} = \frac{\exp(\tilde{S}_{ij})}{\sum_{j'=1}^{L} \exp(\tilde{S}_{ij'})}
$$

Cada fila $A_{i,:}$ es una distribución de probabilidad: $\sum_j A_{ij} = 1$. El **output** de la atención es la combinación convexa de los values según esos pesos:
$$
\text{Attention}(Q, K, V) = A \, V \in \mathbb{R}^{L \times d_v}
$$

La fila $i$ del output es $\sum_j A_{ij} \mathbf{v}_j$: una **mezcla ponderada** de todos los values, donde los pesos $A_{ij}$ indican cuánto "atender" a cada posición $j$.

### 0.5C.3b Enmascaramiento causal (autoregresivo)

En modelos **decoder-only** (GPT, LLaMA), la generación es **autoregresiva**: la posición $i$ solo puede ver posiciones $j \leq i$. Se aplica un **enmascaramiento causal**: antes del softmax, se pone $S_{ij} = -\infty$ para $j > i$. Así, después del softmax, $A_{ij} = 0$ para $j > i$. La matriz $A$ es **triangular inferior**.

$$
\tilde{S}_{ij}^{\text{causal}} = \begin{cases} \tilde{S}_{ij} & \text{si } j \leq i \\ -\infty & \text{si } j > i \end{cases}, \qquad
A = \mathrm{softmax}(\tilde{S}^{\text{causal}})
$$

Sin esto, el modelo "vería" tokens futuros durante el entrenamiento, violando la causalidad. Los agentes que usamos (GPT, etc.) son decoder-only con máscara causal.

**Encoder vs Decoder:** BERT usa atención **bidireccional** (sin máscara causal). T5 usa encoder (bidireccional) + decoder (causal) con **cross-attention** entre ambos. No desarrollamos eso aquí.

---

## 0.5C.4 Analogía con mecánica estadística

### 0.5C.4.1 Los scores como energías de interacción

En mecánica estadística, la probabilidad de que un sistema esté en un estado de energía $E$ a temperatura $T$ es $p \propto e^{-E/(k_B T)}$ (distribución de **Boltzmann**). Invirtiendo el signo: si definimos la "energía de interacción" entre $i$ y $j$ como:
$$
E_{ij} := -\frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}} = -\tilde{S}_{ij}
$$

entonces "relevancia alta" (producto escalar grande) corresponde a $E_{ij}$ **más negativo** (estado más favorable). La matriz de atención es:
$$
A_{ij} = \frac{\exp(-E_{ij}/T)}{Z_i} = \frac{e^{-E_{ij}/T}}{\sum_{j'} e^{-E_{ij'}/T}}
$$

con **temperatura efectiva** $T = 1$ (en unidades $k_B = 1$). En la práctica estándar no se introduce $\tau$ en la atención; pero si se usara $A = \mathrm{softmax}(\tilde{S}/\tau)$, entonces $T = \tau$ y $\tau \to 0$ concentraría la atención (explotación), $\tau \to \infty$ la dispersaría (análogo a §0.10).

**Comparación con §0.2.3:** Allí el softmax actúa sobre logits $z_y$ del vocabulario; aquí sobre scores $\tilde{S}_{ij}$ de posiciones. La forma $e^x/Z$ es idéntica; solo cambia el dominio (tokens vs posiciones).

### 0.5C.4.2 Interpretación física

- **Posiciones** $i, j$ = "sitios" o "partículas".
- **Energía** $E_{ij}$ = cuán desfavorable es atender a $j$ desde $i$.
- **Peso** $A_{ij}$ = probabilidad de "seleccionar" $j$ en un ensemble canónico.
- **Entropía de la fila $i$:** $H(A_{i,:}) = -\sum_j A_{ij} \log A_{ij}$. Entropía baja = atención concentrada en pocas posiciones; entropía alta = atención difusa. Conecta con §0.3.

### 0.5C.4.3 Función de partición y libre energía

$Z_i = \sum_{j} e^{-E_{ij}/T}$ es la función de partición. La **energía libre** de Helmholtz es $F_i = -T \log Z_i$ (con $T=1$: $F_i = -\log Z_i$). Minimizar $F_i$ equivale a maximizar $Z_i$. El entrenamiento ajusta $W_Q, W_K, W_V$ para que las energías reflejen relevancia semántica real.

### 0.5C.4.4 Visualización geométrica y recuperación de información tipo Hopfield/Boltzmann

La atención puede entenderse como un **sistema de recuperación asociativa** basado en energía, análogo a redes de Hopfield o memoria de Boltzmann.

#### Interpretación geométrica

- **Espacio de consultas/keys:** $\mathbb{R}^{d_k}$. Cada $\mathbf{q}_i$ y $\mathbf{k}_j$ es un punto en ese espacio.
- **Query** $\mathbf{q}_i$: vectores que representan "qué busca" la posición $i$.
- **Keys** $\mathbf{k}_j$: vectores que representan "patrones almacenados" en las posiciones $j$.
- **Values** $\mathbf{v}_j$: el "contenido" asociado a cada patrón $\mathbf{k}_j$.

El producto escalar $\mathbf{q}_i \cdot \mathbf{k}_j$ mide el **ángulo** entre query y key (si se normaliza, coincide con la similitud coseno). Keys cercanos a la query en orientación dan score alto; keys ortogonales dan score ≈ 0.

#### Diagrama del flujo de recuperación

```
                    ┌─────────────────────────────────────────────────────────┐
                    │           SISTEMA DE RECUPERACIÓN ASOCIATIVA             │
                    │                                                         │
  Posición i        │   Query q_i ──┐                                         │
  (consultante)     │               │  E_ij = -q_i·k_j/√d_k  (energía)       │
                    │   Keys K ─────┼─────────────────────────────────────┐  │
                    │               │  A_ij = exp(-E_ij/T) / Z_i          │  │
                    │               │       (peso Boltzmann)              │  │
                    │               ▼                                      │  │
                    │   Output_i = Σ_j A_ij · v_j   (recuperación ponderada) │  │
                    │               │                                      │  │
                    │   Values V ────┴───────────────────────────────────────┘  │
                    └─────────────────────────────────────────────────────────┘
```

#### Conexión con Hopfield

En una **red de Hopfield** discreta, la energía del sistema es $E = -\frac{1}{2} \sum_{i,j} w_{ij} s_i s_j$ con $s_i \in \{-1,+1\}$. Los estados de menor energía son **atractores**: patrones almacenados. Una consulta (estado inicial) fluye hacia el atractor más cercano en el paisaje de energía.

En atención: la "energía" $E_{ij} = -\tilde{S}_{ij}$ entre query $i$ y key $j$ define un paisaje. La posición $i$ **no** evoluciona iterativamente hacia un atractor; en su lugar, la distribución Boltzmann $A_{i,:}$ asigna pesos a todas las posiciones según sus energías. La **recuperación** es la suma ponderada $\sum_j A_{ij} \mathbf{v}_j$: se obtiene una mezcla de valores, con más peso en los keys de menor energía (más alineados con la query). Es un "Hopfield suave" (soft): en vez de elegir un único patrón, se hace un promedio ponderado por Boltzmann.

#### Formulación como memoria asociativa

| Concepto Hopfield/Boltzmann | Atención (Q, K, V) |
|-----------------------------|---------------------|
| Patrones almacenados | Keys $\{\mathbf{k}_1, \ldots, \mathbf{k}_L\}$ |
| Consulta | Query $\mathbf{q}_i$ |
| Energía de interacción | $E_{ij} = -\mathbf{q}_i \cdot \mathbf{k}_j / \sqrt{d_k}$ |
| Probabilidad de "activar" patrón $j$ | $A_{ij} = e^{-E_{ij}/T} / Z_i$ |
| Contenido recuperado | $\sum_j A_{ij} \mathbf{v}_j$ |

La diferencia clave: en Hopfield clásico hay **dinámica** (la red converge iterativamente); en atención la "recuperación" es **inmediata** vía softmax. La estructura energía–Boltzmann–recuperación es la misma.

---

## 0.5C.5 Factor de escala $\sqrt{d_k}$: ¿por qué?

### 0.5C.5.1 Derivación de la varianza

Sean $q_{ik}$, $k_{jk}$ las entradas de $\mathbf{q}_i$ y $\mathbf{k}_j$, con $\mathbb{E}[q_{ik}] = \mathbb{E}[k_{jk}] = 0$ y $\mathrm{Var}(q_{ik}) = \mathrm{Var}(k_{jk}) = \sigma^2$ (i.i.d.). El score sin escalar es $S_{ij} = \sum_{k=1}^{d_k} q_{ik} k_{jk}$. Por independencia:
$$
\mathrm{Var}(S_{ij}) = \sum_{k=1}^{d_k} \mathrm{Var}(q_{ik} k_{jk}) = d_k \cdot \mathrm{Var}(q) \mathrm{Var}(k) = d_k \sigma^4
$$

(Para variables centradas, $\mathrm{Var}(XY) = \mathrm{Var}(X)\mathrm{Var}(Y)$.) Por tanto $\mathrm{Var}(\tilde{S}_{ij}) = \mathrm{Var}(S_{ij}/\sqrt{d_k}) = \sigma^4$, independiente de $d_k$.

### 0.5C.5.2 Saturación del softmax y gradientes

Si $S_{ij}$ tiene varianza $O(d_k)$ y $d_k \approx 768$, los scores serían típicamente $\sim \pm 30$ o más. El softmax $e^{30}/(e^{30} + e^{29} + \cdots)$ satura: una entrada domina, las demás $\approx 0$. El **gradiente** del softmax $\frac{\partial A_{ij}}{\partial S_{ij}}$ tiene la forma $A_{ij}(\delta_{jj'} - A_{ij'})$; cuando $A_{ij} \approx 0$ o $\approx 1$, ese gradiente se anula. **Vanishing gradient**: el backpropagation deja de propagar señal útil. Escalar por $\sqrt{d_k}$ mantiene los scores en un rango ($\sim \pm 2\sigma^2$) donde el softmax es sensible.

---

## 0.5C.6 Complejidad computacional

- **$Q K^T$:** matriz $L \times d_k$ por matriz $d_k \times L$ → $O(L^2 d_k)$.
- **Softmax:** $O(L^2)$.
- **$A V$:** matriz $L \times L$ por $L \times d_v$ → $O(L^2 d_v)$.

El cuello de botella es $O(L^2)$ en la longitud de la secuencia. Con $d_k = d_v = d$, el coste total por capa de atención es $O(L^2 d)$. Para un transformer con múltiples capas y $d \sim 10^3$–$10^4$, el término dominante es $O(L^2)$ (o $O(L \cdot d^2)$ si $d^2 \gg L$ en el resto del modelo). Esto justifica la mención en §0.14 de "atención cuadrática".

---

## 0.5C.7 Implementación en Python

```python
import numpy as np

def scaled_dot_product_attention(
    Q: np.ndarray,  # (L, d_k)
    K: np.ndarray,  # (L, d_k)
    V: np.ndarray,  # (L, d_v)
    d_k: int,
    causal: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scaled Dot-Product Attention.
    causal=True: máscara para autoregresión (GPT-style).
    """
    scores = Q @ K.T / np.sqrt(d_k)  # (L, L)
    if causal:
        mask = np.triu(np.ones((scores.shape[0], scores.shape[1])), k=1) * -1e9
        scores = scores + mask  # -inf para j > i
    A = np.exp(scores - scores.max(axis=1, keepdims=True))
    A = A / A.sum(axis=1, keepdims=True)
    output = A @ V
    return output, A

# Ejemplo numérico pequeño: L=3, d_k=d_v=2
Q = np.array([[1., 0.], [0.5, 0.5], [0., 1.]])   # queries
K = np.array([[1., 0.], [0.7, 0.3], [0., 1.]])   # keys
V = np.array([[1., 0.], [0., 1.], [1., 1.]])      # values
d_k = 2
scores = Q @ K.T / np.sqrt(d_k)
# scores[0] con K: q0·k0=1, q0·k1=0.7, q0·k2=0 => atención en pos 0
A = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
print("Scores (escalados):\n", scores)
print("Matriz A (filas=distribuciones):\n", np.round(A, 3))
print("Output[0] = A[0]@V =", A[0] @ V)
```

**Nota:** En la práctica se usa estabilidad numérica: restar el máximo por fila antes del exp (`scores - scores.max(axis=1)`).

---

## 0.5C.8 Múltiples cabezas (Multi-Head Attention)

En la práctica, el transformer usa **varias cabezas** de atención en paralelo ($h$ cabezas), cada una con $d_k = d_v = d/h$:
$$
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V), \quad
\text{MultiHead} = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, W^O
$$
Cada cabeza proyecta a un subespacio distinto; distintas cabezas pueden especializarse en sintaxis, co-referencia, relaciones a largo plazo, etc. La complejidad sigue siendo $O(L^2 d)$ (cada cabeza hace $O(L^2 \cdot d/h)$, hay $h$ cabezas).

---

## 0.5C.8b Estructura completa del bloque (resumen)

Un **bloque** típico del transformer decoder es:
1. **Multi-Head Self-Attention** con máscara causal
2. **Residual + LayerNorm:** $\mathbf{x} \leftarrow \mathrm{LayerNorm}(\mathbf{x} + \mathrm{Attention}(\mathbf{x}))$
3. **FFN (Feed-Forward):** $\mathbf{x} \leftarrow \mathrm{LayerNorm}(\mathbf{x} + \mathrm{FFN}(\mathbf{x}))$

La conexión residual $\mathbf{x} + f(\mathbf{x})$ permite flujo de gradientes sin degradación. El modelo es una pila de $N$ bloques (p. ej. $N=12$–96).

---

## 0.5C.9 Resumen y tabla

| Concepto | Fórmula |
|----------|---------|
| Q, K, V | $Q = HW_Q$, $K = HW_K$, $V = HW_V$ |
| Scores | $\tilde{S} = QK^T / \sqrt{d_k}$ |
| Causal mask | $\tilde{S}_{ij} = -\infty$ si $j > i$ (decoder-only) |
| Energía (analogía) | $E_{ij} = -\tilde{S}_{ij}$ |
| Recuperación Hopfield/Boltzmann | Query→Keys (energía)→Values; §0.5C.4.4 |
| Atención | $A = \mathrm{softmax}(\tilde{S})$, $A_{ij} = e^{-E_{ij}/T}/Z_i$, $T=1$ |
| Output | $\text{Attention}(Q,K,V) = A V$ |
| Var sin escalar | $\mathrm{Var}(S_{ij}) = d_k \sigma^4$ |
| Coste | $O(L^2 d)$ |

---

## 0.5C.10 Referencias cruzadas

- §0.2.3: Softmax en logits de salida (dominio = vocabulario); aquí dominio = posiciones. Misma estructura Boltzmann.
- §0.3: Entropía $H(A_{i,:})$ de la distribución de atención.
- §0.5B: Producto escalar como medida de compatibilidad; embeddings como entrada $H$ al transformer.
- §0.5C.4.4: Visualización geométrica; Q,K,V como recuperación Hopfield/Boltzmann.
- §0.10: Temperatura en sampling; análogo a $\tau$ en softmax de atención.
- §0.14: Coste $O(L^2 d + L d^2)$; el término $L^2$ viene de la atención.

---

## 0.6 Cadenas de Markov

### 0.6.1 Definición

Una secuencia de variables aleatorias $(X_t)_{t \geq 0}$ es una **cadena de Markov** (de primer orden) si satisface la **propiedad de Markov**:

$$
P(X_t \mid X_0, \ldots, X_{t-1}) = P(X_t \mid X_{t-1})
$$

Es decir, el futuro depende del pasado solo a través del presente. El estado actual $X_{t-1}$ es un **resumen suficiente** del historial para predecir $X_t$.

### 0.6.2 Matriz de transición

Para espacios de estados finitos $\mathcal{X} = \{1, \ldots, N\}$, la dinámica se describe por una **matriz de transición** $P$ con entradas:

$$
P_{ij} = P(X_t = j \mid X_{t-1} = i)
$$

Cumple $\sum_j P_{ij} = 1$ (estocástica por filas). La distribución en el instante $t$ es $\boldsymbol{\pi}_t = \boldsymbol{\pi}_0 P^t$.

### 0.6.3 Conexión con el modelo de lenguaje

El modelo de lenguaje **no** es estrictamente una cadena de Markov de primer orden: $P(x_t \mid x_{1:t-1})$ depende de todo el contexto dentro de la ventana. Pero la **factorización** (regla de la cadena) tiene la misma estructura: en cada paso se predice el siguiente símbolo condicionado al pasado. La diferencia es que el "estado" en un LLM es de alta dimensión (todo el contexto) en lugar de un único valor.

### 0.6.4 Cadenas de orden superior

Una cadena de Markov de orden $k$ satisface $P(X_t \mid X_{0:t-1}) = P(X_t \mid X_{t-k:t-1})$. Cualquier cadena de orden $k$ puede reescribirse como cadena de orden 1 definiendo el estado como $\tilde{X}_t = (X_{t-k+1}, \ldots, X_t)$.

---

## 0.7 Procesos de decisión de Markov (MDP)

### 0.7.1 Definición formal

Un **MDP** es una tupla $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ donde:

- $\mathcal{S}$: espacio de estados
- $\mathcal{A}$: espacio de acciones
- $\mathcal{P}(s' \mid s, a)$: probabilidad de transición al estado $s'$ dado estado $s$ y acción $a$
- $\mathcal{R}(s, a, s')$: recompensa (o $r(s,a)$ si solo depende de estado y acción)
- $\gamma \in [0, 1)$: factor de descuento

La **política** $\pi(a \mid s)$ es una distribución sobre acciones dado el estado. En el caso determinista, $\pi: \mathcal{S} \to \mathcal{A}$.

### 0.7.2 Retorno y valor

El **retorno** descontado desde el instante $t$ es:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
$$

El **valor de un estado** bajo la política $\pi$ es $V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$. El **valor de una acción** es $Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t=s, A_t=a]$.

### 0.7.3 Ecuación de Bellman

La **ecuación de Bellman** para $V^\pi$ es:

$$
V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} \mathcal{P}(s'|s,a) \left[ \mathcal{R}(s,a,s') + \gamma V^\pi(s') \right]
$$

Para $Q^\pi$:

$$
Q^\pi(s,a) = \sum_{s'} \mathcal{P}(s'|s,a) \left[ \mathcal{R}(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a') \right]
$$

Una **política óptima** $\pi^*$ maximiza $V^\pi$ en todos los estados. Se cumple $V^*(s) = \max_a Q^*(s,a)$ y la política greedy $\pi^*(s) = \arg\max_a Q^*(s,a)$ es óptima.

### 0.7.4 Relación con el agente

El agente con herramientas puede verse como un MDP donde:
- Estados: historial de mensajes + memoria
- Acciones: emitir texto o llamar una herramienta
- Transiciones: deterministas dadas la acción (la API devuelve una respuesta; la herramienta devuelve un resultado)
- Recompensa: implícita (no la definimos explícitamente; el LLM fue entrenado con RLHF u otro objetivo)

El LLM implementa una **política** $\pi(a|s)$ aproximada. No resolvemos Bellman; la política viene del entrenamiento previo.

---

## 0.8 Espacio de estados del agente

### 0.8.1 Definición del estado

El **estado** del agente en el instante $t$ es la tupla:

$$
s_t = (M_t, H_t, C_t)
$$

- $M_t$: historial de mensajes hasta $t$ (lista de pares rol/contenido)
- $H_t$: buffer de episodios de memoria (acciones y resultados recientes)
- $C_t$: contexto inyectado (p. ej. resumen de memoria)

El espacio $\mathcal{S}$ es discreto pero de cardinalidad astronómicamente grande (combinatoria de secuencias de texto).

### 0.8.2 Función de transición

La transición $s_t \to s_{t+1}$ depende de:
1. Acción del usuario $a_{\text{user}}$ (nuevo mensaje)
2. Respuesta del LLM $a_{\text{llm}}$ (texto o llamada a herramienta)
3. Observación $o_{\text{tool}}$ (resultado de la herramienta si aplica)

$$
s_{t+1} = T(s_t, a_{\text{user}}, a_{\text{llm}}, o_{\text{tool}})
$$

La función $T$ es determinista: concatenar mensajes, actualizar memoria, etc.

### 0.8.3 Espacio de acciones

$$
\mathcal{A} = \mathcal{A}_{\text{text}} \cup \mathcal{A}_{\text{tool}}
$$

- $\mathcal{A}_{\text{text}}$: cadenas de tokens (respuestas en lenguaje natural)
- $\mathcal{A}_{\text{tool}}$: pares $(f, \mathbf{args})$ con $f \in \mathcal{F}$ (herramienta) y $\mathbf{args}$ en el dominio de $f$

---

# 0.5D Teoría de control y estabilidad del bucle del agente

*Un agente no es solo un modelo de lenguaje: es un **sistema de retroalimentación** en lazo cerrado. El bucle ReAct (pensar → actuar → observar) puede analizarse con herramientas de teoría de control: condiciones de convergencia, oscilaciones, estados de error como puntos de inestabilidad. Esta sección formaliza el bucle como un sistema dinámico de control en **tiempo discreto** y estudia cuándo termina (o falla).*

**Notación (coherencia con §0.8):** Usamos $t$ para el **turno** de usuario (cada mensaje del usuario inicia un turno). Dentro de un turno, $k = 0, 1, 2, \ldots$ denota la **iteración** del bucle tool-calling. El estado $s_{t,k}$ o simplemente $s_k$ cuando el turno $t$ está implícito es el estado en la iteración $k$ del bucle de ese turno. La transición $T$ de §0.8 opera a nivel de turno; aquí $T_{\text{loop}}$ modela la evolución **interna** del bucle dentro de un turno.

---

## 0.5D.1 El bucle como sistema de retroalimentación

### 0.5D.1.1 Diagrama de bloques

El bucle tool-calling tiene la estructura de un **sistema de control en lazo cerrado** (tiempo discreto). La analogía con control clásico es **relajada**: no hay bloques físicos que implementen un comparador o un cálculo $e = r - y$; el "error" y el "comparador" son **analógicos**—la política $\pi$ ha sido entrenada para actuar como si hubiera tal comparación implícita.

```
                    ┌─────────────────────────────────────────────────────┐
                    │  LAZO DE CONTROL (analogía conceptual; ver texto)   │
                    │                                                     │
  r (referencia)    │     e (implícito)    u (acción)     y (salida)      │
  (mensaje usuario) ──▶  [Comparador] ──▶ [Controlador] ──▶ [Planta] ──────┼──▶ respuesta
        │                ↑      │              │                │         │      final
        │                │      │              │                │         │
        │                │      └──────────────┴────────────────┘         │
        │                │                    │                           │
        │                │              [Retroalimentación]                │
        │                │         (historial + resultado herramienta)     │
        │                │                    │                           │
        └────────────────┴────────────────────┘                           │
                         comparar estado actual con "objetivo cumplido"    │
```

*El "Comparador" no es un bloque separado; la decisión de continuar o parar está codificada en $\pi$.*

- **Referencia** $r$: objetivo implícito (responder al usuario; contenido del mensaje del usuario).
- **Controlador**: el LLM con política $\pi(a|s)$ que decide la acción (texto o herramienta).
- **Planta**: la ejecución de la herramienta (si aplica) o la emisión de texto.
- **Retroalimentación**: el historial actualizado que vuelve al controlador como nuevo estado $s_{k+1}$.

El "error" $e$ es implícito: el LLM no computa $e = r - y$; la política ha sido entrenada para emitir `message` cuando estima que el objetivo está satisfecho. La decisión de emitir `message` en lugar de `function_call` es el **cierre del lazo**.

### 0.5D.1.2 Conexión con el patrón ReAct (§4)

El ciclo **ReAct** (Reasoning + Acting) coincide con el lazo de control:

| Fase ReAct | Correspondencia en el bucle |
|------------|----------------------------|
| **Pensar (Reason)** | El LLM procesa el estado $s_k$ internamente (no observable). |
| **Actuar (Act)** | Emite $a_k$: `function_call` o `message`. |
| **Observar (Observe)** | Recibe $\text{output}(a_k)$: resultado de la herramienta o confirmación del mensaje. |

La retroalimentación es la **observación** que actualiza el estado. El bucle termina cuando la acción es `message` (respuesta final al usuario).

### 0.5D.1.3 Formalización del bucle

Sea $k = 0, 1, 2, \ldots$ el índice de iteración dentro de un turno. El estado $s_k = (M_k, H_k, C_k)$ (véase §0.8.1) evoluciona según:
$$
s_{k+1} = T_{\text{loop}}(s_k, a_k) = s_k \oplus \text{output}(a_k)
$$
donde $a_k \sim \pi(\cdot \mid s_k)$ es la acción del LLM y $\text{output}(a_k)$ es el resultado (respuesta de la herramienta o el mensaje mismo). La notación $\oplus$ denota concatenación al historial $M_k$. Esta $T_{\text{loop}}$ es la **restricción** de la función de transición $T$ de §0.8 al caso en que $a_{\text{user}}$ está fijo (mensaje del usuario) y solo varía $a_{\text{llm}} = a_k$, $o_{\text{tool}} = \text{output}(a_k)$ si aplica.

**Variable de control discreta:** Definimos $\sigma_k$ como el tipo de acción:
$$
\sigma_k = \begin{cases} \text{tool} & \text{si } a_k \in \mathcal{A}_{\text{tool}} \\ \text{text} & \text{si } a_k \in \mathcal{A}_{\text{text}} \end{cases}
$$
El **objetivo de control** es alcanzar $\sigma_k = \text{text}$ en algún $k$ finito. Si $\sigma_0 = \text{text}$ (el modelo responde sin herramientas), el bucle termina en la iteración 1 ($K = 0$ como índice de parada).

---

## 0.5D.2 Convergencia y condición de parada

### 0.5D.2.1 Definición de convergencia

Indexando desde $k = 0$, el bucle **converge** (en un turno dado) si existe $K \ge 0$ finito tal que $\sigma_K = \text{text}$. Entonces $K$ es el **índice de la iteración en que se emite el mensaje final**. El número total de iteraciones hasta la parada es $K + 1$; el número de llamadas a herramientas es $K$ (si $K = 0$, ninguna).

**Definición formal:** Para un turno con entrada (mensaje del usuario) $r$:
$$
\exists K \in \mathbb{N} : \sigma_K = \text{text} \quad \wedge \quad \forall k < K : \sigma_k = \text{tool}
$$

### 0.5D.2.2 Convergencia no garantizada

A diferencia de un controlador clásico con ley de Lyapunov o garantías de estabilidad, el LLM **no** tiene garantía de convergencia. La política $\pi$ puede:
1. **Converger** en $K$ pasos (caso deseado).
2. **Oscilar**: alternar indefinidamente entre tool-calls que no llevan al objetivo (ej: leer el mismo archivo repetidamente).
3. **Divergir** o entrar en un bucle de herramientas que no satisface al usuario.

La convergencia depende de la **calidad de la política** (entrenamiento) y del **estado actual** (si el objetivo es ambiguo o irrealizable, el agente puede "confundirse").

### 0.5D.2.3 Parada forzada: el límite $K_{\max}$

Para evitar **bucles infinitos**, se impone un límite $K_{\max}$ (`max_tool_calls_per_turn`). Si tras $K_{\max}$ iteraciones con $\sigma_k = \text{tool}$ aún no se ha emitido `message`, se **fuerza** la parada: se inyecta un mensaje del tipo "Responde con lo que tengas" y se espera un `message` en el siguiente paso (estado FORZADO; véase §0.5D.5). Así, el bucle **siempre** termina en un número finito de iteraciones.

Formalmente, el bucle con parada forzada satisface:
$$
\forall \text{ turno }, \quad \exists K \leq K_{\max} + 1 : \sigma_K = \text{text}
$$
(El $+1$ incluye la iteración forzada en que se pide explícitamente la respuesta.)

El sistema es **estable en el sentido de terminación**: no puede ejecutarse indefinidamente.

---

## 0.5D.3 Oscilaciones y bucles patológicos

### 0.5D.3.1 Tipos de oscilación

Una **oscilación** ocurre cuando el estado $s_k$ (o una proyección de él) repite un ciclo sin acercarse a un estado terminal. Ejemplos:

| Patrón | Descripción | Ejemplo |
|--------|-------------|---------|
| **Bucle de herramientas redundantes** | El modelo llama las mismas herramientas con los mismos (o similares) argumentos una y otra vez | `read_file("a.txt")` → resultado → `read_file("a.txt")` → ... |
| **Alternancia sin progreso** | Alternar entre dos herramientas sin avanzar hacia el objetivo | `list_dir` → `read_file` → `list_dir` → ... |
| **Dependencia circular** | El resultado de la herramienta $A$ induce al modelo a llamar $B$, cuyo resultado induce a llamar $A$ de nuevo | Feedback mal formado en el diseño de herramientas |

### 0.5D.3.2 Análisis de estabilidad: función tipo Lyapunov y condiciones formales

En teoría de control, un punto de equilibrio $\bar{x}$ es **estable en el sentido de Lyapunov** si existe una función $V(x) \geq 0$ (positiva definida en un entorno del equilibrio, con $V(\bar{x}) = 0$) tal que $\Delta V = V(x_{k+1}) - V(x_k) \leq 0$ a lo largo de las trayectorias del sistema. Si además $\Delta V < 0$ fuera del equilibrio, la convergencia es asintótica.

#### Función de Lyapunov para el bucle del agente

Definimos una función $V: \mathcal{S} \times \mathbb{N} \to \mathbb{R}_{\ge 0}$ que combina estado e iteración:

$$
V(s_k, k) = \underbrace{\mathbb{1}[\sigma_k = \text{tool}] \cdot \phi(s_k)}_{\text{"distancia" al objetivo}} + \underbrace{\lambda \cdot \max(0,\, k - K_{\max})}_{\text{penalización por exceso de iteraciones}}
$$

donde:
- $\sigma_k$ es el tipo de acción en el paso $k$ (antes de ejecutar; o bien el tipo de la última acción).
- $\phi(s) \geq 0$ es una función que mide "cuánto falta para terminar". Operativamente: $\phi(s) = 0$ si el estado tiene una respuesta final emitida; $\phi(s) > 0$ en caso contrario. En la práctica $\phi$ no es computable de forma exacta (no hay ground truth de "objetivo cumplido"); se puede aproximar con heurísticas (p. ej. $\phi(s) = 1$ si no hay `message` en el último output, $\phi(s) = 0$ si hay).
- $\lambda > 0$ es un peso que penaliza violaciones del límite $K_{\max}$.

**Propiedades deseables:**
1. $V(s, k) \geq 0$.
2. $V(s, k) = 0$ solo cuando $\sigma = \text{text}$ (terminación) y $k \leq K_{\max}$.
3. Si el bucle **converge** (emite `message` en paso $K$), entonces $V(s_{K+1}, K+1) = 0$.
4. Si $k \geq K_{\max}$ y aún $\sigma = \text{tool}$, el término $\lambda \cdot (k - K_{\max})$ crece; la parada forzada interviene y lleva a un estado absorbente, con lo que $k$ no puede superar $K_{\max} + 1$ en ejecución normal.

#### Condición de estabilidad: ausencia de ciclos infinitos

**Proposición (estabilidad por parada forzada):** Con el límite $K_{\max}$ y parada forzada, el bucle **no** puede entrar en un ciclo infinito de tool-calls. Es decir, existe $K \leq K_{\max} + 1$ tal que $\sigma_K = \text{text}$ con probabilidad 1 (determinista, dado que la parada forzada no depende del LLM).

**Demostración (sketch):** Por construcción, si $k = K_{\max}$ y $\sigma_k = \text{tool}$, el sistema transita al estado FORZADO y en el siguiente paso se inyecta un mensaje que fuerza `message`. Por tanto la secuencia de iteraciones está acotada por $K_{\max} + 2$ pasos como máximo. No hay ciclo infinito posible.

#### Condición de no-alucinación en horizonte acotado

La "alucinación" en el sentido de **bucle** (repetir indefinidamente herramientas sin progresar) queda excluida por $K_{\max}$. La "alucinación" como **contenido erróneo** en la respuesta final es distinta: el LLM puede emitir `message` con información falsa. La función $V$ no controla la calidad semántica; solo la **terminación**.

**Versión "suave" de Lyapunov:** Si pudiéramos definir $\phi(s)$ de forma operativa (p. ej. mediante un modelo de valor $V^\pi(s)$ de RL), entonces un decrecimiento esperado $\mathbb{E}_\pi[\Delta V] \leq 0$ indicaría que el agente tiende a acercarse al objetivo. En la práctica no tenemos acceso a tal $\phi$; la parada forzada provee una **Lyapunov externa**: la función $V_{\text{ext}}(k) = k$ es acotada por $K_{\max} + 1$, y el sistema es estable en el sentido de que siempre termina.

#### Resumen

| Concepto | Papel en el agente |
|----------|--------------------|
| **Lyapunov clásica** | $V(x) \geq 0$, $\Delta V \leq 0$ → convergencia |
| **$V(s, k)$ propuesta** | Combina $\phi(s)$ (distancia al objetivo) y penalización por $k > K_{\max}$ |
| **Estabilizador** | $K_{\max}$ actúa como cota que impide ciclos infinitos |
| **Límite** | No controlamos la calidad de la respuesta, solo la terminación |

### 0.5D.3.3 Criterio heurístico de detección

Para detectar oscilaciones en tiempo de ejecución se pueden usar heurísticas:
- **Repetición de (herramienta, argumentos)** en los últimos $n$ pasos.
- **Entropía del historial** de herramientas: si siempre las mismas, sospecha de bucle.
- **Crecimiento de $|M_k|$** sin "nuevo contenido" útil (detección más sofisticada).

Una política de **parada anticipada** podría activarse si se detecta un patrón repetitivo, en lugar de esperar a $K_{\max}$.

---

## 0.5D.4 Estados de error como puntos de inestabilidad

### 0.5D.4.1 Definición (caracterización comportamental)

En lugar de definir "error" de forma circular ("estados desde los que se comporta mal"), caracterizamos $\mathcal{E} \subset \mathcal{S}$ por **comportamiento observable**:
- **No convergencia en horizonte acotado:** Desde $s$, con alta probabilidad bajo $\pi$, el bucle requiere más de $K_{\max}$ iteraciones para alcanzar $\sigma_k = \text{text}$, o llega a $K_{\max}$ sin haber progresado hacia el objetivo.
- **Bucles detectables:** Existe un ciclo en la secuencia de estados $(s_k)$ o en la secuencia de pares (herramienta, argumentos) que se repite sin cambio.
- **Crecimiento descontrolado del contexto:** $|M_k|$ crece rápidamente sin aportar información nueva (ej: repetición de la misma herramienta).

Formalmente: $s \in \mathcal{E}$ si, partiendo de $s_0 = s$, la trayectoria bajo $\pi$ exhibe alguno de estos comportamientos con probabilidad no despreciable. La política $\pi$ no "sabe" que está en $\mathcal{E}$; la caracterización es **externa** (observador) para análisis y mitigación.

### 0.5D.4.2 Ejemplos concretos

| Estado de error | Descripción |
|----------------|-------------|
| **Historial saturado** | $|M_k|$ cercano al límite del contexto; el modelo "olvida" información antigua y toma decisiones incoherentes. |
| **Resultado de herramienta ambiguo o vacío** | La herramienta devuelve `{"files": []}` o un mensaje de error; el modelo no sabe cómo proceder y puede repetir la misma llamada. |
| **Objetivo irrealizable** | El usuario pide algo imposible ("lee el archivo que no existe"); el modelo puede intentar repetidamente. |
| **Prompts adversariales** | Entradas diseñadas para confundir al modelo y provocar bucles o respuestas erráticas. |

### 0.5D.4.3 Mitigaciones

- **Guardrails** (§0.12): restringir las acciones (ej: rutas de archivos) reduce estados alcanzables y evita algunos errores.
- **Parada forzada** $K_{\max}$: evita explosión temporal.
- **Mensajes de sistema** que instruyan al modelo a "admitir derrota" si tras varios intentos no logra el objetivo.
- **Validación de resultados**: si la herramienta devuelve error, inyectar un mensaje que guíe al modelo a no repetir la misma acción ciegamente.

---

## 0.5D.5 Diagrama de estados (simplificado)

Podemos modelar el bucle como un autómata de estados finitos:

```
                    ┌──────────────┐
    mensaje usuario │   INICIO     │
    ───────────────▶│  (s_0)       │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐     function_call
                    │  ESPERANDO  │◀──────────────────┐
                    │  (s_k)      │                   │
                    └──────┬──────┘                   │
                           │                          │
              ┌────────────┼────────────┐             │
              │            │            │             │
              ▼            ▼            ▼             │
        ┌──────────┐ ┌──────────┐ ┌──────────┐       │
        │  TOOL    │ │  MESSAGE │ │  FORZADO │       │
        │  (ejec.) │ │  (fin)   │ │  (K_max) │       │
        └────┬─────┘ └──────────┘ └────┬─────┘       │
             │            │            │              │
             │            │            │              │
             └────────────┴────────────┴──────────────┘
                          resultado / mensaje
```

- **INICIO**: estado tras el mensaje del usuario ($k = 0$).
- **ESPERANDO**: esperando la respuesta del LLM (decisión $\sigma_k$).
- **TOOL**: se ejecutó una herramienta; se vuelve a ESPERANDO con el resultado (transición $s_k \to s_{k+1}$).
- **MESSAGE**: el LLM emitió mensaje final; **estado absorbente** (el turno termina).
- **FORZADO**: se alcanzó $K_{\max}$ tool-calls sin `message`; el sistema inyecta un prompt para forzar la respuesta; el siguiente paso lleva a MESSAGE. **Estado absorbente de fallback**.

**Transición a FORZADO:** Cuando $k = K_{\max}$ y $\sigma_k = \text{tool}$, en lugar de volver a ESPERANDO se transita a FORZADO; desde ahí se fuerza una última llamada al LLM ("Responde ahora") y se obtiene MESSAGE. Los estados MESSAGE y FORZADO son **absorbentes** en ese turno: el bucle no continúa.

---

## 0.5D.6 Pseudocódigo del bucle de control

```python
def run_agent_loop(s0: State, pi: Policy, K_max: int) -> tuple[State, int]:
    """Bucle tool-calling con parada forzada."""
    s = s0
    k = 0
    while True:
        a = pi(s)  # acción del LLM (tool o message)
        if is_message(a):
            return s | output(a), k  # convergencia, K = k
        if k >= K_max:
            s = inject_force_message(s)
            a = pi(s)
            return s | output(a), K_max  # parada forzada
        s = s | output(a)  # T_loop(s, a)
        k += 1
```

La función `output(a)` ejecuta la herramienta si $a \in \mathcal{A}_{\text{tool}}$ y devuelve el resultado; si es mensaje, devuelve el propio mensaje. `s | output(a)` implementa $s \oplus \text{output}(a)$.

---

## 0.5D.7 Resumen y conexiones

| Concepto | Formalización |
|----------|---------------|
| Bucle (tiempo discreto) | $s_{k+1} = T_{\text{loop}}(s_k, a_k)$, $a_k \sim \pi(\cdot \mid s_k)$, $k = 0,1,2,\ldots$ |
| Convergencia | $\exists K \ge 0 : \sigma_K = \text{text}$ |
| Parada forzada | $K \leq K_{\max} + 1$ siempre |
| Oscilación | Ciclo en $(s_k)$ o en (herramienta, args) sin aproximarse a terminal |
| Estabilidad (Lyapunov) | $V(s,k)$ tipo Lyapunov; $K_{\max}$ garantiza terminación (sin ciclos infinitos); §0.5D.3.2 |
| Estado de error | $s \in \mathcal{E}$: comportamiento de no convergencia, bucle o explosión de contexto |

**Referencias cruzadas:**
- §0.7: El agente como MDP; aquí se analiza la **dinámica del lazo** de ese MDP.
- §0.8: Espacio de estados y transición; el bucle usa $T$ iteradamente.
- §3.3: Descripción operativa del bucle tool-calling.
- §4: Patrón ReAct; §0.5D.1.2 conecta explícitamente el lazo con Reason → Act → Observe.
- §0.14: Coste del bucle; $K_{\max}$ limita el coste por turno.
- Anexo A.2: Optimal stopping; la decisión de parar es un problema de parada óptima (no resuelto explícitamente por el LLM).
- Anexo A.6: Modelado con mónadas y funtores; cierre formal de la arquitectura del agente (Cap. 14).

---

## 0.9 Decisión como maximización de utilidad

### 0.9.1 Utilidad esperada

Bajo incertidumbre sobre el resultado $o$ de la acción $a$, una decisión racional (maximización de utilidad esperada) sería:

$$
a^* = \arg\max_{a \in \mathcal{A}} \mathbb{E}_{o \sim P(o|s,a)}\left[ U(s, a, o) \right]
$$

donde $U$ es una función de utilidad. En el LLM, $U$ no está definida explícitamente; emerge del entrenamiento (RLHF, etc.).

### 0.9.2 Generación token a token

En la práctica, el LLM no elige la acción completa de una vez. Genera token a token; la "acción" (texto o JSON de llamada a herramienta) se construye incrementalmente. La decisión en cada paso es qué token emitir, condicionado al contexto actual.

---

## 0.10 Temperatura y sampling

### 0.10.1 Transformación con temperatura

Dada una distribución $p(y)$ sobre el vocabulario (logits $z_y$ con $p(y) \propto e^{z_y}$), la **temperatura** $\tau > 0$ modifica la distribución a:

$$
p_\tau(y) = \frac{e^{z_y / \tau}}{Z_\tau} = \frac{p(y)^{1/\tau}}{Z_\tau}, \quad Z_\tau = \sum_{y} e^{z_y / \tau}
$$

### 0.10.2 Límites

- $\tau \to 0$: $p_\tau$ se concentra en el $\arg\max$ (distribución delta en el modo).
- $\tau = 1$: $p_\tau = p$.
- $\tau \to \infty$: $p_\tau$ tiende a la uniforme sobre $\mathcal{V}$.

### 0.10.3 Divergencia KL inducida

La distribución $p_\tau$ diverge de $p$. La divergencia $D_{\text{KL}}(p \| p_\tau)$ mide el "costo" de usar temperatura en términos de desviación de la distribución original. Para $\tau < 1$, $p_\tau$ está más concentrada (menos entropía); para $\tau > 1$, más dispersa (más entropía).

### 0.10.4 Analogía física

En mecánica estadística, la temperatura $T$ controla la fluctuación: $\langle E^2 \rangle - \langle E \rangle^2 \propto k_B T$. Aquí $\tau$ juega un papel análogo: controla la "fluctuación" (aleatoriedad) de la elección del token.

---

## 0.11 Exploración vs explotación

El trade-off **exploración/explotación** aparece en decisiones secuenciales: explorar (probar alternativas inciertas) vs explotar (elegir la mejor acción conocida). En bandidos multi-brazo y RL se formaliza con índices como UCB. En LLMs:

- **Explotación** ($\tau \approx 0$): elegir siempre el token más probable.
- **Exploración** ($\tau$ alto): mayor diversidad, más riesgo de incoherencia.

Para agentes que ejecutan tareas críticas (herramientas, código), suele preferirse $\tau \approx 0$.

---

## 0.12 Conjuntos admisibles y guardrails

### 0.12.1 Formalización

Los **guardrails** restringen el espacio de acciones a un subconjunto **admisible** $\mathcal{A}_{\text{adm}} \subseteq \mathcal{A}$. Para rutas de archivos $\mathcal{P}$:

$$
\mathcal{P}_{\text{adm}} = \left\{ p \in \mathcal{P} : 
\begin{array}{l}
\text{resolved}(p) \subseteq \text{base}, \\
\text{depth}(p) \leq d_{\max}, \\
\forall \phi \in \Phi : \neg \phi(p)
\end{array}
\right\}
$$

- $\text{resolved}(p) \subseteq \text{base}$: la ruta resuelta cae bajo el directorio base (no path traversal).
- $\text{depth}(p) \leq d_{\max}$: profundidad acotada.
- $\phi(p) = \text{true}$: $p$ coincide con un patrón prohibido (ej. `\.env`, `\.git`).

### 0.12.2 Predicados como filtros

Cada $\phi \in \Phi$ es un predicado (p. ej. expresión regular). La validación es $p \in \mathcal{P}_{\text{adm}} \iff \bigwedge_{\phi} \neg \phi(p)$ y las condiciones geométricas anteriores.

---

## 0.13 Límite termodinámico e irreversibilidad (analogía)

El historial de mensajes crece monótonamente: cada turno añade tokens. Sin compresión:

$$
|M_t| \geq |M_{t-1}|
$$

En términos de entropía: si el estado $s_t$ incluye $M_t$ completo, $H(s_t) \geq H(s_{t-1})$ para historiales no truncados. La **memoria episódica** comprime $M_t$ en $\tilde{M}_t$ de longitud acotada, actuando como un "baño" que disipa información menos relevante y mantiene la entropía del resumen acotada.

---

## 0.14 Complejidad computacional del bucle

### 0.14.1 Coste por iteración

Cada iteración del bucle tool-calling implica:
1. Envío de $\sim |M|$ tokens a la API: $O(|M|)$.
2. Inferencia del LLM: en arquitecturas transformer, $O(L^2 d + L d^2)$ con $L$ = longitud de la secuencia y $d$ = dimensión del modelo. El término $O(L^2 d)$ proviene del **mecanismo de atención** (matriz $L \times L$ de pesos; véase §0.5C); el término $O(L d^2)$ de las proyecciones lineales (FFN) y demás capas.
3. Ejecución de herramienta: $O(1)$ para operaciones locales típicas.

En muchos modelos $L^2 d$ domina cuando $L$ es grande; cuando $d \gg L$, puede dominar $L d^2$. Por simplificación se suele escribir $O(L \cdot d^2)$ o $O(L^2)$ según el contexto.

### 0.14.2 Coste total por turno

Si hay $K$ llamadas a herramientas en un turno:

$$
\text{Coste} = O\left( K \cdot \left( |M| + L^2 d + L d^2 \right) \right)
$$

Por eso se limita `max_tool_calls_per_turn`: controlar tanto el coste económico (API) como la latencia. Desde teoría de control (§0.5D), este límite actúa además como **parada forzada** que garantiza terminación y evita bucles infinitos.

---

## 0.15 Entrenamiento del modelo: minimización de la cross-entropy

El modelo de lenguaje se entrena para **predecir el siguiente token** dados los anteriores. Dado un corpus de secuencias $(x_1^{(i)}, \ldots, x_n^{(i)})$, la **función de pérdida** típica es la **cross-entropy** (equivalente a maximizar la log-verosimilitud):

$$
\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \frac{1}{n_i} \sum_{t=1}^{n_i} \log p_\theta(x_t^{(i)} \mid x_1^{(i)}, \ldots, x_{t-1}^{(i)})
$$

Minimizar $\mathcal{L}$ equivale a maximizar $\prod p_\theta(\text{token verdadero} | \text{contexto})$ sobre el corpus. El gradiente se calcula con **backpropagation**; se usa **descenso por gradiente** (o variantes como Adam) para actualizar $\theta$. Este proceso es lo que "enseña" al modelo la estructura del lenguaje; el agente hereda ese conocimiento y lo usa para generar texto y decidir cuándo llamar herramientas.

## 0.16 Ejemplo numérico ilustrativo

Supongamos un vocabulario simplificado $\mathcal{V} = \{\texttt{A}, \texttt{B}, \texttt{C}\}$ y que en un paso el modelo produce logits $z = (2, 1, 0)$. Las probabilidades son:

$$
p = \mathrm{softmax}(z) = \left( \frac{e^2}{e^2+e^1+e^0}, \frac{e^1}{e^2+e^1+e^0}, \frac{e^0}{e^2+e^1+e^0} \right) \approx (0.67, 0.24, 0.09)
$$

La entropía es $H = -\sum p_i \log p_i \approx 0.82$ nats. Con temperatura $\tau = 0.5$, $p_\tau \propto (e^4, e^2, e^0)$ está más concentrada (menor entropía); con $\tau = 2$, más dispersa. La perplejidad de esta distribución es $\exp(H) \approx 2.27$.

## 0.17 Resumen de fórmulas clave

| Concepto | Fórmula |
|----------|---------|
| Regla de la cadena | $P(X_{1:n}) = \prod_t P(X_t \mid X_{1:t-1})$ |
| Entropía | $H(X) = -\sum_x p(x) \log p(x)$ |
| Entropía condicional | $H(X \mid Y) = -\sum_{x,y} p(x,y) \log p(x \mid y)$ |
| Información mutua | $I(X;Y) = H(X) - H(X \mid Y)$ |
| Divergencia KL | $D_{\text{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$ |
| Perplejidad | $\text{PP} = \exp\left(-\frac{1}{n}\sum_t \log p(x_t \mid \ldots)\right)$ |
| Softmax | $p(y) = e^{z_y} / \sum_{y'} e^{z_{y'}}}$ |
| Temperatura | $p_\tau(y) \propto p(y)^{1/\tau}$ |
| Cross-entropy | $\mathcal{L} = -\frac{1}{n}\sum_t \log p(x_t \mid \ldots)$ |
| Bellman | $V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R + \gamma V^\pi(s')]$ |
| Atención | $A = \mathrm{softmax}(QK^T/\sqrt{d_k})$, $\mathrm{Output} = A V$ (§0.5C) |
| Parada forzada | $K \leq K_{\max} + 1$; garantiza terminación del bucle (§0.5D) |

---

## 0.18 Referencias cruzadas

- §0.1–0.2: Base del modelo de lenguaje (§2 del documento).
- §0.3–0.5: Fundamentan la compresión de memoria (§5).
- §0.5B: Embeddings, geometría, RAG y memoria semántica (§5, Anexo A.3).
- §0.5C: Mecánica estadística de la atención, coste $O(L^2)$.
- §0.5D: Teoría de control y estabilidad del bucle, convergencia, oscilaciones.
- §0.6–0.7: Marco para ReAct y MDPs (§4).
- §0.9–0.11: Decisión, temperatura, exploración.
- §0.12: Implementación en PathGuardrail (§6).

---

# PARTE I — TEORÍA BÁSICA

---

## Capítulo 1: ¿Qué es un agente de IA?

### 1.1 Definición intuitiva

Un **agente de IA** es un sistema de software que:

- Recibe instrucciones (en lenguaje natural).
- Toma decisiones autónomamente sobre qué hacer.
- Puede usar **herramientas** (leer archivos, ejecutar código, buscar en internet, etc.).
- Entrega un resultado final al usuario.

A diferencia de un simple chatbot que solo responde texto, un agente puede **actuar** en el mundo: modificar archivos, enviar emails, consultar bases de datos, etc.

### 1.2 Analogía simple

Imagina que contratas a un asistente humano:

| Característica | Asistente humano | Agente de IA |
|----------------|------------------|--------------|
| Entiende instrucciones | Sí (hablas con él) | Sí (procesa texto) |
| Decide por sí mismo | Sí (elige qué hacer) | Sí (un modelo de lenguaje decide) |
| Usa herramientas | Teléfono, ordenador, etc. | Funciones de código: leer archivo, editar, etc. |
| Recuerda conversaciones | Sí | Depende de cómo lo diseñes |
| Tiene limitaciones | Puede equivocarse | Puede alucinar o tomar decisiones incorrectas |

### 1.3 Definición formal (para lectores con formación matemática)

Un **agente** puede definirse como la tupla $\mathcal{A} = (\mathcal{S}, \mathcal{A}, \mathcal{O}, \pi, \mathcal{T})$ donde:

- $\mathcal{S}$: espacio de estados (historial de mensajes, memoria, etc.)
- $\mathcal{A}$: espacio de acciones (texto + llamadas a herramientas)
- $\mathcal{O}$: espacio de observaciones (respuestas del LLM, outputs de herramientas)
- $\pi: \mathcal{S} \to \Delta(\mathcal{A})$: política (distribución sobre acciones dado el estado)
- $\mathcal{T}: \mathcal{S} \times \mathcal{A} \times \mathcal{O} \to \mathcal{S}$: función de transición de estado

El agente opera en ciclos: observa $s_t$, elige $a_t \sim \pi(\cdot | s_t)$, recibe $o_t$, y actualiza $s_{t+1} = \mathcal{T}(s_t, a_t, o_t)$. La política $\pi$ está implementada por el LLM.

### 1.4 Componentes básicos de un agente

Todo agente típico tiene al menos estos elementos:

1. **Cerebro (orquestador):** Un modelo de lenguaje (LLM) que interpreta las instrucciones y decide qué hacer.
2. **Memoria:** El historial de la conversación y, opcionalmente, hechos almacenados.
3. **Herramientas (tools):** Funciones que el agente puede invocar para interactuar con el exterior.
4. **Bucle de control:** Un ciclo que repite: *interpretar → actuar → observar → repetir o finalizar*.

### 1.5 Tipos de agentes (por complejidad)

| Tipo | Descripción | Ejemplo |
|------|-------------|---------|
| **Chat simple** | Solo responde texto, sin herramientas | ChatGPT básico |
| **Agente con herramientas** | Puede llamar funciones (leer/editar archivos, APIs) | Nuestro agente |
| **Agente ReAct** | Razonamiento explícito antes de actuar | Nuestra versión mejorada |
| **Agente multi‑agente** | Varios agentes cooperando | Equipos de agentes especializados |
| **Agente autónomo** | Puede planificar objetivos a largo plazo y ejecutarlos | Agentes de investigación |

Nuestro proyecto cubre el paso de **agente con herramientas** a **agente ReAct** con memoria y guardrails.

---

## Capítulo 2: Fundamentos técnicos

### 2.1 ¿Qué es un LLM (Large Language Model)?

Un **modelo de lenguaje grande** es un programa entrenado con billones de palabras para:

- Predecir el texto que sigue a un fragmento dado.
- Entender y generar lenguaje natural con coherencia.

Cuando le envías un mensaje, el modelo genera una respuesta token a token. No "piensa" como un humano: calcula probabilidades sobre qué palabra es más plausible a continuación.

**Formalismo:** Sea $\mathbf{c} = (c_1, \ldots, c_m)$ el contexto (prompt + historial) y $\mathbf{y} = (y_1, \ldots, y_n)$ la respuesta generada. La probabilidad conjunta se factoriza como:

$$
P(\mathbf{y} \mid \mathbf{c}; \theta) = \prod_{i=1}^{n} P(y_i \mid \mathbf{c}, y_1, \ldots, y_{i-1}; \theta)
$$

Cada término $P(y_i \mid \cdot)$ es un vector de probabilidades sobre el vocabulario $\mathcal{V}$ (típicamente $|\mathcal{V}| \sim 10^5$). El muestreo puede ser:
- **Greedy:** $y_i = \arg\max P(y_i \mid \cdot)$
- **Temperature sampling:** $P'(y) \propto P(y)^{1/\tau}$ con $\tau$ (temperatura) controlando la aleatoriedad ($\tau \to 0$: determinista; $\tau \to \infty$: uniforme)

### 2.2 El concepto de contexto (context window)

El LLM solo "ve" un número limitado de tokens (palabras o sub-palabras). Eso se llama **ventana de contexto**. Ejemplo: si la ventana es de 128.000 tokens, el modelo puede considerar solo esos caracteres a la vez.

**Límite de información:** Si $L$ es la longitud máxima en tokens, la cantidad de información que puede codificarse está acotada:

$$
\text{Info}_{\max} \propto L \cdot \log_2 |\mathcal{V}| \quad \text{bits}
$$

Para $L = 128\,000$ y $|\mathcal{V}| \approx 100\,000$, esto da del orden de $\sim 2 \times 10^6$ bits. **Implicación:** Si la conversación es muy larga, los mensajes antiguos se pierden o hay que resumirlos (véase §0.5.4 y §0.12–0.13 sobre compresión y memoria).

### 2.2.1 Truncamiento y pérdida de información

Cuando el historial excede $L$, se aplica una función de truncamiento $\mathcal{T}$ que retiene solo los últimos $L - L_{\text{reserva}}$ tokens:

$$
\mathcal{T}(M_t) = (m_{t-k}, \ldots, m_t) \quad \text{con} \quad \sum_{i=t-k}^{t} |m_i| \leq L
$$

La **información perdida** en el truncamiento es $I(M_t; \mathcal{T}(M_t)) < H(M_t)$ para historiales largos. Por eso la memoria episódica (§5) intenta preservar hechos clave en un resumen acotado.

### 2.3 Prompts y roles

Los mensajes que envías al LLM suelen tener **roles**:

| Rol | Uso típico |
|-----|------------|
| `system` | Instrucciones fijas que definen el comportamiento del agente (ej: "Eres un asistente en español, sé conciso") |
| `user` | Lo que dice el usuario |
| `assistant` | Lo que respondió el modelo anteriormente |

El **prompt de sistema** es crucial: define la personalidad, las reglas y las capacidades del agente.

### 2.4 API de OpenAI (y similares)

Para usar un LLM en código, se hace una llamada HTTP a una API. En nuestro proyecto usamos la **API de OpenAI** (Responses API), que acepta:

- `model`: Nombre del modelo (ej: `gpt-4o-mini`, `gpt-5-nano`).
- `input`: Lista de mensajes (historial).
- `tools`: Definiciones de las herramientas que el agente puede usar.

La API devuelve una respuesta que puede contener:

- Texto (mensaje al usuario).
- Llamadas a funciones (el modelo "pide" ejecutar una herramienta con ciertos parámetros).

---

## Capítulo 3: Tool-calling: herramientas para el agente

### 3.1 ¿Qué es una herramienta (tool)?

Una **herramienta** es una función que el agente puede invocar. En nuestro proyecto tenemos tres:

| Herramienta | Descripción | Parámetros |
|-------------|-------------|------------|
| `list_files_in_dir` | Lista archivos en un directorio | `directory` (opcional) |
| `read_file` | Lee el contenido de un archivo | `path` |
| `edit_file` | Edita o crea un archivo | `path`, `prev_text`, `new_text` |

El modelo no ejecuta código: solo devuelve *"quiero llamar a `read_file` con path='config.txt'"*. Nuestro código en Python ejecuta la función real y devuelve el resultado al modelo.

### 3.2 Esquema de una herramienta (JSON Schema)

Cada herramienta se describe con un esquema que el LLM entiende:

```json
{
  "type": "function",
  "name": "read_file",
  "description": "Lee el contenido de un archivo en la ruta especificada.",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {
        "type": "string",
        "description": "Ruta del archivo"
      }
    },
    "required": ["path"]
  }
}
```

- `name`: Nombre que usará el modelo para invocar la función.
- `description`: Ayuda al modelo a saber cuándo usarla.
- `parameters`: Define los argumentos y tipos (JSON Schema).

### 3.3 El bucle tool-calling

El bucle puede describirse como un sistema dinámico discreto. Sea $k$ el índice de iteración dentro de un turno de usuario:

$$
\begin{aligned}
\text{Input}_k &= (\text{messages}_k, \text{tools}) \\
\text{Output}_k &= \text{LLM}(\text{Input}_k) \\
\end{aligned}
$$

Si $\text{Output}_k$ contiene `function_call`, entonces:
$$
\text{messages}_{k+1} = \text{messages}_k \oplus \text{Output}_k \oplus \text{result}(f, \mathbf{args})
$$
donde $\oplus$ denota concatenación y $\text{result}$ es el output de ejecutar la herramienta. El bucle termina cuando $\text{Output}_k$ es de tipo `message` (respuesta final).

**Condición de parada:** Existe $K < \infty$ tal que $\text{Output}_K \in \mathcal{A}_{\text{text}}$, o se alcanza un límite $K = K_{\max}$ (en nuestro caso `max_tool_calls_per_turn`). Para un análisis desde teoría de control (feedback loop, convergencia, oscilaciones), véase §0.5D.

El flujo típico es:

```
Usuario: "Lee el archivo config.txt y dime qué hay"
    ↓
1. Enviamos mensajes + definición de herramientas a la API
2. El modelo responde: "Voy a llamar a read_file(path='config.txt')"
3. Nuestro código ejecuta read_file('config.txt') y obtiene el contenido
4. Añadimos ese resultado al historial como "output de la herramienta"
5. Enviamos de nuevo todo a la API
6. El modelo ve el contenido y responde con un mensaje final al usuario
```

Este ciclo se repite hasta que el modelo decide que ya no necesita más herramientas y devuelve solo texto.

**Diagrama del flujo (simplificado):**

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Usuario   │────▶│  Historial +     │────▶│   API OpenAI    │
│  "Lee X"   │     │  Herramientas    │     │   (LLM)         │
└─────────────┘     └──────────────────┘     └────────┬────────┘
       ▲                        ▲                      │
       │                        │                      ▼
       │                        │              ┌──────────────┐
       │                        │              │ function_call│
       │                        │              │ o message   │
       │                        │              └──────┬───────┘
       │                        │                     │
       │                        │    ┌────────────────┴────────────────┐
       │                        │    │                                 │
       │                        │    ▼                                 ▼
       │                        │  Ejecutar                      Mostrar
       │                        │  herramienta                   respuesta
       │                        │  (read_file, etc.)             al usuario
       │                        │    │                                 │
       │                        └────┘                                 │
       │                     Añadir resultado                          │
       │                     al historial                              │
       └──────────────────────────────────────────────────────────────┘
                          (repetir hasta respuesta final)
```

### 3.4 Ventajas y riesgos del tool-calling

**Ventajas:**
- El agente puede actuar en el mundo real (archivos, APIs, bases de datos).
- Las herramientas son funciones normales que tú controlas.

**Riesgos:**
- El modelo podría pedir acciones peligrosas (borrar archivos, acceder a rutas sensibles).
- Por eso añadimos **guardrails** (Capítulo 6).

---

## Capítulo 4: El patrón ReAct

### 4.1 ¿Qué es ReAct?

**ReAct** significa **Re**asoning + **Act**ing. Es un patrón donde el agente:

1. **Razona** — Piensa qué hacer antes de actuar.
2. **Actúa** — Ejecuta una herramienta.
3. **Observa** — Lee el resultado y decide si necesita más pasos o puede responder.

Es un ciclo explícito que mejora la calidad de las decisiones. Para un análisis del bucle ReAct desde la **teoría de control** (convergencia, oscilaciones, estados de error), véase §0.5D.

**Formalización como MDP (véase §0.7):** El ciclo ReAct corresponde a una política $\pi$ que alterna entre fases. Sea $\mathcal{X}$ el espacio de "pensamientos" internos (texto de razonamiento) y $\mathcal{O}$ el de observaciones. La política compuesta es:

$$
\pi(a \mid s) = \pi_{\text{reason}}(x \mid s) \cdot \pi_{\text{act}}(a \mid s, x) \cdot \mathbb{1}[o = f(a)]
$$

En la práctica, $\pi_{\text{reason}}$ y $\pi_{\text{act}}$ están implícitos en el LLM; el prompt de sistema instruye al modelo para emitir primero razonamiento (en el texto) y luego la llamada a herramienta. La mejora de ReAct respecto a actuar sin razonar puede cuantificarse como reducción en el **número esperado de pasos** hasta alcanzar el objetivo:

$$
\mathbb{E}[T_{\text{ReAct}}] \leq \mathbb{E}[T_{\text{naive}}]
$$

en tareas que requieren planificación multi-paso.

### 4.2 Diferencias con un agente “simple”

| Agente simple | Agente ReAct |
|---------------|--------------|
| Actúa según la primera intuición | Razonamiento explícito en el prompt |
| Menos estructura en el flujo | Ciclo Pensar → Actuar → Observar |
| Más errores por precipitación | Mejor planificación |

### 4.3 Cómo implementamos ReAct en nuestro agente

No cambiamos la API: **modificamos el prompt de sistema** para que el modelo siga el patrón:

```
1. PENSAR: Antes de actuar, razona brevemente sobre el objetivo y el plan.
2. ACTUAR: Usa las herramientas disponibles de forma precisa.
3. OBSERVAR: Lee los resultados y decide si continuar o finalizar.
```

El modelo, al leer estas instrucciones, tiende a estructurar mejor su proceso interno. Las fases (`AgentPhase`) en nuestro código sirven para etiquetar trazas (ej: `ACTING` cuando se ejecuta una herramienta).

### 4.4 Fases del agente (AgentPhase)

En `agent_enhancements.py` definimos:

```python
class AgentPhase(Enum):
    REASONING = "reasoning"
    ACTING = "acting"
    OBSERVING = "observing"
    REFLECTING = "reflecting"
    FINALIZING = "finalizing"
```

- **REASONING:** El modelo está pensando.
- **ACTING:** Se ejecuta una herramienta.
- **OBSERVING:** Se procesa el resultado.
- **REFLECTING:** (preparado para futura autocrítica).
- **FINALIZING:** Respuesta final al usuario.

---

## Capítulo 5: Sistemas de memoria en agentes

### 5.1 El problema de la memoria

Los LLMs no tienen memoria persistente entre sesiones. Todo lo que "recuerdan" es lo que está en el historial de mensajes. Si la conversación es larga:

- Se consume mucho contexto.
- Los mensajes antiguos pueden "perderse" si se truncan.

**Formulación cuantitativa:** Sea $M_t = (m_1, \ldots, m_t)$ el historial hasta el turno $t$, con $|m_i|$ la longitud en tokens del mensaje $i$. El costo en tokens es:

$$
C(t) = \sum_{i=1}^{t} |m_i| + |m_{\text{sistema}}|
$$

Cuando $C(t) > L$ (ventana de contexto), es necesario comprimir. La **memoria episódica** implementa una función de compresión $\mathcal{C}: M_t \mapsto \tilde{M}_t$ con $|\tilde{M}_t| \ll |M_t|$, preservando la información relevante para futuras decisiones. La calidad de $\mathcal{C}$ se mide por la **información mutua** retenida (véase §0.5):

$$
I(M_t; \tilde{M}_t) = H(\tilde{M}_t) - H(\tilde{M}_t \mid M_t)
$$

### 5.2 Tipos de memoria (conceptos)

| Tipo | Descripción | Ejemplo |
|------|-------------|---------|
| **Memoria de trabajo** | El historial inmediato de la conversación | Lista de mensajes |
| **Memoria episódica** | Resúmenes de acciones y resultados pasados | "Ya listamos la carpeta X, contenía A y B" |
| **Memoria semántica** | Hechos abstractos (aún más comprimidos); se implementa típicamente con **embeddings** y búsqueda por similitud (véase §0.5B) | "El usuario prefiere Python" |

### 5.3 Nuestra implementación: EpisodicMemory

La clase `EpisodicMemory` almacena episodios recientes:

```python
{
  "action": "read_file('config.txt')",
  "result": "contenido...",
  "key_facts": ""  # Opcional: hechos extraídos
}
```

Cada vez que se ejecuta una herramienta, se añade un episodio. Se mantiene un máximo de episodios (ej: 20); los más antiguos se eliminan.

**Modelo de buffer circular:** Sea $N$ el tamaño máximo (`max_entries`). El buffer $\mathcal{B} = (e_1, \ldots, e_N)$ se actualiza como cola FIFO:

$$
e_{i+1} \leftarrow e_i \quad \text{para } i = 1, \ldots, N-1; \quad e_N \leftarrow e_{\text{nuevo}}
$$

El resumen inyectado es una función de los últimos $k \leq N$ episodios con longitud acotada $L_{\text{sum}}$:

$$
\tilde{M} = \text{Summarize}(e_{N-k+1}, \ldots, e_N) \quad \text{con } |\tilde{M}| \leq L_{\text{sum}}
$$

En nuestra implementación, `Summarize` es una concatenación truncada de las representaciones de cada episodio.

### 5.4 Inyección de contexto

Antes de enviar un nuevo mensaje del usuario al modelo, inyectamos un resumen de la memoria episódica:

```
[Memoria reciente]
  [1] read_file(config.txt) → contenido...
  [2] list_files_in_dir(.) → ['a.py', 'b.py']
---
Usuario: ¿Qué archivos hay?
```

Así el modelo tiene contexto sin enviar todo el historial completo.

### 5.5 Parámetros configurables

- `max_entries`: Cantidad máxima de episodios.
- `max_summary_length`: Longitud máxima del resumen inyectado (para no saturar el contexto).

---

## Capítulo 6: Seguridad y guardrails

### 6.1 ¿Por qué son necesarios los guardrails?

Un agente con acceso a archivos puede intentar:

- Leer `.env` (claves API).
- Modificar archivos en `/etc/`.
- Usar `../` para escapar del directorio permitido (path traversal).

Los **guardrails** son restricciones que impiden acciones peligrosas.

**Formalización (detalle en §0.12):** Sea $\mathcal{P}$ el espacio de todas las rutas (strings sobre el alfabeto de caracteres de rutas). Un guardrail define el conjunto **admisible** $\mathcal{P}_{\text{adm}} \subset \mathcal{P}$. Una ruta $p$ es aceptada si y solo si:

$$
p \in \mathcal{P}_{\text{adm}} \iff \underbrace{p \succeq p_{\text{base}}}_{\text{prefijo válido}} \wedge \underbrace{\text{depth}(p) \leq d_{\max}}_{\text{profundidad}} \wedge \underbrace{\bigwedge_{\phi \in \Phi} \neg \phi(p)}_{\text{sin patrones prohibidos}}
$$

donde $p \succeq p_{\text{base}}$ significa que la ruta resuelta está contenida bajo el directorio base, y $\Phi$ es el conjunto de predicados de patrones bloqueados (regex).

### 6.2 Nuestra clase PathGuardrail

`PathGuardrail` valida todas las rutas antes de usarlas:

1. **Base permitida:** Solo rutas bajo un directorio base (ej: el proyecto).
2. **Profundidad máxima:** Evita rutas demasiado profundas.
3. **Patrones bloqueados:** Expresiones regulares que deniegan acceso a rutas sensibles.

### 6.3 Patrones bloqueados por defecto

```
\.\./        → Path traversal (salir del directorio)
/etc/
/proc/
/sys/
\.env        → Claves de API
\.git/       → Repositorio
__pycache__
node_modules
```

### 6.4 Flujo de validación

```
Usuario pide: read_file("../../.env")
    ↓
PathGuardrail.validate("../../.env")
    ↓
resolve_safe() → detecta ".." → retorna None
    ↓
Resultado: {"error": "Ruta no permitida...", "content": None}
```

Nunca se ejecuta la operación real sobre una ruta no permitida.

### 6.5 Personalización

Puedes añadir más patrones o cambiar el directorio base:

```python
guardrail = PathGuardrail(
    allowed_base="/home/usuario/proyecto",
    block_patterns=[r"secreto", r"\.key$"]
)
```

---

## Capítulo 7: Persistencia y estado

### 7.1 ¿Qué es persistir estado?

**Persistir** significa guardar el estado actual (mensajes, turno, etc.) en disco para poder continuar más tarde.

**Perspectiva de sistemas dinámicos:** El estado $s_t$ del agente evoluciona según una dinámica $s_{t+1} = F(s_t, u_t)$ donde $u_t$ es la entrada del usuario. La **persistencia** implementa un mapeo:

$$
\Psi: \mathcal{S} \to \mathcal{X} \quad \text{(serialización)}, \qquad \Psi^{-1}: \mathcal{X} \to \mathcal{S} \quad \text{(deserialización)}
$$

con $\mathcal{X}$ siendo el espacio de representaciones en disco (p. ej. strings JSON). La **idempotencia** ideal sería $\Psi^{-1}(\Psi(s)) = s$; en la práctica, tipos no serializables (handles, conexiones) se excluyen.

### 7.2 Qué guardamos

En nuestro agente guardamos:

- `messages`: Lista completa de mensajes (sistema, usuario, asistente, outputs de herramientas).
- `turn`: Número de turno actual.

Todo en un archivo JSON (por defecto `.agent_state.json`).

El estado serializado tiene entropía $H(\Psi(s))$ que crece con la longitud de la conversación. Para $t$ turnos con mensajes de longitud media $\ell$, el tamaño esperado es $O(t \cdot \ell)$.

### 7.3 Comandos guardar y cargar

En el bucle interactivo:

- **guardar**: Llama a `agent.save_state()` y escribe el JSON.
- **cargar**: Llama a `agent.load_state()` y restaura `messages` y `turn`.

### 7.4 Consideraciones de seguridad

El archivo de estado puede contener contenido sensible (conversaciones, rutas). Conviene:

- Añadirlo a `.gitignore` si contiene datos privados.
- No compartir el archivo sin revisar.

---

## Capítulo 8: Observabilidad y trazas

### 8.1 ¿Qué es observabilidad?

**Observabilidad** es la capacidad de entender qué está haciendo el sistema: qué se ejecutó, cuánto tardó, si hubo errores.

**Formalización (teoría de sistemas):** Un sistema con estado $s_t$ y salidas $y_t$ es **observable** si el estado puede reconstruirse a partir de las salidas en un número finito de pasos. En nuestro caso, las **trazas** son las salidas observables: cada traza $\tau_i$ captura

$$
\tau_i = (t_i, \phi_i, a_i, \mathbf{x}_i, o_i, \Delta t_i, \sigma_i)
$$

donde $t_i$ = turno, $\phi_i$ = fase, $a_i$ = acción, $\mathbf{x}_i$ = argumentos, $o_i$ = resultado, $\Delta t_i$ = duración, $\sigma_i \in \{0,1\}$ = éxito/fallo. La secuencia $\{\tau_i\}$ forma un **proceso puntual** en el tiempo (discreto), análogo a una trayectoria en el espacio de estados del agente.

### 8.2 ExecutionTrace

Cada ejecución de una herramienta genera un registro `ExecutionTrace`:

```python
ExecutionTrace(
    turn=1,
    phase=AgentPhase.ACTING,
    action="read_file",
    input_data={"path": "config.txt"},
    output={"content": "..."},
    duration_ms=2.5,
    success=True,
    metadata={}
)
```

### 8.3 Comando trazas

El comando `trazas` en el REPL muestra un resumen de las últimas ejecuciones, por ejemplo:

```
Turn 1 | acting | read_file | 3ms
Turn 1 | acting | edit_file | 15ms
```

Útil para depurar y entender el comportamiento del agente.

### 8.4 Usos avanzados

Las trazas se pueden exportar a logs, métricas o sistemas de monitoreo (Prometheus, etc.) para producción.

---

# PARTE II — IMPLEMENTACIÓN Y USO

---

## Capítulo 9: Implementación detallada

### 9.1 Estructura del módulo agent_enhancements.py

```
agent_enhancements.py
├── Tipos: AgentPhase, ExecutionTrace, TaskDecomposition
├── PathGuardrail (seguridad)
├── EpisodicMemory (memoria)
├── EnhancedAgent (agente principal)
├── run_enhanced_loop() (bucle REPL)
└── decompose_task() (utilidad de planificación)
```

### 9.2 Clase EnhancedAgent — Resumen

| Componente | Función |
|------------|---------|
| `__init__` | Configura modelo, guardrail, memoria, herramientas |
| `setup_tools` | Define las 3 herramientas con sus schemas |
| `_safe_path` | Valida rutas usando PathGuardrail |
| `list_files_in_dir`, `read_file`, `edit_file` | Herramientas con validación |
| `_record_trace` | Registra cada ejecución |
| `_record_episode` | Añade episodio a la memoria |
| `_inject_memory_context` | Inserta resumen de memoria en el mensaje del usuario |
| `process_response` | Interpreta la respuesta del modelo, ejecuta herramientas, devuelve (llamó_herramienta, mensaje_final) |
| `save_state`, `load_state` | Persistencia |

### 9.3 Flujo completo de process_response

```
response (de la API)
    ↓
Para cada output en response.output:
    Si type == "function_call":
        - Extraer nombre y argumentos
        - Ejecutar la herramienta correspondiente
        - Validar rutas con _safe_path
        - Registrar traza y episodio
        - Añadir function_call_output al historial
        - Retornar (True, "")
    Si type == "message":
        - Extraer texto de content
        - Retornar (False, texto)
```

### 9.4 Formato de salida de las herramientas

Todas las herramientas devuelven diccionarios estructurados:

**list_files_in_dir:**
```python
{"files": [...], "path": "/ruta/resuelta"}
# o en error: {"error": "mensaje", "files": []}
```

**read_file:**
```python
{"content": "...", "path": "/ruta"}
# o: {"error": "mensaje", "content": None}
```

**edit_file:**
```python
{"success": True, "action": "editado"|"creado", "path": "..."}
# o: {"error": "mensaje", "success": False}
```

Esto facilita que el modelo interprete errores y el código maneje casos fallidos.

### 9.5 TaskDecomposition y decompose_task

`TaskDecomposition` representa una tarea dividida en sub-tareas:

```python
TaskDecomposition(
    goal="Crear un script y documentarlo",
    sub_tasks=["Crear script", "Documentar"],
    dependencies={0: [], 1: [0]},  # La 2 depende de la 1
    status=["pending", "pending"]
)
```

La función `decompose_task(goal)` es un placeholder heurístico que divide por frases. En producción se podría usar un LLM para planificar.

---

## Capítulo 10: Uso, personalización y extensión

### 10.1 Ejecución básica

```bash
python main_enhanced.py
```

o:

```python
from agent_enhancements import EnhancedAgent, run_enhanced_loop
agent = EnhancedAgent()
run_enhanced_loop(agent)
```

### 10.2 Comandos del REPL

| Comando | Acción |
|---------|--------|
| `salir`, `exit`, `bye` | Termina el programa |
| `guardar` | Guarda el estado en `.agent_state.json` |
| `cargar` | Restaura el estado guardado |
| `trazas` | Muestra las últimas ejecuciones de herramientas |

### 10.3 Parámetros del constructor de EnhancedAgent

```python
EnhancedAgent(
    base_dir=".",                    # Directorio base para rutas
    model="gpt-4o-mini",             # Modelo de OpenAI
    max_tool_calls_per_turn=5,       # Límite de herramientas por turno
    enable_reflection=True,          # (Reservado)
    enable_memory=True,               # Activar memoria episódica
    state_file=".agent_state.json"   # Archivo de persistencia
)
```

### 10.4 Cómo añadir una nueva herramienta

1. **Añadir el schema en `setup_tools`:**
```python
{
    "type": "function",
    "name": "mi_herramienta",
    "description": "Descripción clara para el modelo",
    "parameters": {...}
}
```

2. **Implementar el método en la clase:**
```python
def mi_herramienta(self, param1: str, param2: int) -> dict:
    # Lógica
    return {"success": True, "data": ...}
```

3. **Registrar en `process_response`:**
```python
elif fn_name == "mi_herramienta":
    result = self.mi_herramienta(**args)
```

### 10.5 Cambiar el modelo

Si tu entorno usa `gpt-5-nano` u otro modelo:

```python
agent = EnhancedAgent(model="gpt-5-nano")
```

### 10.6 Desactivar memoria o cambiar directorio base

```python
agent = EnhancedAgent(
    enable_memory=False,
    base_dir="/ruta/segura/proyecto"
)
```

---

## Capítulo 11: Comparación — Agente original vs mejorado

### 11.1 Tabla comparativa

| Aspecto | Agente original (`agent.py`) | Agente mejorado (`agent_enhancements.py`) |
|---------|------------------------------|------------------------------------------|
| **Razonamiento** | Implícito, sin estructura | ReAct explícito en el prompt |
| **Memoria** | Solo historial de mensajes | Historial + memoria episódica con resúmenes |
| **Seguridad** | Sin restricciones de ruta | PathGuardrail bloquea rutas peligrosas |
| **Salida de herramientas** | `{"files": result}` genérico | Dicts estructurados: `content`, `error`, `success` |
| **Persistencia** | No | Comandos `guardar` / `cargar` |
| **Observabilidad** | Solo prints de herramienta | Trazas con duración, turno, fase |
| **Límite de herramientas** | Sin límite | `max_tool_calls_per_turn` configurable |
| **Manejo de errores** | Básico | Errores estructurados en la respuesta |

### 11.2 Migración del agente original al mejorado

Para usar el agente mejorado sin cambiar `main.py`:

```python
# En main.py, sustituir:
# from agent import Agent
# agent = Agent()

from agent_enhancements import EnhancedAgent, run_enhanced_loop
agent = EnhancedAgent(model="gpt-5-nano")  # o el modelo que uses
run_enhanced_loop(agent)
```

O ejecutar directamente `python main_enhanced.py`.

### 11.3 Compatibilidad de API

El agente mejorado usa la misma API de OpenAI (`client.responses.create`) y el mismo formato de mensajes. La única diferencia es que las herramientas devuelven diccionarios con estructura definida en lugar de valores crudos.

---

## Capítulo 12: Preguntas frecuentes y solución de problemas

### ¿Por qué mi modelo no existe?

Si usas `gpt-5-nano` y obtienes un error de modelo no encontrado, puede ser que tu cuenta de OpenAI no tenga acceso. Prueba con `gpt-4o-mini` o `gpt-4o`:

```python
agent = EnhancedAgent(model="gpt-4o-mini")
```

### ¿Cómo aumento el límite de herramientas por turno?

```python
agent = EnhancedAgent(max_tool_calls_per_turn=10)
```

Si el agente necesita muchas operaciones en un solo turno, aumenta este valor. Ten en cuenta el coste de tokens.

### El agente intenta acceder a una ruta bloqueada

Es el comportamiento esperado de los guardrails. Si necesitas permitir una ruta específica, crea un PathGuardrail personalizado sin ese patrón en `block_patterns`, o ajusta `allowed_base`.

### La memoria no parece funcionar

La memoria episódica solo se inyecta cuando hay episodios previos (tras al menos una ejecución de herramienta). En el primer mensaje no hay nada que inyectar.

### ¿Dónde se guarda el estado?

Por defecto en `.agent_state.json` en el directorio actual. Cambia con `state_file="otro_archivo.json"`.

### ¿Puedo usar otro proveedor de LLM (Anthropic, etc.)?

Sí, pero necesitarás adaptar `run_enhanced_loop` para usar su API. La estructura de mensajes y tool-calling puede variar. El núcleo del agente (`EnhancedAgent`, `PathGuardrail`, `EpisodicMemory`) es independiente del proveedor.

### El archivo de estado es muy grande

Cada turno añade mensajes. Para conversaciones largas, el archivo puede crecer. Una mejora futura sería truncar o resumir mensajes antiguos al guardar.

---

## Capítulo 13: Evaluación rigurosa de agentes

*Casi todos los textos sobre IA describen agentes, pero pocos explican **cómo saber si el agente es "bueno"** más allá de la intuición. Este capítulo formaliza la evaluación: métricas deterministas (Pass@k), evaluación basada en modelos superiores (LLM-as-a-judge) y análisis de robustez frente a la temperatura $\tau$. El objetivo es dar herramientas rigurosas y reproducibles para medir el rendimiento de un agente.*

---

### 13.0 Motivación: Benchmarks vs. tests de unidad

**Benchmarks** y **tests de unidad** son complementarios pero distintos:

| Aspecto | Tests de unidad | Benchmarks |
|---------|-----------------|------------|
| **Objetivo** | Verificar que un componente concreto funciona según su contrato | Medir el rendimiento global del sistema en tareas variadas |
| **Granularidad** | Función, clase, módulo | Tarea completa, flujo de usuario |
| **Ground truth** | Expectativas fijas (asserts) | Respuestas de referencia o criterios externos (juicio humano, otro modelo) |
| **Ejecución** | En CI/CD, por cambio de código | Periódica o antes de releases; más costosa |
| **Interpretación** | Binario: pasa/falla | Métricas agregadas (Pass@k, tasa de éxito, etc.) |

Un **test de unidad** típico: `assert agent.read_file("a.txt")["success"]` — verifica una función aislada. Un **benchmark** típico: "Dado el prompt 'Lista los archivos en src/ y resume el contenido de main.py', ¿el agente completa la tarea correctamente?" — verifica el comportamiento end-to-end. Los benchmarks requieren una **definición operativa de corrección** (¿qué cuenta como "correcto"?), que es el foco de este capítulo.

**¿Por qué medir?** Un agente que "funciona" en un ejemplo manual puede fallar sistemáticamente en otros. La pregunta central es:

> **¿Cómo caracterizamos el rendimiento de un agente de forma objetiva, repetible y comparable entre versiones o modelos?**

Sin benchmarks rigurosos, la evaluación queda reducida a "probé y parece bien" — no escalable ni comparable. La respuesta implica tres niveles de evaluación:

1. **Métricas deterministas** — Resultados verificables algorítmicamente (ej: código ejecutable, tests que pasan).
2. **Evaluación por modelo superior** — Un LLM más capaz juzga la calidad/racionalidad de las salidas del agente.
3. **Análisis de robustez** — Cómo varía el éxito cuando cambiamos hiperparámetros (temperatura $\tau$, prompt, etc.).

---

### 13.1 El problema de evaluación: espacio de resultados

Sea $\mathcal{T}$ un conjunto de **tareas** (inputs, problemas, prompts). Cada tarea $t \in \mathcal{T}$ tiene un **resultado esperado** (ground truth) o un **criterio de corrección**. El agente, dado $t$, produce una salida $y = f_\theta(t)$ (donde $\theta$ engloba modelo, temperatura, prompt). Definimos:

- **Espacio de salidas** $\mathcal{Y}$: texto, código, acciones, trazas.
- **Función de evaluación** $\mathcal{E}: \mathcal{T} \times \mathcal{Y} \to \mathbb{R}$ o $\mathcal{E}: \mathcal{T} \times \mathcal{Y} \to \{0,1\}$ (éxito/fracaso binario).

El **rendimiento** del agente en $\mathcal{T}$ es la esperanza de $\mathcal{E}$ sobre tareas y (si hay sampling) sobre realizaciones de $f_\theta$:

$$
R(\theta) = \mathbb{E}_{t \sim \mathcal{T}}\left[ \mathbb{E}_{y \sim f_\theta(t)}\left[ \mathcal{E}(t, y) \right] \right] \approx \frac{1}{N} \sum_{i=1}^{N} \frac{1}{M} \sum_{m=1}^{M} \mathcal{E}(t_i, y_{i,m})
$$

**Tipos de $\mathcal{E}$:**
- **Determinista:** $\mathcal{E}$ es computable sin modelos (test que pasa/falla, regex, comparación exacta).
- **Basado en modelo:** $\mathcal{E}$ usa otro LLM (LLM-as-a-judge).
- **Híbrido:** Ej. compilar + tests para código; luego juez para legibilidad.

**Nota sobre la muestra:** Si $\mathcal{T}$ es grande, se usa un **subconjunto de evaluación** $\mathcal{T}_{\text{eval}} \subset \mathcal{T}$. La elección de $\mathcal{T}_{\text{eval}}$ afecta la generalización del reporte: un benchmark "fácil" infla $R$; uno "difícil" lo deflaciona. Lo ideal es que $\mathcal{T}_{\text{eval}}$ sea representativo del uso real.

---

### 13.2 Métricas deterministas

#### 13.2.1 Criterios binarios: exact match, ejecución correcta y alternativas

La forma más simple: $\mathcal{E}(t, y) \in \{0, 1\}$.

##### Exact match

$\mathcal{E}(t, y) = \mathbb{1}[y = y_{\text{ref}}]$. Solo aplicable cuando la respuesta correcta es única y determinista. **Problemas:** Espacios, mayúsculas, puntuación distinta; sinonimia; reformulaciones válidas. **Mitigaciones:** Normalizar (lowercase, strip, colapsar espacios) antes de comparar; o usar matching por regex si la estructura es conocida.

##### Ejecución correcta (código)

El código extraído de $y$ se ejecuta en un entorno controlado y pasa tests predefinidos. **Requisitos prácticos:**
- **Sandbox:** Ejecutar en contenedor o proceso aislado para evitar daño al sistema.
- **Timeout:** Limitar tiempo de ejecución (ej. 5 s); si hay bucle infinito, fallar.
- **Extracción de código:** Si $y$ es texto mixto (explicación + código), extraer el bloque de código (p. ej. entre \`\`\`python y \`\`\`) antes de ejecutar.

**Estructura típica:** Para cada problema $i$ hay tests $T_i$. Se ejecuta `exec(code)` o `subprocess` con el código; se corre `pytest` o asserts embebidos. $\mathcal{E}(t, y) = 1$ si todos los tests pasan.

##### Métricas continuas para texto (cuando exact match falla)

Cuando hay varias respuestas válidas, se usan métricas de similitud:
- **BLEU:** Precisión de n-gramas entre $y$ y referencias; orientado a traducción.
- **ROUGE:** Recall de n-gramas; orientado a resumen.
- **Edit distance (Levenshtein):** Distancia de edición mínima; $\mathcal{E} = 1$ si $\text{edit}(y, y_{\text{ref}}) \leq \ell_{\max}$.

**Limitación:** Ninguna captura equivalencia semántica. "La capital de Francia es París" y "París es la capital de Francia" pueden tener BLEU bajo aunque ambas sean correctas. Para evaluar significado, suele ser necesario LLM-as-a-judge o anotación humana.

#### 13.2.2 Pass@k: probabilidad de al menos un acierto en k intentos

En tareas donde el agente genera **código** (o una salida verificable) y podemos **re muestrear** (generar $k$ intentos independientes), la métrica **Pass@k** mide la probabilidad de que **al menos uno** de los $k$ intentos pase los tests.

##### 13.2.2.1 Definición y problema del estimador naive

Sea $p$ la **probabilidad verdadera** de que una muestra aleatoria del modelo pase los tests para un problema dado. Pass@k es la probabilidad de que **al menos 1** de $k$ muestras i.i.d. pase:
$$
\text{Pass@}k = 1 - (1-p)^k
$$
(ya que $(1-p)^k$ es la probabilidad de que las $k$ fallen).

**Problema:** No conocemos $p$. Si generamos $c$ muestras y $n$ pasan, el estimador **naive** sería $\hat{p} = n/c$, y entonces:
$$
\widehat{\text{Pass@}k}_{\text{naive}} = 1 - (1 - n/c)^k
$$
Este estimador es **sesgado hacia abajo**. La razón: cuando usamos las mismas $c$ muestras para estimar $p$ y para simular "elegir $k$ de ellas", estamos **reutilizando** las muestras. La probabilidad condicional de que al menos 1 pase en una subsample de $k$ de las $c$ muestras, dado que vimos $n$ aciertos en $c$, no es $(1 - n/c)^k$ sino exactamente:
$$
P(\text{ninguna de } k \text{ elegidas al azar pasa} \mid n \text{ de } c \text{ pasan}) = \frac{\binom{c-n}{k}}{\binom{c}{k}}
$$
porque elegimos $k$ muestras sin reemplazo de $c$, y la probabilidad de que las $k$ sean todas del grupo de $c-n$ que fallaron es precisamente ese cociente combinatorio.

##### 13.2.2.2 Deducción del estimador imparcial

Dado que generamos $c$ muestras y observamos $n$ que pasan, la variable $N \sim \text{Binomial}(c, p)$ (número de aciertos). Queremos estimar $\theta = 1 - (1-p)^k = \text{Pass@}k$.

**Clave:** En lugar de estimar $p$ y luego calcular $1-(1-\hat{p})^k$, usamos el hecho de que si **subsampleamos** $k$ muestras sin reemplazo de las $c$, la probabilidad de que al menos una pase es exactamente:
$$
\theta = 1 - \frac{\binom{c-n}{k}}{\binom{c}{k}} \quad \text{(valor poblacional dado } n,c\text{)}
$$
Pero $n$ es aleatorio. El **estimador** que proponemos es:
$$
\widehat{\theta} = 1 - \frac{\binom{c - N}{k}}{\binom{c}{k}}
$$
 con convención $\binom{a}{k}=0$ si $a < k$. Se puede demostrar (Chen et al., 2021, HumanEval) que $\mathbb{E}[\widehat{\theta}] = \theta$ bajo $N \sim \text{Bin}(c,p)$:
$$
\mathbb{E}\left[1 - \frac{\binom{c-N}{k}}{\binom{c}{k}}\right] = 1 - (1-p)^k
$$
La demostración usa identidades combinatorias y la función generadora de momentos del binomio; el resultado es que este estimador corrige exactamente el sesgo del naive.

**Comparación con el naive:** Para $c=10$, $n=3$, $k=1$:
- Naive: $\widehat{\text{Pass@}1}_{\text{naive}} = 1 - (1 - 3/10)^1 = 0.3$ (en este caso coincide por casualidad).
- Para $k=5$: Naive $= 1 - (7/10)^5 \approx 0.832$; imparcial $= 1 - \binom{7}{5}/\binom{10}{5} = 1 - 21/252 \approx 0.917$. El naive subestima porque asume reemplazo (las $k$ muestras son i.i.d.), pero en realidad las $k$ se eligen de las $c$ sin reemplazo, lo que incrementa la probabilidad de acertar al menos una vez cuando $k$ es una fracción significativa de $c$.

##### 13.2.2.3 Ejemplo numérico detallado

$c = 10$ muestras, $n = 3$ pasan. Para distintos $k$:

| $k$ | $\binom{c-n}{k}/\binom{c}{k}$ | Pass@k |
|-----|------------------------------|--------|
| 1   | $\binom{7}{1}/\binom{10}{1} = 7/10$ | $0.3$ |
| 2   | $\binom{7}{2}/\binom{10}{2} = 21/45$ | $1 - 21/45 \approx 0.533$ |
| 3   | $\binom{7}{3}/\binom{10}{3} = 35/120$ | $\approx 0.708$ |
| 5   | $\binom{7}{5}/\binom{10}{5} = 21/252$ | $\approx 0.917$ |

Interpretación: con 3 soluciones correctas de 10 intentos, si el usuario tuviera **5 intentos gratuitos** (como en code completion), la probabilidad de acertar al menos una vez sería ~91.7%.

##### 13.2.2.4 Agregación sobre múltiples problemas

Para un benchmark con $P$ problemas, se genera $c$ muestras por problema. Sea $n_i$ el número de aciertos en el problema $i$. El Pass@k **por problema** es $\hat{\theta}_i = 1 - \binom{c-n_i}{k}/\binom{c}{k}$. El **Pass@k del benchmark** es la media:
$$
\widehat{\text{Pass@}k}_{\text{benchmark}} = \frac{1}{P} \sum_{i=1}^{P} \hat{\theta}_i
$$
No se debe promediar primero $n_i$ y luego aplicar la fórmula; eso sería incorrecto. Cada problema tiene su propia $\hat{\theta}_i$.

**Implementación en Python:**

```python
import math
from typing import List

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Estimador imparcial de Pass@k.
    n = número de muestras que pasan los tests
    c = número total de muestras generadas
    k = número de intentos "gratuitos" que se considerarían
    """
    if n > c or k > c or n < 0:
        return 0.0
    if k == 0:
        return 0.0
    if n >= k:
        return 1.0  # al menos k pasan, Pass@k = 1
    # P(al menos 1 pase en k intentos) = 1 - P(ninguno pase)
    # P(ninguno pase) = C(c-n, k) / C(c, k)
    return 1.0 - math.comb(c - n, k) / math.comb(c, k)

def pass_at_k_batch(results: List[bool], k: int) -> float:
    """results[i] = True si la muestra i pasó los tests."""
    n = sum(results)
    c = len(results)
    return pass_at_k(n, c, k)

# Ejemplo: 10 intentos, 3 pasan
print(pass_at_k(3, 10, 1))   # ~0.3
print(pass_at_k(3, 10, 3))   # ~0.708
```

##### 13.2.2.5 Relación con la política y la temperatura

El agente tiene política $\pi_\tau(a|s)$ con temperatura $\tau$ (§0.10). Pass@k promedia sobre las realizaciones de esa política. Para $\tau \to 0$ (greedy), cada problema produce **una sola** salida determinista; en ese caso Pass@k con $k > 1$ carece de sentido a nivel de un único problema (no hay diversidad). Para obtener Pass@k con $k>1$ en modo greedy hay que variar **algo**: semilla, orden de los ejemplos en el prompt, o múltiples formulaciones del problema. Para $\tau > 0$, múltiples muestras del mismo prompt dan diversidad natural; típicamente $c = 10$ o $c = 100$ para problemas tipo HumanEval.

**Trade-off coste–precisión:** Cada muestra requiere una llamada a la API. Para $P$ problemas y $c$ muestras por problema, el coste es $P \cdot c$ llamadas. Aumentar $c$ reduce la varianza del estimador pero aumenta el coste linealmente. En la práctica, $c=20$–$50$ es un compromiso razonable para reportes; para publicaciones se suele usar $c \geq 100$.

##### 13.2.2.6 Benchmarks estándar

- **HumanEval:** 164 problemas de Python; cada uno pide implementar una función; los tests son asserts en el docstring. Pass@1 y Pass@100 son estándar.
- **MBPP:** Problemas más simples; útil para modelos pequeños.
- **APPS:** Problemas de competición; más difíciles; Pass@5, Pass@10.

##### 13.2.2.7 Varianza e intervalos de confianza

El estimador $\widehat{\text{Pass@}k}$ tiene varianza $\text{Var}(\widehat{\theta}) = \mathbb{E}[\widehat{\theta}^2] - \theta^2$; la expresión exacta es compleja. En la práctica se usan:

1. **Bootstrap sobre problemas:** Remuestrear con reemplazo los $P$ problemas, recalcular la media de $\hat{\theta}_i$ en cada réplica, y tomar percentiles 2.5 y 97.5.
2. **Bootstrap sobre muestras (por problema):** Dado $n_i$ de $c$ muestras en el problema $i$, remuestrear las $c$ etiquetas (pasa/no pasa), recalcular $\hat{\theta}_i$, repetir para todos los problemas y promediar; repetir el proceso global para el IC.

```python
import numpy as np

def pass_at_k_ci(n: int, c: int, k: int, n_bootstrap: int = 1000) -> tuple[float, float, float]:
    """Estima Pass@k y IC 95% por bootstrap."""
    if c == 0 or k > c:
        return 0.0, 0.0, 0.0
    obs = [1.0] * n + [0.0] * (c - n)  # n pasan, c-n fallan
    estimates = []
    rng = np.random.default_rng()
    for _ in range(n_bootstrap):
        boot = rng.choice(obs, size=c, replace=True)
        n_boot = int(boot.sum())
        estimates.append(pass_at_k(n_boot, c, k))
    return (
        pass_at_k(n, c, k),
        np.percentile(estimates, 2.5),
        np.percentile(estimates, 97.5),
    )
```

---

#### 13.2.3 Extensión a agentes con herramientas: evaluación de trazas

Cuando el agente **ejecuta herramientas** (read_file, run_code, etc.), la evaluación no es solo la salida final sino la **trayectoria completa**. Una **traza** es la secuencia:
$$
\text{trace} = (a_0, o_0, a_1, o_1, \ldots, a_K, o_K, r)
$$
donde $a_i$ es la acción (tool-call o mensaje), $o_i$ el resultado, y $r$ la respuesta final al usuario.

**Criterios de corrección para trazas:**

1. **Corrección de la respuesta final:** $\mathcal{E}_{\text{final}}(t, r) \in \{0,1\}$ — la respuesta $r$ satisface el objetivo (tests, pregunta, etc.).
2. **Validez de las acciones:** Cada $a_i$ tiene argumentos que pasan validación (rutas dentro de base, tipos correctos); si un guardrail bloqueó, $a_i$ no debe haber sido intentado con argumentos prohibidos.
3. **Eficiencia (opcional):** Número de tool-calls $K$ por debajo de un umbral; ausencia de ciclos redundantes (no llamar `read_file("x")` dos veces seguidas).
4. **Seguridad:** La traza no contiene intentos de path traversal, ejecución de código arbitrario fuera de sandbox, etc.

**Criterio compuesto:**
$$
\mathcal{E}(t, \text{trace}) = \mathbb{1}[\mathcal{E}_{\text{final}}(t, r) = 1] \cdot \mathbb{1}[\text{validez}(\text{trace})] \cdot \mathbb{1}[\text{seguridad}(\text{trace})]
$$

**Ejemplo concreto:** Tarea $t$: "Cuenta cuántas líneas tiene el archivo main.py en el directorio actual". Una traza correcta podría ser:
1. `list_dir(".")` → lista de archivos
2. `read_file("main.py")` → contenido
3. El agente cuenta y responde "42 líneas"

Una traza **inválida** sería: `read_file("../.env")` (path traversal). Una traza **correcta en objetivo pero ineficiente**: llamar `list_dir` cinco veces antes de `read_file`. La evaluación puede ponderar la eficiencia con un factor $0 < w \leq 1$: $\mathcal{E} = \mathcal{E}_{\text{correcto}} \cdot (1 - \alpha \cdot \mathbb{1}[\text{redundante}])$ con $\alpha$ pequeño.

---

### 13.3 LLM-as-a-Judge: evaluación por modelo superior

#### 13.3.1 Idea central y cuándo usarlo

Cuando no hay ground truth verificable (respuestas abiertas, razonamiento, calidad de prosa, adecuación al contexto), se usa un **modelo LLM más capaz** como **juez** para evaluar las salidas del agente. Formalmente:

- **Agente bajo evaluación:** $\mathcal{M}_{\text{agent}}$ con política $\pi_{\text{agent}}$.
- **Juez:** $\mathcal{M}_{\text{judge}}$ (típicamente más grande: GPT-4 evalúa respuestas de GPT-3.5).
- **Tarea:** Dado $(t, y)$, el juez produce $v \in \mathcal{V}$ (binario, Likert 1–5, o texto).

**Cuándo usar LLM-as-a-judge:** No existe test automático (ej. "¿es la explicación didáctica?"); el espacio de respuestas correctas es amplio (varias formulaciones válidas); se quieren criterios subjetivos (claridad, tono, racionalidad). **Cuándo NO:** Existe criterio determinista (tests de código, exact match) — usar eso primero; coste prohibitivo; sesgo fuerte del juez en el dominio.

#### 13.3.2 Diseño del prompt del juez

El prompt es el **instrumento de medida**; un mal diseño invalida los resultados.

**Estructura recomendada:** (1) **Rol:** "Eres un evaluador experto." (2) **Contexto:** La tarea original $t$. (3) **Objeto:** La traza y/o respuesta $y$. (4) **Criterios explícitos:** Lista numerada de qué se evalúa. (5) **Formato de salida:** JSON `{"score": 1-5, "reason": "..."}` para facilitar parsing.

**Ejemplo de prompt bien especificado:**

```
Eres un evaluador. Evalúa si la respuesta del agente resuelve correctamente la tarea.

TAREA: "Lista los archivos .py en el directorio actual y dime cuántos hay."

TRAZA DEL AGENTE:
- Acción: list_dir(".")
- Resultado: {"files": ["main.py", "utils.py", "config.json"]}
- Respuesta final: "Hay 2 archivos Python: main.py y utils.py."

CRITERIOS:
1. Correctitud: ¿Se identificaron correctamente los archivos .py?
2. Completitud: ¿Se respondió a ambas partes (listar y contar)?
3. Precisión: ¿Se excluyeron correctamente los no-.py?

Responde en JSON: {"correct": true/false, "score": 1-5, "reason": "..."}
```

**Errores comunes:** Criterios vagos ("¿Es buena la respuesta?"); anchoring por orden de presentación; prompt demasiado largo que hace perder criterios.

#### 13.3.3 Evaluación de racionalidad del agente

Criterios operativos: **Coherencia** — ¿La respuesta es consistente con los datos de las herramientas? **Relevancia** — ¿Cada tool fue necesaria? **Orden** — ¿El orden de acciones es lógico? **Completitud** — ¿Todos los sub-objetivos? **Seguridad** — ¿Intentos de acciones prohibidas? Pedir al juez que justifique con referencias a la traza: "Señala la acción que indica incoherencia, o 'N/A' si es coherente."

#### 13.3.4 Ejemplo trabajado completo

**Tarea:** "¿Qué contiene el archivo README.md en la raíz?"

**Traza correcta:** `list_dir(".")` → `["README.md", ...]`; `read_file("README.md")` → contenido; respuesta: "El README contiene X e Y." **Veredicto esperado:** `{"correct": true, "score": 5, "reason": "..."}`.

**Traza incorrecta:** Responde "No encontré README" sin haber llamado a `list_dir` ni `read_file`. **Veredicto esperado:** `{"correct": false, "reason": "El agente no ejecutó herramientas para verificar."}`.

#### 13.3.5 Sesgos: diagnóstico y mitigación

| Sesgo | Efecto | Mitigación |
|-------|--------|------------|
| **Longitud** | Puntúa más alto respuestas largas | "No puntúes más por verbosidad" |
| **Posición** | Orden A/B influye en comparaciones | Alternar aleatoriamente qué va primero |
| **Conocimiento** | Rellena información no explícita | "Evalúa únicamente lo explícito" |
| **Regresión** | Juez indulgente si mismo modelo base | Usar juez de otra familia o superior |
| **Inconsistencia** | $\tau>0$ da veredictos distintos | $\tau_{\text{judge}}=0$ o promediar $M$ evaluaciones |

#### 13.3.6 Agregación, coste y validación

**Agregación:** Media $\bar{v}$ y desviación estándar; para binario, proporción e IC Wilson. **Coste:** $N \cdot M$ llamadas; 100 tareas × 3 muestras × ~500 tokens ≈ \$4–5 (GPT-4). **Validación:** Comparar el juez con anotaciones humanas en 50–100 ejemplos; correlación Spearman $\rho \gtrsim 0.8$ con humanos es objetivo razonable (Zheng et al., 2023).

---

### 13.4 Análisis de robustez frente a la temperatura $\tau$

#### 13.4.1 Conexión entre $\tau$ y la distribución de salidas

La temperatura $\tau$ (§0.10) modifica la distribución sobre el espacio de secuencias (o tokens):
$$
p_\tau(y | t) \propto p(y | t)^{1/\tau}, \quad Z_\tau = \sum_y p(y|t)^{1/\tau}
$$

**Efecto en la entropía:** La entropía de $p_\tau$ es $H_\tau = -\sum_y p_\tau(y) \log p_\tau(y)$. Se cumple:
- $\tau \to 0$: $H_\tau \to 0$ (concentración en el modo); la salida es prácticamente determinista.
- $\tau = 1$: $H_\tau = H_1$ (entropía de la distribución original).
- $\tau > 1$: $H_\tau$ aumenta; más masa en colas, más diversidad.

Para un **agente** que emite secuencias de acciones (texto + tool-calls), cada decisión token-a-token usa $p_\tau$. Una secuencia completa es un producto de condicionales; la diversidad **acumulada** crece con la longitud. Por eso $\tau$ alto en agentes produce rápidamente incoherencia: un token fuera de lugar propaga error en el resto de la secuencia.

#### 13.4.2 Curva éxito–temperatura: definición y estimación

Definimos la **tasa de éxito** en función de $\tau$:
$$
S(\tau) = \mathbb{E}_{t \sim \mathcal{T},\, y \sim \pi_\tau(\cdot|t)}\left[ \mathcal{E}(t, y) \right]
$$

En la práctica, con $N$ tareas y $M$ muestras por $(\tau, t)$:
$$
\hat{S}(\tau_j) = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{M} \sum_{m=1}^{M} \mathcal{E}(t_i, y_{i,m}^{(\tau_j)})
$$

**Comportamiento típico:** $S(\tau)$ suele tener forma de campana o meseta:
- $\tau \to 0$: $S(\tau)$ fijo (modo determinista); puede ser subóptimo si la solución correcta no es el modo.
- $\tau$ óptimo (ej. 0.3–0.7): a veces $S(\tau) > S(0)$ por exploración (Pass@k mejora cuando hay múltiples soluciones correctas).
- $\tau$ alto: $S(\tau)$ cae por incoherencia, tool-calls erróneos, bucles (§0.5D.3).

#### 13.4.3 Sensibilidad, elasticidad y banda de robustez

**Sensibilidad:** $\frac{\partial S}{\partial \tau}$. Un agente **robusto** tiene $|\partial S / \partial \tau|$ pequeño en el rango de operación.

**Elasticidad:** $\eta_\tau = \frac{\tau}{S(\tau)} \frac{\partial S}{\partial \tau}$. Mide el cambio porcentual de $S$ ante un cambio porcentual de $\tau$. Si $|\eta_\tau| < 0.5$ en $\tau \in [0.2, 0.8]$, el agente es relativamente insensible.

**Banda de robustez al $\alpha\%$:** El conjunto de $\tau$ tal que $S(\tau) \geq (1-\alpha) \cdot \max_{\tau'} S(\tau')$. Ejemplo: robustez al 10% significa que $S(\tau)$ no cae más del 10% respecto al óptimo en ese rango.

**Varianza entre ejecuciones:** Para $\tau$ fijo, $\mathcal{E}(t, y)$ es Bernoulli (o multivaluada) con $y \sim \pi_\tau$. La varianza por tarea es $S(\tau)(1-S(\tau))$ para binario. La varianza del estimador $\hat{S}(\tau)$ es aproximadamente $\sigma^2/MN$ donde $\sigma^2$ es la varianza de $\mathcal{E}$ sobre tareas y muestras. Para detectar una caída de 5% en $S(\tau)$ con potencia 80%, se necesitan del orden de cientos de evaluaciones (depende del tamaño del efecto).

#### 13.4.4 Múltiples dimensiones de robustez

La robustez no es solo $\tau$. Otras dimensiones:

| Dimensión | Variación | Qué medir |
|-----------|-----------|-----------|
| **Temperatura** | $\tau \in [0, 1.5]$ | $S(\tau)$, banda al 90% |
| **Formulación del prompt** | Parafraseos de $t$ | Media y desviación de $S$ sobre parafraseos |
| **Ruido en la entrada** | Typos, palabras extra | Tasa de degradación |
| **Orden de ejemplos** | Few-shot con distinto orden | Varianza de $S$ |
| **Límite de tools** | $K_{\max} = 3, 5, 10$ | ¿Converge con menos llamadas? |

Un agente **robusto en sentido amplio** mantiene $S$ estable ante estas variaciones. Reportar solo $S(0)$ es insuficiente; al menos $S(\tau)$ para $\tau \in \{0, 0.5, 1.0\}$ da una imagen más fiel.

#### 13.4.5 Protocolo y código

1. Fijar benchmark $\mathcal{T}$ (50+ tareas). 2. Variar $\tau \in \{0, 0.2, 0.5, 0.7, 1.0, 1.2\}$. 3. Para cada $(\tau, t)$: $M \geq 5$ muestras. 4. Calcular $\hat{S}(\tau)$, IC por bootstrap (remuestrear tareas). 5. Reportar curva y banda de robustez al 90%.

```python
def robustness_sweep(tasks, eval_fn, agent_fn, temps, samples_per_task=5):
    """Barre temperaturas; retorna mean, std e IC por bootstrap sobre tareas."""
    results = {tau: [] for tau in temps}
    for tau in temps:
        for task in tasks:
            successes = [eval_fn(task, agent_fn(task, tau)) for _ in range(samples_per_task)]
            results[tau].append(np.mean(successes))
    out = {}
    for tau, vals in results.items():
        out[tau] = {"mean": np.mean(vals), "std": np.std(vals)}
        # Bootstrap IC: remuestrear tareas
        boots = [np.mean(np.random.choice(vals, size=len(vals), replace=True)) for _ in range(1000)]
        out[tau]["ci_low"] = np.percentile(boots, 2.5)
        out[tau]["ci_high"] = np.percentile(boots, 97.5)
    return out
```

#### 13.4.6 Conexión con teoría de control (§0.5D)

Con $\tau$ alto, la política $\pi_\tau$ tiene más entropía; el agente explora más pero puede **oscilar** (§0.5D.3): llamar las mismas herramientas repetidamente, entrar en bucles, o no converger antes de $K_{\max}$. La parada forzada asegura terminación, pero la **calidad** de la respuesta puede degradarse. La curva $S(\tau)$ captura ese trade-off: no solo "¿termina?" sino "¿termina bien?".

---

### 13.5 Resumen y tabla de métricas

| Métrica | Tipo | Ventaja | Limitación |
|---------|------|---------|------------|
| **Exact match** | Determinista | Simple, reproducible | Muy estricto para texto libre |
| **Pass@k** | Determinista | Estándar en código; estimador imparcial | Requiere tests; coste de muestras |
| **LLM-as-a-judge** | Modelo | Flexible; aplica a tareas abiertas | Sesgo, coste, no ground truth |
| **Curva $S(\tau)$** | Robustez | Detecta fragilidad a hiperparámetros | Coste computacional (barridos) |

---

### 13.6 Implementación práctica: esquema de evaluación mínima

##### Estructura del directorio de evaluación

```
eval/
├── tasks/           # Definición de tareas
│   ├── tasks.json   # [{"id": "1", "prompt": "...", "gold": "..."}]
│   └── tests/       # Tests por tarea (si aplica)
│       └── task_1.py
├── evaluate.py      # Script principal
├── metrics.py       # pass_at_k, exact_match, etc.
└── results/         # Salidas (JSON, CSV)
```

##### Flujo del script de evaluación

1. **Cargar tareas** desde `tasks.json` o módulo.
2. **Para cada** $(\tau, \text{tarea})$: invocar agente, capturar salida (y traza si hay herramientas).
3. **Evaluar** con $\mathcal{E}$ (determinista o juez).
4. **Agregar** por tarea y por $\tau$; calcular Pass@k si hay múltiples muestras.
5. **Exportar** reporte (JSON/CSV) con tasas, IC, y metadatos.

##### Código de ejemplo ampliado

```python
# eval/evaluate.py
import json
import numpy as np
from pathlib import Path

def run_eval(agent, tasks_path: Path, eval_fn, temperatures=(0.0, 0.5, 1.0), samples_per_task=1):
    with open(tasks_path) as f:
        tasks = json.load(f)
    report = []
    for tau in temperatures:
        tau_results = []
        for task in tasks:
            successes = []
            for _ in range(samples_per_task):
                out = agent.run(task["prompt"], temperature=tau)
                successes.append(eval_fn(task, out))
            tau_results.extend(successes)
        rate = np.mean(tau_results)
        # IC 95% aproximado para proporción (Wilson)
        n = len(tau_results)
        z = 1.96
        p_hat = rate
        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
        report.append({
            "tau": tau,
            "success_rate": rate,
            "ci_low": max(0, center - margin),
            "ci_high": min(1, center + margin),
            "n": n,
        })
    return report
```

##### Integración con el agente actual

El `EnhancedAgent` del proyecto no expone temperatura directamente en la API de tool-calling; la API de OpenAI permite `temperature` en la llamada a `responses.create`. Para evaluar con $\tau$ variable, habría que añadir el parámetro en el flujo de `run_enhanced_loop` o en la capa que invoca al modelo.

---

### 13.7 Referencias cruzadas

- §0.10: Definición formal de temperatura $p_\tau$ y su efecto en la distribución.
- §0.5D: Convergencia del bucle, oscilaciones; alto $\tau$ puede incrementar comportamiento errático.
- §0.7: MDP; la evaluación mide el retorno esperado bajo la política.
- §0.12: Guardrails; las evaluaciones pueden incluir verificación de que no se violaron.
- Cap. 8: Trazas; útiles para alimentar al juez (secuencia de acciones).

---

## Capítulo 14: Arquitectura de software — El agente en producción

*Para el desarrollador, `agent_enhancements.py` es solo el inicio. Un agente que funciona en un REPL local no es lo mismo que un agente **listo para producción**: múltiples usuarios concurrentes, APIs externas lentas, herramientas que fallan. Este capítulo introduce tres patrones de diseño que, **combinados**, convierten el prototipo en un sistema robusto: **inyección de dependencias** (desacoplar el orquestador de las herramientas), **manejo de errores estructurado** (permitir reintentos y decisiones informadas) y **programación asíncrona** (no bloquear el sistema durante I/O). Los tres forman un único flujo de datos coherente que se desarrolla a lo largo del capítulo.*

---

### 14.1 Por qué importa la arquitectura

El agente actual tiene tres limitaciones arquitectónicas que se amplifican en producción:

| Limitación | Efecto en producción |
|------------|----------------------|
| **Herramientas acopladas** | Cambiar o añadir una herramienta exige modificar `process_response` y `setup_tools`; no se puede "enchufar" un módulo distinto sin tocar el orquestador. |
| **Errores genéricos** | Un `{"error": str(e)}` no distingue 404 de timeout ni de permiso denegado; el LLM y el código no pueden reaccionar de forma diferenciada (reintentar, buscar alternativa, abortar). |
| **I/O bloqueante** | `client.responses.create()` y las llamadas a herramientas son síncronas: el hilo se bloquea hasta que la API responde. Con múltiples usuarios, un agente lento bloquea a los demás. |

Los tres patrones abordan estas limitaciones y, al mismo tiempo, **se apoyan entre sí**: la inyección de dependencias permite cambiar el conjunto de herramientas; el manejo estructurado de errores exige un contrato de salida (`ToolResult`) que las herramientas o su wrapper deben cumplir; la programación asíncrona afecta la forma en que el orquestador invoca tanto al LLM como a las herramientas. Todas las piezas confluyen en el **flujo de ejecución** que veremos en §14.5.

---

### 14.2 Inyección de dependencias: herramientas intercambiables

#### 14.2.1 El problema: acoplamiento fuerte

En `agent_enhancements.py`, las herramientas están **acopladas** al agente:

```python
# Código actual (simplificado)
def process_response(self, response):
    # ...
    if fn_name == "list_files_in_dir":
        result = self.list_files_in_dir(**args)
    elif fn_name == "read_file":
        result = self.read_file(**args)
    elif fn_name == "edit_file":
        result = self.edit_file(**args)
    else:
        result = {"error": f"Herramienta desconocida: {fn_name}"}
```

**Consecuencias:** Para añadir una herramienta nueva (ej. `search_web`), hay que (1) añadir el schema en `setup_tools`, (2) implementar el método en la clase, (3) añadir un `elif` en `process_response`. El **orquestador** (el bucle que decide qué ejecutar) está ligado a la lista concreta de herramientas. No se puede ejecutar un "agente de solo lectura" (solo `list_files_in_dir` y `read_file`) sin duplicar código o usar flags.

#### 14.2.2 Idea: el orquestador no debe conocer las herramientas

El **principio de inversión de dependencias** (DIP) dice: los módulos de alto nivel no deben depender de los de bajo nivel; ambos deben depender de abstracciones. Aquí, el orquestador no debería depender de `list_files_in_dir` o `read_file` concretos, sino de una **abstracción**: "algo invocable por nombre y argumentos que devuelve un resultado".

**Contrato de una Tool:**
- `name: str` — nombre que el LLM usa para invocar.
- `description: str` — descripción para el schema de la API.
- `parameters: dict` — schema OpenAPI: `{"type": "object", "properties": {...}, "required": [...]}`.
- `execute(args: dict) -> dict` — devuelve un dict con los datos (éxito) o lanza excepción. Las excepciones se traducirán a `ToolResult` en §14.3.

El orquestador recibe un **registry** que mapea nombre → herramienta y solo invoca `registry.execute(name, args)`; no conoce las implementaciones concretas.

#### 14.2.3 Implementación: Tool y ToolRegistry

```python
from typing import Callable
from dataclasses import dataclass

@dataclass
class Tool:
    """Representa una herramienta inyectable."""
    name: str
    description: str
    parameters: dict  # {"type": "object", "properties": {...}, "required": [...]}
    execute: Callable[[dict], dict]  # Puede ser sync o async; véase §14.4.5

class ToolRegistry:
    """Registro de herramientas. El orquestador depende solo de este."""
    
    def __init__(self):
        self._tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)
    
    def get_schemas(self) -> list[dict]:
        """Para la API de OpenAI: tools como lista de schemas OpenAPI."""
        return [
            {"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._tools.values()
        ]
    
    def execute(self, name: str, args: dict) -> dict:
        """Invoca la herramienta. Retorna dict (o propaga si la herramienta lanza).
        Para captura de excepciones y ToolResult, usar execute_safe (§14.3.7)."""
        tool = self.get(name)
        if tool is None:
            return {"success": False, "error": f"Herramienta desconocida: {name}", "error_code": "validation_error"}
        return tool.execute(args)
```

**Ejemplo de schema completo para `read_file`:**

```python
READ_FILE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Ruta del archivo"}
    },
    "required": ["path"]
}

def read_file_impl(args: dict) -> dict:
    path = args["path"]
    # ... validación con guardrail, lectura
    return {"content": content, "path": str(resolved)}

registry.register(Tool("read_file", "Lee el contenido de un archivo", READ_FILE_SCHEMA, read_file_impl))
```

#### 14.2.4 Inyección en el constructor y composición

El agente recibe el registry en el constructor. Para compatibilidad con el código existente, se ofrece un registry por defecto:

```python
class EnhancedAgent:
    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        ...
    ):
        self.tool_registry = tool_registry or self._default_registry()
    
    def _default_registry(self) -> ToolRegistry:
        reg = ToolRegistry()
        reg.register(Tool("list_files_in_dir", "...", {...}, self.list_files_in_dir))
        reg.register(Tool("read_file", "...", {...}, self.read_file))
        reg.register(Tool("edit_file", "...", {...}, self.edit_file))
        return reg
    
    def process_response(self, response):
        # ...
        result = self.tool_registry.execute(fn_name, args)
```

En producción: `agent = EnhancedAgent(tool_registry=my_custom_registry)` sin modificar la clase. Las herramientas son **plug-in**.

#### 14.2.5 Resumen

| Concepto | Significado |
|----------|-------------|
| **Acoplamiento** | El orquestador conoce cada herramienta; añadir una obliga a tocar su código. |
| **Inyección** | El orquestador recibe las herramientas desde fuera; no las crea. |
| **Registry** | Mapa nombre → Tool; el orquestador solo llama `registry.execute(name, args)`. |
| **Beneficio** | Cambiar herramientas = cambiar configuración; el orquestador es reutilizable. |

---

### 14.3 Manejo de errores estructurado y self-healing

Esta sección se integra con la anterior: el registry (o un wrapper encima) no solo invoca la herramienta sino que **captura excepciones** y las convierte en resultados estructurados. Así el orquestador y el LLM reciben siempre un formato interpretable.

#### 14.3.1 El problema: errores planos

Actualmente, cuando una herramienta falla:

```python
return {"error": str(e), "content": None}
```

`str(e)` puede ser `"FileNotFoundError: [Errno 2] No such file or directory: 'foo.txt'"` o `"Timeout"` o `"Permission denied"`. El LLM recibe una cadena opaca; el **código** no puede distinguir casos para aplicar lógicas distintas (reintentar en timeout, sugerir crear el archivo en 404, abortar en permiso denegado).

#### 14.3.2 ToolResult: contrato de salida estructurado

Definimos una **jerarquía de códigos** y un tipo de resultado único que el orquestador y el LLM interpretan:

| Código | ¿Reintentar? | Acción sugerida al LLM |
|--------|--------------|------------------------|
| `not_found` | No | "¿Crear el recurso o usar otra ruta?" |
| `timeout` | Sí | "Reintentar una vez." |
| `permission_denied` | No | "Sin permisos. No insista." |
| `validation_error` | No | "Argumentos inválidos. Corregir." |
| `rate_limit` | Sí (backoff) | "Esperar y reintentar." |
| `service_unavailable` | Sí | "Reintentar en breve." |

```python
from dataclasses import dataclass
from enum import Enum

class ToolErrorCode(Enum):
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"
    VALIDATION = "validation_error"
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"
    UNKNOWN = "unknown"

@dataclass
class ToolResult:
    """Resultado estructurado de una herramienta. Siempre es el contrato de salida."""
    success: bool
    data: dict | None = None
    error_code: ToolErrorCode | None = None
    error_message: str = ""
    retryable: bool = False
    
    def to_api_format(self) -> dict:
        """Formato para inyectar en el historial del LLM."""
        if self.success:
            return {"success": True, **(self.data or {})}
        return {
            "success": False,
            "error": self.error_message,
            "error_code": (self.error_code or ToolErrorCode.UNKNOWN).value,
            "retryable": self.retryable,
        }
```

El LLM recibe un JSON claro: `{"success": false, "error_code": "timeout", "retryable": true}`.

#### 14.3.3 Wrapper: de excepción o dict a ToolResult

La herramienta (`tool.execute`) devuelve `dict` en éxito o **lanza** en fallo. Un **wrapper** captura las excepciones y produce `ToolResult`. Así la herramienta no necesita conocer `ToolResult`; el orquestador sí.

```python
import asyncio

def _handle_tool_exception(e: Exception) -> ToolResult:
    """Mapea excepciones a ToolResult estructurado."""
    if isinstance(e, FileNotFoundError):
        return ToolResult(False, error_code=ToolErrorCode.NOT_FOUND, error_message=str(e), retryable=False)
    if isinstance(e, PermissionError):
        return ToolResult(False, error_code=ToolErrorCode.PERMISSION_DENIED, error_message=str(e), retryable=False)
    if isinstance(e, asyncio.TimeoutError) or "timeout" in str(e).lower():
        return ToolResult(False, error_code=ToolErrorCode.TIMEOUT, error_message="Operación excedió el tiempo límite", retryable=True)
    if "rate limit" in str(e).lower() or "429" in str(e):
        return ToolResult(False, error_code=ToolErrorCode.RATE_LIMIT, error_message=str(e), retryable=True)
    return ToolResult(False, error_code=ToolErrorCode.UNKNOWN, error_message=str(e), retryable=False)

def _safe_execute(tool: Tool, args: dict) -> ToolResult:
    """Ejecuta la herramienta y convierte excepciones en ToolResult."""
    try:
        out = tool.execute(args)
        return ToolResult(True, data=out if isinstance(out, dict) else {"result": out})
    except Exception as e:
        return _handle_tool_exception(e)
```

**Flujo:** `tool.execute(args)` → dict (éxito) o excepción → `_safe_execute` → siempre `ToolResult`.

#### 14.3.4 Self-healing: reintentos automáticos

Para errores `retryable`, el orquestador **reintenta** antes de pasar el fallo al LLM. Política: backoff exponencial ($t$, $2t$, $4t$, …).

```python
import time

def execute_with_retry(
    tool: Tool,
    args: dict,
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> ToolResult:
    """Ejecuta con reintentos si el resultado es retryable."""
    for attempt in range(max_retries + 1):
        result = _safe_execute(tool, args)
        if result.success or not result.retryable:
            return result
        if attempt < max_retries:
            time.sleep(base_delay * (2 ** attempt))
    return result
```

**Cuándo aplicar:** Solo para I/O externo (APIs, red). Operaciones locales (leer archivo): un 404 no se soluciona reintentando.

#### 14.3.5 Mensaje al LLM: guiar la recuperación

El prompt del sistema puede instruir:

```
Si recibes error con retryable=true (timeout, rate_limit, service_unavailable), puedes sugerir reintentar.
Si recibes not_found, considera si crear el recurso o usar otra ruta.
Si recibes permission_denied, no insistas; informa al usuario.
```

El LLM "auto-corrige" a nivel de decisión: ruta alternativa, verificar permisos, etc. **Self-healing a nivel de agente**.

#### 14.3.6 Reflexión de errores: self-healing guiado por el LLM

Más allá del **reintento mecánico** con backoff exponencial, existe una estrategia avanzada: **reflexión de errores** (error reflection). En lugar de reintentar la misma llamada ciegamente, el orquestador inyecta el error estructurado en el historial y permite que el LLM **razone** sobre la causa antes de decidir la siguiente acción.

**Flujo:**
1. La herramienta falla y devuelve `ToolResult` con `error_code`, `error_message`, `retryable`.
2. En lugar de reintentar automáticamente (si `retryable`), el orquestador **no** reintenta en código.
3. Se añade el error al historial del LLM en formato estructurado: `{"success": false, "error_code": "not_found", "error_message": "File 'foo.txt' not found", "retryable": false}`.
4. El LLM recibe el error y, en su siguiente respuesta, puede:
   - **Razonar:** "El archivo no existe. Tal vez la ruta es incorrecta; probaré list_dir primero para ver el contenido del directorio."
   - **Adaptar:** Llamar a `list_dir` en lugar de repetir `read_file("foo.txt")`.
   - **Abortar:** "No encuentro el archivo; puede que no exista en este directorio."

**Ventaja sobre reintento mecánico:** Un timeout puede solucionarse con reintento. Un `not_found` no: reintentar `read_file("foo.txt")` una y otra vez es inútil. La reflexión permite que el agente **cambie de estrategia**: buscar en otro directorio, preguntar al usuario, o usar una herramienta distinta.

**Implementación:** Se desactiva el reintento automático para ciertos códigos (p. ej. `not_found`, `validation_error`) y siempre se pasa el error al LLM. Para `timeout` o `rate_limit` se puede usar una política híbrida: 1–2 reintentos mecánicos (rápidos, sin LLM) y, si fallan, pasar al LLM para que reflexione. El prompt del sistema ya incluye instrucciones para interpretar `error_code`; la reflexión explota esa capacidad.

**Ejemplo de prompt para reflexión:**

```
Cuando una herramienta devuelve un error:
1. Lee el error_code y error_message.
2. Razona brevemente: ¿por qué pudo fallar? (ruta incorrecta, recurso inexistente, permisos, etc.)
3. Decide: ¿reintentar con otros argumentos? ¿usar otra herramienta? ¿informar al usuario?
4. No repitas la misma llamada con los mismos argumentos si el error indica que no tiene sentido (ej. not_found).
```

**Resumen:** La reflexión de errores es **self-healing guiado por el LLM** — el agente usa su capacidad de razonamiento para recuperarse, no solo un backoff ciego.

#### 14.3.7 Conexión con el registry

El registry puede exponer un método que ya incluye el wrapper:

```python
def execute_safe(self, name: str, args: dict) -> ToolResult:
    """Ejecuta y devuelve ToolResult. Usar para orquestador con reintentos."""
    tool = self.get(name)
    if tool is None:
        return ToolResult(False, error_code=ToolErrorCode.VALIDATION, error_message=f"Herramienta desconocida: {name}")
    return _safe_execute(tool, args)
```

Así el orquestador llama `registry.execute_safe(name, args)` → `ToolResult` → `result.to_api_format()` para el historial del LLM.

---

### 14.4 Async/Await: no bloquear el sistema

El registry y el manejo de errores funcionan tanto en modo síncrono como asíncrono. En producción, sin embargo, las llamadas al LLM y a APIs externas son **I/O-bound**: el hilo espera respuestas de red. Convertir el flujo a **async** permite que un solo proceso atienda muchas peticiones concurrentes sin bloquearse.

#### 14.4.1 El problema: I/O bloqueante

En el bucle actual:

```python
response = client.responses.create(...)   # Bloquea 2–30 segundos
result = self.read_file(path)             # Bloquea si disco lento
```

Mientras esperamos la respuesta de la API de OpenAI o la lectura del disco, el **hilo de ejecución** está bloqueado. En un servidor web con un solo worker, una petición lenta bloquea a todas las demás. En un REPL interactivo, el usuario no puede hacer nada hasta que termine.

#### 14.4.2 Idea: concurrencia con async/await

En Python, `async`/`await` permiten **concurrencia cooperativa**: cuando una operación espera I/O (red, disco), se "cede" el control para que otras corrutinas avancen. No hay paralelismo real (un solo hilo), pero el hilo no se bloquea inútilmente esperando.

**Analogía:** En un restaurante con un solo camarero (hilo), si el camarero espera en la cocina hasta que el plato esté listo (bloqueante), no atiende a otros clientes. Con async, el camarero toma el pedido, deja la orden en cocina, y va a atender a otro cliente; cuando el plato está listo, vuelve. Varios clientes se atienden "en paralelo" sin más empleados.

#### 14.4.3 Sintaxis básica

```python
import asyncio

async def fetch_user(id: int) -> dict:
    # Simula llamada HTTP lenta
    await asyncio.sleep(1)
    return {"id": id, "name": "Alice"}

async def main():
    # Secuencial (2 segundos)
    u1 = await fetch_user(1)
    u2 = await fetch_user(2)

    # Concurrente (1 segundo): lanzar ambas y esperar
    t1 = asyncio.create_task(fetch_user(1))
    t2 = asyncio.create_task(fetch_user(2))
    u1, u2 = await asyncio.gather(t1, t2)

asyncio.run(main())
```

- `async def` define una corrutina.
- `await` cede el control hasta que la operación asíncrona termine.
- `asyncio.create_task` lanza una corrutina sin esperar; `asyncio.gather` espera varias.

#### 14.4.4 Cliente OpenAI asíncrono

OpenAI ofrece un cliente asíncrono:

```python
from openai import AsyncOpenAI

async def run_agent_turn(agent, messages):
    client = AsyncOpenAI()
    response = await client.responses.create(
        model=agent.model,
        input=messages,
        tools=agent.tools,
    )
    return response
```

`await client.responses.create(...)` no bloquea el hilo; mientras la API procesa, el event loop puede ejecutar otras corrutinas.

#### 14.4.5 Herramientas asíncronas

Si una herramienta llama a una API externa (HTTP, base de datos), debe ser asíncrona:

```python
import httpx

async def search_web_async(args: dict) -> dict:
    query = args.get("query", "")
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"https://api.example.com/search?q={query}")
        resp.raise_for_status()
        return {"results": resp.json()}
```

El orquestador debe soportar herramientas sync y async. Una función auxiliar:

```python
async def execute_tool_async(registry: ToolRegistry, name: str, args: dict) -> dict | ToolResult:
    """Invoca la herramienta; soporta sync y async. Retorna dict o ToolResult según diseño."""
    tool = registry.get(name)
    if tool is None:
        return {"error": f"Unknown: {name}"}
    out = tool.execute(args)
    if asyncio.iscoroutine(out):
        return await out
    return out
```

#### 14.4.6 Bucle principal asíncrono

El REPL o el endpoint HTTP deben correr en un event loop. Como `input()` es bloqueante, se ejecuta en otro hilo con `asyncio.to_thread` (Python 3.9+):

```python
async def run_enhanced_loop_async(agent: EnhancedAgent):
    client = AsyncOpenAI()
    while True:
        user_input = await asyncio.to_thread(input, "Tú: ")
        if not user_input.strip():
            continue
        # ...
        response = await client.responses.create(...)
        result = await execute_tool_async(agent.tool_registry, fn_name, args)
```

Para un servidor web (FastAPI, aiohttp), cada petición es una corrutina; el servidor gestiona muchas conexiones concurrentes con pocos workers.

#### 14.4.7 Timeouts en operaciones async

Un timeout evita que una operación colgue indefinidamente:

```python
try:
    response = await asyncio.wait_for(
        client.responses.create(...),
        timeout=60.0
    )
except asyncio.TimeoutError:
    # Devolver ToolResult con error_code=TIMEOUT, retryable=True
```

#### 14.4.8 Diagrama de secuencia: ejecución paralela con asyncio.gather

Cuando el LLM emite **múltiples tool-calls** en una sola respuesta (la API permite parallel tool calls), el orquestador puede ejecutarlos **en paralelo** con `asyncio.gather` en lugar de secuencialmente. El siguiente diagrama muestra el flujo:

```
   Usuario    Orquestador      Event Loop      LLM        Tool A      Tool B
      |            |                |           |            |            |
      |--mensaje-->|                |           |            |            |
      |            |---await create()--------->|            |            |
      |            |                |           | (procesa)  |            |
      |            |                |           |--response->|            |
      |            |                |           | (fn_a, fn_b, args)     |
      |            |                |           |            |            |
      |            | create_task(execute_tool("A",...))     |            |
      |            | create_task(execute_tool("B",...))     |            |
      |            |----------------await gather(t1,t2)------------------->|
      |            |                |           |            |            |
      |            |                |     (paralelo)  |--executar->|       |
      |            |                |                |<-resultado-|       |
      |            |                |                |       |--executar->|
      |            |                |                |       |<-resultado-|
      |            |<--------------------(t1,t2 listos)------------------|
      |            | append outputs to messages            |            |
      |            |---await create()--------->| (siguiente turno)       |
      |            |                |           |            |            |
```

**Código correspondiente:**

```python
async def process_parallel_tool_calls(
    registry: ToolRegistry,
    tool_calls: list[tuple[str, dict]],  # [(name, args), ...]
    max_retries: int = 2,
) -> list[ToolResult]:
    """Ejecuta múltiples tool-calls en paralelo con asyncio.gather."""
    tasks = [
        execute_with_retry_async(registry, name, args, max_retries)
        for name, args in tool_calls
    ]
    return await asyncio.gather(*tasks)
```

**Ventaja:** Si Tool A tarda 2 s y Tool B tarda 3 s, la ejecución secuencial tomaría 5 s; en paralelo, ~3 s (el máximo de ambos). El event loop cede el control mientras cada herramienta espera I/O, permitiendo que ambas avancen de forma entrelazada.

**Nota:** La API de OpenAI permite `parallel_tool_calls=True` en la creación de la respuesta; el modelo puede devolver varios `function_call` en un solo turno. El orquestador los recolecta, lanza las tareas con `create_task`, y agrupa los resultados antes de añadirlos al historial y re-enviar al LLM.

#### 14.4.9 Resumen didáctico

| Concepto | Significado |
|----------|-------------|
| **Bloqueante** | El hilo espera sin hacer nada; otras peticiones no avanzan. |
| **async/await** | Concurrencia cooperativa: durante I/O, otras corrutinas ejecutan. |
| **asyncio.gather** | Ejecuta varias corrutinas en paralelo y espera todas; tiempo total ≈ máximo de las latencias. |
| **AsyncOpenAI** | Cliente que usa `await`; no bloquea. |
| **Herramientas async** | Si llaman a redes/APIs, definirlas con `async def` y `await`. |
| **Event loop** | `asyncio.run()` o el loop del servidor; ejecuta todas las corrutinas. |

---

### 14.5 Integración: flujo unificado y esqueleto ProductionAgent

Los tres patrones confluyen en un **flujo de datos único**:

```
Usuario / API  →  Orquestador  →  LLM (AsyncOpenAI)
                    ↓
              ToolRegistry.execute_safe(name, args)
                    ↓
              _safe_execute(tool, args)  ← captura excepciones
                    ↓
              ToolResult  →  to_api_format()  →  historial del LLM
                    ↓
              (si retryable) execute_with_retry_async con backoff
```

El orquestador (1) recibe el registry por inyección, (2) invoca herramientas con reintentos cuando `retryable`, (3) ejecuta todo en async para no bloquear.

#### Esqueleto ProductionAgent

```python
async def _safe_execute_async(tool: Tool, args: dict) -> ToolResult:
    """Versión async de _safe_execute: soporta herramientas sync y async."""
    try:
        out = tool.execute(args)
        result = await out if asyncio.iscoroutine(out) else out
        return ToolResult(True, data=result if isinstance(result, dict) else {"result": result})
    except Exception as e:
        return _handle_tool_exception(e)

async def execute_with_retry_async(
    registry: ToolRegistry,
    name: str,
    args: dict,
    max_retries: int,
    base_delay: float = 0.5,
) -> ToolResult:
    """Ejecuta con reintentos. Integra los tres patrones."""
    tool = registry.get(name)
    if not tool:
        return ToolResult(False, error_code=ToolErrorCode.VALIDATION, error_message=f"Unknown tool: {name}")
    last = ToolResult(False, error_code=ToolErrorCode.UNKNOWN, error_message="")
    for attempt in range(max_retries + 1):
        last = await _safe_execute_async(tool, args)
        if last.success or not last.retryable:
            return last
        if attempt < max_retries:
            await asyncio.sleep(base_delay * (2 ** attempt))
    return last

class ProductionAgent:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        llm_client: AsyncOpenAI,
        max_retries_per_tool: int = 2,
        request_timeout: float = 60.0,
    ):
        self.tool_registry = tool_registry
        self.llm_client = llm_client
        self.max_retries = max_retries_per_tool
        self.request_timeout = request_timeout

    async def process_turn(self, messages: list) -> tuple[bool, str]:
        try:
            response = await asyncio.wait_for(
                self.llm_client.responses.create(
                    model="gpt-4o-mini",
                    input=messages,
                    tools=self.tool_registry.get_schemas(),
                ),
                timeout=self.request_timeout,
            )
        except asyncio.TimeoutError:
            # Timeout del LLM → ToolResult-like para el historial
            messages.append({"role": "user", "content": json.dumps({
                "success": False, "error": "LLM timeout", "retryable": True
            })})
            return True, ""  # forzar otro intento o mensaje de error

        final_message = ""
        for output in response.output:
            if getattr(output, "type", None) == "function_call":
                args = json.loads(getattr(output, "arguments", "{}"))
                result = await execute_with_retry_async(
                    self.tool_registry, output.name, args, self.max_retries
                )
                payload = result.to_api_format()
                messages.append({
                    "type": "function_call_output",
                    "call_id": getattr(output, "call_id", ""),
                    "output": json.dumps(payload),
                })
                return True, ""
            elif getattr(output, "type", None) == "message":
                parts = getattr(output, "content", [])
                texts = [getattr(p, "text", str(p)) for p in parts if hasattr(p, "text")]
                final_message = "\n".join(texts)
        return False, final_message
```

**Resumen de la integración:**
- **Inyección:** `tool_registry` y `llm_client` inyectados en el constructor.
- **Errores:** `ToolResult` con `error_code` y `retryable`; `execute_with_retry_async` reintenta con backoff.
- **Async:** Todo el flujo es `async`; el timeout del LLM se maneja con `asyncio.wait_for`.

---

### 14.6 Resumen: cómo encajan los patrones

| Patrón | Aborda | Piezas clave |
|--------|--------|--------------|
| **Inyección** | Acoplamiento herramientas–orquestador | `Tool`, `ToolRegistry`, `register`, `get_schemas`, `execute` / `execute_safe` |
| **Errores estructurados** | Errores planos no interpretables | `ToolResult`, `ToolErrorCode`, `_handle_tool_exception`, `_safe_execute`, `execute_with_retry` |
| **Reflexión de errores** | Reintento ciego ineficaz | Pasar error al LLM; que razone y adapte (cambio de herramienta, otros args); §14.3.6 |
| **Async** | I/O bloqueante | `AsyncOpenAI`, `async def`, `await`, `asyncio.gather`, `execute_with_retry_async` |

El **contrato unificado** es: las herramientas devuelven `dict` o lanzan; el wrapper produce `ToolResult`; el orquestador serializa con `to_api_format()` para el historial del LLM. Los reintentos ocurren antes de que el error llegue al LLM, salvo cuando el propio LLM timeout se inyecta como mensaje para dar otra oportunidad.

---

### 14.7 Referencias cruzadas

- Cap. 9: Estructura actual de `EnhancedAgent` y `process_response`.
- Cap. 6: Guardrails; los errores de validación de ruta pueden mapearse a `ToolErrorCode.VALIDATION`.
- §0.5D: Parada forzada; los reintentos tienen un límite análogo a $K_{\max}$.
- Cap. 13: Evaluación; las trazas con errores estructurados facilitan métricas de "tasa de recuperación".
- Anexo A.6: Modelado con mónadas y funtores (State monad, Kleisli); cierre formal de la arquitectura.

---

# ANEXOS

---

## Anexo A: Temas matemáticos avanzados

### A.1 Rate-Distortion y compresión óptima de memoria

El problema de comprimir el historial $M$ en un resumen $\tilde{M}$ de longitud acotada puede formularse en términos de **rate-distortion** (Shannon). Sea $d(M, \tilde{M})$ una función de distorsión (p. ej. pérdida de información relevante para la tarea). El problema óptimo es:

$$
\min_{P(\tilde{M}|M)} \mathbb{E}[d(M, \tilde{M})] \quad \text{sujeto a } I(M; \tilde{M}) \leq R
$$

donde $R$ es el presupuesto de bits (rate). La solución teórica involucra el **blow-up** de rate-distortion; en la práctica, la compresión por resúmenes o extracción de hechos es una aproximación heurística.

### A.2 Proceso de parada óptima

¿Cuándo debe el agente dejar de llamar herramientas y dar la respuesta final? En el marco de **optimal stopping** (e.g. problema del secretario), se busca maximizar $\mathbb{E}[U]$ donde $U$ depende del momento de parada $\tau$:

$$
\tau^* = \arg\max_\tau \mathbb{E}[U_\tau]
$$

El LLM no resuelve esto explícitamente; la decisión emerge del entrenamiento. Para el análisis del bucle como sistema de control, convergencia, oscilaciones y parada forzada, véase §0.5D.

### A.3 Espacios de embedding (desarrollo completo en §0.5B)

Muchos sistemas de agentes usan **embeddings** para recuperación semántica: los mensajes o documentos se proyectan a vectores $\mathbf{v} \in \mathbb{R}^d$ con un modelo de embeddings. La **Parte 0** incluye un capítulo exhaustivo (§0.5B) que cubre:
- Definición formal de embeddings y proyección a $\mathbb{R}^d$
- Espacios de alta dimensión y concentración de la medida
- Métricas: similitud coseno vs distancia euclidiana
- Hipótesis del colector (manifold hypothesis)
- Aplicación a memoria semántica y RAG

Nuestra memoria episódica actual no usa embeddings; es un buffer FIFO. Una extensión futura podría usar búsqueda por similitud (véase §0.5B.7) para recuperar episodios relevantes, implementando **memoria asociativa** y capacidades tipo RAG.

### A.4 Gradiente de la política (referencia para RL)

En sistemas que entrenan agentes con refuerzo, la política $\pi_\theta(a|s)$ se optimiza maximizando el retorno esperado. El **policy gradient** tiene la forma:

$$
\nabla_\theta J = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]
$$

donde $G_t$ es el retorno descontado desde $t$. Los LLMs que usamos están pre-entrenados; no hacemos fine-tuning en este proyecto, pero la estructura ReAct puede verse como una política estructurada que favorece cierto orden de acciones.

### A.5 Divergencia KL y alineamiento

Cuando se aplica temperature sampling, la distribución $P_\tau$ diverge de la original $P$. La **divergencia de Kullback-Leibler**:

$$
D_{\text{KL}}(P \| P_\tau) = \sum_y P(y) \log \frac{P(y)}{P_\tau(y)}
$$

mide cuánto se desvía. Para $\tau < 1$, $P_\tau$ está más concentrada que $P$; para $\tau > 1$, más dispersa.

### A.6 Modelado del agente con teoría de categorías (mónadas y funtores)

*Para lectores con formación en matemáticas, el flujo del agente (estado → acción → observación) admite una formalización elegante en términos de **mónadas** y **funtores**. Esto proporciona un cierre formal a la arquitectura de software del Capítulo 14.*

#### Categoría de tipos y efectos

En programación funcional, los **efectos** (estado mutable, I/O, aleatoriedad) se modelan con mónadas. Sea $\mathbf{Set}$ la categoría de conjuntos y funciones. Para el agente:

- **Estados:** $\mathcal{S}$ (historial, memoria, contexto).
- **Acciones:** $\mathcal{A}$ (texto, tool-calls).
- **Observaciones:** $\mathcal{O}$ (resultados de herramientas, respuestas del LLM).

El bucle del agente toma un estado $s$ y produce una nueva acción y un estado actualizado (o una observación intermedia). Esto se modela con el **State monad** clásico.

#### Mónada State

La mónada $\mathsf{State}(\mathcal{S})$ asocia a cada tipo $A$ el tipo de funciones $A \to \mathcal{S} \to (A \times \mathcal{S})$: "dado un valor de tipo $A$ y un estado, producir un resultado y un nuevo estado". Equivalentemente, $\mathsf{State}(A) = \mathcal{S} \to (A \times \mathcal{S})$.

**Unidad (return):** $\eta : A \to \mathsf{State}(A)$ envuelve un valor en una función que deja el estado intacto: $\eta(a) = \lambda s.\, (a, s)$.

**Multiplicación (join):** Dado $\mathsf{State}(\mathsf{State}(A))$ — es decir, $\mathcal{S} \to ((\mathcal{S} \to (A \times \mathcal{S})) \times \mathcal{S})$ — se colapsa a $\mathsf{State}(A)$ ejecutando la transformación de estado en secuencia.

#### Flujo del agente como Kleisli

El bucle ReAct (estado → acción → observación → nuevo estado) es una composición en la categoría de Kleisli de la mónada. Una **flecha Kleisli** $A \to B$ en la mónada $T$ es una función $A \to T(B)$. Para el agente:

- $\mathsf{act} : \mathcal{S} \to \mathsf{State}(\mathcal{A} \times \mathcal{O})$: dado estado, produce (acción, observación) y actualiza estado.
- $\mathsf{observe} : \mathcal{A} \to \mathsf{State}(\mathcal{O})$: dado acción (tool-call), ejecuta y devuelve observación actualizando estado.
- $\mathsf{decide} : \mathcal{S} \to \mathsf{State}(\mathcal{A})$: la política $\pi$; dado estado, muestrea acción (estocástico: en rigor usaríamos una mónada que combine State con Probability).

La composición $\mathsf{decide} \ggg \mathsf{observe} \ggg \mathsf{update}$ (donde $\ggg$ es el operador Kleisli) da un paso del bucle. El bucle completo es el **fixpoint** de esta composición hasta que la acción sea `message`.

#### Funtores y composición

Un **funtor** $F : \mathcal{C} \to \mathcal{D}$ preserva estructura: objetos a objetos, morfismos a morfismos, identidades e identidades, composición a composición. La mónada $T$ es un funtor $\mathbf{Set} \to \mathbf{Set}$ con transformaciones naturales $\eta$ (unidad) y $\mu$ (multiplicación) que satisfacen las leyes de mónada.

El **funtor de herramientas** (Cap. 14): dado un conjunto de herramientas $\mathcal{F}$, el registro $\mathsf{ToolRegistry}$ puede verse como un funtor que a cada "configuración" de herramientas asocia el orquestador que las usa. Cambiar de herramienta = cambiar de objeto en la categoría de configuraciones; el mismo orquestador (morfismo) opera en cualquier configuración inyectada.

#### Resumen

| Concepto | Papel en el agente |
|----------|--------------------|
| **State monad** | $\mathcal{S} \to (A \times \mathcal{S})$; modela paso de estado a través del bucle |
| **Flechas Kleisli** | `decide` $;$ `observe` $;$ `update` = un paso del bucle |
| **Mónada** | Encapsula el efecto "estado mutable" de forma composable |
| **Funtor** | El registry mapea configuraciones; la inyección es funtorial |

**Referencia:** Para un desarrollo completo, véase "Monads for functional programming" (Wadler) y la relación con el patrón de inyección de dependencias en lenguajes funcionales (Cap. 14).

---

## Glosario

| Término | Significado |
|---------|-------------|
| **Argmax** | Argumento que maximiza: $\arg\max_x f(x) = \{ x : f(x) = \max_y f(y) \}$ |
| **Atención** | Mecanismo que pondera posiciones por relevancia: $A = \mathrm{softmax}(QK^T/\sqrt{d_k})$, Output $= AV$ (§0.5C) |
| **Agente** | Sistema que percibe, decide y actúa de forma autónoma |
| **Convergencia (bucle)** | Existencia de $K < \infty$ tal que el agente emite respuesta final (texto) en el paso $K$; no garantizada sin parada forzada (§0.5D) |
| **Context window** | Cantidad máxima de tokens que el modelo puede considerar |
| **Episodio** | Unidad de memoria: acción + resultado |
| **Guardrail** | Restricción de seguridad que bloquea acciones peligrosas |
| **LLM** | Large Language Model, modelo de lenguaje grande |
| **Path traversal** | Ataque que usa `../` para acceder a rutas no permitidas |
| **Prompt** | Texto de entrada que se envía al modelo |
| **ReAct** | Patrón Reasoning + Acting para agentes |
| **Token** | Unidad de texto (palabra o sub-palabra) que procesa el modelo |
| **Tool / Herramienta** | Función que el agente puede invocar |
| **Tool-calling** | Mecanismo por el que el modelo solicita ejecutar herramientas |
| **Embedding** | Función $\mathbf{e}: \mathcal{X} \to \mathbb{R}^d$ que asigna texto/tokens a vectores; la similitud semántica se traduce en proximidad geométrica (§0.5B) |
| **Entropía $H$** | $H(X) = -\sum_x P(x) \log P(x)$; mide incertidumbre de una variable aleatoria |
| **Información mutua $I$** | $I(X;Y) = H(X) - H(X|Y)$; información compartida entre $X$ e $Y$ |
| **MDP** | Proceso de Decisión de Markov: estados, acciones, transiciones, recompensas |
| **Policy $\pi$** | Mapeo $\pi(a|s)$ que asigna probabilidad a cada acción dado el estado |
| **RAG** | Retrieval-Augmented Generation: recuperar documentos por similitud de embeddings e inyectarlos en el contexto del LLM antes de generar (§0.5B.7) |
| **Pass@k** | Métrica de evaluación: probabilidad de que al menos 1 de $k$ intentos pase los tests; estimador imparcial $\widehat{\text{Pass@}k} = 1 - \binom{c-n}{k}/\binom{c}{k}$ con $n$ aciertos en $c$ muestras (§13) |
| **LLM-as-a-judge** | Protocolo de evaluación donde un modelo LLM superior evalúa la calidad o racionalidad de las salidas de un agente; útil cuando no hay ground truth verificable (§13) |
| **Robustez (evaluación)** | Estabilidad del éxito del agente ante variación de hiperparámetros (ej. temperatura $\tau$); se mide con la curva $S(\tau)$ y la sensibilidad $\partial S/\partial \tau$ (§13) |
| **Inyección de dependencias** | Patrón donde el orquestador recibe las herramientas (o clientes) desde fuera; permite intercambiar implementaciones sin modificar el código (§14) |
| **ToolRegistry** | Registro nombre → herramienta; el orquestador invoca `registry.execute(name, args)` sin conocer las herramientas concretas (§14) |
| **ToolResult** | Resultado estructurado de una herramienta: `success`, `error_code`, `retryable`; permite al LLM y al código reaccionar de forma diferenciada (§14) |
| **Self-healing** | Capacidad del sistema de recuperarse ante fallos transitorios mediante reintentos y decisión del agente según `error_code` (§14) |
| **Reflexión de errores** | Self-healing avanzado: el LLM recibe el error estructurado, razona sobre la causa y adapta la estrategia (otra herramienta, otros args) en lugar de reintentar mecánicamente (§14.3.6) |
| **async/await** | Concurrencia cooperativa en Python; durante I/O el hilo no bloquea y otras corrutinas pueden ejecutarse (§14) |

---

## Referencias

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Paper original del patrón ReAct
- [Evaluating Large Language Models Trained on Code (HumanEval, Pass@k)](https://arxiv.org/abs/2107.03374) — Origen de la métrica Pass@k
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — Evaluación de LLM como juez
- [OpenAI API Documentation](https://platform.openai.com/docs) — Documentación oficial
- [How to Build an Agent (Ampcode)](https://ampcode.com/how-to-build-an-agent) — Inspiración del proyecto base
- [LangChain](https://www.langchain.com/) — Framework de agentes y cadenas
- [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT) — Ejemplo de agente autónomo

---

## Diagrama del ciclo ReAct

```
     ┌──────────────────────────────────────────────────────────┐
     │                      CICLO REACT                         │
     └──────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
   ┌──────────┐         ┌──────────┐         ┌──────────┐
   │ PENSAR   │   ──▶   │ ACTUAR   │   ──▶   │ OBSERVAR │
   │(Reason)  │         │(Tool)    │         │(Result)  │
   └──────────┘         └──────────┘         └────┬─────┘
         ▲                    │                    │
         │                    │                    │
         │              ¿Necesita más?              │
         │                    │                    │
         │              Sí ───┴─── No              │
         │               │         │               │
         └───────────────┘         └───────────────▶│
                                                    ▼
                                            ┌──────────────┐
                                            │ FINALIZAR    │
                                            │(Respuesta)   │
                                            └──────────────┘
```

---

## Changelog de la documentación

| Fecha | Cambios |
|-------|---------|
| 2025-03-22 | Versión inicial. 13 secciones, teoría + implementación. |
| 2025-03-22 | v1.1: Parte 0 (fundamentos matemáticos), ecuaciones en todos los capítulos, Anexo A (temas avanzados), extensión para lectores con formación en física. |
| 2025-03-22 | v1.2: Parte 0 muy expandida: regla de la cadena con derivación, Bayes, entropía y propiedades, Jensen, info mutua, KL, rate-distortion, cadenas de Markov, MDP completo con Bellman, logits/softmax, top-k/top-p, ejemplo numérico. |
| 2025-03-22 | v1.3: §0.5B "El salto de la probabilidad a la geometría": embeddings, espacios de alta dimensión, similitud coseno vs euclidiana, hipótesis del colector, memoria semántica, RAG. |
| 2025-03-22 | v1.4: Revisión crítica de §0.5B: fórmulas de concentración de medida, derivación dist. vectores aleatorios, Cauchy-Schwarz, embeddings estáticos vs contextuales, código Python (cosine_similarity, top_k, RAG, intrinsic_dim), manifold con definición formal, implementaciones prácticas. |
| 2025-03-22 | v1.5: §0.5C Mecánica estadística de la atención: Scaled Dot-Product Attention, Q/K/V, analogía Boltzmann (energía $E_{ij}$, función de partición), factor $\sqrt{d_k}$, complejidad $O(L^2 d)$, código Python, Multi-Head. §0.14 actualizado con desglose de coste. |
| 2025-03-22 | v1.6: Revisión crítica de §0.5C: §0.5C.0 tabla softmax atención vs logits, §0.5C.2b encoding posicional, §0.5C.3b máscara causal (decoder-only), derivación Var($S_{ij}$), saturación y vanishing gradient, entropía $H(A_{i,:})$, conexión §0.5B, residual+LayerNorm, ejemplo numérico, código con causal=True. |
| 2025-03-22 | v1.7: §0.5D Teoría de control y estabilidad del agente: bucle como feedback loop, diagrama de bloques, convergencia no garantizada, parada forzada $K_{\max}$, oscilaciones y bucles patológicos, analogía Lyapunov, estados de error, diagrama de estados (autómata), mitigaciones. |
| 2025-03-22 | v1.8: Capítulo 13 "Evaluación rigurosa de agentes": métricas deterministas (exact match, Pass@k con estimador imparcial y código Python), LLM-as-a-judge (formalización, sesgos, racionalidad), análisis de robustez frente a $\tau$ (curva $S(\tau)$, sensibilidad, protocolo, código), tabla resumen, referencias cruzadas. Glosario: Pass@k, LLM-as-a-judge, Robustez. |
| 2025-03-22 | v1.9: Cap 13 ampliado en profundidad: Pass@k con derivación del estimador imparcial vs naive sesgado, agregación sobre problemas, trade-off coste–precisión, IC bootstrap; criterios deterministas con BLEU/ROUGE/edit distance, ejecución en sandbox; evaluación de trazas (criterios compuestos, ejemplo concreto); LLM-as-a-judge con diseño de prompt, ejemplo completo, tabla de sesgos; robustez con $H_\tau$, elasticidad, banda al 90%, múltiples dimensiones (prompt, ruido, $K_{\max}$), código con IC; estructura eval/, script con IC Wilson. |
| 2025-03-22 | v1.10: Cap 14 "Arquitectura de software — El agente en producción": inyección de dependencias con ToolRegistry y patrón plug-in; manejo de errores estructurado (ToolResult, ToolErrorCode, mapeo de excepciones); self-healing con reintentos y backoff exponencial; async/await con AsyncOpenAI, herramientas async, event loop, timeouts; esqueleto ProductionAgent integrando los tres patrones. Glosario: Inyección de dependencias, ToolRegistry, ToolResult, Self-healing, async/await. |
| 2025-03-22 | v1.11: Revisión crítica Cap 14: hilo conductor único, flujo de datos unificado (Tool→exception/dict→_safe_execute→ToolResult→to_api_format); definición explícita de _safe_execute y conexión con registry.execute_safe; _safe_execute_async para async; ProductionAgent completo con manejo de timeout del LLM; diagrama de flujo en §14.5; tabla resumen §14.6; transiciones entre secciones; eliminación de referencias indefinidas y redundancias. |
| 2025-03-22 | v1.12: Parte 0 reforzada: §0.5C.4.4 visualización geométrica de Q,K,V como recuperación Hopfield/Boltzmann (diagrama, tabla memoria asociativa); §0.5D.3.2 análisis Lyapunov formal ($V(s,k)$, condiciones de estabilidad, proposición sobre ausencia de ciclos infinitos, límites); Anexo A.6 modelo del agente con teoría de categorías (State monad, Kleisli, funtores, conexión Cap 14). |
| 2025-03-22 | v1.13: Cap 14 ampliado: §14.4.8 diagrama de secuencia asíncrono (ejecución paralela con asyncio.gather, parallel tool calls, código process_parallel_tool_calls); §14.3.6 reflexión de errores (self-healing guiado por LLM: razonar sobre error_code, adaptar estrategia, prompt para reflexión). Tabla resumen actualizada. |
