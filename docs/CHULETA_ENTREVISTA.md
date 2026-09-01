# Chuleta — defensa oral de la prueba técnica

**Autor:** Justin Alejandro Zambrano Lucas
**Repo:** `prueba-ia-justin-zambrano`
**Stack:** FastAPI · SQLAlchemy · scikit-learn · TensorFlow/Keras · LangGraph + Gemini · MCP · JWT · Docker

> Guion para la entrevista de validación. La prueba avisa que hay que **explicar el código
> y modificarlo en vivo**, así que la sección 9 (modificaciones en vivo) es la más importante:
> practícala antes, con el proyecto levantado.

---

## 0. Elevator pitch (45 segundos)

> Construí un sistema de atención al cliente para telecomunicaciones con cuatro piezas:
> **(1)** un clasificador de tickets en TECH/BILL/PLAN/CNCL/OTHR con scikit-learn,
> **(2)** un modelo de churn con probabilidades y explicabilidad,
> **(3)** dos redes en Keras — una LSTM de sentimiento y una red multi-input que estima el
> tiempo de resolución — y **(4)** un agente conversacional en LangGraph que **usa esos
> modelos** para decidir la intención, detectar frustración y escalar a un humano.
> Todo se expone en una API REST con JWT por roles y en un servidor MCP para que otros
> agentes de IA consuman las mismas capacidades. Corre entero con `docker compose up`,
> y hay un demo web en `/demo` para verlo funcionando sin tocar Swagger.

**Si preguntan por qué es interesante:** la integración. Los modelos no son cuatro scripts
sueltos: el agente los consume en tiempo real y el resultado cambia el flujo del grafo
(escalar o no escalar).

---

## 1. Mapa mental del repo

```
app/
├── core/          config (Pydantic Settings), database, security (JWT), exceptions, seed
├── models/        entidades SQLAlchemy — el esquema del enunciado
├── schemas/       Pydantic v2: request/response + validaciones
├── repositories/  acceso a datos (patrón Repository)
├── services/      lógica de negocio
├── api/v1/        routers REST: auth, customers, tickets, ml, agent
├── mcp/           servidor MCP (capabilities, resources, tools/execute)
├── agent/         grafo LangGraph: state.py, nodes.py, graph.py, llm.py
└── ml_runtime/    capa de inferencia compartida (API y agente usan la misma)
ml_training/       Parte 1 — scikit-learn
dl_training/       Parte 2 — TensorFlow/Keras
saved_models/      artefactos entrenados + reportes JSON + gráficas
sql/init.sql       DDL + 1 función + 1 stored procedure PL/pgSQL
static/            portal del cliente (/portal) y panel técnico (/demo)
tests/             44 tests (pytest + TestClient + SQLite)
```

**La frase para explicar la arquitectura en una línea:**
> Arquitectura por capas: `api → services → repositories → models`. El router solo valida y
> delega, el service tiene las reglas de negocio, el repository es el único que toca la BD.
> `ml_runtime/` es transversal: lo consumen tanto la API como el agente, así los modelos se
> cargan una sola vez por proceso (`lru_cache`).

**Si preguntan por DDD:** el enunciado permite "arquitectura por capas **o** DDD". Elegí capas
porque para el alcance de la prueba un dominio con entidades puras y ports agregaba
indirección sin beneficio real. Sé explicar la diferencia: en DDD el dominio no conocería
SQLAlchemy; aquí las entidades SQLAlchemy sí son el modelo.

---

## 2. Recorrido de demo (5 minutos)

1. `docker compose up --build` → API en `:8000`, Postgres en `:5432`.
2. **`http://localhost:8000/portal`** con `cliente@telecom.com` / `Cliente123!`.
   Sale un chat a pantalla completa: **el cliente no ve formularios ni métricas.**
3. Escribir *"El internet se corta cada media hora desde el lunes, ya reinicié el router"*.
   El agente responde y aparece una tarjeta: **Solicitud #1 · Soporte Técnico · respuesta
   estimada 11.6 h**. Explicar que el ticket lo abrió el grafo, no el navegador.
4. Insistir con el mismo problema → *"La sumé a tu solicitud #1, que sigue abierta"*.
   **No se duplica el ticket**: es una regla de negocio, no un accidente.
5. Salir y entrar con `agente@telecom.com` / `Agente123!` → **la misma URL muestra otra
   cosa**: la bandeja de soporte, con ese ticket ya clasificado y etiquetado con
   sentimiento, riesgo de churn y tiempo estimado. Tomar el caso o resolverlo.
6. `/demo` para la vista técnica (probabilidades por clase, MCP con su sobre JSON-RPC).

> **El punto que hay que dejar claro:** un solo mensaje del cliente activa los cuatro
> modelos. El clasificador decide la categoría y el equipo, el de sentimiento decide la
> prioridad y si se escala, el de churn entra en esa misma decisión, y el de tiempo de
> resolución produce la promesa de respuesta. El cliente ve una frase amable; el agente
> ve el caso ya priorizado.

> Si preguntan *"¿esto es solo maquetación?"*: el portal llama a `/agent/chat` y
> `/tickets/queue`, y todo queda en Postgres —`GET /api/v1/tickets?customer_id=1` lo
> demuestra—. La creación del ticket vive en `app/agent/nodes.py`, no en el HTML.

## 3. Parte 1 — ML con scikit-learn

**Qué hice:** `ml_training/train_ticket_classifier.py` y `ml_training/train_churn_model.py`.

| Requisito | Dónde está |
|---|---|
| Preprocesamiento de texto | `app/ml_runtime/text_preprocessing.py:28` (`normalize_text`) |
| 2 modelos comparados | LogisticRegression vs LinearSVC calibrado |
| `sklearn.pipeline.Pipeline` | `build_pipelines()` |
| CV 5 folds | `StratifiedKFold(n_splits=5)` + `cross_val_score` |
| Métricas por categoría | `classification_report(output_dict=True)` → JSON |
| Matriz de confusión | `saved_models/ml/confusion_matrix_tickets.png` |
| joblib | `saved_models/ml/ticket_classifier.joblib` |

**Preguntas probables**

- *¿Por qué TF-IDF y no embeddings?* Con ~800 ejemplos sintéticos y 5 clases, TF-IDF + un
  lineal es más fuerte y explicable; los embeddings pedirían más datos. Además el enunciado
  pedía comparar 2 modelos, y aquí la comparación es honesta.
- *¿Por qué calibrar el LinearSVC?* Porque `LinearSVC` no tiene `predict_proba`, y el
  enunciado exige devolver la probabilidad de cada clase. `CalibratedClassifierCV` con
  sigmoide (Platt) lo resuelve.
- *¿Por qué `class_weight="balanced"`?* Las categorías no están perfectamente balanceadas y
  evita que el modelo se sesgue hacia la mayoritaria, sin tener que hacer resampling.
- *Tu accuracy es 1.00, ¿no es sospechoso?* **Sí, y lo digo yo primero.** Los datos son
  sintéticos, generados por plantillas por categoría, así que las clases son casi
  separables: es un techo del dataset, no del modelo. Con datos reales esperaría
  solapamiento (sobre todo BILL vs PLAN) y métricas bastante menores. Está escrito en el
  README, sección 9.
- *Feature engineering en churn:* dos features derivados —
  `charge_per_tenure = total_charges / tenure_months` (cliente "caro" en relación a su
  antigüedad) y `tickets_per_tenure` (densidad de problemas). El primero es la feature más
  importante del modelo (0.24).
- *¿Nulos y desbalanceo?* `SimpleImputer` (mediana para numéricas, moda para categóricas)
  dentro del `ColumnTransformer`, y `class_weight="balanced"` para el 90/10 de churn.
- *¿Por qué AUC-ROC 0.73 y no más?* Los datos de churn los generé **con ruido a propósito**,
  justamente para no repetir el 1.00 del clasificador. Un AUC de 0.73 sobre clases 90/10 es
  un resultado realista.

---

## 4. Parte 2 — Deep Learning con Keras

**2.1 Sentimiento** (`dl_training/train_sentiment_model.py`):
`Embedding(mask_zero=True) → LSTM(64, dropout=0.2) → Dense(32, relu) → Dropout(0.4) → Dense(3, softmax)`.
Tokenizer con `num_words=10000`, `pad_sequences(maxlen=200)`, y los tres callbacks pedidos:
`EarlyStopping(patience=3, restore_best_weights=True)`, `ModelCheckpoint(save_best_only=True)`,
`ReduceLROnPlateau(factor=0.5, patience=2)`. Exportado en `.keras`.

**2.2 Tiempo de resolución** (`dl_training/train_resolution_time_model.py`): API funcional con
5 entradas — texto (Embedding + GlobalAveragePooling), categoría one-hot, prioridad one-hot,
y hora/día en **codificación cíclica** (`sin`/`cos`). Salida `Dense(1, linear)`, pérdida MSE.
MAE ≈ 1.75 h, RMSE ≈ 2.53 h, R² ≈ 0.58.

**Preguntas probables**

- *¿Por qué codificación cíclica para la hora?* Porque la hora 23 y la hora 0 están a una
  hora de distancia, no a 23. Con `sin`/`cos` la red ve esa continuidad; con un entero
  crudo, no. Lo mismo con el día de la semana (domingo↔lunes).
- *¿Por qué `mask_zero=True`?* Para que el padding no contamine el estado de la LSTM.
- *¿Por qué `sparse_categorical_crossentropy`?* Las etiquetas van como enteros
  (`LabelEncoder`), no one-hot; evita expandirlas.
- *¿Por qué LSTM y no un Transformer?* Volumen de datos y tiempo de la prueba. El enunciado
  pedía "al menos una capa LSTM o GRU, o CNN para texto".
- *¿Por qué el modelo de resolución tiene peor R² que el de sentimiento accuracy?* Porque le
  inyecté ruido log-normal al generar los tiempos: un tiempo de resolución real no es
  determinista. Un R² de 0.58 sobre datos ruidosos es honesto.
- *¿Dónde se usa el 2.2?* En `POST /api/v1/ml/predict-resolution-time`, en la tool MCP
  `predict_resolution_time`, y en el panel correspondiente del demo.

---

## 5. Parte 3 — Agente LangGraph

**Estado** (`app/agent/state.py`): `TypedDict` con `messages`, `customer_id`, `intent`,
`context`, `escalate`, `response` y `error` (este último para degradar sin romper el grafo).

**Grafo** (`app/agent/graph.py:37`), los 7 nodos del enunciado:

```
START → classify_intent
          ├─ (saludo/despedida) ──────────────→ generate_response → END
          └─ get_customer_info
                ├─ handle_account_query      ┐
                ├─ handle_technical_support  ├→ check_escalation → generate_response → END
                └─ handle_general_info       ┘
```

Dos `add_conditional_edges`: `_route_after_intent` (línea 23) ataja saludos y despedidas sin
pasar por BD ni escalamiento; `_route_by_intent` (línea 29) enruta por intención.

**Integración con ML (la pregunta estrella — 3.2 del enunciado):**

| Modelo | Dónde entra | Para qué |
|---|---|---|
| Clasificador de tickets | `classify_intent` | La categoría (TECH/BILL/…) se mapea a la intención vía `CATEGORY_TO_INTENT` |
| Sentimiento (LSTM) | `classify_intent` | `is_frustrated` = negative con confianza > 0.6 (`FRUSTRATION_THRESHOLD`) |
| Churn | `get_customer_info` | Si el cliente está identificado, se calcula su riesgo |

**Reglas de escalamiento** (`check_escalation`): escala si el cliente **pide un humano**
explícitamente, si **hay frustración**, si es una **cancelación** (categoría CNCL) o si el
**riesgo de churn es alto**. Además guarda `escalation_reason` en texto, que es lo que se ve
en el demo.

**Preguntas probables**

- *¿Cómo recuerda la conversación?* No uso el checkpointer de LangGraph: la fuente de verdad
  es la tabla `agent_session` (que pedía el esquema del enunciado). En cada turno,
  `AgentService.chat` (`app/services/agent_service.py:19`) reconstruye `messages` desde la BD,
  invoca el grafo y persiste los dos mensajes nuevos. **Ventaja:** la conversación sobrevive
  a un reinicio del proceso y es consultable por `GET /agent/sessions/{id}`.
  **Trade-off:** no aprovecho las features de checkpointing de LangGraph (time-travel, replay).
- *¿Cómo inyectas la sesión de BD en los nodos?* Por el `RunnableConfig` de LangGraph:
  `graph.invoke(state, config={"configurable": {"db": self.db}})`, y cada nodo la lee de
  `config["configurable"]["db"]`. Así los nodos no importan la sesión global ni dependen de FastAPI.
- *¿Qué pasa si el LLM no está disponible?* `app/agent/llm.py` devuelve `None` si no hay
  `GOOGLE_API_KEY`, y `generate_response` cae a respuestas por plantilla. El sistema y los
  tests nunca dependen de una llamada de red externa. **Es una decisión, no un olvido.**
- *¿Y si un modelo falla?* Cada nodo está envuelto en `try/except`: registra el error en
  `state["error"]` y sigue con valores por defecto. El grafo nunca aborta a mitad.
- *¿Por qué Gemini y no OpenAI?* Es la API key que tengo. El wrapper está aislado en
  `llm.py`, cambiar de proveedor es tocar un archivo.

---

## 6. Parte 4 — API REST y MCP

**28 endpoints**: los 24 que pide el enunciado (20 bajo `/api/v1` + 4 de MCP) más
`predict-resolution-time`, `GET /auth/me` (el portal necesita saber el rol y el cliente
de la sesión: el JWT solo lleva email y rol) y `GET /tickets/queue` (la bandeja del agente).

- **Validaciones Pydantic:** email con `EmailStr`; teléfono con regex `^09\d{8,}$`
  (`app/schemas/customer.py`); descripción de ticket `min_length=20, max_length=500`.
- **JWT:** access (30 min) + refresh (7 días), con `type` dentro del payload para que un
  refresh token no sirva como access token (`app/core/security.py:72`). Roles vía
  `require_roles(...)` como dependencia del router.
- **Errores estandarizados:** `app/core/exceptions.py` — todo sale como
  `{"error": {"code": ..., "message": ..., "details": ...}}` con el HTTP correcto.
- **Eliminación lógica:** `is_active` + `deleted_at`; el repositorio filtra por `is_active`
  por defecto (`app/repositories/customer_repository.py:56`).
- **SQL:** `sql/init.sql` con la función `fn_customer_churn_summary(customer_id)` y el
  procedure `sp_refresh_category_avg_resolution()`. Se cargan solos al crear el volumen de
  Postgres en docker-compose.

**MCP:** `GET /mcp/capabilities`, `GET /mcp/resources`, `GET /mcp/resources/{id}`,
`POST /mcp/tools/execute`, con las 5 tools pedidas + `predict_resolution_time`.

- *¿Por qué un error de tool devuelve HTTP 200?* Porque en MCP el error de ejecución es parte
  del **resultado** (`result.isError = true`), no un fallo de transporte. Un 500 significaría
  que el servidor MCP se cayó, que es distinto a "la tool falló".

---

## 7. Preguntas transversales

- *¿Por qué Postgres y no SQLite?* Porque el requisito de stored procedure pedía PL/pgSQL
  real. Los tests sí usan SQLite, por velocidad y para no depender de infraestructura.
- *¿Los tests cubren qué?* 44 tests sobre los endpoints críticos: auth (incluido rechazo de
  refresh token como access), CRUD de clientes con sus validaciones, tickets, los 4 modelos,
  el agente (incluidos 401 y 403 por rol) y MCP.
- *¿Cómo garantizas que el modelo carga en producción?* `ml_runtime` cachea con `lru_cache`
  y lanza excepciones tipadas (`*Unavailable`) que la API traduce a **503**, no a 500: si
  falta un artefacto, el error dice qué script hay que correr.
- *¿Y si te piden desplegarlo?* La imagen ya está lista y probada; faltaría un registry y
  variables de entorno reales (`JWT_SECRET`, `DATABASE_URL`, `GOOGLE_API_KEY`).

---

## 8. Debilidades conocidas — dilas tú antes de que las encuentren

Reconocer un límite con la solución al lado suma; que te lo descubran, resta.

1. **Datos sintéticos ⇒ métricas optimistas** en tickets y sentimiento. Ya explicado arriba.
2. **`avg_satisfaction` está fijo en 3.5** al predecir churn de un cliente concreto
   (`app/services/customer_service.py:60`), pese a ser la 2ª feature más importante. La
   función SQL `fn_customer_churn_summary` ya calcula el promedio real: cablearla es el
   siguiente paso natural. **Buena respuesta si preguntan "¿qué mejorarías?"**.
3. **El `config` de LangGraph no llegaba a los nodos** hasta que lo detecté al implementar
   la apertura de tickets: la firma `config: RunnableConfig | None = None` hace que
   LangGraph **no** inyecte el config (lo decide inspeccionando la firma). El síntoma era
   silencioso: `get_customer_info` caía siempre en "sin sesión de base de datos" y el
   agente nunca leía datos del cliente. Corregido anotando `config: RunnableConfig`.
   **Si preguntan por un bug difícil que hayas encontrado, cuenta este.**
4. **Sin Alembic.** Las tablas se crean con `Base.metadata.create_all`; para producción haría
   falta migraciones versionadas.
5. **Sin checkpointer de LangGraph** (decisión consciente, ver sección 5).
6. **Sin rate limiting ni paginación en tickets** (clientes sí tiene paginación).

---

## 9. Modificaciones en vivo — ensáyalas

Lo más probable es que pidan algo pequeño y transversal. Practica estas cuatro; todas tocan
varias capas, que es justo lo que quieren ver que entiendes.

**a) "Agrega un campo nuevo al cliente"** (p. ej. `city`)
1. `app/models/customer.py` → columna SQLAlchemy.
2. `app/schemas/customer.py` → campo en `CustomerBase` (+ `CustomerUpdate`).
3. `sql/init.sql` → columna en el DDL.
4. Reiniciar: `Base.metadata.create_all` no altera tablas existentes → borrar el volumen
   (`docker compose down -v`) o añadir la columna a mano. **Saber esto de antemano evita el
   momento incómodo.**

**b) "Agrega una categoría de ticket"** (p. ej. `SALE`)
1. `Literal` en `app/schemas/ticket.py`.
2. `CATEGORIES` en `ml_training/train_ticket_classifier.py` + plantillas en
   `scripts/generate_synthetic_data.py`.
3. Reentrenar. **Punto clave:** el modelo solo predice clases que vio en entrenamiento;
   añadir el `Literal` no basta.

**c) "Cambia la regla de escalamiento"** — todo vive en `check_escalation`
(`app/agent/nodes.py`). Ejemplo: bajar `FRUSTRATION_THRESHOLD`, o escalar solo si además
`churn_risk == "high"`. Es la edición más fácil y la más vistosa.

**d) "Agrega un endpoint"** — el camino siempre es el mismo:
schema en `app/schemas/` → método en el service → ruta con `require_roles` → test.
Lo hice para `predict-resolution-time`; puedes mostrar ese commit como plantilla.

**e) "Agrega una tool MCP"** — tres puntos en `app/mcp/router.py`: el `MCPToolDescriptor`,
la función `_tool_x`, y la entrada en `TOOL_HANDLERS`.

---

## 10. Datos duros

**Credenciales**

| Email | Password | Rol |
|---|---|---|
| admin@telecom.com | Admin123! | admin |
| agente@telecom.com | Agente123! | agent |
| cliente@telecom.com | Cliente123! | customer |

**URLs:** `/portal` (cliente o agente, según el rol) · `/demo` (panel técnico) · `/docs` (Swagger) · `/` (health)

**Métricas para citar de memoria**

- Clasificador: F1-macro CV 5-fold 0.998; accuracy test 1.00 *(dataset sintético)*.
- Churn: AUC-ROC 0.731, AP 0.273; top feature `charge_per_tenure` (0.24).
- Sentimiento: accuracy test 1.00 *(sintético)*.
- Resolución: MAE 1.75 h, RMSE 2.53 h, R² 0.58.

**Comandos**

```bash
docker compose up --build              # stack completo
pytest tests/ -v                       # 44 tests
python ml_training/train_churn_model.py    # reentrenar
docker exec -it telecom_support_db psql -U postgres -d telecom_support \
  -c "SELECT * FROM fn_customer_churn_summary(1);"   # demostrar la función SQL
```

---

## 11. Si preguntan por el uso de IA

El README lo dice de frente, así que la respuesta debe ser tranquila y consistente:

> Usé asistencia de IA para acelerar el desarrollo, igual que usaría documentación o un
> colega. Las decisiones técnicas — arquitectura por capas, calibrar el SVC para tener
> probabilidades, codificación cíclica de la hora, persistir la conversación en BD en vez de
> usar el checkpointer, degradar a plantillas sin API key — las puedo justificar una por una,
> y también sé cuáles son los puntos débiles del proyecto.

Y luego demuéstralo con la sección 8 y con una modificación en vivo. **La defensa no es el
discurso: es poder tocar el código delante de ellos.** Si no sabes algo, dilo y explica cómo
lo averiguarías; es mejor que improvisar una explicación falsa sobre tu propio repo.
