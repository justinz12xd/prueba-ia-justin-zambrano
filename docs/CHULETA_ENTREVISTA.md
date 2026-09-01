# Chuleta — defensa oral de la prueba técnica

**Autor:** Justin Alejandro Zambrano Lucas
**Repo:** `prueba-ia-justin-zambrano`
**Stack:** FastAPI · SQLAlchemy · scikit-learn · TensorFlow/Keras · LangGraph + Gemini · MCP · JWT · Docker

> **¿Te preguntan dónde está algo?** Ve directo a la **sección 12**: mapa de cada
> requisito del enunciado con archivo y línea.
> Guion para la entrevista de validación. La prueba avisa que hay que **explicar el código
> y modificarlo en vivo**, así que la sección 9 (modificaciones en vivo) es la más importante:
> practícala antes, con el proyecto levantado.

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
├── mcp/           servidor MCP (capabilities, resources, tools/     execute)
├── agent/         grafo LangGraph: state.py, nodes.py, graph.py, llm.py
└── ml_runtime/    capa de inferencia compartida (API y agente usan la misma)
ml_training/       Parte 1 — scikit-learn
dl_training/       Parte 2 — TensorFlow/Keras
saved_models/      artefactos entrenados + reportes JSON + gráficas
sql/init.sql       DDL + 1 función + 1 stored procedure PL/pgSQL
static/            portal del cliente (/portal) y panel técnico (/demo)
tests/             61 tests (pytest + TestClient + SQLite)
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
   riesgo de churn y tiempo estimado. Tomar el caso o resolverlo.
6. **El traspaso a humano** (lo más vistoso): en la bandeja, *Abrir chat con el cliente*
   muestra el hilo completo que tuvo con el bot. Escribe una respuesta como asesor y
   **vuelve a la pestaña del cliente sin recargarla**: el mensaje aparece solo, en una
   burbuja verde marcada "Asesor". Es el circuito cerrado: el bot atiende, clasifica y
   abre el ticket; cuando no puede, una persona entra al mismo hilo.
7. `/demo` para la vista técnica (probabilidades por clase, MCP con su sobre JSON-RPC).

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
MAE ≈ 1.68 h, RMSE ≈ 2.54 h, R² ≈ 0.63.

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
  determinista. Un R² de 0.63 sobre datos ruidosos es honesto.
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
  Postgres en docker-compose. **La función no es decorativa**: `CustomerService.ticket_stats()`
  la llama cuando el motor es PostgreSQL (con fallback ORM en SQLite para los tests) y de
  ahí salen los tickets abiertos y la satisfacción promedio que alimentan el churn.
  *Demostrable en vivo:* `SELECT * FROM fn_customer_churn_summary(1);` en psql devuelve lo
  mismo que usa el endpoint.

**MCP:** `GET /mcp/capabilities`, `GET /mcp/resources`, `GET /mcp/resources/{id}`,
`POST /mcp/tools/execute`, con las 5 tools pedidas + `predict_resolution_time`.

- *¿Por qué un error de tool devuelve HTTP 200?* Porque en MCP el error de ejecución es parte
  del **resultado** (`result.isError = true`), no un fallo de transporte. Un 500 significaría
  que el servidor MCP se cayó, que es distinto a "la tool falló".

---

## 7. Preguntas transversales

- *¿Por qué Postgres y no SQLite?* Porque el requisito de stored procedure pedía PL/pgSQL
  real. Los tests sí usan SQLite, por velocidad y para no depender de infraestructura.
- *¿Los tests cubren qué?* 61 tests sobre los endpoints críticos: auth (incluido rechazo de
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
   Relacionado: el modelo de sentimiento se entrenó con mensajes **emocionales**, así que
   sobre una descripción factual de avería su salida es ruido (llegó a decir "positivo").
   Por eso la bandeja solo muestra la etiqueta cuando `is_frustrated` es verdadero:
   **prefiero no mostrar nada antes que mostrar un dato engañoso.** Si preguntan por
   criterio de producto, esta es una buena respuesta.
2. **El dataset no tiene histórico temporal**: el churn se predice con una foto del
   cliente, no con su evolución. Con datos reales usaría ventanas (tickets del último
   trimestre, variación del consumo) en vez de acumulados.
3. **El `config` de LangGraph no llegaba a los nodos** hasta que lo detecté al implementar
   la apertura de tickets: la firma `config: RunnableConfig | None = None` hace que
   LangGraph **no** inyecte el config (lo decide inspeccionando la firma). El síntoma era
   silencioso: `get_customer_info` caía siempre en "sin sesión de base de datos" y el
   agente nunca leía datos del cliente. Corregido anotando `config: RunnableConfig`.
   **Si preguntan por un bug difícil que hayas encontrado, cuenta este.**

   Y si preguntan por **seguridad**: la primera versión tenía un IDOR. Todos los
   endpoints validaban el rol pero ninguno la pertenencia, así que el cliente 1 podía
   pedir `/customers/2`, `/tickets?customer_id=2` o la sesión de otro y leer nombre,
   teléfono, tickets y conversaciones ajenas. Lo reproduje, lo cerré con una dependencia
   de alcance (`app/api/deps.py`) y lo fijé con cuatro tests. Detalle a mencionar:
   devuelve **404 y no 403**, porque un 403 ya confirma que ese cliente existe.

   El otro que vale la pena contar: el modelo de sentimiento marcaba como frustrado
   *"Una última cosa, ¿cuál es el horario?"* con **confianza 1.0**. Descarté subir el
   umbral demostrando que no servía —el error era *más* confiado que los aciertos
   reales— y aislé el token culpable probando la frase con y sin prefijo: la palabra
   *"una"* aparecía en el **100 %** de los mensajes negativos del dataset y en ninguno
   de las otras clases, porque las plantillas eran *"esperando una respuesta"* y
   *"nadie me da una solución"*. Con 124 textos únicos y 155 palabras de vocabulario,
   el modelo se agarró a un artículo indefinido. La solución fue rehacer el generador
   de forma composicional para que las palabras funcionales se repartan entre clases.
4. **Sin Alembic.** Las tablas se crean con `Base.metadata.create_all`; para producción haría
   falta migraciones versionadas.
5. **Sin checkpointer de LangGraph** (decisión consciente, ver sección 5).
6. **El agente con LLM puede alucinar datos concretos** (inventó un horario de oficinas
   que no existe, pese a que el prompt se lo prohíbe). La solución real es RAG sobre una
   base de conocimiento, no más prompting. Buena respuesta si preguntan por los riesgos
   de poner un LLM de cara al cliente.
7. **Sin rate limiting ni paginación en tickets** (clientes sí tiene paginación).

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
- Sentimiento: accuracy test 1.00 *(sintético)*; 10/10 en frases nuevas escritas a mano.
- Resolución: MAE 1.68 h, RMSE 2.54 h, R² 0.63.

**Si piden probar el clasificador en vivo** — no muestres el 100 % del CSV, muestra esto:

```bash
docker exec -it telecom_support_api python scripts/probar_clasificador.py
```

12 frases escritas a mano: **11 aciertos**, con las probabilidades de las 5 clases y el
desglose del fallo. Es mucho mejor respuesta que un 100 % sobre datos sintéticos, porque
demuestra que sabes dónde falla tu modelo y por qué. El fallo a explicar:
*"solo quería felicitar al técnico que vino ayer"* → TECH con 47 %, porque TF-IDF pondera
palabras sueltas y "técnico" pesa más que la intención de la frase. La solución sería
embeddings contextuales (que entienden "felicitar"), a cambio de más datos y más cómputo.

Con `--interactivo` puedes pedirles a ellos que dicten la frase: es la demostración más
convincente, porque el ejemplo no lo elegiste tú.

**Comandos**

```bash
docker compose up --build              # stack completo
pytest tests/ -v                       # 61 tests
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

---

## 12. Mapa: cada requisito del enunciado → dónde está en el código

Tabla de rastreo para la entrevista. Si te preguntan *"¿dónde implementaste X?"*, la
respuesta está aquí con archivo y línea. Todas las referencias están verificadas contra
el código actual.

### PARTE 1.1 — Clasificación de tickets

| Lo que pide el enunciado | Dónde está | Qué hace |
|---|---|---|
| Preprocesamiento (tokenización, limpieza, vectorización) | `app/ml_runtime/text_preprocessing.py:28` `normalize_text` | Minúsculas, quita acentos y signos, colapsa espacios. Vive en un módulo compartido porque joblib lo serializa dentro del vectorizador |
| Stopwords en español | `app/ml_runtime/text_preprocessing.py:15` | Lista propia, evita depender de NLTK |
| **2 modelos diferentes y comparación** | `ml_training/train_ticket_classifier.py:54` `build_pipelines` | LogisticRegression vs LinearSVC calibrado; se elige el de mejor F1-macro en test |
| Calibración para tener probabilidades | `ml_training/train_ticket_classifier.py:71` | `CalibratedClassifierCV`, porque LinearSVC no tiene `predict_proba` |
| **Pipeline con `sklearn.pipeline.Pipeline`** | `ml_training/train_ticket_classifier.py:54-77` | TF-IDF (1-2 gramas, min_df=2, 5000 features) + clasificador |
| **Validación cruzada con 5 folds** | `ml_training/train_ticket_classifier.py:87,93` | `StratifiedKFold(n_splits=5)` + `cross_val_score` con F1-macro |
| Accuracy, precision, recall, F1 **por categoría** | `ml_training/train_ticket_classifier.py:97` | `classification_report(output_dict=True)` → JSON del reporte |
| **Matriz de confusión** | `ml_training/train_ticket_classifier.py:122` | PNG en `saved_models/ml/confusion_matrix_tickets.png` |
| **Guardar con joblib** | `ml_training/train_ticket_classifier.py:131` | `saved_models/ml/ticket_classifier.joblib` |
| Validación: **mínimo 10 caracteres** | `app/ml_runtime/text_preprocessing.py:38` + `app/schemas/ticket.py:50` | Se valida en el modelo *y* en el schema Pydantic; la API responde 422 |
| Texto en español con tildes | `app/ml_runtime/text_preprocessing.py:28` | El normalizador quita acentos, así que "señal"/"senal" caen en el mismo token |
| **Retornar categoría y probabilidad de cada clase** | `app/ml_runtime/ticket_classifier.py:42-47` `classify_ticket` | Devuelve `(categoría, {clase: probabilidad})` |
| Probarlo a mano | `scripts/probar_clasificador.py` | 4 modos: batería de frases nuevas, frase suelta, interactivo, dataset completo |

### PARTE 1.2 — Predicción de churn

| Lo que pide el enunciado | Dónde está | Qué hace |
|---|---|---|
| **Análisis exploratorio** (distribución de churn, correlaciones) | `ml_training/train_churn_model.py:63` `run_eda` | Distribución de clases, correlación de cada feature con el target, conteo de nulos. Queda en el JSON del reporte |
| **Manejo de valores nulos** | `ml_training/train_churn_model.py:89-96` | `SimpleImputer` mediana (numéricas) y moda (categóricas), dentro del `ColumnTransformer` |
| **Datos desbalanceados** | `ml_training/train_churn_model.py:100-103` | `class_weight="balanced"` sobre un 90/10, sin resampling externo |
| **≥2 features derivados** | `ml_training/train_churn_model.py:53` `engineer_features` | `charge_per_tenure` (cargo por mes de antigüedad) y `tickets_per_tenure` (densidad de incidencias) |
| Modelo de clasificación con probabilidades | `ml_training/train_churn_model.py:100-118` | RandomForest vs GradientBoosting, se elige por AUC-ROC; `predict_proba` |
| **AUC-ROC** | `ml_training/train_churn_model.py:114` | `roc_auc_score`, además de `average_precision_score` |
| **Curva precision-recall** | `ml_training/train_churn_model.py:130` | PNG en `saved_models/ml/churn_roc_pr_curves.png` |
| **Explicabilidad: feature importance** | `ml_training/train_churn_model.py:143` | Gráfica ordenada + ranking en el JSON |
| **Guardar modelo y preprocesador** | `ml_training/train_churn_model.py:153` | Un solo `.joblib`: el `Pipeline` incluye el `ColumnTransformer`, así no pueden desincronizarse |
| Features del cliente en tiempo real | `app/services/customer_service.py:47` `ticket_stats` | Tickets abiertos y satisfacción promedio reales (vía función SQL en Postgres) |

### PARTE 2.1 — Red de sentimiento

| Lo que pide el enunciado | Dónde está | Qué hace |
|---|---|---|
| **Capa de Embedding** | `dl_training/train_sentiment_model.py:82` | `Embedding(vocab, 64, mask_zero=True)`; la máscara evita que el padding contamine la LSTM |
| **Capa LSTM o GRU** | `dl_training/train_sentiment_model.py:83` | `LSTM(64, dropout=0.2, recurrent_dropout=0.2)` |
| **Dense con Dropout** | `dl_training/train_sentiment_model.py:84-85` | `Dense(32, relu)` + `Dropout(0.4)` |
| **Salida con activación apropiada** | `dl_training/train_sentiment_model.py:86` | `Dense(3, softmax)` para 3 clases |
| **Tokenizer de tf.keras** | `dl_training/train_sentiment_model.py:67` | `Tokenizer(num_words=10000, oov_token="<OOV>")` |
| **Padding de secuencias** | `dl_training/train_sentiment_model.py:72` | `pad_sequences(maxlen=200, padding="post")` |
| **Vocabulario máximo 10.000** | `dl_training/train_sentiment_model.py:45` | `MAX_VOCAB = 10_000` |
| **Longitud máxima 200** | `dl_training/train_sentiment_model.py:46` | `MAX_LEN = 200` |
| **EarlyStopping (patience=3)** | `dl_training/train_sentiment_model.py:98` | Con `restore_best_weights=True` |
| **ModelCheckpoint** | `dl_training/train_sentiment_model.py:99` | `save_best_only=True` → `sentiment_model_best.keras` |
| **ReduceLROnPlateau** | `dl_training/train_sentiment_model.py:100` | Factor 0.5, patience 2 |
| **Curvas de entrenamiento (loss y accuracy)** | `dl_training/train_sentiment_model.py:133` | `sentiment_training_curves.png` |
| **Matriz de confusión en test** | `dl_training/train_sentiment_model.py:139` | `sentiment_confusion_matrix.png` |
| **Exportar en .keras** | `dl_training/train_sentiment_model.py:147` | `saved_models/dl/sentiment_model.keras` |
| Inferencia desde la API | `app/ml_runtime/sentiment_model.py` | Expone `analyze_sentiment` → `(etiqueta, probabilidades, is_frustrated)` |

### PARTE 2.2 — Tiempo de resolución

| Lo que pide el enunciado | Dónde está | Qué hace |
|---|---|---|
| **Red neuronal para regresión** | `dl_training/train_resolution_time_model.py:108` | `Dense(1, activation="linear")`, pérdida MSE |
| Input: **descripción (embeddings)** | `dl_training/train_resolution_time_model.py:94-97` | Embedding + `GlobalAveragePooling1D` + Dense |
| Input: **categoría one-hot** | `dl_training/train_resolution_time_model.py:60,99` | Helper `one_hot` + `Input(shape=(5,))` |
| Input: **prioridad** | `dl_training/train_resolution_time_model.py:100` | One-hot de 3 niveles |
| Input: **hora del día y día de la semana** | `dl_training/train_resolution_time_model.py:55,101-102` | `cyclical_encode` (sin/cos): la hora 23 y la 0 quedan contiguas |
| **Manejar inputs mixtos** | `dl_training/train_resolution_time_model.py:104` | API funcional con `Concatenate` de las 5 ramas |
| **MAE, RMSE, R²** | `dl_training/train_resolution_time_model.py:138-140` | Las tres al JSON del reporte |
| Exportar el modelo | `dl_training/train_resolution_time_model.py:158` | `.keras` + tokenizer y encoders en `.joblib` |
| Consumo real | `app/api/v1/ml.py:40` y `app/mcp/router.py` (tool `predict_resolution_time`) | Endpoint REST + tool MCP; se muestra al cliente en el portal |

### PARTE 3.1 — Agente LangGraph

| Lo que pide el enunciado | Dónde está | Qué hace |
|---|---|---|
| **State con TypedDict** | `app/agent/state.py:7` | `AgentState(TypedDict, total=False)` |
| ├ `messages` | `app/agent/state.py:8` | Historial completo de la conversación |
| ├ `customer_id` (opcional) | `app/agent/state.py:9` | |
| ├ `intent` | `app/agent/state.py:10` | greeting / farewell / account_query / technical_support / general_info |
| ├ `context` | `app/agent/state.py:12` | Datos del cliente + señales de los modelos |
| ├ `escalate` | `app/agent/state.py:13` | |
| └ `response` | `app/agent/state.py:14` | (+ `error` en la línea 15, añadido para degradar sin romper el grafo) |
| **Nodo `classify_intent`** | `app/agent/nodes.py:71` | Regex para saludo/despedida; si no, el clasificador ML decide la categoría y de ahí sale la intención. También calcula sentimiento |
| **Nodo `get_customer_info`** | `app/agent/nodes.py:113` | Lee el cliente de la BD (sesión inyectada por el config) y calcula su riesgo de churn |
| **Nodo `handle_account_query`** | `app/agent/nodes.py:238` | Marca cancelaciones y registra el ticket |
| **Nodo `handle_technical_support`** | `app/agent/nodes.py:247` | Registra el ticket técnico |
| **Nodo `handle_general_info`** | `app/agent/nodes.py:254` | Consultas que no abren ticket |
| **Nodo `check_escalation`** | `app/agent/nodes.py:264` | Decide escalar y redacta el motivo |
| **Nodo `generate_response`** | `app/agent/nodes.py:335` | Gemini si hay API key; si no, plantillas |
| **Edges condicionales según la intención** | `app/agent/graph.py:49,53` | `_route_after_intent` (línea 23) ataja saludos; `_route_by_intent` (línea 29) enruta por intención |
| **Manejo de errores y estados inválidos** | `app/agent/nodes.py:104,164,232,368` | Cada nodo en `try/except`: el error va a `state["error"]` y se sigue con defaults |
| **Saludar y despedirse** | `app/agent/nodes.py:35,39` + `TEMPLATE_RESPONSES` | Regex de saludo/despedida que saltan el resto del grafo |
| **Recordar contexto de la conversación** | `app/services/agent_service.py:20` `chat` | Reconstruye `messages` desde la tabla `agent_session` en cada turno |
| **Escalar si detecta frustración o no puede resolver** | `app/agent/nodes.py:267` | 4 disparadores: pide humano, frustración, cancelación o churn alto |
| **Dar información del cliente si se identifica** | `app/agent/nodes.py:113` | Nombre, plan, antigüedad y riesgo al contexto |

### PARTE 3.2 — Integración con los modelos

| Lo que pide el enunciado | Dónde está | Qué hace |
|---|---|---|
| **Usar el clasificador para categorizar consultas** | `app/agent/nodes.py:85` | La categoría se mapea a intención con `CATEGORY_TO_INTENT` |
| **Usar el modelo de sentimiento para detectar frustración** | `app/agent/nodes.py:92` | `is_frustrated` (negativo con confianza > 0.6) alimenta el escalamiento |
| **Consultar churn para clientes de alto riesgo** | `app/agent/nodes.py:149` | Si el riesgo es alto, se escala aunque el mensaje sea cordial |
| Extra: tiempo de resolución | `app/agent/nodes.py:219` | Estima la respuesta del ticket que abre el agente |

### PARTE 4.1 — API REST

| Lo que pide el enunciado | Dónde está |
|---|---|
| `POST /auth/login` · `POST /auth/refresh` | `app/api/v1/auth.py:14,21` |
| `GET/POST /customers`, `GET/PUT/DELETE /customers/{id}` | `app/api/v1/customers.py:16,34,26,41,49` |
| `GET /customers/{id}/churn-prediction` | `app/api/v1/customers.py:57` |
| `GET/POST /tickets`, `GET/PUT /tickets/{id}` | `app/api/v1/tickets.py:15,43,36,52` |
| *(extra)* `GET /tickets/{id}/conversation`, `POST /tickets/{id}/reply` | traspaso a asesor humano |
| `POST /tickets/classify` | `app/api/v1/tickets.py:60` |
| `POST /ml/predict-churn` · `classify-ticket` · `analyze-sentiment` · `GET /ml/models/info` | `app/api/v1/ml.py:16,24,32,53` |
| `POST /agent/chat` · `GET/DELETE /agent/sessions/{id}` | `app/api/v1/agent.py:14,23,32` |
| *(extra)* `GET /auth/me`, `GET /tickets/queue`, `POST /ml/predict-resolution-time` | `app/api/v1/auth.py:28`, `app/api/v1/tickets.py:24`, `app/api/v1/ml.py:40` |

**Requisitos de la API**

| Lo que pide el enunciado | Dónde está | Qué hace |
|---|---|---|
| **Schemas de request y response** | `app/schemas/` | Uno por recurso: auth, customer, ticket, ml, agent, mcp, common |
| **Validación de email** | `app/schemas/customer.py:16,42` | `EmailStr` de Pydantic |
| **Teléfono: mínimo 10 dígitos, solo números, empieza con 09** | `app/schemas/customer.py:9,27,53` | Regex `^09\d{8,}$`, validado en creación y actualización |
| **Descripción de tickets: 20 a 500 caracteres** | `app/schemas/ticket.py:15,29` | `min_length=20, max_length=500` |
| **JWT con expiración configurable** | `app/core/security.py:42` + `app/core/config.py:31` | `ACCESS_TOKEN_EXPIRE_MINUTES` por entorno |
| **Refresh tokens** | `app/core/security.py:48` + `app/services/auth_service.py` | El payload lleva `type`, y un refresh no sirve como access |
| **Roles admin / agent / customer** | `app/core/security.py:18,79` | `require_roles(...)` como dependencia de cada ruta |
| *(y autorización por pertenencia)* | `app/api/deps.py` | El rol da acceso al endpoint; el *alcance* decide de quién son los datos. Un cliente solo ve lo suyo, y se le fuerza el filtro en listados y creación |
| **Swagger UI habilitado** | `/docs` (FastAPI lo monta solo) | |
| **Descripción de cada endpoint** | Todos los decoradores `@router.*` | Los 25 tienen `summary` y `description` |
| **Ejemplos de request/response** | `app/schemas/*.py` | `examples=[...]` en los campos y `json_schema_extra` en los 8 schemas de respuesta |
| **Respuestas de error estandarizadas** | `app/core/exceptions.py:25` | Todo sale como `{"error": {"code", "message", "details"}}` |
| **Códigos HTTP apropiados** | `app/core/exceptions.py:10` | Mapa estado → código (401, 403, 404, 409, 422, 503…) |
| **SQLAlchemy como ORM** | `app/models/` + `app/repositories/` | Modelos declarativos 2.0 y patrón Repository |
| **PostgreSQL o SQLite** | `docker-compose.yml` (Postgres) / `tests/conftest.py` (SQLite) | |
| **Al menos 1 stored procedure o función SQL** | `sql/init.sql:100` y `:140` | `fn_customer_churn_summary(id)` y `sp_refresh_category_avg_resolution()` |
| *(y se consume de verdad)* | `app/services/customer_service.py:47,57` | En Postgres llama a la función; en SQLite cae a una consulta ORM equivalente |

### PARTE 4.2 — Protocolo MCP

| Lo que pide el enunciado | Dónde está | Qué hace |
|---|---|---|
| `GET /mcp/capabilities` | `app/mcp/router.py:114` | Nombre, versión, tools con su JSON Schema, recursos |
| `POST /mcp/tools/execute` | `app/mcp/router.py:225` | Despacha por nombre de tool |
| `GET /mcp/resources` | `app/mcp/router.py:121` | |
| `GET /mcp/resources/{id}` | `app/mcp/router.py:128` | |
| **Tools: predict_churn, classify_ticket, get_customer_info, create_ticket, chat_with_agent** | `app/mcp/router.py:30` (descriptores) y `:215` (handlers) | Las 5 exigidas + `predict_resolution_time` |
| **Formato de respuesta JSON-RPC** (`jsonrpc`, `id`, `result.content`, `isError`) | `app/schemas/mcp.py:43,46` | Literal `"2.0"`; los errores de tool viajan con HTTP 200 e `isError=true` |

### Consideraciones y entrega

| Lo que pide el enunciado | Dónde está |
|---|---|
| **Arquitectura por capas o DDD** | `api → services → repositories → models`, más `ml_runtime` transversal |
| **Eliminaciones lógicas** (`is_active` / `deleted_at`) | `app/repositories/customer_repository.py:56`; los listados filtran por `is_active` |
| **ORM recomendable** | SQLAlchemy 2.0 con `Mapped[...]` |
| **Dockerizar el backend** | `Dockerfile` + `docker-compose.yml` (API + Postgres con healthcheck) |
| **Tests básicos de endpoints críticos** | `tests/` — 46 tests: auth, clientes, tickets, ML, agente y MCP |
| **README.md con instrucciones** | `README.md` |
| **requirements.txt** | `requirements.txt` |
| **docker-compose.yml funcional** | `docker-compose.yml` — verificado levantando el stack completo |
| **Script SQL con stored procedures** | `sql/init.sql` — se carga solo al crear el volumen de Postgres |
| **Esquema de datos completo** (customer, ticket, interaction, ticket_category, prediction, agent_session) | `app/models/` — las 6 tablas existen y **todas se escriben**: `prediction` en `app/services/customer_service.py:94`, `interaction` en `app/services/agent_service.py:51`, `ticket_category` en `app/core/seed.py:38` |
