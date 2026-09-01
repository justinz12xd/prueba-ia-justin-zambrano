# Sistema Inteligente de Atención al Cliente — Telecomunicaciones

> Prueba Técnica: Desarrollador de Inteligencia Artificial — Viamatica
> **Autor:** Justin Alejandro Zambrano Lucas ([@justinz12xd](https://github.com/justinz12xd))

Sistema que integra ML clásico (scikit-learn), Deep Learning (TensorFlow/Keras), un
agente conversacional (LangGraph + Google Gemini) y una API REST (FastAPI) con
soporte del protocolo MCP, para atención al cliente de una empresa de
telecomunicaciones.

## Estado de avance

- [x] Parte 1 — ML clásico (clasificación de tickets + churn)
- [x] Parte 2 — Deep Learning (sentimiento + tiempo de resolución)
- [x] Parte 3 — Agente LangGraph
- [x] Parte 4 — API FastAPI + MCP
- [x] Docker / Tests / Documentación final

---

## 1. Stack técnico

| Capa | Tecnología |
|---|---|
| API | FastAPI + Pydantic v2, JWT (roles admin/agent/customer) |
| Base de datos | PostgreSQL vía SQLAlchemy 2.0 (Docker o Supabase CLI) |
| ML clásico | scikit-learn (TF-IDF + LogisticRegression/LinearSVC, RandomForest/GradientBoosting) |
| Deep Learning | TensorFlow/Keras (LSTM para sentimiento, red multi-input para regresión) |
| Agente conversacional | LangGraph + `langchain-google-genai` (Gemini) |
| Infraestructura | Docker + docker-compose, pytest |

## 2. Estructura del proyecto

```
app/
  core/          config, database, security (JWT), exceptions, seed
  models/        entidades SQLAlchemy (Customer, Ticket, Interaction, Prediction, AgentSession, UserAccount)
  schemas/       Pydantic (request/response + validaciones)
  repositories/  acceso a datos (patrón Repository)
  services/      lógica de negocio (capa de aplicación)
  api/v1/        routers REST (auth, customers, tickets, ml, agent)
  mcp/           servidor MCP (capabilities, tools/execute, resources)
  agent/         grafo LangGraph (state, nodes, graph, llm)
  ml_runtime/    wrappers de inferencia para los 4 modelos entrenados
ml_training/     scripts de entrenamiento scikit-learn (Parte 1)
dl_training/     scripts de entrenamiento TensorFlow/Keras (Parte 2)
saved_models/    modelos entrenados (.joblib / .keras) + reportes/gráficas
data/            datasets sintéticos generados (scripts/generate_synthetic_data.py)
sql/init.sql     DDL + función y stored procedure PL/pgSQL
static/          portal del cliente (/portal) y panel técnico (/demo)
docs/            chuleta de defensa oral para la entrevista
tests/           pytest (39 tests) con TestClient + SQLite
```

## 3. Cómo ejecutar

### Opción A — Docker (recomendado, todo incluido)

```bash
cp .env.example .env
# (opcional) completar GOOGLE_API_KEY en .env para respuestas del agente con LLM real;
# sin la key, el agente sigue funcionando con respuestas basadas en plantillas.
docker compose up --build
```

- API: http://localhost:8000
- **Portal del cliente: http://localhost:8000/portal**
- **Panel técnico de modelos: http://localhost:8000/demo**
- Swagger UI: http://localhost:8000/docs
- Postgres queda expuesto en `localhost:5432` (usuario/clave `postgres`/`postgres`, DB `telecom_support`)
- `sql/init.sql` se ejecuta automáticamente al crear el volumen de Postgres (DDL + función + stored procedure).
- Los modelos ML/DL ya vienen entrenados y versionados en `saved_models/`, así que la API funciona de inmediato sin paso de entrenamiento previo.

### Opción B — Local sin Docker (Python 3.11)

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # ajustar DATABASE_URL a un Postgres local o SQLite
uvicorn app.main:app --reload
```

### Opción C — Base de datos con Supabase CLI

```bash
supabase start
# copiar la cadena de conexión que imprime el comando (puerto 54322) a DATABASE_URL en .env
# opcionalmente correr sql/init.sql manualmente desde el SQL editor / psql
```

### Reentrenar los modelos (opcional)

```bash
python scripts/generate_synthetic_data.py     # regenera data/*.csv
python ml_training/train_ticket_classifier.py
python ml_training/train_churn_model.py
python dl_training/train_sentiment_model.py
python dl_training/train_resolution_time_model.py
```

### Tests

```bash
pytest tests/ -v      # 39 tests, corren contra SQLite en un archivo temporal
```

## 4. Credenciales de prueba

Sembradas automáticamente al arrancar la app (`app/core/seed.py`):

| Email | Password | Rol |
|---|---|---|
| admin@telecom.com | Admin123! | admin |
| agente@telecom.com | Agente123! | agent |
| cliente@telecom.com | Cliente123! | customer |

También se crea un cliente demo (`customer_id=1`) asociado al usuario `customer`.

## 5. Endpoints principales

Todos documentados con ejemplos en Swagger (`/docs`). Resumen:

- **Auth:** `POST /api/v1/auth/login`, `POST /api/v1/auth/refresh`, `GET /api/v1/auth/me`
- **Customers:** CRUD completo + `DELETE` lógico + `GET /{id}/churn-prediction`
- **Tickets:** CRUD + `POST /tickets/classify` (clasificación sin crear ticket)
- **ML:** `POST /ml/predict-churn`, `POST /ml/classify-ticket`, `POST /ml/analyze-sentiment`, `POST /ml/predict-resolution-time`, `GET /ml/models/info`
- **Agente:** `POST /agent/chat`, `GET/DELETE /agent/sessions/{id}`
- **MCP:** `GET /mcp/capabilities`, `GET /mcp/resources(/{id})`, `POST /mcp/tools/execute`

Ejemplo de llamada MCP:
```bash
curl -X POST http://localhost:8000/mcp/tools/execute \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"req-1","tool":"classify_ticket","arguments":{"description":"No tengo internet desde ayer"}}'
```

## 5.1 Interfaces web

Además de Swagger, la API sirve dos páginas que consumen sus propios endpoints:

**`/portal` — Centro de Ayuda (simulación del producto).** Es la vista del cliente
final: escribe su problema en lenguaje natural y, mientras teclea, el clasificador
de tickets detecta la categoría y le dice a qué equipo se derivará
(*"Detectamos: Consulta de facturación → Facturación, confianza 87%"*). Al enviar,
se crea el ticket real con la categoría inferida y se le muestra el tiempo estimado
de respuesta calculado por la red de la Parte 2.2. Incluye el asistente conversacional
—que avisa cuando deriva a una persona— y el historial de solicitudes del cliente.

**`/demo` — Panel técnico.** Vista para evaluar los modelos: probabilidades por clase,
sentimiento con su `is_frustrated`, churn con nivel de riesgo, tiempo de resolución y
ejecución de tools MCP con el sobre JSON-RPC completo.

El portal usa `GET /api/v1/auth/me` para resolver a qué cliente pertenece la sesión
(el JWT solo transporta email y rol). La clasificación en vivo va con *debounce* de
450 ms para no lanzar una petición por tecla, y si falla no bloquea el envío: es una
ayuda, no un requisito.

## 6. Resultados de los modelos

**1.1 Clasificador de tickets** (TF-IDF + comparación LogisticRegression vs LinearSVC calibrado, CV 5-fold):

| Modelo | F1-macro (CV 5-fold) | Accuracy test | F1-macro test |
|---|---|---|---|
| LogisticRegression (elegido) | 0.998 | 1.00 | 1.00 |
| LinearSVC calibrado | 0.998 | 1.00 | 1.00 |

> El 100% de accuracy es esperable con el dataset sintético (frases generadas a
> partir de plantillas por categoría, muy separables); ver sección de
> limitaciones más abajo.

**1.2 Predicción de churn** (RandomForest vs GradientBoosting, clases desbalanceadas 90/10):

| Modelo | AUC-ROC | Average Precision |
|---|---|---|
| RandomForest | 0.709 | 0.233 |
| GradientBoosting (elegido) | **0.731** | **0.273** |

Top features por importancia: `charge_per_tenure` (0.24), `avg_satisfaction` (0.20),
`total_charges` (0.14), `tickets_per_tenure` (0.14), `monthly_charge` (0.11).

**2.1 Sentimiento (LSTM):** accuracy de test 1.00 (dataset sintético con
mensajes bien diferenciados por polaridad).

**2.2 Tiempo de resolución (red multi-input):** MAE ≈ 1.75h, RMSE ≈ 2.53h, R² ≈ 0.58
sobre datos con ruido log-normal inyectado deliberadamente.

Gráficas y reportes completos en `saved_models/ml/*.png|json` y `saved_models/dl/*.png|json`.

## 7. Decisiones técnicas y justificación

- **Datos sintéticos en español:** no se proporcionó `tickets_train.csv`, así que se
  generó un dataset sintético (plantillas + variabilidad + ruido) para las 4 tablas
  necesarias (tickets, clientes/churn, interacciones/sentimiento, tiempos de
  resolución), con semilla fija para reproducibilidad (`scripts/generate_synthetic_data.py`).
- **Arquitectura por capas:** `api → services → repositories → models`, separando
  reglas de negocio (services) del acceso a datos (repositories) y de la
  serialización HTTP (schemas), en vez de DDD "puro" para no sobre-diseñar dado
  el tiempo disponible.
- **Eliminación lógica uniforme:** `is_active` + `deleted_at` en `customer` y
  `ticket`, consistente con lo pedido en "Consideraciones".
- **Autenticación uniforme en todos los routers, incluido el agente:** `/api/v1/agent/*`
  también exige JWT. `chat` y la lectura de una sesión están abiertos a los tres roles
  (`admin`/`agent`/`customer`), mientras que borrar una sesión queda restringido a
  personal interno (`admin`/`agent`), mismo criterio que el `DELETE` de clientes. La
  alternativa era dejar el chat público (widget para clientes anónimos), pero eso
  implicaba exponer `GET/DELETE /agent/sessions/{id}` —que devuelve y destruye el
  historial completo de la conversación— sin identificar a quien llama.
- **Gemini vía LangChain:** se usó `langchain-google-genai` (el usuario cuenta con
  API key de Gemini) en vez de OpenAI/Anthropic. El nodo `generate_response`
  degrada automáticamente a respuestas basadas en plantillas si no hay
  `GOOGLE_API_KEY` configurada, para que el sistema (y los tests) no dependan de
  una llamada de red externa.
- **DB en Postgres, no SQLite:** se eligió Postgres (compatible con Supabase CLI)
  para poder cumplir el requisito de función/stored procedure en PL/pgSQL real;
  los tests automatizados sí usan SQLite por velocidad y para no depender de
  infraestructura externa en CI.
- **`ml_runtime/` como capa de inferencia compartida:** tanto la API como el
  agente LangGraph consumen los mismos wrappers de carga/predicción (`joblib`/
  `tf.keras.models.load_model`, cacheados con `lru_cache`), evitando cargar los
  modelos más de una vez por proceso y evitando duplicar lógica de inferencia.
- **MCP con envoltura JSON-RPC en `result.content[].data`:** se siguió literalmente
  el formato de respuesta del enunciado; los errores de ejecución de una tool se
  devuelven con HTTP 200 e `isError:true` (igual que el protocolo MCP real: el
  error es parte del resultado, no un fallo de transporte).

## 8. Dificultades encontradas (y cómo se resolvieron)

1. **`joblib` no podía cargar el clasificador de tickets fuera del script de
   entrenamiento.** La función de preprocesamiento (`normalize_text`), usada como
   `preprocessor` del `TfidfVectorizer`, quedaba serializada como
   `__main__.normalize_text` al ejecutar el script directamente, por lo que
   fallaba al importarse desde la API. Se movió a un módulo compartido
   (`app/ml_runtime/text_preprocessing.py`) importado tanto por el entrenamiento
   como por la inferencia, y se reentrenó el modelo.
2. **FastAPI: `AssertionError` al registrar `DELETE /agent/sessions/{id}`** por
   combinar `status_code=204` con un `response_model` inferido del tipo de
   retorno. Se resolvió pasando `response_model=None` explícitamente.
3. **Errores 500 al formatear respuestas 422 de Pydantic v2:** cuando un
   `field_validator` lanza `ValueError` (p. ej. validación de teléfono),
   `RequestValidationError.errors()` incluye la excepción cruda dentro de `ctx`,
   que no es serializable a JSON. Se normaliza `ctx` a texto en el handler de
   excepciones antes de responder.
4. **Modelo de tiempo de resolución sin consumidor.** El modelo de la Parte 2.2
   quedó entrenado y guardado, pero durante un repaso se detectó que ningún
   endpoint, tool MCP ni nodo del agente lo llamaba: solo aparecía como metadata
   en `/ml/models/info`. Se expuso en `POST /api/v1/ml/predict-resolution-time`
   y como tool MCP `predict_resolution_time`, y se añadió al demo web.

## 9. Limitaciones conocidas / próximos pasos

- El dataset sintético de tickets/sentimiento está generado por plantillas, por
  lo que los modelos de clasificación llegan a accuracy ≈ 1.0: es un techo
  esperado de los datos, no una garantía de desempeño en producción. Con datos
  reales se esperaría más solapamiento entre clases y métricas más moderadas.
- No se implementaron migraciones (Alembic incluido en `requirements.txt` pero
  no configurado); las tablas se crean con `Base.metadata.create_all` al
  arrancar, más `sql/init.sql` como documentación/alternativa manual.
- El checkpointer de LangGraph no se usó explícitamente: el historial de la
  conversación se reconstruye en cada turno desde `agent_session` (Postgres),
  que es la fuente de verdad pedida por el esquema de datos.

## 10. Tiempo dedicado

| Sección | Tiempo aproximado |
|---|---|
| Estructura del proyecto + generación de datos sintéticos | ~0.5 h |
| Parte 1 — ML con scikit-learn (clasificador + churn) | ~1.5 h |
| Parte 2 — Deep Learning (sentimiento + tiempo de resolución) | ~1.5 h |
| Parte 3 — Agente LangGraph e integración con los modelos | ~1.5 h |
| Parte 4 — API FastAPI, MCP, seguridad y Docker | ~2 h |
| Tests, demo web y documentación | ~1 h |
| **Total** | **~8 h** |

Nota de transparencia: el desarrollo se apoyó en **asistencia de IA (Claude Code)**,
por lo que el tiempo de reloj efectivo fue menor al de un desarrollo manual; la
tabla refleja el esfuerzo relativo de cada sección. Las decisiones técnicas están
justificadas una a una en la sección 7 y las limitaciones reconocidas en la 9.

**Preparación para la entrevista de validación:** en `docs/CHULETA_ENTREVISTA.md`
hay un guion de defensa oral —recorrido de demo, preguntas probables por sección,
debilidades conocidas y ejercicios de modificación en vivo— centrado en
`app/agent/` (flujo LangGraph) y `app/ml_runtime/` (conexión entre los modelos
entrenados y la API), que son las partes más específicas del enunciado.
