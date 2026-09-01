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
- [x] Extras: portal del cliente y consola del asesor, traspaso a humano, endpoint y
      tool MCP para el modelo de tiempo de resolución, script de prueba manual del
      clasificador y guion de defensa oral (`docs/CHULETA_ENTREVISTA.md`)

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
tests/           pytest (61 tests) con TestClient + SQLite
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
- **No hace falta entrenar nada ni cargar datos a mano.** Los cuatro modelos vienen
  entrenados y versionados en `saved_models/` (2.6 MB en total), y al arrancar la app
  siembra sola los tres usuarios de prueba, un cliente demo y las cinco categorías de
  tickets (`app/core/seed.py`). También instala la función y el procedimiento PL/pgSQL
  de `sql/init.sql` si no existen, de modo que un `docker compose up` deja el sistema
  listo para usarse.

### Opción B — Local sin Docker (Python 3.11)

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # ajustar DATABASE_URL a un Postgres local o SQLite
uvicorn app.main:app --reload
```

### Opción C — Base de datos gestionada (Supabase u otro Postgres)

La app funciona con cualquier PostgreSQL: basta con apuntar `DATABASE_URL` a él. Las
tablas se crean al arrancar y la función y el procedimiento PL/pgSQL se instalan solos,
así que no hay migración manual que hacer.

```bash
DATABASE_URL=postgresql://postgres:[password]@db.[proyecto].supabase.co:5432/postgres
```

### Opción D — Supabase CLI en local

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

### Probar el clasificador a mano

```bash
# Batería de 12 frases escritas a mano, distintas de las plantillas del generador
python scripts/probar_clasificador.py

python scripts/probar_clasificador.py "el wifi anda lentísimo en el cuarto"   # frase suelta
python scripts/probar_clasificador.py --interactivo                            # escribir en vivo
python scripts/probar_clasificador.py --dataset                                # CSV completo

# Con el stack levantado, sin instalar nada:
docker exec -it telecom_support_api python scripts/probar_clasificador.py --interactivo
```

Sobre frases nuevas el modelo acierta **11 de 12** con confianzas de 47-89 %, muy por
debajo del 100 % que da sobre el CSV (que contiene las filas de entrenamiento). El fallo
recurrente es ilustrativo: *"solo quería felicitar al técnico que vino ayer"* se predice
como TECH, porque TF-IDF pondera palabras sueltas y "técnico" arrastra la predicción
aunque la intención sea otra.

### Tests

```bash
pytest tests/ -v      # 61 tests, corren contra SQLite en un archivo temporal
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
- **Tickets:** CRUD + `POST /tickets/classify` (clasificación sin crear ticket) + `GET /tickets/queue` (bandeja del agente) + `GET /tickets/{id}/conversation` y `POST /tickets/{id}/reply` (traspaso a un asesor humano)
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

Además de Swagger, la API sirve dos páginas que consumen sus propios endpoints.
**`/portal` decide la vista según el rol del usuario autenticado**, porque un cliente
y un agente no necesitan lo mismo:

**Rol `customer` → solo un chat.** Pantalla completa de conversación, sin formularios ni
métricas internas. El cliente describe su problema en lenguaje natural y **es el agente
LangGraph, en el backend, quien decide registrar el ticket**: los nodos
`handle_technical_support` y `handle_account_query` lo abren con la categoría que infirió
el clasificador y una prioridad derivada de las señales del mensaje (frustración,
cancelación o churn alto ⇒ alta). Si el cliente ya tiene un ticket abierto de esa
categoría, se reutiliza en vez de duplicarlo. El chat muestra el resultado como una
tarjeta: número de solicitud, equipo asignado y tiempo estimado de respuesta.

**Roles `agent` / `admin` → consola de soporte.** Bandeja de tickets sin resolver, cada
uno enriquecido con las señales de los otros modelos (frustración detectada sobre el
último mensaje del cliente, riesgo de churn y tiempo estimado), con acciones para tomar el caso o marcarlo resuelto. Se arma
con una sola llamada a `GET /api/v1/tickets/queue`: el trabajo pesado lo hace el backend
en vez de N peticiones desde el navegador.

**`/demo` — Panel técnico.** Vista para evaluar los modelos uno por uno: probabilidades
por clase, `is_frustrated`, churn, tiempo de resolución y ejecución de tools MCP con el
sobre JSON-RPC completo.

El portal usa `GET /api/v1/auth/me` para resolver el rol y a qué cliente pertenece la
sesión (el JWT solo transporta email y rol).

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

**2.1 Sentimiento (LSTM):** accuracy de test 1.00 sobre el conjunto apartado. El
número es alto porque los datos son sintéticos, pero el dataset se rehízo tras
detectar un problema real: la primera versión tenía 1800 filas con solo **124
textos únicos** (vocabulario de 155 palabras), y con tan poco material el modelo
aprendió atajos —la palabra *"una"* aparecía únicamente en frases negativas, así que
*"Una última cosa, ¿cuál es el horario?"* salía **negative con confianza 1.0** y
disparaba un escalamiento. Ahora los mensajes se componen de apertura + núcleo +
cierre, con aperturas y cierres **compartidos entre las tres clases**, de modo que las
palabras funcionales no puedan llevar señal: 2400 filas, **2205 textos únicos**,
vocabulario de 411 palabras. La clase neutra incluye además reportes de avería en tono
calmado, para que describir un problema no se confunda con estar enfadado.

**2.2 Tiempo de resolución (red multi-input):** MAE ≈ 1.68 h, RMSE ≈ 2.54 h, R² ≈ 0.63
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
- **Autorización por pertenencia, además de por rol:** `require_roles(...)` responde a
  "¿puedes usar este endpoint?", pero no a "¿son tuyos estos datos?". Sin lo segundo, un
  cliente autenticado podía leer la ficha, los tickets y las conversaciones de cualquier
  otro cliente cambiando el id de la URL. `app/api/deps.py` resuelve el *alcance* del
  usuario (`None` para admin/agent, su `customer_id` para un cliente) y lo aplica en
  clientes, tickets y sesiones. A un cliente se le **fuerza** el filtro en los listados y
  al crear tickets o conversar, para que no pueda actuar en nombre de otro. Se responde
  **404 y no 403**: un 403 confirmaría que ese recurso existe.
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
- **La función SQL se consume desde la aplicación, no solo se entrega:**
  `CustomerService.ticket_stats()` llama a `fn_customer_churn_summary` cuando el motor es
  PostgreSQL, y cae a una consulta ORM equivalente en SQLite (los tests). De ahí salen el
  número de tickets abiertos y la satisfacción promedio **reales** que alimentan el modelo
  de churn; antes `avg_satisfaction` iba fija en 3.5 pese a ser la 2ª feature más
  importante. Los dos caminos devuelven lo mismo, verificado contra ambos motores.
- **La tabla `interaction` del esquema se alimenta desde el agente:** cada turno que queda
  ligado a un ticket registra `customer_msg`, `agent_response`, el `sentiment` detectado y
  el tiempo estimado. Sin esto la tabla del enunciado existía pero quedaba siempre vacía.
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

5. **LangGraph no inyectaba el `config` en los nodos.** Los nodos estaban firmados como
   `def nodo(state, config: RunnableConfig | None = None)`, y LangGraph inspecciona la
   firma para decidir si inyecta el config: con una unión y un valor por defecto **no lo
   inyecta**, así que los nodos recibían `config=None` y `get_customer_info` caía siempre
   en la rama "sin sesión de base de datos". Se detectó al implementar la apertura de
   tickets desde el chat, que también necesitaba la sesión. La corrección fue anotar el
   parámetro exactamente como `RunnableConfig`, sin unión ni default, en los 7 nodos.

## 9. Limitaciones conocidas / próximos pasos

- El dataset sintético de tickets/sentimiento está generado por plantillas, por
  lo que los modelos de clasificación llegan a accuracy ≈ 1.0: es un techo
  esperado de los datos, no una garantía de desempeño en producción. Con datos
  reales se esperaría más solapamiento entre clases y métricas más moderadas.
- No se implementaron migraciones (Alembic incluido en `requirements.txt` pero
  no configurado); las tablas se crean con `Base.metadata.create_all` al
  arrancar, más `sql/init.sql` como documentación/alternativa manual.
- **El agente puede inventar datos concretos.** Con `GOOGLE_API_KEY` configurada, ante
  *"¿cuál es el horario de atención los sábados?"* Gemini respondió *"de 9:00 a 14:00"*:
  un dato que no existe en el sistema, y lo hizo pese a la instrucción explícita del
  prompt de no inventar precios, horarios ni fechas. Es alucinación de manual y no se
  arregla con prompting: haría falta una base de conocimiento que el agente consulte
  (RAG) y responder "no dispongo de ese dato" cuando no la encuentre. Sin la API key el
  agente usa plantillas y el problema no se da, porque no improvisa.
- **Sentimiento:** el modelo ya distingue una consulta informativa, un reporte de avería
  en calma y un cliente enfadado (10/10 en frases escritas a mano fuera del generador),
  pero sigue entrenado con datos sintéticos: con mensajes reales, llenos de ironía,
  jerga y errores de tipeo, cabe esperar bastante menos. La bandeja del agente muestra la
  etiqueta solo cuando `is_frustrated` es verdadero (negativo con confianza > 0.6), para
  no llenar la vista de datos poco accionables.
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
