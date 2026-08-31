# Sistema Inteligente de Atención al Cliente — Telecomunicaciones

> Prueba Técnica: Desarrollador de Inteligencia Artificial — Viamatica
> **Autor:** Justin (justinz12xd)

Sistema que integra ML clásico (scikit-learn), Deep Learning (TensorFlow/Keras), un
agente conversacional (LangGraph + Gemini) y una API REST (FastAPI) con soporte del
protocolo MCP, para atención al cliente de una empresa de telecomunicaciones.

> ⚠️ Este README se completa progresivamente a medida que avanza el desarrollo.
> La sección final con tiempos, decisiones técnicas y dificultades se agrega al cierre.

## Estado de avance

- [ ] Parte 1 — ML clásico (clasificación de tickets + churn)
- [ ] Parte 2 — Deep Learning (sentimiento + tiempo de resolución)
- [ ] Parte 3 — Agente LangGraph
- [ ] Parte 4 — API FastAPI + MCP
- [ ] Docker / Tests / Documentación final

## Stack

- **API:** FastAPI + Pydantic v2 + JWT (roles: admin, agent, customer)
- **DB:** PostgreSQL (SQLAlchemy ORM), compatible con Postgres local vía Docker o
  vía [Supabase CLI](https://supabase.com/docs/guides/cli) (`supabase start`)
- **ML:** scikit-learn (TF-IDF + clasificadores, churn con probabilidades)
- **DL:** TensorFlow/Keras (LSTM/GRU para sentimiento, red multi-input para regresión)
- **Agente:** LangGraph + Google Gemini (`langchain-google-genai`)
- **Infra:** Docker + docker-compose

## Ejecución rápida

```bash
cp .env.example .env   # completar GOOGLE_API_KEY, JWT_SECRET, etc.
docker compose up --build
```

- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs

(Instrucciones detalladas al final de este documento, sección "Cómo ejecutar".)
