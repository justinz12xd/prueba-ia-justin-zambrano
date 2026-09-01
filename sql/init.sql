-- =============================================================================
-- Sistema Inteligente de Atención al Cliente — Script SQL (PostgreSQL)
-- =============================================================================
-- Nota: en la app, SQLAlchemy crea estas mismas tablas automáticamente al
-- arrancar (Base.metadata.create_all), por lo que este script NO es
-- estrictamente necesario para levantar el sistema. Se entrega como
-- documentación del esquema y para satisfacer el requisito de incluir al
-- menos 1 stored procedure / función SQL, ejecutable manualmente contra
-- cualquier Postgres (docker-compose, o una instancia de Supabase CLI vía
-- `supabase start` + `psql` / SQL editor).
-- =============================================================================

CREATE TABLE IF NOT EXISTS customer (
    customer_id     SERIAL PRIMARY KEY,
    name            VARCHAR(120) NOT NULL,
    email           VARCHAR(150) NOT NULL UNIQUE,
    phone           VARCHAR(15) NOT NULL,
    plan_type       VARCHAR(60) NOT NULL,
    monthly_charge  FLOAT NOT NULL DEFAULT 0,
    tenure_months   INTEGER NOT NULL DEFAULT 0,
    total_charges   FLOAT NOT NULL DEFAULT 0,
    contract_type   VARCHAR(30) NOT NULL DEFAULT 'month-to-month',
    payment_method  VARCHAR(30) NOT NULL DEFAULT 'credit_card',
    churn_status    INTEGER NOT NULL DEFAULT 0,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    deleted_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ticket_category (
    category_id     SERIAL PRIMARY KEY,
    category_name   VARCHAR(10) NOT NULL UNIQUE,
    description     VARCHAR(255),
    avg_resolution  FLOAT
);

CREATE TABLE IF NOT EXISTS ticket (
    ticket_id       SERIAL PRIMARY KEY,
    customer_id     INTEGER NOT NULL REFERENCES customer(customer_id),
    category        VARCHAR(10) NOT NULL,
    description     TEXT NOT NULL,
    priority        VARCHAR(10) NOT NULL DEFAULT 'medium',
    status          VARCHAR(20) NOT NULL DEFAULT 'open',
    satisfaction    INTEGER,
    -- Conversacion de la que nacio el ticket: permite que un asesor humano entre
    -- al chat desde la bandeja de soporte y continue el mismo hilo con el cliente.
    agent_session_id VARCHAR(36),  -- FK añadida más abajo: agent_session se crea después
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    deleted_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at     TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS interaction (
    interaction_id  SERIAL PRIMARY KEY,
    ticket_id       INTEGER NOT NULL REFERENCES ticket(ticket_id),
    agent_response  TEXT,
    customer_msg    TEXT NOT NULL,
    sentiment       VARCHAR(10),
    resolution_time FLOAT,
    "timestamp"     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS prediction (
    prediction_id   SERIAL PRIMARY KEY,
    customer_id     INTEGER NOT NULL REFERENCES customer(customer_id),
    churn_prob      FLOAT NOT NULL,
    risk_level      VARCHAR(10) NOT NULL,
    model_version   VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_session (
    session_id      VARCHAR(36) PRIMARY KEY,
    customer_id     INTEGER REFERENCES customer(customer_id),
    conversation    JSONB NOT NULL DEFAULT '[]',
    tokens_used     INTEGER NOT NULL DEFAULT 0,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at        TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS user_account (
    user_id         SERIAL PRIMARY KEY,
    email           VARCHAR(150) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    role            VARCHAR(20) NOT NULL DEFAULT 'customer',
    customer_id     INTEGER REFERENCES customer(customer_id),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- La clave foránea se declara aquí y no en el CREATE TABLE porque `ticket` se crea
-- antes que `agent_session`. DO $$ ... $$ la hace idempotente al re-ejecutar el script.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_ticket_agent_session') THEN
        ALTER TABLE ticket
            ADD CONSTRAINT fk_ticket_agent_session
            FOREIGN KEY (agent_session_id) REFERENCES agent_session(session_id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_ticket_customer_id ON ticket(customer_id);
CREATE INDEX IF NOT EXISTS idx_ticket_agent_session ON ticket(agent_session_id);
CREATE INDEX IF NOT EXISTS idx_interaction_ticket_id ON interaction(ticket_id);
CREATE INDEX IF NOT EXISTS idx_prediction_customer_id ON prediction(customer_id);

-- =============================================================================
-- Función 1: resumen de riesgo de churn de un cliente
-- Retorna, en una sola llamada, la antigüedad, el número de tickets abiertos,
-- la satisfacción promedio y la última predicción de churn registrada.
-- Uso:  SELECT * FROM fn_customer_churn_summary(1);
-- =============================================================================
CREATE OR REPLACE FUNCTION fn_customer_churn_summary(p_customer_id INTEGER)
RETURNS TABLE (
    customer_id     INTEGER,
    name            VARCHAR,
    tenure_months   INTEGER,
    open_tickets    BIGINT,
    avg_satisfaction NUMERIC,
    last_churn_prob FLOAT,
    last_risk_level VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.customer_id,
        c.name,
        c.tenure_months,
        COUNT(t.ticket_id) FILTER (WHERE t.status NOT IN ('resolved', 'closed')) AS open_tickets,
        ROUND(AVG(t.satisfaction)::NUMERIC, 2) AS avg_satisfaction,
        p.churn_prob AS last_churn_prob,
        p.risk_level AS last_risk_level
    FROM customer c
    LEFT JOIN ticket t ON t.customer_id = c.customer_id AND t.is_active = TRUE
    LEFT JOIN LATERAL (
        SELECT churn_prob, risk_level
        FROM prediction pr
        WHERE pr.customer_id = c.customer_id
        ORDER BY pr.created_at DESC
        LIMIT 1
    ) p ON TRUE
    WHERE c.customer_id = p_customer_id
    GROUP BY c.customer_id, c.name, c.tenure_months, p.churn_prob, p.risk_level;
END;
$$ LANGUAGE plpgsql STABLE;

-- =============================================================================
-- Procedimiento 2: recalcula avg_resolution en ticket_category a partir de los
-- tickets realmente resueltos (created_at -> resolved_at). Se puede invocar
-- periódicamente (cron / job) para mantener la tabla de categorías actualizada.
-- Uso:  CALL sp_refresh_category_avg_resolution();
-- =============================================================================
CREATE OR REPLACE PROCEDURE sp_refresh_category_avg_resolution()
LANGUAGE plpgsql AS $$
BEGIN
    UPDATE ticket_category tc
    SET avg_resolution = sub.avg_hours
    FROM (
        SELECT
            category,
            AVG(EXTRACT(EPOCH FROM (resolved_at - created_at)) / 3600.0) AS avg_hours
        FROM ticket
        WHERE resolved_at IS NOT NULL AND is_active = TRUE
        GROUP BY category
    ) AS sub
    WHERE tc.category_name = sub.category;
END;
$$;

-- =============================================================================
-- Seed mínimo de categorías (idempotente)
-- =============================================================================
INSERT INTO ticket_category (category_name, description, avg_resolution) VALUES
    ('TECH', 'Problemas técnicos: internet lento, sin conexión', 8),
    ('BILL', 'Consultas de facturación', 3),
    ('PLAN', 'Cambio de plan o servicios', 4),
    ('CNCL', 'Cancelación de servicio', 5),
    ('OTHR', 'Otros', 2)
ON CONFLICT (category_name) DO NOTHING;
