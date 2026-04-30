-- KeaBuilder ML Schema Design
-- Stores user inputs and ML model predictions

CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

-- =====================
-- USERS TABLE
-- =====================
CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email       VARCHAR(255) UNIQUE NOT NULL,
    name        VARCHAR(255),
    plan        VARCHAR(50) DEFAULT 'free',
    created_at  TIMESTAMP DEFAULT NOW(),
    updated_at  TIMESTAMP DEFAULT NOW()
);

-- =====================
-- USER INPUTS TABLE
-- =====================
CREATE TABLE user_inputs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id  UUID,
    input_text  TEXT NOT NULL,
    input_type  VARCHAR(50) NOT NULL,
    source      VARCHAR(50),
    metadata    JSONB DEFAULT '{}',
    ip_address  INET,
    created_at  TIMESTAMP DEFAULT NOW()
);

-- =====================
-- PREDICTIONS TABLE
-- =====================
CREATE TABLE predictions (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    input_id          UUID NOT NULL REFERENCES user_inputs(id) ON DELETE CASCADE,
    model_name        VARCHAR(100) NOT NULL,
    model_version     VARCHAR(20) NOT NULL,
    prediction_label  VARCHAR(100),
    confidence        FLOAT,
    raw_output        JSONB DEFAULT '{}',
    latency_ms        INTEGER,
    status            VARCHAR(20) DEFAULT 'completed',
    error_message     TEXT,
    created_at        TIMESTAMP DEFAULT NOW()
);

-- =====================
-- EMBEDDINGS TABLE (pgvector)
-- =====================
CREATE TABLE embeddings (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    input_id    UUID NOT NULL REFERENCES user_inputs(id) ON DELETE CASCADE,
    user_id     UUID NOT NULL REFERENCES users(id),
    embedding   vector(384),
    model_name  VARCHAR(100) DEFAULT 'all-MiniLM-L6-v2',
    input_type  VARCHAR(50),
    created_at  TIMESTAMP DEFAULT NOW()
);

-- =====================
-- INDEXES
-- =====================
CREATE INDEX idx_user_inputs_user_id ON user_inputs(user_id);
CREATE INDEX idx_user_inputs_type ON user_inputs(input_type);
CREATE INDEX idx_user_inputs_created ON user_inputs(created_at DESC);

CREATE INDEX idx_predictions_input_id ON predictions(input_id);
CREATE INDEX idx_predictions_label ON predictions(prediction_label);
CREATE INDEX idx_predictions_model ON predictions(model_name, model_version);
CREATE INDEX idx_predictions_created ON predictions(created_at DESC);

CREATE INDEX idx_embeddings_vector ON embeddings
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_embeddings_user ON embeddings(user_id);
CREATE INDEX idx_embeddings_type ON embeddings(input_type);
