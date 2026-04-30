-- =====================
-- SAMPLE QUERIES
-- =====================

-- 1. Get all HOT leads for a user
SELECT
    ui.input_text,
    ui.source,
    p.prediction_label,
    p.confidence,
    p.raw_output->>'response' AS ai_response,
    ui.created_at
FROM user_inputs ui
JOIN predictions p ON p.input_id = ui.id
WHERE ui.user_id = 'USER_UUID_HERE'
  AND p.prediction_label = 'HOT'
  AND p.model_name = 'lead-classifier-v1'
ORDER BY ui.created_at DESC;

-- 2. Model performance tracking (accuracy by label)
SELECT
    prediction_label,
    COUNT(*) AS total,
    AVG(confidence) AS avg_confidence,
    AVG(latency_ms) AS avg_latency_ms
FROM predictions
WHERE model_name = 'lead-classifier-v1'
  AND created_at > NOW() - INTERVAL '7 days'
GROUP BY prediction_label
ORDER BY total DESC;

-- 3. Vector similarity search (find similar user inputs)
SELECT
    ui.id,
    ui.input_text,
    ui.input_type,
    1 - (e.embedding <=> '[0.1, 0.2, 0.3]'::vector) AS similarity_score
FROM embeddings e
JOIN user_inputs ui ON ui.id = e.input_id
WHERE e.user_id = 'USER_UUID_HERE'
  AND e.input_type = 'text'
ORDER BY similarity_score DESC
LIMIT 5;

-- 4. Failed predictions in last 24 hours
SELECT
    p.model_name,
    p.error_message,
    ui.input_text,
    p.created_at
FROM predictions p
JOIN user_inputs ui ON ui.id = p.input_id
WHERE p.status = 'failed'
  AND p.created_at > NOW() - INTERVAL '24 hours'
ORDER BY p.created_at DESC;

-- 5. Lead conversion funnel (inputs by type + source)
SELECT
    input_type,
    source,
    COUNT(*) AS total_inputs,
    COUNT(DISTINCT user_id) AS unique_users
FROM user_inputs
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY input_type, source
ORDER BY total_inputs DESC;
