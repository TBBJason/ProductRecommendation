-- Model Performance Dashboard
WITH daily_metrics AS (
    SELECT
        DATE(recommendation_timestamp) as date,
        COUNT(*) as total_recommendations,
        SUM(CASE WHEN clicked = 1 THEN 1 ELSE 0 END) as clicks,
        SUM(CASE WHEN purchased = 1 THEN 1 ELSE 0 END) as purchases,
        AVG(ranking_score) as avg_score
    FROM gold.recommendation_events
    WHERE recommendation_timestamp >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY 1
)
SELECT
    date,
    total_recommendations,
    clicks,
    purchases,
    1.0 * clicks / total_recommendations as ctr,
    1.0 * purchases / clicks as conversion_rate,
    avg_score
FROM daily_metrics
ORDER BY date DESC;