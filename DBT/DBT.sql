{{
    config(
        materialized='incremental',
        unique_key='product_id',
        on_schema_change='sync_all_columns'
    )
}}

WITH product_base AS (
    SELECT 
        product_id,
        title_clean,
        description_clean,
        category,
        price,
        brand,
        created_at
    FROM {{ ref('silver_products_cleaned') }}
    {% if is_incremental() %}
    WHERE created_at > (SELECT MAX(created_at) FROM {{ this }})
    {% endif %}
),

category_stats AS (
    SELECT
        category,
        AVG(price) as avg_category_price,
        COUNT(*) as category_product_count
    FROM product_base
    GROUP BY category
),

brand_stats AS (
    SELECT
        brand,
        AVG(price) as avg_brand_price,
        COUNT(*) as brand_product_count
    FROM product_base
    GROUP BY brand
)

SELECT
    p.*,
    cs.avg_category_price,
    cs.category_product_count,
    bs.avg_brand_price,
    bs.brand_product_count,
    CASE 
        WHEN p.price > cs.avg_category_price THEN 'premium'
        WHEN p.price < cs.avg_category_price * 0.7 THEN 'budget'
        ELSE 'mid-range'
    END as price_segment
FROM product_base p
LEFT JOIN category_stats cs ON p.category = cs.category
LEFT JOIN brand_stats bs ON p.brand = bs.brand


{{
    config(
        materialized='incremental',
        unique_key=['user_id', 'product_id', 'interaction_date']
    )
}}

SELECT
    user_id,
    product_id,
    interaction_type,  -- view, click, purchase, add_to_cart
    DATE(interaction_timestamp) as interaction_date,
    COUNT(*) as interaction_count,
    MAX(interaction_timestamp) as last_interaction
FROM {{ ref('silver_user_events') }}
{% if is_incremental() %}
WHERE interaction_timestamp > (SELECT MAX(last_interaction) FROM {{ this }})
{% endif %}
GROUP BY 1, 2, 3, 4