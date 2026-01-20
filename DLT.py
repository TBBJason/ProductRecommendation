import dlt
from delta.tables import DeltaTable

# End-to-end pipeline
@dlt.table(name="gold_product_embeddings")
@dlt.expect_or_fail("valid_embedding", "SIZE(embedding) = 896")
def generate_product_embeddings():
    """
    Full pipeline: read processed products, generate embeddings
    """
    return (
        spark.read.format("delta").load("/mnt/gold/products_processed")
        .withColumn("embedding", generate_embeddings_udf(col("image_preprocessed"), col("description_clean")))
        .withColumn("embedding_timestamp", current_timestamp())
    )

# Data quality monitoring
@dlt.table(name="data_quality_metrics")
def quality_metrics():
    return spark.sql("""
        SELECT
            'products' as table_name,
            COUNT(*) as total_records,
            SUM(CASE WHEN embedding IS NULL THEN 1 ELSE 0 END) as null_embeddings,
            AVG(price) as avg_price,
            CURRENT_TIMESTAMP() as check_timestamp
        FROM LIVE.gold_product_embeddings
    """)
