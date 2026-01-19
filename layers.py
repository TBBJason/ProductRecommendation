import dlt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp

@dlt.table(
    name="bronze_products_raw",
    comment="Raw product data with images and descriptions"
)
def ingest_products():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.schemaLocation", "/mnt/schemas/products")
        .load("/mnt/raw/products/")
        .withColumn("ingestion_timestamp", current_timestamp())
    )

@dlt.table(name="bronze_images_raw")
def ingest_images():
    # Images stored as paths in cloud storage
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "binaryFile")
        .load("/mnt/raw/images/")
    )

@dlt.table(
    name="silver_products_cleaned",
    comment="Validated and cleaned product data"
)

@dlt.expect_or_drop("valid_product_id", "product_id IS NOT NULL")
@dlt.expect_or_drop("valid_description", "LENGTH(description) > 10")
def clean_products():
    return (
        dlt.read_stream("bronze_products_raw")
        .select(
            col("product_id"),
            col("title"),
            col("description"),
            col("category"),
            col("price"),
            col("brand"),
            col("image_url")
        )
        .dropDuplicates(["product_id"])
    )