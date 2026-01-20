from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

spark = (
    SparkSession.builder.appName("LocalRecommendations")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

print("✓ Spark session created (local mode)")

df_products = spark.read.csv("data/raw/products.csv", header=True, inferSchema=True)
print(f"Loaded {df_products.count()} products")

df_clean = (
    df_products.filter(col("product_id").isNotNull())
    .filter(col("description").isNotNull())
    .dropDuplicates(["product_id"])
)


@udf(StringType())
def add_image_path(product_id):
    return f"data/raw/images/{product_id}.jpg"


df_clean = df_clean.withColumn("image_path", add_image_path(col("product_id")))

df_clean.write.mode("overwrite").parquet("data/processed/products_clean.parquet")

print("✓ Cleaned data saved to data/processed/products_clean.parquet")
print(f"✓ {df_clean.count()} products ready for embedding generation")

df_clean.show(5)
spark.stop()
