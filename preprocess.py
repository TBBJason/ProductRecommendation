from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, FloatType
from PIL import Image
import io

spark = SparkSession.builder \
    .appName("ProductPreprocessing") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
    .getOrCreate()

# Image preprocessing UDF
def preprocess_image_bytes(image_bytes):
    """Resize and normalize images"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        img = img.resize((224, 224))  # CLIP input size
        # Return as bytes for storage
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        return buffer.getvalue()
    except Exception as e:
        return None

preprocess_udf = udf(preprocess_image_bytes, BinaryType())

# Text preprocessing
from pyspark.sql.functions import lower, regexp_replace, trim

df_products = spark.read.format("delta").load("/mnt/silver/products_cleaned")

df_processed = df_products \
    .withColumn("description_clean", 
                lower(trim(regexp_replace(col("description"), "[^a-zA-Z0-9\\s]", "")))) \
    .withColumn("title_clean",
                lower(trim(col("title"))))

# Join with images
df_images = spark.read.format("binaryFile").load("/mnt/silver/images/")

df_final = df_processed.join(
    df_images,
    df_processed.image_url == df_images.path,
    "left"
).select(
    "product_id",
    "title_clean",
    "description_clean",
    "category",
    "price",
    "brand",
    preprocess_udf(col("content")).alias("image_preprocessed")
)

df_final.write.format("delta").mode("overwrite").save("/mnt/gold/products_processed")