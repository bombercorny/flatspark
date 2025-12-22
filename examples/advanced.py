from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    ArrayType,
    StringType,
    BooleanType,
    DoubleType,
)


import pyspark.sql.functions as F
from flatspark import get_flattened_dataframes

spark = SparkSession.builder.getOrCreate()

# the initial nested schema does include arrays and structs
# note that elements do not have a natural unique identifier such as order_id or item_id
schema = StructType(
    [
        StructField("has_discount", BooleanType(), True),
        StructField("order_region", StringType(), True),
        StructField("currency", StringType(), True),
        StructField(
            "items",
            ArrayType(
                StructType(
                    [
                        StructField("item_name", StringType(), True),
                        StructField("price", DoubleType(), True),
                    ]
                )
            ),
        ),
    ]
)

data = [
    (
        True,
        "US",
        "USD",
        [{"item_name": "item1", "price": 10.0}, {"item_name": "item2", "price": 20.0}],
    ),
    (False, "EU", "EUR", [{"item_name": "item1", "price": 12.0}]),
    (False, "EU", "EUR", []),
]

df = spark.createDataFrame(data, schema).coalesce(1)


# additional transformations can be added during the flattening process
def with_lit_value(df: DataFrame, lit_value: str) -> DataFrame:
    return df.withColumn("lit_value_column", F.lit(lit_value))


# incremental loads may required to continue technical_id from previous max ids
existing_max_tech_ids = {"main_technical_id": 100, "items_technical_id": 200}


flat_dfs = get_flattened_dataframes(
    df=df,
    standard_columns=[
        "region",
        "lit_value_column",
    ],  # some columns may be usefull in all flattened tables
    root_name="orders",
    existing_max_tech_ids=existing_max_tech_ids,
    explode_strategy=F.explode_outer,  # control the explode strategy
    additional_transformations={
        "orders": {
            "function": with_lit_value,
            "kwargs": {"lit_value": "some_value"},
        }
    },
)


for name, flat_df in flat_dfs.items():
    print(f"Table: {name}")
    flat_df.printSchema()
    flat_df.show(truncate=False)
