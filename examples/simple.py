from pyspark.sql import SparkSession
from flatspark import get_flattened_dataframes

spark = SparkSession.builder.getOrCreate()

# the initial nested schema does include 2 arrays
df = spark.createDataFrame(
    [
        ("Alice", ["reading", "hiking"], ["Mike"]),
        ("Bob", ["cooking"], ["Sara", "Tom"]),
        ("Charlie", [], []),
    ],
    ["name", "hobbies", "friends"],
)

# flatten the dataframe results in 3 dataframes: users, hobbies, friends
# technical ids (primary & foreign keys) are automatically added
flat_dfs = get_flattened_dataframes(df=df, root_name="users")

for table_name, flat_df in flat_dfs.items():
    print(f"Table: {table_name}")
    flat_df.printSchema()
    flat_df.show(truncate=False)

# the technical ids can be used for joins between the flattened tables
print("Joined DataFrame:")
flat_dfs["users"].join(other=flat_dfs["hobbies"], on="users_technical_id").show(
    truncate=False
)
