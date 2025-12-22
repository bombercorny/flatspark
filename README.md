# flatspark
An opinionated pyspark package to handle deeply nested DataFrames with ease. 

## Key Features
- Flatten deeply nested DataFrames with arrays and structs
- Automatic generation of technical IDs for joins
- Customizable explode strategies for arrays
- Support for incremental loads with existing technical IDs
- Apply additional transformations during flattening
- Select standard columns across all flattened tables

## Getting Started

`pip install flatspark`

```
from pyspark.sql import SparkSession
from flatspark import get_flattened_dataframes

spark = SparkSession.builder.getOrCreate()

# inital DataFrame contains one array
df = spark.createDataFrame(
    [
        ("Alice", ["reading", "hiking"]),
        ("Bob", ["cooking"]),
        ("Charlie", []),
    ],
    ["name", "hobbies"],
)

flat_dfs = get_flattened_dataframes(df=df, root_name="friends")
```

A primary key column has been added to the root DataFrame `friends`: 

```
flat_dfs["friends"].show()

+-------+--------------------+                                                  
|name   |friends_technical_id|
+-------+--------------------+
|Alice  |0                   |
|Bob    |1                   |
|Charlie|2                   |
+-------+--------------------+
```

The array `hobbies` has been seperated in its own DataFrame with a primary key and a foreign key column to enable joins: 

```
+--------------------+-------+--------------------+
|friends_technical_id|hobbies|hobbies_technical_id|
+--------------------+-------+--------------------+
|0                   |reading|0                   |
|0                   |hiking |1                   |
|1                   |cooking|2                   |
|2                   |NULL   |3                   |
+--------------------+-------+--------------------+
```

### Additional Examples
Additional examples showing the features of the package can be found here: 
- [simple example](examples/simple.py)
- [advanced example](examples/advanced.py)



