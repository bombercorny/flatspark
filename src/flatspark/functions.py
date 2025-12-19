import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType
from typing import Callable, Optional
# from pyspark.sql.connect.dataframe import DataFrame as DataFrame_c


def flatten_struct_columns(nested_df: DataFrame, seperator: str = "__"):
    """ "Flatten all subfields in StructType columns. Arrays column remain unchanged.

    Subfields in the StructType columns are renamed using '__' as a seperator.
    """
    stack = [((), nested_df)]
    columns = []

    while len(stack) > 0:
        parents, df = stack.pop()

        flat_cols = [
            F.col(".".join(parents + (c[0],))).alias(seperator.join(parents + (c[0],)))
            for c in df.dtypes
            if c[1][:6] != "struct"
        ]

        nested_cols = [c[0] for c in df.dtypes if c[1][:6] == "struct"]
        columns.extend(flat_cols)
        for nested_col in nested_cols:
            projected_df = df.select(nested_col + ".*")
            stack.append((parents + (nested_col,), projected_df))  # type: ignore
    return nested_df.select(columns)


def get_exploded_array_column(
    df: DataFrame,
    array_column: str,
    technical_id_columns: list[str],
    standard_columns: list[str],
) -> DataFrame:
    available_standard_columns = [col for col in standard_columns if col in df.columns]
    return df.select(
        *technical_id_columns, *available_standard_columns, array_column
    ).withColumn(array_column, F.explode_outer(array_column))


def add_technical_id(
    df: DataFrame, column_name: str, max_existing_id: Optional[dict[str, int]]
) -> DataFrame:
    """Adds a technical id to the dataframe.

    The technical id is a monotonically increasing id starting
    from 0 or the maximum value of the column if provided.

    Args:
        df: The dataframe to add the technical id to
        column_name:
            The name of technical id column. E.g. businessPartners__addresses_technical_id.
        max_existing_id:
            A dictionary containing the maximum value of each technical id column from previous loads.
            If the column does not exist, it is initialized to an empty dictionary.
            E.g. {"addresses_technical_id": 100}

    Important:
        column_name is using the long form including the the full path to the column
        while max_existing_id uses the final column names after prefixes have been removed!
        This needs to be accounted for when checking the max values for different columns.
    """
    starting_id = 0
    if max_existing_id is None:
        max_existing_id = {}

    # the short column name is always a part of the long column name
    for short_column, previous_max_id in max_existing_id.items():
        if short_column.lower() in column_name.lower():
            starting_id = previous_max_id + 1
            break

    df_with_tech_id = df.withColumn(
        column_name, F.monotonically_increasing_id() + starting_id
    )
    return df_with_tech_id


def get_array_column_names(df: DataFrame) -> list[str]:
    return [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, ArrayType)
    ]


def get_dataframe_dicts(
    df: DataFrame,
    standard_columns: list[str],
    level: str = "main",
    technical_id_suffix: str = "_technical_id",
    additional_transformations: Optional[
        dict[str, Callable[[DataFrame, dict], DataFrame]]
    ] = None,
    flattened_dfs: Optional[dict[str, DataFrame]] = None,
    technical_id_columns: Optional[list[str]] = None,
    seperator: str = "__",
    existing_max_tech_ids: Optional[dict[str, int]] = None,
) -> dict[str, DataFrame]:
    """Flattens all nested StructTypes and Arrays"""
    if not additional_transformations:
        additional_transformations = {}

    tech_id_column = f"{level}{technical_id_suffix}"
    if not technical_id_columns:
        technical_id_columns = [tech_id_column]
    # add the logic here to address technical_ids
    df = add_technical_id(df, tech_id_column, max_existing_id=existing_max_tech_ids)

    if not flattened_dfs:
        flattened_dfs = {level: df}

    updated_dfs = {}
    for current_level, df in flattened_dfs.items():
        flat_with_arrays_df = flatten_struct_columns(df, seperator)
        array_columns = get_array_column_names(flat_with_arrays_df)

        # some transformations need to take place during flattening
        # in order to include these columns in follow up flattenings
        # e.g. adding additional standard columns like 'isMainScoring'
        if additional_transformations.get(current_level):
            flat_with_array_transformed_df = additional_transformations[current_level][
                "function"
            ](
                flat_with_arrays_df,
                **additional_transformations[current_level].get("kwargs", {}),
            )
        else:
            flat_with_array_transformed_df = flat_with_arrays_df

        updated_dfs[current_level] = flat_with_array_transformed_df.drop(*array_columns)

        for array_column in array_columns:
            exploded_array_df = get_exploded_array_column(
                flat_with_array_transformed_df,
                array_column,
                technical_id_columns,
                standard_columns,
            )
            updated_dfs[array_column] = get_dataframe_dicts(
                df=exploded_array_df,
                standard_columns=standard_columns,
                level=array_column,
                flattened_dfs=None,
                technical_id_columns=technical_id_columns
                + [f"{array_column}{technical_id_suffix}"],
                additional_transformations=additional_transformations,
                seperator=seperator,
                existing_max_tech_ids=existing_max_tech_ids,
            )
    return updated_dfs


def flatten_dicts(
    nested_dict: dict,
) -> dict:
    """Restructures the dictionnary of nested dicts in order to return a flat dict"""
    flat_dict = {}

    def _flatten(nested_dict):
        for key, value in nested_dict.items():
            if isinstance(value, DataFrame):
                flat_dict[key.replace("-", "_")] = value
            elif isinstance(value, dict):
                _flatten(value)

    _flatten(nested_dict)
    return flat_dict


def get_flattened_dataframes(
    df: DataFrame,
    standard_columns: list[str],
    level: str = "main",
    technical_id_suffix: str = "_technical_id",
    additional_transformations: Optional[
        dict[str, Callable[[DataFrame], DataFrame]]
    ] = None,
    seperator: str = "__",
    existing_max_tech_ids: Optional[dict[str, int]] = None,
) -> dict[str, DataFrame]:
    """Flattens all nested structs and arrays and returns a dictionnary of the flattened DataFrames"""
    dataframe_dicts = get_dataframe_dicts(
        df=df,
        standard_columns=standard_columns,
        level=level,
        technical_id_suffix=technical_id_suffix,
        additional_transformations=additional_transformations,
        seperator=seperator,
        existing_max_tech_ids=existing_max_tech_ids,
    )
    flattened_dataframes = flatten_dicts(dataframe_dicts)
    return flattened_dataframes
