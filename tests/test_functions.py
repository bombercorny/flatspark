import pytest


from chispa import assert_df_equality
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from pyspark.sql import DataFrame, SparkSession
from flatspark.functions import flatten_struct_columns


@pytest.fixture(scope="module")
def spark_session():
    spark_session = SparkSession.builder.getOrCreate()
    yield spark_session
    spark_session.stop()


@pytest.fixture(scope="module")
def df1(spark_session):
    """Setup some test data frames to be used in all test cases"""
    df = spark_session.createDataFrame(
        [
            (
                1,
                "John Doe",
                30,
                {
                    "street": "123 Elm St",
                    "city": "Springfield",
                    "zip": "12345",
                    "coordinates": {
                        "latitude": "40.7128N",
                        "longitude": "74.0060W",
                    },
                    "phone_numbers": [
                        {
                            "type": "home",
                            "number": "555-1234",
                            "phone_details": {
                                "carrier": "Verizon",
                                "country_code": "1",
                            },
                            "tags": ["personal", "emergency"],
                        },
                        {
                            "type": "mobile",
                            "number": "555-5678",
                            "phone_details": {
                                "carrier": "AT&T",
                                "country_code": "1",
                            },
                            "tags": ["work"],
                        },
                    ],
                },
                ["john.doe@example.com", "j.doe@example.com"],
            ),
            (
                2,
                "Jane Smith",
                25,
                {
                    "street": "456 Oak St",
                    "city": "Shelbyville",
                    "zip": "67890",
                    "coordinates": {
                        "latitude": "34.0522N",
                        "longitude": "118.2437W",
                    },
                    "phone_numbers": [
                        {
                            "type": "home",
                            "number": "555-8765",
                            "phone_details": {
                                "carrier": "T-Mobile",
                                "country_code": "1",
                            },
                            "tags": ["home"],
                        },
                        {
                            "type": "work",
                            "number": "555-4321",
                            "phone_details": {
                                "carrier": "Verizon",
                                "country_code": "1",
                            },
                            "tags": ["work", "client"],
                        },
                    ],
                },
                ["jane.smith@example.com"],
            ),
            (
                3,
                "Alice Johnson",
                35,
                {
                    "street": "789 Pine St",
                    "city": "Capital City",
                    "zip": "11223",
                    "coordinates": {
                        "latitude": "37.7749N",
                        "longitude": "122.4194W",
                    },
                    "phone_numbers": [
                        {
                            "type": "home",
                            "number": "555-9876",
                            "phone_details": {
                                "carrier": "Sprint",
                                "country_code": "1",
                            },
                            "tags": ["personal"],
                        }
                    ],
                },
                [
                    "alice.j@example.com",
                    "a.johnson@example.com",
                    "alice@example.com",
                ],
            ),
        ],
        schema=StructType(
            [
                StructField("id", IntegerType(), True),
                StructField("name", StringType(), True),
                StructField("age", IntegerType(), True),
                StructField(
                    "address",
                    StructType(
                        [  # StructType column
                            StructField("street", StringType(), True),
                            StructField("city", StringType(), True),
                            StructField("zip", StringType(), True),
                            StructField(
                                "coordinates",
                                StructType(
                                    [  # Nested StructType
                                        StructField("latitude", StringType(), True),
                                        StructField("longitude", StringType(), True),
                                    ]
                                ),
                                True,
                            ),
                            StructField(
                                "phone_numbers",
                                ArrayType(  # ArrayType inside StructType
                                    StructType(
                                        [
                                            StructField("type", StringType(), True),
                                            StructField("number", StringType(), True),
                                            StructField(
                                                "phone_details",
                                                StructType(
                                                    [  # Nested Struct within Array of Structs
                                                        StructField(
                                                            "carrier",
                                                            StringType(),
                                                            True,
                                                        ),
                                                        StructField(
                                                            "country_code",
                                                            StringType(),
                                                            True,
                                                        ),
                                                    ]
                                                ),
                                                True,
                                            ),
                                            StructField(
                                                "tags",
                                                ArrayType(StringType()),
                                                True,
                                            ),  # Array within Array of Structs
                                        ]
                                    )
                                ),
                                True,
                            ),
                        ]
                    ),
                    True,
                ),
                StructField(
                    "email_addresses", ArrayType(StringType()), True
                ),  # ArrayType column
            ]
        ),
    )

    return df


def test_flatten_struct_columns(df1: DataFrame, spark_session: SparkSession):
    expected_schema = StructType(
        [
            StructField("id", IntegerType(), True),
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("address__street", StringType(), True),
            StructField("address__city", StringType(), True),
            StructField("address__zip", StringType(), True),
            StructField("address__coordinates__latitude", StringType(), True),
            StructField("address__coordinates__longitude", StringType(), True),
            StructField(
                "address__phone_numbers",
                ArrayType(
                    StructType(
                        [
                            StructField("type", StringType(), True),
                            StructField("number", StringType(), True),
                            StructField(
                                "phone_details",
                                StructType(
                                    [
                                        StructField("carrier", StringType(), True),
                                        StructField("country_code", StringType(), True),
                                    ]
                                ),
                                True,
                            ),
                            StructField("tags", ArrayType(StringType()), True),
                        ]
                    )
                ),
                True,
            ),
            StructField("email_addresses", ArrayType(StringType()), True),
        ]
    )
    expected_data = [
        {
            "id": 1,
            "name": "John Doe",
            "age": 30,
            "address__street": "123 Elm St",
            "address__city": "Springfield",
            "address__zip": "12345",
            "address__coordinates__latitude": "40.7128N",
            "address__coordinates__longitude": "74.0060W",
            "address__phone_numbers": [
                {
                    "type": "home",
                    "number": "555-1234",
                    "phone_details": {"carrier": "Verizon", "country_code": "1"},
                    "tags": ["personal", "emergency"],
                },
                {
                    "type": "mobile",
                    "number": "555-5678",
                    "phone_details": {"carrier": "AT&T", "country_code": "1"},
                    "tags": ["work"],
                },
            ],
            "email_addresses": ["john.doe@example.com", "j.doe@example.com"],
        },
        {
            "id": 2,
            "name": "Jane Smith",
            "age": 25,
            "address__street": "456 Oak St",
            "address__city": "Shelbyville",
            "address__zip": "67890",
            "address__coordinates__latitude": "34.0522N",
            "address__coordinates__longitude": "118.2437W",
            "address__phone_numbers": [
                {
                    "type": "home",
                    "number": "555-8765",
                    "phone_details": {"carrier": "T-Mobile", "country_code": "1"},
                    "tags": ["home"],
                },
                {
                    "type": "work",
                    "number": "555-4321",
                    "phone_details": {"carrier": "Verizon", "country_code": "1"},
                    "tags": ["work", "client"],
                },
            ],
            "email_addresses": ["jane.smith@example.com"],
        },
        {
            "id": 3,
            "name": "Alice Johnson",
            "age": 35,
            "address__street": "789 Pine St",
            "address__city": "Capital City",
            "address__zip": "11223",
            "address__coordinates__latitude": "37.7749N",
            "address__coordinates__longitude": "122.4194W",
            "address__phone_numbers": [
                {
                    "type": "home",
                    "number": "555-9876",
                    "phone_details": {"carrier": "Sprint", "country_code": "1"},
                    "tags": ["personal"],
                }
            ],
            "email_addresses": [
                "alice.j@example.com",
                "a.johnson@example.com",
                "alice@example.com",
            ],
        },
    ]
    expected_df = spark_session.createDataFrame(expected_data, expected_schema)
    resulted_df = flatten_struct_columns(df1)
    # return resulted_df,expected_df
    assert_df_equality(
        expected_df, resulted_df, ignore_row_order=True, ignore_column_order=True
    )
