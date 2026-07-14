from os import environ
from pathlib import Path
from sys import executable
from tomllib import load

import pyarrow as pa
from datasets import load_dataset
from dotenv import dotenv_values
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pyspark.sql import functions as F, Window, dataframe as df, SparkSession
from pyspark.sql.pandas.types import from_arrow_schema


def _search_file(cur_path: Path | str, filename: str) -> Path | None:
    cur_path = Path(cur_path).resolve()
    if cur_path.is_file():
        cur_path = cur_path.parent

    while True:
        for path in cur_path.iterdir():
            if path.is_file() and path.name == filename:
                return path
        cur_path = cur_path.parent
        if cur_path == cur_path.parent:
            break

    return None


class DataBaseConfig(BaseSettings):
    url: str = Field(
        default='jdbc:postgresql://postgres:postgres@localhost:5432/postgres',
        description='database url',
        validation_alias='URL'
    )
    username: str = Field(default='postgres', description='database username', validation_alias='USERNAME')
    password: str = Field(default='postgres', description='database password', validation_alias='PASSWORD')

    model_config = SettingsConfigDict(
        env_file='.env.db',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    @staticmethod
    def load_config():
        cur_path = Path(__file__)
        env_file = _search_file(cur_path, '.env.db')

        if env_file is not None and env_file.exists():
            values = dotenv_values(env_file)
            return DataBaseConfig(**values)
        return DataBaseConfig()


class Config(BaseModel):
    random_state: int = Field(default=42, description='random seed', validation_alias='random-state')
    database: DataBaseConfig = Field(default=DataBaseConfig.load_config(), description='database config')

    hf_write_token: str = Field(description='hugging face write token', validation_alias='WRITE_TOKEN')
    hf_read_token: str = Field(description='hugging face read token', validation_alias='READ_TOKEN')
    hf_dataset: str = Field(description='hugging face dataset path', validation_alias='dataset-path')

    dagshub_token: str = Field(description='dagshub token', validation_alias='DAGSHUB_TOKEN')
    dagshub_uri: str = Field(description='dagshub uri', validation_alias='dagshub-uri')

    @staticmethod
    def load_config():
        cur_path = Path(__file__)
        toml_path = _search_file(cur_path, 'pyproject.toml')
        env_file = _search_file(cur_path, '.env')
        data = {}

        if toml_path is not None and toml_path.exists():
            with open(toml_path, 'rb') as toml_file:
                data: dict = load(toml_file).get('tool', {}).get('config', {})

        if env_file is not None and env_file.exists():
            data.update(dotenv_values(env_file))

        environ['MLFLOW_TRACKING_USERNAME'] = 'jstnoname'
        environ['MLFLOW_TRACKING_PASSWORD'] = data['DAGSHUB_TOKEN']
        environ["PYSPARK_PYTHON"] = executable
        environ["PYSPARK_DRIVER_PYTHON"] = executable

        return Config(**data)


def time_train_test_split(
        dataframe: df.DataFrame, test_size: float = 0.2
) -> tuple[df.DataFrame, df.DataFrame, dict[str, int]]:
    days = dataframe \
        .select('departure_month', 'departure_day') \
        .distinct() \
        .withColumn('day', F.dense_rank().over(Window.orderBy('departure_month', 'departure_day'))) \
        .cache()

    all_days = days.count()
    train_days_threshold = int((1 - test_size) * all_days)

    train = days.filter(F.col('day') <= train_days_threshold).drop('day').cache()
    train = dataframe.join(train, ['departure_month', 'departure_day'], 'inner')
    test = days.filter(F.col('day') > train_days_threshold).drop('day').cache()
    test = dataframe.join(test, ['departure_month', 'departure_day'], 'inner')

    meta = {
        "all_days": all_days,
        "train_size": train_days_threshold
    }

    days.unpersist()

    return train, test, meta


def wape_metric(predictions: df.DataFrame, label_col: str, prediction_col: str):
    errors_sum = predictions \
        .select(
            F.sum(F.abs(F.col(label_col) - F.col(prediction_col))).alias("abs_err_sum"),
            F.sum(F.col(label_col)).alias("actual_sum")
        ) \
        .first()

    if errors_sum is not None and errors_sum.actual_sum != 0:
        return errors_sum.abs_err_sum / errors_sum.actual_sum
    return 0


def read_parquet(
        spark: SparkSession, url: str, token: str | None = None, batch_size: int = 10000, partition_size: int = 20
) -> df.DataFrame:
    dataset = load_dataset(url, token=token, streaming=True)['train']
    arrow_schema = pa.schema([(name, feature.pa_type) for name, feature in dataset.features.items()])
    spark_schema = from_arrow_schema(arrow_schema)

    def string_generator():
        for batch in dataset.to_pandas(batch_size=batch_size, batched=True):
            rows = batch.itertuples(index=False, name=None)
            for row in rows:
                yield row

    rdd = spark.sparkContext.parallelize(string_generator(), numSlices=partition_size)
    dataframe = spark.createDataFrame(rdd, schema=spark_schema)
    return dataframe
