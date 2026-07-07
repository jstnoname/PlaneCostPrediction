from pathlib import Path
from tomllib import load

from dotenv import dotenv_values
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def search_file(cur_path: Path | str, filename: str) -> Path | None:
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
        env_file = search_file(cur_path, '.env.db')

        if env_file is not None and env_file.exists():
            values = dotenv_values(env_file)
            return DataBaseConfig(**values)
        return DataBaseConfig()


class Config(BaseModel):
    random_state: int = Field(default=42, description='random seed')
    database: DataBaseConfig = Field(default=DataBaseConfig.load_config(), description='database config')

    @staticmethod
    def load_config():
        cur_path = Path(__file__)
        toml_path = search_file(cur_path, 'pyproject.toml')
        toml_data = {}

        if toml_path is not None and toml_path.exists():
            with open(toml_path, 'rb') as toml_file:
                toml_data: dict = load(toml_file).get('tool', {}).get('config', {})

        return Config(**toml_data)
