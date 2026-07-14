import mlflow
import pandas as pd
from datasets import load_dataset

from utility_data.utils import Config


class Inference:
    def __init__(self, model_name: str, model_version: str, mlflow_uri: str):
        mlflow.set_tracking_uri(mlflow_uri)

        self.model_name = model_name
        self.model_version = model_version
        self.model_uri = f'models:/{model_name}/{model_version}'
        self.pipeline = mlflow.pyfunc.load_model(self.model_uri)

    def run(self, dataframe: pd.DataFrame):
        return self.pipeline.predict(dataframe)


if __name__ == '__main__':
    if __name__ == '__main__':
        config = Config.load_config()
        inference = Inference('PlaneCostPredictor', '1', config.dagshub_uri)
        data = load_dataset(config.hf_dataset)['train'].to_pandas()
        preds = inference.run(data)

        print(preds.head(10))
