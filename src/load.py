import pandas as pd
from src import data_dir
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from src.transform import get_best_transform, Transform
from typing import Any, Type


@dataclass
class DataLoader(ABC):
    file_path: str = field(init=False)
    df: pd.DataFrame = field(init=False)
    text_encoding_schema: dict[str, float] = field(init=False)
    transformation_schema: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.load_data()
        self.preprocess_data()
        self.create_schemas()

    def load_data(self) -> None:
        self.df = pd.read_csv(data_dir() / self.file_path, low_memory=False)

    @abstractmethod
    def preprocess_data(self) -> None:
        pass

    @abstractmethod
    def create_schemas(self) -> None:
        pass

    @staticmethod
    def transform_data(numerical_column: str = None) -> Type[Transform]:
        if numerical_column is not None:
            return get_best_transform(numerical_column)

    @property
    def transformed_df(self) -> pd.DataFrame:
        transformed_df = self.df.copy()
        for col, obj in self.transformation_schema.items():
            transformed_df[col] = obj.data
        return transformed_df

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()
