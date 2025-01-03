import faiss
import numpy as np
from time import time
from abc import ABC, abstractmethod
from src import root_dir
from src.load import DataLoader
from src.embed import TextEmbedder
from src.encode import ModalityEncoder


class Experiment(ABC):
    def __init__(
        self,
        dataset: DataLoader,
        model_path_or_name: str,
        main_mod: str,
        aux_mods: list[str],
    ) -> None:
        self.dataset = dataset
        self.text_embedder = self.get_embedder(model_path_or_name)
        self.main_mod = main_mod
        self.aux_encoding_schema = self.make_aux_encoding_schema(aux_mods)
        self.client = None
        self.encoder = None

    @staticmethod
    def get_client(data_dimensionality: int) -> object:
        return faiss.IndexFlatIP(data_dimensionality)

    @abstractmethod
    def get_encoder(self) -> ModalityEncoder:
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    def get_embedder(model_path_or_name: str) -> TextEmbedder:
        if model_path_or_name.startswith(
            "/"
        ):  # local model, needs to be prepended with "/"
            model_path_or_name = str(root_dir()) + model_path_or_name
        try:
            return TextEmbedder(model_path_or_name)
        except Exception:
            Exception(f'Model path "{model_path_or_name}" not found.')

    def make_aux_encoding_schema(self, aux_mods: list[str]) -> dict[str, str]:
        aux_encoding_schema = {}
        for col in aux_mods:
            dtype = self.dataset.df[col].dtype
            if self.dataset.df[col].nunique() == 2:
                aux_encoding_schema[col] = "binary"
            elif dtype in ("object", "category"):
                if self.dataset.df[col].apply(lambda x: isinstance(x, tuple)).any():
                    aux_encoding_schema[col] = "geolocation"
                else:
                    aux_encoding_schema[col] = "sparse"
            elif dtype in ("float64", "int64"):
                aux_encoding_schema[col] = "dense"
            else:
                raise ValueError(f"{dtype} data type is not supported.")
        return aux_encoding_schema

    def index_products(self, data_vectors: np.array) -> float:
        print(
            f"\nData dimensionality: {data_vectors.shape[0]} x {data_vectors.shape[1]}\n"
        )
        print(f"Indexing data...\n")
        start_time = time()
        self.client.add(data_vectors)
        print("Done.")
        return time() - start_time

    def init_experiment(self) -> None:
        if self.client is None:
            self.encoder = self.get_encoder()
            print("Vectorizing data...")
            data_vectors = self.encoder.encode_products(
                self.dataset.transformed_df, save_dir=self.dataset.name
            )
            print("Done.")
            self.client = self.get_client(data_dimensionality=data_vectors.shape[1])
            index_runtime = self.index_products(data_vectors)
            print(f"Indexing time: {index_runtime}")

    def print_experiment(
        self, res_ids: np.array, res_rel: np.array, random_mods: list[str]
    ) -> None:
        for i in range(len(res_ids)):
            title = f", text: {self.dataset.df.iloc[res_ids[i]][self.main_mod]}, "
            attributes = ", ".join(
                [
                    f"{mod}: {self.dataset.df.iloc[res_ids[i]][mod]}"
                    for mod in random_mods
                ]
            )
            print(
                f"Top {i}: id: {res_ids[i]}, rel: {res_rel[i]:.4f}"
                + title
                + "{ "
                + attributes
                + " }"
            )

    @abstractmethod
    def get_results(
        self, random_id: int, random_mods: list[str], limit: int = 10
    ) -> (list[dict], float):
        raise NotImplementedError("Subclasses must implement this method")

    def run_experiment(
        self, random_id: int, random_mods: list[str], limit: int = 10
    ) -> (list[dict], float):
        self.init_experiment()
        res_ids, res_rel = self.get_results(random_id, random_mods, limit)
        self.print_experiment(res_ids, res_rel, random_mods)
        return res_ids
