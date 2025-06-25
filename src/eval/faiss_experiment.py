import numpy as np
import pandas as pd
from typing import Any
from src.load import DataLoader
from src.encode import ModalityEncoder
from src.eval.experiment import Experiment


class FaissExperiment(Experiment):
    def __init__(
        self,
        dataset: DataLoader,
        model_path_or_name: str,
        main_mod: str,
        aux_mods: list[str],
        num_harmonics: int,
        interval_epsilon: float,
    ) -> None:
        super().__init__(dataset, model_path_or_name, main_mod, aux_mods)
        self.num_harmonics = num_harmonics
        self.interval_epsilon = interval_epsilon

    def get_encoder(self) -> ModalityEncoder:
        if self.encoder is not None:
            return self.encoder
        encoder = ModalityEncoder(
            text_embedding_dir=self.dataset.name,
            text_embedder=self.text_embedder,
            text_encoding_schema=self.dataset.text_encoding_schema,
            aux_encoding_schema=self.aux_encoding_schema,
            num_harmonics=self.num_harmonics,
            interval_epsilon=self.interval_epsilon,
        )
        return encoder

    def make_filters(self, random_id: int, random_mods: list[str]) -> dict[str, Any]:
        filters = dict.fromkeys(self.aux_encoding_schema, (None, 1.0))
        if len(self.aux_encoding_schema) > 0:
            for col in random_mods:
                value = self.dataset.df.loc[random_id, col]
                if self.aux_encoding_schema[col] == "dense":
                    # if the sampled number is None, assign it to max
                    if value is None or pd.isna(value):
                        value = self.dataset.df[col].max()
                    lower_bound = self.dataset.df[col].min()
                    upper_bound = value
                    if col in self.dataset.transformation_schema:
                        lower_bound = self.dataset.transformation_schema[col].transform(lower_bound)
                        upper_bound = self.dataset.transformation_schema[col].transform(value)
                    filters[col] = (lower_bound, upper_bound, False), 1.0
                elif self.aux_encoding_schema[col] == "sparse":
                    filters[col] = ([value], False), 1.0
                elif self.aux_encoding_schema[col] == "binary":
                    filters[col] = value, 1.0
                elif self.aux_encoding_schema[col] == "geolocation":
                    location = self.dataset.df.loc[random_id, "Location"]
                    filters[col] = location + (False,), 1.0
        return filters

    def get_results(self, random_id: int, random_mods: list[str], limit: int = 10) -> (np.array, np.array):
        # Let query be the product name of the sampled row
        query_text = self.dataset.df.loc[random_id, self.main_mod]
        print(f"\nQuery: {query_text}\n")

        print("Searching in Faiss...")
        query_vector = self.encoder.encode_query(query_text, self.make_filters(random_id, random_mods))
        res = self.client.search(query_vector.reshape(1, -1), k=limit)
        print("Done.")

        res_ids = res[1].flatten()
        res_rel = res[0].flatten()
        print(f"Number of results: {len(res_ids)}")

        return res_ids, res_rel
