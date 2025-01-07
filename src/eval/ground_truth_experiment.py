import numpy as np
import pandas as pd
from src.load import DataLoader
from src.encode import ModalityEncoder
from src.eval.experiment import Experiment


class GroundTruthExperiment(Experiment):
    def __init__(
        self,
        dataset: DataLoader,
        model_path_or_name: str,
        main_mod: str,
        aux_mods: list[str],
    ) -> None:
        super().__init__(dataset, model_path_or_name, main_mod, aux_mods)

    def get_encoder(self) -> ModalityEncoder:
        if self.encoder is not None:
            return self.encoder
        encoder = ModalityEncoder(
            text_embedding_dir=self.dataset.name,
            text_embedder=self.text_embedder,
            text_encoding_schema=self.dataset.text_encoding_schema,
            aux_encoding_schema={},
        )
        return encoder

    def get_filtered_ids(self, random_id: int, random_mods: list[str]) -> (list[int], str):
        filters = []
        df = self.dataset.df.copy()
        df.reset_index(drop=True, inplace=True)
        for col in random_mods:
            value = self.dataset.df.loc[random_id, col]
            if self.aux_encoding_schema[col] == "dense":
                # if the sampled number is None, assign it to max
                if value == "nan" or pd.isna(value):
                    value = self.dataset.df[col].max()
                df = df[df[col] <= value]
                filters.append(f"{col} <= {value}")
            elif self.aux_encoding_schema[col] == "sparse":
                if value is None:
                    df = df[df[col].isnull()]
                else:
                    df = df[df[col] == value]
                filters.append(f"{col} == {value}")
            elif self.aux_encoding_schema[col] == "binary":
                df = df[df[col] == value]
                filters.append(f"{col} == {value}")
        return df.index.tolist(), " & ".join(filters)

    def get_results(self, random_id: int, random_mods: list[str], limit: int = 10) -> (list[dict], float):
        # Let query be the product name of the sampled row
        query_text = self.dataset.df.loc[random_id, self.main_mod]
        print("*******************")
        print(f"Query: {query_text}")
        filtered_ids, filters = self.get_filtered_ids(random_id, random_mods)
        print(filters)
        print("*******************")

        print("Computing ground truth...")
        aux_data = {}  # use empty filters
        query_vector = self.encoder.encode_query(query_text, aux_data)
        res = self.client.search(query_vector.reshape(1, -1), k=len(self.dataset.df))  # get relevance for each product
        print("Done.")

        res_ids = res[1].flatten()
        res_rel = res[0].flatten()
        mask = np.isin(res_ids, filtered_ids)
        res_ids = res_ids[mask][:limit]
        res_rel = res_rel[mask][:limit]

        return res_ids, res_rel
