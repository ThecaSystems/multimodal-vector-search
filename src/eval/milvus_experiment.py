import json
import tempfile
import numpy as np
import pandas as pd
from time import time
from tqdm.auto import tqdm
from pymilvus import MilvusClient
from src.eval.experiment import Experiment
from src.load import DataLoader
from src.encode import ModalityEncoder

class MilvusExperiment(Experiment):
    def __init__(
            self,
            dataset: DataLoader,
            model_path_or_name: str,
            main_mod: str,
            aux_mods: list[str],
            num_harmonics,
            interval_epsilon,
    ) -> None:
        super().__init__(dataset, model_path_or_name, main_mod, aux_mods)
        self.num_harmonics = num_harmonics
        self.interval_epsilon = interval_epsilon
        self.temp_files = []

    def get_client(self, data_dimensionality: int) -> (MilvusClient, object):
        db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        client = MilvusClient(db_file.name)
        client.create_collection(
            collection_name=self.dataset.name,
            dimension=data_dimensionality,
            metric_type="IP",
            index_type="IVF_FLAT"
        )
        return client, db_file

    def index_products(self, data_vectors: np.array) -> float:
        data = []
        print("Building Milvus data structure...")
        for i in tqdm(range(len(self.dataset.df)), position=0, leave=True, ascii=True):
            milvus_metadata = {}
            for col in self.aux_encoding_schema.keys():
                col_name = col.replace(" ", "_").lower()
                col_value = self.dataset.df.iloc[i][col]

                if self.aux_encoding_schema[col] == "dense":
                    milvus_metadata[col_name] = float(col_value)
                elif self.aux_encoding_schema[col] in ("sparse", "binary"):
                    milvus_metadata[col_name] = str(col_value)
                else:
                    raise ValueError(f"{col} data type is not supported. Value: {col_value}")
            item = {
                "id": i,
                "vector": data_vectors[i],
                "text": self.dataset.df.iloc[i][self.main_mod],
                **milvus_metadata
            }
            data.append(item)
        print("Done.")

        print(f"\nData dimensionality: {data_vectors.shape[0]} x {data_vectors.shape[1]}\n")

        chunk_size = int(268435456 / (data_vectors.shape[1] * 8))  # to avoid "resource exhausted" error
        print(f"Indexing data...(chunk size: {chunk_size})\n")
        start_time = time()
        for i in tqdm(range(0, len(data), chunk_size), position=0, leave=True, ascii=True):
            self.client[0].insert(collection_name=self.dataset.name, data=data[i:i + chunk_size])
        print("Done.")
        return time() - start_time

    def make_filters(self, random_id: int, random_mods: list[str]) -> str:
        filters = []
        if len(self.aux_encoding_schema) > 0:
            for col in random_mods:
                col_milvus = col.replace(" ", "_").lower()
                value = self.dataset.df.loc[random_id, col]
                if self.aux_encoding_schema[col] == "dense":
                    # if the sampled number is None, assign it to max
                    if value == 'nan' or pd.isna(value):
                        value = self.dataset.df[col].max()
                    filters.append(f"{col_milvus} <= {value}")
                elif self.aux_encoding_schema[col] in ("sparse", "binary"):
                    if value is None:
                        value = ''
                    # json handles ' and " in value
                    filters.append(f"{col_milvus} == {json.dumps(str(value), ensure_ascii=False)}")
        return " && ".join(filters)

    def __del__(self):
        for db_file in self.temp_files:
            db_file.close()

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

    def get_results(self, random_id: int, random_mods: list[str], limit: int = 10) -> (np.array, np.array):
        # Let query be the product name of the sampled row
        query_text = self.dataset.df.loc[random_id, self.main_mod]
        print(f"\nQuery: {query_text}\n")

        print("Searching in Milvus...")
        query_vector = self.encoder.encode_query(query_text, {})
        filters = self.make_filters(random_id, random_mods)
        print(filters)
        res = self.client[0].search(
            collection_name=self.dataset.name,
            data=[query_vector.tolist()],
            filter=filters,
            limit=limit,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["text"],
        )
        print("Done.")

        res_ids = []
        res_rel = []
        print(f"Number of results: {len(res[0])}")

        for i, r in enumerate(res[0]):
            res_ids.append(r['id'])
            res_rel.append(r['distance'])

        return res_ids, res_rel
