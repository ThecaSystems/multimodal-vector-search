import os
import json
import pickle
import tempfile
import numpy as np
import pandas as pd
from scipy import stats
from time import time
from typing import Any
from functools import reduce
from tqdm.auto import tqdm
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, Collection, connections
from torchvision.models.vision_transformer import Encoder

from src import root_dir
from src.load import DataLoader
from src.embed import TextEmbedder
from src.encode import ModalityEncoder

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('macosx')


class MilvusExperiment:
    def __init__(
            self,
            dataset: DataLoader,
            model_path_or_name: str,
            main_mod: str,
            aux_mods: list[str],
    ) -> None:
        self.dataset = dataset
        # dataset.df.reset_index(drop=True, inplace=True)
        self.text_embedder = self.get_embedder(model_path_or_name)
        self.main_mod = main_mod
        self.aux_encoding_schema = self.make_aux_encoding_schema(aux_mods)
        self.client = None
        self.encoder = None
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

    @staticmethod
    def get_embedder(model_path_or_name: str) -> TextEmbedder:
        if model_path_or_name.startswith("/"):  # local model, needs to be prepended with "/"
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
                    continue
                else:
                    aux_encoding_schema[col] = "sparse"
            elif dtype in ("float64", "int64"):
                aux_encoding_schema[col] = "dense"
            else:
                raise ValueError(f"{dtype} data type is not supported.")
        return aux_encoding_schema

    def index_products(self, data_vectors: np.array) -> float:
        data = []
        print("Building Milvus data structure...")
        for i in tqdm(range(len(self.dataset.df)), position=0, leave=True, ascii=True):
            milvus_metadata = {}
            for col in self.aux_encoding_schema.keys():
                #ToDo: perhaps we can check for aux_encoding_schema's values instead of column dtype?
                col_name = col.replace(" ", "_").lower()
                col_value = self.dataset.df.iloc[i][col]

                if self.dataset.df[col].dtype in ("float64", "int64"):
                    milvus_metadata[col_name] = float(col_value)
                elif self.dataset.df[col].dtype == "bool":
                    milvus_metadata[col_name] = bool(col_value)
                elif self.dataset.df[col].dtype in ("object", "category"):
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
            self.client.insert(collection_name=self.dataset.name, data=data[i:i + chunk_size])
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
                elif self.aux_encoding_schema[col] == "sparse":
                    if value is None:
                        value = ''
                    filters.append(f"{col_milvus} == {json.dumps(value, ensure_ascii=False)}")  # json handles ' and " in value
                elif self.aux_encoding_schema[col] == "binary":
                    if self.dataset.df[col].dtype in ("object", "category"):
                        value = json.dumps(value, ensure_ascii=False)
                    filters.append(f"{col_milvus} == {value}")
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

    # def analyze(
    #         self,
    #         runtimes_native: dict[int, list[float]],
    #         runtimes_fuserank: dict[int, list[float]],
    # ) -> None:
    #     if runtimes_native is None:
    #         with open('runtimes_native_milvus.pkl', 'rb') as file:
    #             runtimes_native = pickle.load(file)
    #     if runtimes_fuserank is None:
    #         with open('runtimes_fuserank_milvus.pkl', 'rb') as file:
    #             runtimes_fuserank = pickle.load(file)
    #
    #     runtimes_native_np = np.ma.masked_invalid(list(runtimes_native.values()))
    #     runtimes_fuserank_np = np.ma.masked_invalid(list(runtimes_fuserank.values()))
    #     runtimes_native_means = runtimes_native_np.mean(axis=1)
    #     runtimes_fuserank_means = runtimes_fuserank_np.mean(axis=1)
    #     native_std_err = np.std(runtimes_native_np, axis=1) / np.sqrt(runtimes_native_np.shape[1])
    #     fuserank_std_err = np.std(runtimes_fuserank_np, axis=1) / np.sqrt(runtimes_fuserank_np.shape[1])
    #
    #     fig, ax1 = plt.subplots()
    #     plt.errorbar(list(runtimes_native.keys()), runtimes_native_means, yerr=native_std_err, fmt='-o', label='Native', color='b')
    #     plt.errorbar(list(runtimes_fuserank.keys()), runtimes_fuserank_means, yerr=fuserank_std_err, fmt='-o', label='Fuserank', color='g')
    #
    #     plt.xlabel('Number of filters')
    #     plt.ylabel('Runtime mean (s)')
    #     plt.title('Comparison of runtimes for varying filter sizes (BIN_IVF_FLAT)')
    #     plt.legend()
    #     plt.show()
    #
    #     fig.tight_layout()
    #     plt.show(block=True)

    # def vectorize_query(self, filters: dict[str, tuple[Any, float]] | str) -> np.array:
    #     is_native = isinstance(filters, str)
    #     return self.encoder[is_native].encode_query(self.query_text, {} if is_native else filters)

    def run_experiment(self, random_id: int, random_mods: list[str], limit: int = 10) -> (list[dict], float):
        if self.client is None:
            self.encoder = self.get_encoder()
            print("Vectorizing data...")
            data_vectors = self.encoder.encode_products(self.dataset.transformed_df, save_dir=self.dataset.name)
            print("Done.")
            self.client, db_file = self.get_client(data_dimensionality=data_vectors.shape[1])
            self.temp_files.append(db_file)
            index_runtime = self.index_products(data_vectors)
            print(f"Indexing time: {index_runtime}")

        # Let query be the product name of the sampled row
        query_text = self.dataset.df.loc[random_id, self.main_mod]
        print(f"\nQuery: {query_text}\n")

        print("Searching in Milvus...")
        start_time = time()
        query_vector = self.encoder.encode_query(query_text, {})
        filters = self.make_filters(random_id, random_mods)
        print(filters)
        res = self.client.search(
            collection_name=self.dataset.name,
            data=[query_vector.tolist()],
            filter=filters,
            limit=limit,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["text"],  # + [col.replace(" ", "_").lower() for col in selected_mods],
        )
        runtime = time() - start_time
        print("Done.")

        print(f"Query execution time: {runtime}")
        print(f'Number of results: {len(res[0])}')
        res_ids = []

        for i, r in enumerate(res[0]):
            title = f", text: {self.dataset.df.iloc[r['id']][self.main_mod]}, "
            attributes = ", ".join([f'{mod}: {self.dataset.df.iloc[r["id"]][mod]}' for mod in random_mods])
            print(f"Top {i}: id: {r['id']}, rel: {r['distance']:.4f}" + title + "{ " + attributes + " }")
            res_ids.append(r['id'])

        # for i, r in enumerate(res[0]):
        #     print("Top {}: {}".format(i, r))
        #     res_ids.append(r['id'])

        return res_ids, runtime
