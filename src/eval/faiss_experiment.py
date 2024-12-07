import os
import json
import pickle
import tempfile
import faiss
import numpy as np
import pandas as pd
from scipy import stats
from time import time
from typing import Any
from functools import reduce
from tqdm.auto import tqdm
from geopy.geocoders import Nominatim

from src import root_dir
from src.load import DataLoader
from src.embed import TextEmbedder
from src.encode import ModalityEncoder

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('macosx')


class FaissExperiment:
    def __init__(
            self,
            dataset: DataLoader,
            model_path_or_name: str,
            main_mod: str,
            aux_mods: list[str],
            num_harmonics: int = 200
    ) -> None:
        self.dataset = dataset
        self.text_embedder = self.get_embedder(model_path_or_name)
        self.main_mod = main_mod
        self.num_harmonics = num_harmonics or 200  # in case num_harmonics is None
        self.aux_encoding_schema = self.make_aux_encoding_schema(aux_mods)
        self.client = None
        self.encoder = None
        self.nominatum = Nominatim(user_agent="fuserank")

    @staticmethod
    def get_client(data_dimensionality: int) -> object:
        return faiss.IndexFlatIP(data_dimensionality)

    def get_encoder(self) -> ModalityEncoder:
        if self.encoder is not None:
            return self.encoder
        encoder = ModalityEncoder(
            text_embedding_dir=self.dataset.name,
            text_embedder=self.text_embedder,
            text_encoding_schema=self.dataset.text_encoding_schema,
            aux_encoding_schema=self.aux_encoding_schema,
            num_harmonics=self.num_harmonics,
        )
        return encoder

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
                    aux_encoding_schema[col] = "geolocation"
                else:
                    aux_encoding_schema[col] = "sparse"
            elif dtype in ("float64", "int64"):
                aux_encoding_schema[col] = "dense"
            else:
                raise ValueError(f"{dtype} data type is not supported.")
        return aux_encoding_schema

    def index_products(self, data_vectors: np.array) -> float:
        print(f"\nData dimensionality: {data_vectors.shape[0]} x {data_vectors.shape[1]}\n")
        print(f"Indexing data...\n")
        start_time = time()
        self.client.add(data_vectors)
        print("Done.")
        return time() - start_time

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
                    if self.dataset.df[col].dtype in ("object", "bool"):
                        # value = json.dumps(value, ensure_ascii=False)
                        filters[col] = value, 1.0
                elif self.aux_encoding_schema[col] == "geolocation":
                    location = self.dataset.df.loc[random_id, "Location"]
                    filters[col] = location + (False,), 1.0
                    # filters[col] = (longitude, latitude, False), 1.0

                    # city = self.dataset.df.loc[random_id, "City"]
                    # if city is None:
                    #     city = ''
                    # location = self.nominatum.geocode(city).raw
                    # if location:
                    #     latitude, longitude = float(location['lat']), float(location['lon'])
                    #     filters[col] = (longitude, latitude, False), 1.0
                    # else:
                    #     print("Address not found. Please enter a valid address.")

                    # df_city = self.dataset.df[self.dataset.df["City"] == city]
                    # lon, lat = zip(*df_city['Location'].tolist())
                    # lon_mean, lat_mean = sum(lon) / len(lon), sum(lat) / len(lat)
                    # filters[col] = (lon_mean, lat_mean, False), 1.0
        return filters

    def run_experiment(self, random_id: int, random_mods: list[str], limit: int = 10) -> (list[dict], float):
        if self.client is None:
            self.encoder = self.get_encoder()
            print("Vectorizing data...")
            data_vectors = self.encoder.encode_products(self.dataset.transformed_df, save_dir=self.dataset.name)
            print("Done.")
            self.client = self.get_client(data_dimensionality=data_vectors.shape[1])
            index_runtime = self.index_products(data_vectors)
            print(f"Indexing time: {index_runtime}")

        # Let query be the product name of the sampled row
        query_text = self.dataset.df.loc[random_id, self.main_mod]
        print(f"\nQuery: {query_text}\n")

        print("Searching in Faiss...")
        start_time = time()
        query_vector = self.encoder.encode_query(query_text, self.make_filters(random_id, random_mods))
        res = self.client.search(query_vector.reshape(1, -1), k=limit)
        runtime = time() - start_time
        print("Done.")

        print(f"Query execution time: {runtime}")
        # res_ids = res[1][0].tolist()
        res_ids = res[1].flatten()
        res_rel = res[0].flatten()
        print(f'Number of results: {len(res_ids)}')

        for i in range(len(res_ids)):
            title = f", text: {self.dataset.df.iloc[res_ids[i]][self.main_mod]}, "
            attributes = ", ".join([f'{mod}: {self.dataset.df.iloc[res_ids[i]][mod]}' for mod in random_mods])
            # city = f" {self.dataset.df.iloc[res_ids[i]]['City']} "
            # print(f"Top {i}: id: {res_ids[i]}, rel: {res_rel[i]:.4f}" + title + "{ " + attributes + city + " }")
            print(f"Top {i}: id: {res_ids[i]}, rel: {res_rel[i]:.4f}" + title + "{ " + attributes + " }")

        return res_ids, runtime
