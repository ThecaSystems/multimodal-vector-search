import pandas as pd
import numpy as np
import torch
import sys
from scipy.integrate import quad
from tqdm import tqdm
from typing import Any
from src import data_dir
from src.embed import TextEmbedder


class ModalityEncoder:
    """
    This class enables multimodal retrieval and re-ranking.

    When used for retrieval, it encodes all modalities into a single vector representation to enable vector search.
    When used for re-ranking, it encodes relevance scores of each returned item with linear combination of modalities.

    Supported auxiliary encodings for multimodal retrieval:
    'sparse': a one-hot encoding for categorical data; allows the selection of multiple categories at query time
    'dense': a 2-dimensional dense vector encoding for numerical data
    'binary': a binary encoding for boolean data, takes values {-1, 1}
    """

    def __init__(
        self,
        text_embedder: TextEmbedder,
        text_embedding_dir: str = None,
        text_encoding_schema: dict[str, float] = None,
        aux_encoding_schema: dict[str, str] = None,
        num_harmonics: int = 50,
        epsilon: float = 0.0,
    ):
        assert (
            text_embedding_dir is not None or text_encoding_schema is not None
        ), "Either the text embedding schema or the folder containing saved text embeddings must be specified."
        self.text_embedding_dir = text_embedding_dir
        self.text_encoding_schema = text_encoding_schema
        self.aux_encoding_schema = aux_encoding_schema
        self.text_embedder = text_embedder
        self.num_harmonics = num_harmonics
        self.epsilon = epsilon
        self.product_data = None

    def encode_products(self, data: pd.DataFrame, batch_size: int = 10, save_dir: str = None,
                        method: str = "Retrieval") -> np.array:
        """
        Encode a dataframe of products according to the encoding schema.

        :param data: pd.DataFrame: The dataframe to be encoded
        :param batch_size: int: Batch size used when creating text embeddings. Defaults to 10.
        :param save_dir: str: Sub-directory where to save the embeddings. Defaults to None, which implies root folder.
        :param method: str: Operation mode. 'Retrieval' retrieves new items, while 'Re-ranking' re-ranks original list.

        :return: A numpy array of encoded products
        """
        assert method in ("Re-ranking", "Retrieval"), 'Only "Retrieval" and "Re-ranking" methods are supported.'

        self.product_data = data

        encoded_data = self._load_text_embeddings()
        if encoded_data is None:
            print("Creating text embeddings...")
            encoded_data = self._create_text_embeddings(self.text_encoding_schema, batch_size, save_dir)

        if method == "Retrieval":
            for column, encoding in self.aux_encoding_schema.items():
                if encoding == "sparse":
                    modality = pd.get_dummies(data[column], dtype=int).to_numpy()
                elif encoding == "binary":
                    values = data[column].dropna().unique()
                    mapping = {values[0]: -1, values[1]: 1, np.nan: 0}
                    modality = data[column].replace(mapping).to_numpy().reshape(-1, 1)
                elif encoding == "geolocation":
                    modality = np.concatenate(
                        (
                            np.vstack(data[column].apply(self._geospatial_to_cartesian)),
                            np.vstack(np.ones(data.shape[0])),  # shift dimension
                            np.vstack(self._null_indicator_dim(data[column]))  # null indicator dimension
                        ),
                        axis=1,
                    )
                elif encoding == "dense":
                    min_value, max_value = data[column].min(), data[column].max()
                    scaled_data = 2 * ((data[column] - min_value) / (max_value - min_value)) - 1
                    full_circle_encoding = self._scalar_to_fourier_series(scaled_data.to_numpy())
                    half_circle_encoding = self._scalar_to_fourier_series(
                        scaled_data.to_numpy(), freq=np.pi / 2, num_harmonics=1
                    )
                    modality = np.concatenate(
                        (
                            np.vstack(half_circle_encoding),
                            np.vstack(np.ones(data.shape[0])),  # shift dimension
                            np.vstack(full_circle_encoding),
                            np.vstack(np.ones(data.shape[0])),  # shift dimension
                            np.vstack(self._null_indicator_dim(data[column]))  # null indicator dimension
                        ),
                        axis=1,
                    )
                else:
                    continue
                encoded_data = np.concatenate((encoded_data, modality), axis=1)

        return encoded_data

    def encode_query(self, query_text: str, aux_data: dict[str, tuple[Any, float]],
                     method: str = "Retrieval") -> np.array:
        """
        Encode a query to a vector that is compatible with the product vector.

        :param query_text: str: The textual input.
        :param aux_data: dict[str, tuple[Any, float]: Auxiliary data to be blended into the query vector.
        The key is the name of the modality column.
        The first element of a tuple is the value of the modality. The second element is the weight of the modality.
        The value of any auxiliary modality can be set to None, in which case the modality is ignored in the dot product
        (equivalent to setting its weight to 0).
        The weight of any auxiliary modality is relative to the textual modality, which is set to 1. Therefore, if the
        weight is less than 1 we suppress the modality, and when it is greater than 1 we boost it.
        :param method: str: Operation mode. 'Retrieval' retrieves new items, while 'Re-ranking' re-ranks original list.

        :return: A numpy array holding the encoding of a query
        """
        assert aux_data.keys() == self.aux_encoding_schema.keys(), "The query does not comply with the encoding schema"
        assert self.product_data is not None, "The product data hasn't been encoded yet. Please call encode_products()"
        assert method in ("Re-ranking", "Retrieval"), 'Only "Retrieval" and "Re-ranking" methods are supported.'

        with torch.no_grad():
            encoded_data = self.text_embedder.embed(query_text).numpy()
            encoded_data /= np.linalg.norm(encoded_data)
        if method == "Retrieval":
            for key, (value, weight) in aux_data.items():
                encoding = self.aux_encoding_schema[key]
                if encoding == "sparse":
                    unique_values = sorted(self.product_data[key].dropna().unique())  # pd.get_dummies sorts values
                    selection, is_negated = value
                    if selection is None or selection == []:
                        modality = np.zeros(len(unique_values))  # don't do anything if no value was provided
                    else:
                        modality = np.array([1 if v in selection else -1 for v in unique_values])
                        if is_negated:
                            modality = -modality
                elif encoding == "binary":
                    if value is None:
                        modality = np.zeros(1)
                    else:
                        values = self.product_data[key].dropna().unique()
                        modality = np.array([1 if value == values[1] else -1])
                elif encoding == "geolocation":
                    if value is None:
                        modality = np.zeros(5)  # 3 dim for cartesian coordinates + shift dim + null indicator dim
                    else:
                        longitude, latitude, is_negated = value
                        points = self.product_data[key].values
                        distances = self._haversine_distance(points, (longitude, latitude))
                        farthest_point = self.product_data[key].iloc[np.nanargmax(distances)]
                        modality = self._geospatial_encoding((longitude, latitude), farthest_point)
                        if is_negated:
                            modality = -modality
                        modality = np.append(modality, 1)  # null indicator dimension
                elif encoding == "dense":
                    modality = np.zeros(self.num_harmonics * 2 + 4)  # 2 shift dimensions + 2 half-circle dimensions = 4
                    if value is not None:
                        min_value, max_value = self.product_data[key].min(), self.product_data[key].max()
                        if len(value) < 3:  # not an interval filter -> use half-circle encoding
                            try:
                                v, is_negated = value
                            except ValueError:
                                v, is_negated = value[0], False
                            if v is not None:
                                scaled_value = 2 * ((v - min_value) / (max_value - min_value)) - 1
                                modality[:3] = self._centroid_encoding(scaled_value)
                                if is_negated:
                                    modality[:3] = -modality[:3]
                        else:
                            lower_bound, upper_bound, is_negated = value
                            lower_bound = 2 * ((lower_bound - min_value) / (max_value - min_value)) - 1 - self.epsilon
                            upper_bound = 2 * ((upper_bound - min_value) / (max_value - min_value)) - 1 + self.epsilon
                            modality[3:] = self._interval_encoding(lower_bound, upper_bound)
                            if is_negated:
                                modality[3:] = -modality[3:]
                    modality = np.append(modality, 1)  # null indicator dimension
                else:
                    continue
                encoded_data = np.concatenate((encoded_data.flatten(), modality * weight))

        return encoded_data

    def encode_result(self, search_result: pd.DataFrame, aux_data: dict[str, tuple[Any, float]]) -> None:
        """
        Encode the relevance scores of search results using the linear combination of modalities, for re-ranking.
        Missing values are prioritized over non-matching values (in categorical and binary columns)

        :param search_result: pd.DataFrame: The dataframe of search results. Must contain the 'relevance' column.
        :param aux_data: dict[str, tuple[Any, float]: Auxiliary data to be used for re-ranking search results.

        :return: None. The 'relevance' column of the passed dataframe gets updated.
        """
        if "relevance" not in search_result:
            raise ValueError("The search results dataframe must contain the 'relevance' column.")

        for key, (value, weight) in aux_data.items():
            if value not in (None, []):
                encoding = self.aux_encoding_schema[key]
                # product_data contains transformed values, we need them for "dense" filters
                col = search_result.join(self.product_data, lsuffix="_drop")[key]
                if encoding == "dense":
                    if len(value) < 3:
                        try:
                            v, is_negated = value
                        except ValueError:
                            v, is_negated = value[0], False
                        if v is None:
                            continue
                        elif v == self.product_data[key].max():
                            is_ascending = True
                        elif v == self.product_data[key].min():
                            is_ascending = False
                        else:  # centroid filter
                            col = abs(v - col)
                            col = col.fillna(sys.float_info.max)
                            is_ascending = is_negated
                    else:
                        lower_bound, upper_bound, is_negated = value
                        # we assign smaller number to values inside the interval to give them better ranking
                        col = col.apply(lambda x: sys.float_info.min if lower_bound <= x <= upper_bound else x)
                        col = col.fillna(sys.float_info.max)
                        is_ascending = is_negated
                    col_rank = (col.rank(ascending=is_ascending) - 1) / (len(search_result) - 1)
                elif encoding == "geolocation":
                    longitude, latitude, is_negated = value
                    is_ascending = not is_negated
                    distances = self._haversine_distance(points=col.values, ref_point=(longitude, latitude))
                    farthest_point = col.values[np.argmax(distances)]
                    query_vec = self._geospatial_encoding((longitude, latitude), farthest_point)
                    col_vec = np.concatenate(
                        (
                            np.vstack(col.apply(self._geospatial_to_cartesian)),
                            np.vstack(np.ones(col.shape[0])),
                        ),
                        axis=1,
                    )
                    col = pd.Series(np.dot(col_vec, query_vec), index=col.index)
                    col_rank = (col.rank(ascending=is_ascending) - 1) / (len(search_result) - 1)
                elif encoding == "binary":
                    if isinstance(value, str):
                        col = col.replace(value, " ")  # assign value to the smallest char
                        col = col.fillna("\'")  # assign nans to the next smallest char
                    else:
                        col = col.replace(value, -np.inf)  # assign value to the smallest number
                        col = col.fillna(-sys.float_info.max)  # assign value to the next smallest number
                    col_rank = (col.rank(ascending=False) - 1) / (len(search_result) - 1)
                elif encoding == "sparse":
                    selection, is_negated = value
                    if any([v in col.values for v in selection]):
                        # Empty string has top ranking, and 'z' has bottom ranking
                        col = col.replace(selection, " ")
                        # col = col.replace(set(col.values) - set(selection), "_")  # treat all mismatches the same
                        col = col.fillna("\'")  # treat missing values differently (either above or below the rest)
                        col_rank = (col.rank(ascending=is_negated) - 1) / (len(search_result) - 1)
                    else:
                        continue
                else:
                    continue
                search_result["relevance"] += col_rank * weight

    def _load_text_embeddings(self) -> np.array:
        """Load pickled torch tensor from disk."""
        try:
            if self.text_embedding_dir is None:
                return
            pt_files = list((data_dir() / self.text_embedding_dir).glob("*.pt"))
            if len(pt_files) == 0:
                print(f"No .pt files found in the folder.")
                return
            text_embeddings = torch.load(pt_files[0])
            if len(pt_files) > 1:
                print(f"Warning! Multiple .pt files found in the folder. Loaded {pt_files[0]}")
            return text_embeddings.numpy()
        except (FileNotFoundError, ValueError):
            print("The provided text embeddings file is either invalid or does not exist.")
            return

    def _create_text_embeddings(self, text_encoding_schema: dict[str, float], batch_size: int,
                                save_dir: str = None) -> np.array:
        def embed_text() -> np.array:
            num_chunks = len(self.product_data) // batch_size
            remainder = (len(self.product_data) % batch_size)  # the remaining data points are distributed over chunks
            chunk_start = 0
            embedding_list = []
            for chunk_index in tqdm(range(num_chunks)):
                chunk_end = (chunk_start + batch_size + (1 if chunk_index < remainder else 0))
                column_embeddings = []
                for col in text_encoding_schema.keys():
                    text_batch = self.product_data[chunk_start:chunk_end][col.strip()].to_list()
                    with torch.no_grad():
                        embedding_batch = self.text_embedder.embed(text_batch)
                    column_embeddings.append(embedding_batch)
                weights = text_encoding_schema.values()
                weighted_embedding = sum([w * column_embeddings[i] for i, w in enumerate(weights)])
                normalized_embedding = torch.nn.functional.normalize(weighted_embedding, p=2, dim=-1)
                embedding_list.extend(normalized_embedding)
                chunk_start = chunk_end
            # Save to disk before returning
            file_name = f"text_embeddings_{self.text_embedder.model_name.split('/')[-1]}.pt"
            try:
                torch.save(torch.stack(embedding_list), data_dir() / (save_dir or "") / file_name)
            except RuntimeError:
                torch.save(torch.stack(embedding_list), data_dir() / file_name)
                print(
                    "Error saving text embedding file to the specified directory. Check if the directory exists. "
                    f"The file has been saved to the root of the data directory: {data_dir()}"
                )
            return np.array(embedding_list)

        return embed_text()

    def _interval_encoding(self, lower_bound: float, upper_bound: float) -> np.ndarray:
        vector = np.array([self._integrate(lower_bound, upper_bound, k + 1)
                           for k in range(self.num_harmonics)]).reshape(-1)
        # Normalization via scaling and shifting
        point_in = (lower_bound + upper_bound) / 2
        if abs(lower_bound + 1) > abs(1 - upper_bound):
            point_out = (lower_bound - 1) / 2
        else:
            point_out = (upper_bound + 1) / 2
        dot_in = np.dot(self._scalar_to_fourier_series(point_in), vector)
        dot_out = np.dot(self._scalar_to_fourier_series(point_out), vector)
        scaling_factor = 2 / (dot_in - dot_out)
        shift = (dot_in + dot_out) / 2
        vector_scaled = vector * scaling_factor
        return np.append(vector_scaled, -scaling_factor * shift)

    def _centroid_encoding(self, value: float) -> np.ndarray:
        vector = self._scalar_to_fourier_series(value, freq=np.pi / 2, num_harmonics=1)
        farthest_value = 1 if value < 0 else -1
        farthest_vector = self._scalar_to_fourier_series(farthest_value, freq=np.pi / 2, num_harmonics=1)
        return self._scale_and_shift(vector, farthest_vector)

    def _geospatial_encoding(self, query_point: tuple[float, float], farthest_point: tuple[float, float]) -> np.ndarray:
        vector = self._geospatial_to_cartesian(query_point)
        farthest_vector = self._geospatial_to_cartesian(farthest_point)
        return self._scale_and_shift(vector, farthest_vector)

    @staticmethod
    def _scale_and_shift(input_vector: np.array, farthest_vector: np.array) -> np.ndarray:
        """Normalize dot product range via scaling and shifting the input vector with respect to the farthest vector."""
        dot_min = np.dot(input_vector, farthest_vector)
        scaling_factor = 2 / (1 - dot_min)
        vector_scaled = input_vector * scaling_factor
        dot_max = np.dot(input_vector, vector_scaled)
        return np.append(vector_scaled, 1 - dot_max)

    @staticmethod
    def _null_indicator_dim(column: pd.Series) -> np.ndarray:
        """Create dimension with -1 for missing entries and 0 for the rest."""
        nan_mask = pd.isna(column)
        nan_dim = np.zeros(column.shape[0])
        nan_dim[nan_mask] = -1
        return nan_dim

    @staticmethod
    def _haversine_distance(points: np.ndarray[tuple[float, float]], ref_point: tuple[float, float]) -> np.array:
        """Compute the distance (in km) between a set of points and a reference point using Haversine formula
        Note: this formula assumes spherical Earth, which results in errors of up to 0.5%
        https://www.geeksforgeeks.org/program-distance-two-points-earth/
        """
        distances = np.empty(points.shape[0])
        nan_mask = pd.isnull(points)
        distances[nan_mask] = np.nan
        lon1, lat1 = np.radians(ref_point)
        lon2, lat2 = np.radians(np.stack(points[~nan_mask])[:, 0]), np.radians(np.stack(points[~nan_mask])[:, 1])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        R = 6371  # Radius of Earth
        distances[~nan_mask] = c * R
        return distances

    @staticmethod
    def _geospatial_to_cartesian(point: tuple[float, float]) -> np.ndarray:
        """Converts a point designated by a longitude / latitude pair to Cartesian coordinates on a unit sphere."""
        if point is None:
            return np.zeros(3)
        else:
            lon, lat = np.radians(point)
            x = np.cos(lat) * np.cos(lon)
            y = np.cos(lat) * np.sin(lon)
            z = np.sin(lat)
            return np.array([x, y, z])

    def _scalar_to_fourier_series(self, values: np.ndarray | float | int, freq: float = np.pi,
                                  num_harmonics: int = None) -> np.ndarray:
        """Create Fourier series in a vectorized manner."""
        if num_harmonics is None:
            num_harmonics = self.num_harmonics
        if np.any(np.abs(values) > 1):
            raise ValueError("All values should be in the range [-1, 1]")

        harmonics = np.arange(1, num_harmonics + 1)
        sigmas = self._sigma(harmonics, freq)

        sin_components = np.sin(harmonics * freq * np.array(values).reshape(-1, 1))
        cos_components = np.cos(harmonics * freq * np.array(values).reshape(-1, 1))

        # For each passed number, the Fourier series must alternate between sin and cos components
        sin_cos_interleaved = np.hstack(
            (sin_components * sigmas, cos_components * sigmas)
        ).reshape(-1, sin_components.shape[1]).T
        fourier_series = np.vstack(
            np.split(sin_cos_interleaved, sin_components.shape[0], axis=1)
        ).reshape(sin_components.shape[0], -1)

        fourier_series[np.isnan(fourier_series)] = 0  # handle missing values

        if isinstance(values, float | int):
            return fourier_series.flatten()
        else:
            return fourier_series

    def _integrate(self, lower_bound: float, upper_bound: float, k: int) -> tuple[float, float]:
        norm_factor = 1 / (upper_bound - lower_bound)
        return (
            norm_factor * quad(lambda x: np.sin(k * np.pi * x) * self._sigma(k)[0], lower_bound, upper_bound,)[0],
            norm_factor * quad(lambda x: np.cos(k * np.pi * x) * self._sigma(k)[0], lower_bound, upper_bound,)[0]
        )

    def _sigma(self, harmonics: np.ndarray | int, freq: float = np.pi) -> np.ndarray:
        """Lanczos sigma factor for reducing oscillations (Gibbs phenomenon) at discontinuities of Fourier series."""
        if isinstance(harmonics, int):
            harmonics = np.array([harmonics])
        if len(harmonics) == 1 and harmonics[0] == 1:
            return np.ones(1)
        else:
            return np.sin(freq * harmonics / self.num_harmonics) / (freq * harmonics / self.num_harmonics)
