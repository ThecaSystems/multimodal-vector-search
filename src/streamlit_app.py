import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

with st.spinner("Loading libraries..."):
    import sys
    import importlib
    import faiss
    import argparse
    import requests
    from src import root_dir, load, encode
    from embed import TextEmbedder
    from typing import Any
    from time import time


@st.cache_data
def load_data(dataset: str) -> load.DataLoader:
    try:
        module_name = f"load_{dataset.lower()}"
        dataset_module = importlib.import_module(module_name)
        return getattr(dataset_module, dataset.title())()
    except (ModuleNotFoundError, AttributeError):
        st.error(f"Dataset {dataset} not found.", icon="🚨")
        sys.exit(1)


@st.cache_resource
def load_model(model_path_or_name: str) -> TextEmbedder:
    if model_path_or_name.startswith(
        "/"
    ):  # local model, needs to be prepended with "/"
        model_path_or_name = str(root_dir()) + model_path_or_name
    try:
        return TextEmbedder(model_path_or_name)
    except Exception:
        st.error(f'Model path "{model_path_or_name}" not found.', icon="🚨")
        sys.exit(1)


def get_aux_encoding_schema() -> dict[str, str]:
    aux_encoding_schema = {}
    for col in st.session_state["modalities"]:
        dtype = data.df[col].dtype
        if data.df[col].nunique() == 2:
            aux_encoding_schema[col] = "binary"
        elif dtype in ("object", "category"):
            if isinstance(data.df[col].iloc[0], tuple):
                aux_encoding_schema[col] = "geolocation"
            else:
                aux_encoding_schema[col] = "one_hot"
        elif dtype in ("float64", "int64"):
            aux_encoding_schema[col] = "dense"
        else:
            raise ValueError(f"{dtype} data type is not supported.")
    return aux_encoding_schema


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of the dataset to load")
    parser.add_argument(
        "--model", type=str, help="Name or path of the text embedding model to use"
    )
    return parser.parse_args()


def init_session_vars(var_names: list[str]) -> None:
    for var in var_names:
        if var not in st.session_state:
            st.session_state[var] = None


def do_search(
    query_text: str, aux_data: dict[str, Any], display_columns: list[str]
) -> None:
    if query_text:
        with st.spinner("Searching..."):
            start_time = time()
            encoded_query = st.session_state["encoder"].encode_query(
                query_text, aux_data, st.session_state["method"]
            )
            distances, indices = st.session_state["index"].search(
                encoded_query.reshape(1, -1), k=num_results
            )
            result_df = data.df.iloc[indices[0]][:]
            result_df["relevance"] = distances.reshape(-1, 1)
            if st.session_state["method"] == "Re-ranking":
                st.session_state["encoder"].encode_result(result_df, aux_data)
            result_df.sort_values(by="relevance", ascending=False, inplace=True)
            st.dataframe(
                result_df[display_columns + ["relevance"]],
                hide_index=True,
                use_container_width=True,
            )
            st.caption(f":watch: {time() - start_time:.3f} sec")
            # Plot geolocation data
            for key, value in aux_data.items():
                if st.session_state["aux_encoding_schema"][key] == "geolocation":
                    plot_locations(result_df, key, value[0])


def plot_locations(
    result_df: pd.DataFrame, geolocation_column: str, point_q: tuple[float, float]
) -> None:
    if point_q is not None:
        longitudes = result_df[geolocation_column].apply(lambda x: x[0]).tolist()
        latitudes = result_df[geolocation_column].apply(lambda x: x[1]).tolist()
        colors = ["#3498db"] * len(latitudes)  # blue color for products
        colors.append("#2ecc71")  # green color for query
        longitudes.append(point_q[0])
        latitudes.append(point_q[1])
        map_data = pd.DataFrame(
            {"longitude": longitudes, "latitude": latitudes, "colors": colors}
        )
        map_placeholder[geolocation_column].map(map_data, color="colors", zoom=4)


args = get_args()

session_vars = [
    "modalities",
    "display_columns",
    "aux_encoding_schema",
    "index",
    "encoder",
    "method",
    "ranking",
]
init_session_vars(session_vars)

with st.spinner("Loading dataset..."):
    data = load_data(args.dataset)


with st.spinner("Loading model..."):
    text_embedder = load_model(args.model)


col1, col2, col3 = st.columns([0.25, 0.5, 0.25], gap="large")


with col1:
    st.subheader("Index products", divider="blue")
    columns = [
        column
        for column in data.df.columns
        if column not in data.text_encoding_schema
        and data.df[column].nunique(dropna=True) > 1
    ]
    st.session_state["modalities"] = st.multiselect("Choose modalities", columns)
    method = st.radio("Choose method", ["Retrieval", "Re-ranking"])
    if method != st.session_state["method"]:
        st.session_state["encoder"] = None
    st.session_state["method"] = method
    if st.button("Index products"):
        with st.status("Indexing dataset..."):
            start_time = time()
            st.session_state["aux_encoding_schema"] = get_aux_encoding_schema()
            st.write("Encoding products...")
            st.session_state["encoder"] = encode.ModalityEncoder(
                text_embedder=text_embedder,
                text_embedding_dir=args.dataset,
                text_encoding_schema=data.text_encoding_schema,
                aux_encoding_schema=st.session_state["aux_encoding_schema"],
            )
            encoded_products = st.session_state["encoder"].encode_products(
                data=data.transformed_df,
                save_dir=args.dataset,
                method=st.session_state["method"],
            )
            st.write("Indexing products...")
            st.session_state["index"] = faiss.IndexFlatIP(encoded_products.shape[1])
            st.session_state["index"].add(encoded_products)
            st.write(f"Elapsed time: {time() - start_time}")
        st.session_state["display_columns"] = [
            list(data.text_encoding_schema.keys())[0]
        ] + st.session_state["modalities"]
    if st.session_state["encoder"] is not None:
        if st.session_state["method"] == "Retrieval":
            expander = st.expander("Show/hide encoding schema")
            expander.write(st.session_state["aux_encoding_schema"])
        st.divider()
        st.subheader("Display options", divider="blue")
        num_results = st.slider("Number of products to retrieve", 5, 50, 10)
        display_columns = st.multiselect(
            "Select columns to display",
            data.df.columns,
            st.session_state["display_columns"],
        )

        with col3:
            col3_title = (
                "Rank by modalities"
                if st.session_state["method"] == "Retrieval"
                else "Re-rank by modalities"
            )
            st.subheader(col3_title, divider="blue")
            aux_data = dict.fromkeys(
                st.session_state["aux_encoding_schema"].keys(), (None, 1.0)
            )
            map_placeholder = dict()
            for index, (column, encoding) in enumerate(
                st.session_state["aux_encoding_schema"].items()
            ):
                values = None
                st.markdown(f'Modality: :violet["**{column}**"]')
                if encoding == "one_hot":
                    selection = st.multiselect(
                        column,
                        options=sorted(data.df[column].dropna().unique()),
                        label_visibility="collapsed",
                        default=[],
                        key=column,
                    )
                    negation = st.checkbox("Negate", key=column + "_not")
                    values = (selection, negation)
                elif encoding == "binary":
                    values = st.radio(
                        "Value",
                        data.df[column].dropna().unique(),
                        index=None,
                        key=column,
                    )
                elif encoding == "geolocation":
                    address = st.text_input("Enter your address:", key=column)
                    negation = st.checkbox("Negate", key=column + "_not")
                    if address:
                        osm_url = f"https://nominatim.openstreetmap.org/search?format=json&q={address}"
                        location_data = requests.get(osm_url).json()
                        if location_data:
                            latitude, longitude = float(location_data[0]["lat"]), float(
                                location_data[0]["lon"]
                            )
                            map_placeholder[column] = st.empty()
                            values = (longitude, latitude, negation)
                        else:
                            st.error("Address not found. Please enter a valid address.")
                elif encoding == "dense":
                    min_value = data.df[column].min()
                    max_value = data.df[column].max()
                    col31, col32 = st.columns(2)
                    with col31:
                        dense_filter = st.radio(
                            "Select range",
                            ["Low/High", "From - To", "From centroid"],
                            index=0,
                            key=column + "_filter",
                        )

                    if dense_filter == "Low/High":
                        with col32:
                            ranking = st.radio(
                                "Low/High",
                                ["Lowest", "Highest"],
                                index=None,
                                label_visibility="hidden",
                                key=column + "_low/high",
                            )
                        if ranking == "Lowest":
                            values = (data.transformed_df[column].min(),)
                        elif ranking == "Highest":
                            values = (data.transformed_df[column].max(),)

                    elif dense_filter == "From - To":
                        col31, col32 = st.columns(2)
                        with col31:
                            lower_bound = st.number_input(
                                "From",
                                min_value=min_value,
                                max_value=max_value,
                                key=column + "_from",
                            )
                        with col32:
                            upper_bound = st.number_input(
                                "To",
                                min_value=min_value,
                                max_value=max_value,
                                value=max_value,
                                key=column + "_to",
                            )
                        negation = st.checkbox("Negate", key=column + "_not")
                        if lower_bound != min_value or upper_bound != max_value:
                            if column in data.transformation_schema:
                                lower_bound = data.transformation_schema[
                                    column
                                ].transform(lower_bound)
                                upper_bound = data.transformation_schema[
                                    column
                                ].transform(upper_bound)
                            values = (lower_bound, upper_bound, negation)

                    elif dense_filter == "From centroid":
                        with col32:
                            value = st.number_input(
                                "Centroid value",
                                min_value=min_value,
                                max_value=max_value,
                                key=column + "_from_centroid",
                                value=None,
                            )
                        negation = st.checkbox("Negate", key=column + "_not")
                        if value is not None and column in data.transformation_schema:
                            value = data.transformation_schema[column].transform(value)
                        values = (value, negation)

                weight = st.slider(
                    "Modality weight", 0.0, 10.0, 1.0, key=column + "_weight"
                )
                aux_data[column] = (values, weight)
                st.divider()
            print(aux_data)

with col2:
    st.subheader(f"Search {args.dataset.title()}", divider="blue")
    if st.session_state["encoder"] is not None:
        query_text = st.text_input("Enter your search query", disabled=False)
        do_search(query_text, aux_data, display_columns)
    else:
        query_text = st.text_input(
            "Search for products", "Index products to enable search", disabled=True
        )
