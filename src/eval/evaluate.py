import os
import sys
import yaml
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.load import DataLoader
from src.load_bygghemma import Bygghemma
from src.load_restaurants import Restaurants
from src.load_flipkart import Flipkart
from src.eval.faiss_experiment import FaissExperiment
from src.eval.milvus_experiment import MilvusExperiment
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("macosx")

with open("src/eval/config.yaml", "r") as file:
    config = yaml.safe_load(file)


def save_results(rankings: dict[int, list[int]], runtimes: dict[int, list[float]], exp_name: str) -> None:
    dataset_name = dataset.__class__.__name__.lower()
    folder = f"{dataset_name}"
    if not os.path.exists(f"results/{folder}"):
        os.makedirs(f"results/{folder}")
    with open(f"results/{folder}/rankings_{exp_name}.pkl", "wb") as pickle_file:
        pickle.dump(rankings, pickle_file)
    with open(f"results/{folder}/runtimes_{exp_name}.pkl", "wb") as pickle_file:
        pickle.dump(runtimes, pickle_file)


def setup_data(dataset_name: str) -> (pd.DataFrame, str, list[str]):
    aux_modalities = []
    if dataset_name == "restaurants":
        dataset = Restaurants()
        text_modality = dataset.df.columns[0]
        if "categorical" in config["modalities"]:
            aux_modalities.extend(["City", "Cuisines", "Rating text"])
        if "binary" in config["modalities"]:
            aux_modalities.extend(["Has Table booking", "Has Online delivery"])
        if "numerical" in config["modalities"]:
            aux_modalities.extend(
                [
                    "Average Cost for two",
                    "Price range",
                    "Aggregate rating",
                    "Votes",
                ]
            )
        if config["modalities"] == "geolocation":
            aux_modalities.extend(["Location"])
    elif dataset_name == "flipkart":
        dataset = Flipkart()
        text_modality = "product_name"
        if "categorical" in config["modalities"]:
            aux_modalities.extend(
                [
                    "brand",
                    "product_category_1",
                    "product_category_2",
                    "product_category_3",
                ]
            )
        if "binary" in config["modalities"]:
            aux_modalities.extend(["is_FK_Advantage_product"])
        if "numerical" in config["modalities"]:
            aux_modalities.extend(["retail_price", "discounted_price"])
    elif dataset_name == "bygghemma":
        dataset = Bygghemma()
        text_modality = "name"
        if "categorical" in config["modalities"]:
            aux_modalities.extend(
                [
                    "brand",
                    "category",
                    "seller",
                    "availability",
                ]
            )
        if "binary" in config["modalities"]:
            aux_modalities.extend(["itemCondition"])
        if "numerical" in config["modalities"]:
            aux_modalities.extend(["price", "lowPrice", "highPrice", "aggregateRatingValue"])
    else:
        exit(f"Dataset {dataset_name} does not exist!")
    return dataset, text_modality, aux_modalities


def recall(list1: list[int], list2: list[int]) -> float:
    """Get recall for list2 with respect to list1."""
    if len(list1) == 0:
        return 0.0
    true_positives = len(set(list1) & set(list2))
    false_negatives = len(set(list1) - set(list2))
    if true_positives + false_negatives > 0:
        return true_positives / (true_positives + false_negatives)
    else:
        return 0.0


def r_precision(list1: list[int], list2: list[int]) -> float:
    """Get r-precision for list2 with respect to list1."""
    if len(list1) == 0:
        return 0.0
    k = len(list1)
    true_positives = len(set(list1) & set(list2[:k]))
    return true_positives / k


def print_results(results: list[pd.DataFrame]):
    metric_name = config["metric"]
    for i, df in enumerate(results):
        print(f"\nResults for {config['dataset'][i]}:")
        if "numerical" in config["modalities"]:
            print(
                df.groupby(["num_harmonics", "interval_epsilon", "num_modalities"])
                .agg(**{metric_name: (metric_name, "mean")})
                .reset_index()
            )
            print(f"{str.capitalize(metric_name)} vs. num_harmonics")
            print(df.groupby("num_harmonics").agg(**{metric_name: (metric_name, "mean")}).reset_index())
            print(f"{str.capitalize(metric_name)} vs. interval_epsilon")
            print(df.groupby("interval_epsilon").agg(**{metric_name: (metric_name, "mean")}).reset_index())
        else:
            print(f"{str.capitalize(metric_name)} vs. num_modalities")
            print(df.groupby("num_modalities").agg(**{metric_name: (metric_name, "mean")}).reset_index())


def plot_results(results: list[pd.DataFrame], x_column: str, x_label: str):
    df = pd.concat(results, axis=0, ignore_index=True)
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x=x_column,
        y=config["metric"],
        hue="dataset",
        style="dataset",
        markers=True,
        dashes=False,
        markersize=10,
        estimator="mean",
    )
    plt.xticks(df[x_column], fontsize=12)
    plt.tick_params(axis="y", labelsize=12)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(f"{config['metric']} (mean)", fontsize=14)
    plt.title(f"{config['metric']} for numerical filters", fontsize=16)
    plt.grid(axis="y")
    plt.legend(fontsize="14")
    plt.show(block=True)


def evaluate(
    dataset: DataLoader, text_modality: str, aux_modalities: list[str], result_data: list[dict], **kwargs
) -> None:
    num_harmonics = kwargs.get("num_harmonics", 200)
    interval_epsilon = kwargs.get("interval_epsilon", 0.015)
    num_modalities = len(aux_modalities)
    num_repetitions = config["num_repetitions"]
    if dataset.name == "bygghemma":
        model = "/model"
    else:
        model = "mixedbread-ai/mxbai-embed-large-v1"
    exp_milvus = MilvusExperiment(dataset, model, text_modality, aux_modalities, num_harmonics, interval_epsilon)
    exp_faiss = FaissExperiment(dataset, model, text_modality, aux_modalities, num_harmonics, interval_epsilon)
    print(f"\nChosen metric: {config['metric']}")
    for num_modalities in range(1, num_modalities + 1):
        print(f"Number of modalities: {num_modalities}")
        measurements = []
        for i in tqdm(range(num_repetitions), desc="Repetitions"):
            seed = int(str(num_modalities) + str(i))
            rng = np.random.default_rng(seed=seed)
            random_id = dataset.df.sample(random_state=seed).index[0]
            random_mods = rng.choice(
                aux_modalities,
                size=num_modalities if num_modalities > 0 else 1,
                replace=False,
            )
            print(f"\nSelected modalities: {random_mods}")

            # Milvus's max cutoff: 16384 (https://milvus.io/api-reference/pymilvus/v2.5.x/MilvusClient/Vector/search.md)
            limit = 16384 if config["num_results"] == [] else min(16384, config["num_results"][0])

            ranking_milvus = exp_milvus.run_experiment(random_id, random_mods.tolist(), limit)[0]
            if len(ranking_milvus) > 0:
                ranking_faiss = exp_faiss.run_experiment(random_id, random_mods.tolist(), limit)[0]
                metric_function = config["metric"]
                measurement = globals()[metric_function](ranking_milvus, ranking_faiss)
                measurements.append(measurement)
        for measurement in measurements:
            result_data.append(
                {
                    "num_harmonics": num_harmonics,
                    "interval_epsilon": interval_epsilon,
                    "num_modalities": num_modalities,
                    f"{config['metric']}": measurement,
                    "dataset": dataset.name.title(),
                }
            )


if __name__ == "__main__":
    results = []
    results_harmonics = []
    results_epsilon = []
    for dataset_name in config["dataset"]:
        dataset, text_modality, aux_modalities = setup_data(dataset_name)
        if "numerical" in config["modalities"]:
            out_data_harmonics = []
            out_data_epsilon = []
            for i in config["num_harmonics"]:
                evaluate(
                    dataset,
                    text_modality,
                    aux_modalities,
                    out_data_harmonics,
                    num_harmonics=i,
                )
            results_harmonics.append(pd.DataFrame(out_data_harmonics))
            for i in config["interval_epsilon"]:
                evaluate(
                    dataset,
                    text_modality,
                    aux_modalities,
                    out_data_epsilon,
                    interval_epsilon=i,
                )
            results_epsilon.append(pd.DataFrame(out_data_epsilon))
        else:
            out_data = []
            evaluate(dataset, text_modality, aux_modalities, out_data)
            results.append(pd.DataFrame(out_data))
    if "numerical" in config["modalities"]:
        print_results(results_harmonics)
        print_results(results_epsilon)
        plot_results(results_harmonics, x_column="num_harmonics", x_label="Number of harmonics")
        plot_results(results_epsilon, x_column="interval_epsilon", x_label="Interval offset ε")
    else:
        print_results(results)
