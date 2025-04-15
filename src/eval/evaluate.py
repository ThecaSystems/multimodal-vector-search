import os
import sys
import yaml
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.load import DataLoader
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
    if dataset_name == "restaurants":
        dataset = Restaurants()
        text_modality = dataset.df.columns[0]
        if config["modalities"] == "categorical":
            aux_modalities = ["City", "Cuisines", "Rating text"]
        elif config["modalities"] == "binary":
            aux_modalities = ["Has Table booking", "Has Online delivery"]
        elif config["modalities"] == "numerical":
            aux_modalities = [
                "Average Cost for two",
                "Price range",
                "Aggregate rating",
                "Votes",
            ]
        elif config["modalities"] == "geolocation":
            aux_modalities = ["Location"]
        else:
            sys.exit(f"Modality type {config['modalities']} does not exist!")
    elif dataset_name == "flipkart":
        dataset = Flipkart()
        text_modality = "product_name"
        if config["modalities"] == "categorical":
            aux_modalities = [
                "brand",
                "product_category_1",
                "product_category_2",
                "product_category_3",
            ]
        elif config["modalities"] == "binary":
            aux_modalities = ["is_FK_Advantage_product"]
        elif config["modalities"] == "numerical":
            aux_modalities = ["retail_price", "discounted_price"]
        else:
            sys.exit(f"Modality type {config['modalities']} does not exist!")
    else:
        sys.exit(f"Dataset {dataset_name} does not exist!")
    return dataset, text_modality, aux_modalities

def get_recall(list1: list[int], list2: list[int]) -> float:
    """Get recall for list2 with respect to list1."""
    recall = 0.0
    if len(list1) > 0:
        true_positives = len(set(list1) & set(list2))
        false_negatives = len(set(list1) - set(list2))

        # Calculate recall
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0.0  # Handle case where there are no true positives
    return recall


def print_results(results: list[pd.DataFrame]):
    for i, df in enumerate(results):
        print(f"\nResults for {config['dataset'][i]}:")
        if config["modalities"] == "numerical":
            print(
                df.groupby(["num_harmonics", "interval_epsilon", "num_modalities"])
                .agg(recall=("recall", "mean"))
                .reset_index()
            )
            print("Recall vs. num_harmonics")
            print(df.groupby("num_harmonics").agg(recall=("recall", "mean")).reset_index())
            print("Recall vs. interval_epsilon")
            print(df.groupby("interval_epsilon").agg(recall=("recall", "mean")).reset_index())
        else:
            print("Recall vs. num_modalities")
            print(df.groupby("num_modalities").agg(recall=("recall", "mean")).reset_index())


def plot_results(results: list[pd.DataFrame], x_column: str, x_label: str):
    df = pd.concat(results, axis=0, ignore_index=True)
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x=x_column,
        y="recall",
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
    plt.ylabel("Recall (mean)", fontsize=14)
    plt.title("Recall for numerical filters", fontsize=16)
    plt.grid(axis="y")
    plt.legend(fontsize="14")
    plt.show(block=True)


def evaluate(
        dataset: DataLoader,
        text_modality: str,
        aux_modalities: list[str],
        result_data: list[dict],
        num_harmonics: int = 200,
        interval_epsilon: float = 0.015,
) -> None:
    num_modalities = len(aux_modalities)
    num_repetitions = config["num_repetitions"]
    model = "mixedbread-ai/mxbai-embed-large-v1"
    exp_milvus = MilvusExperiment(dataset, model, text_modality, aux_modalities, num_harmonics, interval_epsilon)
    exp_faiss = FaissExperiment(dataset, model, text_modality, aux_modalities, num_harmonics, interval_epsilon)
    for num_modalities in range(1, num_modalities + 1):
        print(f"Number of modalities: {num_modalities}")
        recalls = []
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
            ranking_milvus = exp_milvus.run_experiment(random_id, random_mods.tolist(), limit=config["num_results"])
            if len(ranking_milvus) > 0:
                ranking_faiss = exp_faiss.run_experiment(random_id, random_mods.tolist(), limit=config["num_results"])
                recalls.append(get_recall(ranking_milvus, ranking_faiss))
        for recall in recalls:
            result_data.append(
                {
                    "num_harmonics": num_harmonics,
                    "interval_epsilon": interval_epsilon,
                    "num_modalities": num_modalities,
                    "recall": recall,
                    "dataset": dataset.name.title(),
                }
            )


if __name__ == "__main__":
    results = []
    results_harmonics = []
    results_epsilon = []
    for dataset_name in config["dataset"]:
        dataset, text_modality, aux_modalities = setup_data(dataset_name)
        if config["modalities"] == "numerical":
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
    if config["modalities"] == "numerical":
        print_results(results_harmonics)
        print_results(results_epsilon)
        plot_results(results_harmonics, x_column="num_harmonics", x_label="Number of harmonics")
        plot_results(results_epsilon, x_column="interval_epsilon", x_label="Interval offset ε")
    else:
        print_results(results)
