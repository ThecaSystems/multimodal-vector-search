import os
import yaml
import numpy as np
import pickle

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('macosx')

#%%
def plot_runtimes(folder: str) -> None:
    fig, ax1 = plt.subplots()

    for pickle_file in os.listdir(f'results/{folder}'):
        file_name, engine_name = pickle_file.split(".")[0].split("_")
        if file_name == "runtimes":
            with open(f'results/{folder}/{pickle_file}', 'rb') as f:
                runtimes = pickle.load(f)
            runtimes_np = np.ma.masked_invalid(list(runtimes.values()))
            runtimes_means = runtimes_np.mean(axis=1)
            runtimes_std_err = np.std(runtimes_np, axis=1) / np.sqrt(runtimes_np.shape[1])
            plt.errorbar(list(runtimes.keys()), runtimes_means, yerr=runtimes_std_err, fmt='-o', label=engine_name)
    plt.xlabel('Number of filters')
    plt.ylabel('Runtime mean (s)')
    plt.title(f'Runtimes for varying filter sizes ({folder.title()})')
    plt.legend()
    fig.tight_layout()
    plt.show(block=True)

#%%
def get_recall(list1: list[int], list2: list[int]) -> float:
    """Get recall for list2 with respect to list1."""

    if len(list1) > 0:
        true_positives = len(set(list1) & set(list2))
        false_negatives = len(set(list1) - set(list2))

        # Calculate recall
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0.0  # Handle case where there are no true positives

    return recall

#%%
def get_mean_recall(list1: list[list[int]], list2: list[list[int]]) -> float:
    """Get mean recall for list2 with respect to list1."""
    recalls = []
    for sublist1, sublist2 in zip(list1, list2):
        recalls.append(get_recall(sublist1, sublist2))
    return np.mean(recalls)

#%%
def evaluate_rankings(folder: str):
    engine_rankings = dict()
    for pickle_file in os.listdir(f'results/{folder}'):
        file_name, engine_name = pickle_file.split(".")[0].split("_")
        if file_name == "rankings":
            with open(f'results/{folder}/{pickle_file}', 'rb') as f:
                engine_rankings[engine_name] = pickle.load(f)
    filter_sizes = engine_rankings['faiss'].keys()
    recalls_per_size = dict.fromkeys(filter_sizes)
    for size in filter_sizes:
        recalls_per_size[size] = get_mean_recall(engine_rankings['groundtruth'][size], engine_rankings['faiss'][size])
    return recalls_per_size
    # for rankings1, rankings2 in zip(engine_rankings[0], engine_rankings[1]):

#%%
if __name__ == "__main__":
    # plot_runtimes('bygghemma')
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    recalls = evaluate_rankings(config['dataset'])
    print(recalls)
    print(f'Total recall: {np.mean(list(recalls.values()))}')
