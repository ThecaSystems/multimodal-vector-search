import os
import json
import sys

import yaml
import pickle
import copy
import numpy as np
import pandas as pd
from fontTools.misc.cython import returns
from scipy import stats
from time import time
from typing import Any
from tqdm import tqdm
from functools import reduce
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, Collection, connections
from collections import defaultdict

from src.load import DataLoader
from src.load_restaurants import Restaurants
from src.load_flipkart import Flipkart
from src.load_bygghemma import Bygghemma
from src.embed import TextEmbedder
from src import encode
from src.eval.milvus_experiment import MilvusExperiment
from src.eval.faiss_experiment import FaissExperiment
from src.eval.ground_truth_experiment import GroundTruthExperiment
from src.eval.analyze import get_recall

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('macosx')

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#%%
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

#%%
# Set up dataset
# if config['dataset'] == 'restaurants':
#     dataset = Restaurants()
#     text_modality = dataset.df.columns[0]
#     if config['modalities'] == 'categorical':
#         aux_modalities = ['City', 'Cuisines', 'Rating text']
#     elif config['modalities'] == 'binary':
#         aux_modalities = ['Has Table booking', 'Has Online delivery']
#     elif config['modalities'] == 'numerical':
#         aux_modalities = ['Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']
#     elif config['modalities'] == 'geolocation':
#         aux_modalities = ['Location']
#     else:
#         sys.exit(f"Modality type {config['modalities']} does not exist!")
# elif config['dataset'] == 'flipkart':
#     dataset = Flipkart()
#     text_modality = 'product_name'
#     if config['modalities'] == 'categorical':
#         aux_modalities = ["brand", "product_category_1", "product_category_2", "product_category_3"]
#     elif config['modalities'] == 'binary':
#         aux_modalities = ['is_FK_Advantage_product']
#     elif config['modalities'] == 'numerical':
#         aux_modalities = ['retail_price', 'discounted_price']
#     else:
#         sys.exit(f"Modality type {config['modalities']} does not exist!")
# else:
#     sys.exit(f"Dataset {config['dataset']} does not exist!")

#%%
# dataset = Flipkart()
# model = 'mixedbread-ai/mxbai-embed-large-v1'
# excluded_columns = ['product_name', 'description', 'product_specifications']
# # aux_modalities = [col for col in dataset.df.columns if col not in excluded_columns]
# aux_modalities = ["retail_price", "discounted_price"]
# # aux_modalities = ["is_FK_Advantage_product"]
# # aux_modalities = ["brand", "product_category_1", "product_category_2", "product_category_3"]
# text_modality = 'product_name'

#%%
# dataset = Restaurants()
# # dataset.df.reset_index(drop=True, inplace=True)
# model = 'mixedbread-ai/mxbai-embed-large-v1'
# # aux_modalities = dataset.df.columns[1:-1].tolist()
# # aux_modalities = ['City', 'Cuisines', 'Rating text']  # categorical
# # aux_modalities = ['Has Table booking', 'Has Online delivery']  # boolean
# aux_modalities = ['Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']  # numerical
# # aux_modalities = ['Location']
# text_modality = dataset.df.columns[0]

#%%s
# dataset = Bygghemma()
# model = '/model'
# # excluded_columns = ['aggregateRatingValue', 'aggregateRatingCount', 'name', 'description', 'itemCondition', 'lowPrice', 'highPrice', 'geolocation', 'id', 'priceCurrency']
# # aux_modalities = [col for col in dataset.df.columns if col not in excluded_columns]
# # aux_modalities = ['brand', 'category', 'seller', 'availability']  # categorical
# aux_modalities = ['price']  # numerical
# text_modality = 'name'

#%%
# Experiment setup
# num_modalities = len(aux_modalities)
# num_repetitions = config['num_repetitions']


#%%
def save_results(rankings: dict[int, list[int]], runtimes: dict[int, list[float]], exp_name: str) -> None:
    dataset_name = dataset.__class__.__name__.lower()
    # engine_name = experiment.__class__.__name__.split('Experiment')[0].lower()
    folder = f'{dataset_name}'
    if not os.path.exists(f'results/{folder}'):
        os.makedirs(f'results/{folder}')
    with open(f'results/{folder}/rankings_{exp_name}.pkl', 'wb') as pickle_file:
        pickle.dump(rankings, pickle_file)
    with open(f'results/{folder}/runtimes_{exp_name}.pkl', 'wb') as pickle_file:
        pickle.dump(runtimes, pickle_file)

#%%
# # Experiment loop
# model = 'mixedbread-ai/mxbai-embed-large-v1'
# if config['experiment'] == 'faiss':
#     exp = FaissExperiment(dataset, model, text_modality, aux_modalities, config['num_harmonics'])
# elif config['experiment'] == 'truth':
#     exp = GroundTruthExperiment(dataset, model, text_modality, aux_modalities)
# else:
#     sys.exit(f"Experiment type {config['experiment']} does not exist!")
#
# # for exp in experiments:
# rankings = defaultdict(list)
# runtimes = defaultdict(list)
# for num_modalities in range(0, num_modalities + 1):
#     print(f"Number of modalities: {num_modalities}")
#     for i in tqdm(range(num_repetitions), desc="Repetitions"):
#         # exp = FaissExperiment(dataset, model, text_modality, aux_modalities)
#         # exp = MilvusExperiment(dataset, model, text_modality, aux_modalities)
#         seed = int(str(num_modalities) + str(i))
#         rng = np.random.default_rng(seed=seed)
#         random_id = dataset.df.sample(random_state=seed).index[0]
#         random_mods = rng.choice(aux_modalities, size=num_modalities if num_modalities > 0 else 1, replace=False)
#         print(f"\nSelected modalities: {random_mods}")
#         ranking, runtime = exp.run_experiment(random_id, random_mods.tolist(), limit=config['num_results'])
#         if num_modalities > 0:
#             rankings[num_modalities].append(ranking)
#             runtimes[num_modalities].append(runtime)

#%%
# # Experiment loop
# model = 'mixedbread-ai/mxbai-embed-large-v1'
# # if config['experiment'] == 'faiss':
# #     exp = FaissExperiment(dataset, model, text_modality, aux_modalities, config['num_harmonics'])
# # elif config['experiment'] == 'truth':
# #     exp = GroundTruthExperiment(dataset, model, text_modality, aux_modalities)
# # else:
# #     sys.exit(f"Experiment type {config['experiment']} does not exist!")
#
# # for exp in experiments:
# # rankings = defaultdict(list)
# # runtimes = defaultdict(list)
#
# data = []
#
# for num_harmonics in config['num_harmonics']:
#     exp_truth = GroundTruthExperiment(dataset, model, text_modality, aux_modalities)
#     exp_faiss = FaissExperiment(dataset, model, text_modality, aux_modalities, num_harmonics)
#     for num_modalities in range(1, num_modalities + 1):
#         print(f"Number of modalities: {num_modalities}")
#         recalls = []
#         for i in tqdm(range(num_repetitions), desc="Repetitions"):
#             seed = int(str(num_modalities) + str(i))
#             rng = np.random.default_rng(seed=seed)
#             random_id = dataset.df.sample(random_state=seed).index[0]
#             random_mods = rng.choice(aux_modalities, size=num_modalities if num_modalities > 0 else 1, replace=False)
#             print(f"\nSelected modalities: {random_mods}")
#             ranking_truth, _ = exp_truth.run_experiment(random_id, random_mods.tolist(), limit=config['num_results'])
#             ranking_faiss, _ = exp_faiss.run_experiment(random_id, random_mods.tolist(), limit=config['num_results'])
#             recalls.append(get_recall(ranking_truth, ranking_faiss))
#         for recall in recalls:
#             data.append({'num_harmonics': num_harmonics, 'num_modalities': num_modalities, 'recall': recall})
#         # data.append({
#         #     'num_harmonics': num_harmonics,
#         #     'num_modalities': num_modalities,
#         #     'recall': np.mean(recalls),
#         #     'std_dev': np.std(recalls)
#         # })
#
# df = pd.DataFrame(data)

#%%
def setup_data(dataset_name: str) -> (pd.DataFrame, str, list[str]):
    if dataset_name== 'restaurants':
        dataset = Restaurants()
        text_modality = dataset.df.columns[0]
        if config['modalities'] == 'categorical':
            aux_modalities = ['City', 'Cuisines', 'Rating text']
        elif config['modalities'] == 'binary':
            aux_modalities = ['Has Table booking', 'Has Online delivery']
        elif config['modalities'] == 'numerical':
            aux_modalities = ['Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']
        elif config['modalities'] == 'geolocation':
            aux_modalities = ['Location']
        else:
            sys.exit(f"Modality type {config['modalities']} does not exist!")
    elif dataset_name == 'flipkart':
        dataset = Flipkart()
        text_modality = 'product_name'
        if config['modalities'] == 'categorical':
            aux_modalities = ["brand", "product_category_1", "product_category_2", "product_category_3"]
        elif config['modalities'] == 'binary':
            aux_modalities = ['is_FK_Advantage_product']
        elif config['modalities'] == 'numerical':
            aux_modalities = ['retail_price', 'discounted_price']
        else:
            sys.exit(f"Modality type {config['modalities']} does not exist!")
    else:
        sys.exit(f"Dataset {dataset_name} does not exist!")
    return dataset, text_modality, aux_modalities

#%%
# def run_evaluation(dataset: DataLoader, text_modality: str, aux_modalities: list[str]) -> pd.DataFrame:
#     num_modalities = len(aux_modalities)
#     num_repetitions = config['num_repetitions']
#     model = 'mixedbread-ai/mxbai-embed-large-v1'
#     data = []
#     for num_harmonics in config['num_harmonics']:
#         exp_truth = GroundTruthExperiment(dataset, model, text_modality, aux_modalities)
#         exp_faiss = FaissExperiment(dataset, model, text_modality, aux_modalities, num_harmonics)
#         for num_modalities in range(1, num_modalities + 1):
#             print(f"Number of modalities: {num_modalities}")
#             recalls = []
#             for i in tqdm(range(num_repetitions), desc="Repetitions"):
#                 seed = int(str(num_modalities) + str(i))
#                 rng = np.random.default_rng(seed=seed)
#                 random_id = dataset.df.sample(random_state=seed).index[0]
#                 random_mods = rng.choice(aux_modalities, size=num_modalities if num_modalities > 0 else 1, replace=False)
#                 print(f"\nSelected modalities: {random_mods}")
#                 ranking_truth, _ = exp_truth.run_experiment(random_id, random_mods.tolist(), limit=config['num_results'])
#                 ranking_faiss, _ = exp_faiss.run_experiment(random_id, random_mods.tolist(), limit=config['num_results'])
#                 recalls.append(get_recall(ranking_truth, ranking_faiss))
#             for recall in recalls:
#                 data.append({
#                     'num_harmonics': num_harmonics,
#                     'num_modalities': num_modalities,
#                     'recall': recall,
#                     'dataset': dataset.name.title()
#                 })
#     return pd.DataFrame(data)

#%%
# def evaluate(dataset: DataLoader, text_modality: str, aux_modalities: list[str]) -> pd.DataFrame:
#     num_modalities = len(aux_modalities)
#     num_repetitions = config['num_repetitions']
#     model = 'mixedbread-ai/mxbai-embed-large-v1'
#     data = []
#     exp_truth = GroundTruthExperiment(dataset, model, text_modality, aux_modalities)
#     exp_faiss = FaissExperiment(dataset, model, text_modality, aux_modalities, num_harmonics)
#     for num_modalities in range(1, num_modalities + 1):
#         print(f"Number of modalities: {num_modalities}")
#         recalls = []
#         for i in tqdm(range(num_repetitions), desc="Repetitions"):
#             seed = int(str(num_modalities) + str(i))
#             rng = np.random.default_rng(seed=seed)
#             random_id = dataset.df.sample(random_state=seed).index[0]
#             random_mods = rng.choice(aux_modalities, size=num_modalities if num_modalities > 0 else 1, replace=False)
#             print(f"\nSelected modalities: {random_mods}")
#             ranking_truth, _ = exp_truth.run_experiment(random_id, random_mods.tolist(), limit=config['num_results'])
#             ranking_faiss, _ = exp_faiss.run_experiment(random_id, random_mods.tolist(), limit=config['num_results'])
#             recalls.append(get_recall(ranking_truth, ranking_faiss))
#         for recall in recalls:
#             data.append({
#                 'num_harmonics': num_harmonics,
#                 'num_modalities': num_modalities,
#                 'recall': recall,
#                 'dataset': dataset.name.title()
#             })
#     return pd.DataFrame(data)

#%%
def evaluate(dataset: DataLoader, text_modality: str, aux_modalities: list[str]) -> pd.DataFrame:
    num_modalities = len(aux_modalities)
    num_repetitions = config['num_repetitions']
    harmonics = config['harmonics'] if config['modalities'] == 'numerical' else [None]
    model = 'mixedbread-ai/mxbai-embed-large-v1'
    data = []
    for num_harmonics in harmonics:
        exp_truth = GroundTruthExperiment(dataset, model, text_modality, aux_modalities)
        exp_faiss = FaissExperiment(dataset, model, text_modality, aux_modalities, num_harmonics)
        for num_modalities in range(1, num_modalities + 1):
            print(f"Number of modalities: {num_modalities}")
            recalls = []
            for i in tqdm(range(num_repetitions), desc="Repetitions"):
                seed = int(str(num_modalities) + str(i))
                rng = np.random.default_rng(seed=seed)
                random_id = dataset.df.sample(random_state=seed).index[0]
                random_mods = rng.choice(aux_modalities, size=num_modalities if num_modalities > 0 else 1, replace=False)
                print(f"\nSelected modalities: {random_mods}")
                ranking_truth, _ = exp_truth.run_experiment(random_id, random_mods.tolist(), limit=config['num_results'])
                ranking_faiss, _ = exp_faiss.run_experiment(random_id, random_mods.tolist(), limit=config['num_results'])
                recalls.append(get_recall(ranking_truth, ranking_faiss))
            for recall in recalls:
                data.append({
                    'num_harmonics': num_harmonics,
                    'num_modalities': num_modalities,
                    'recall': recall,
                    'dataset': dataset.name.title()
                })
    return pd.DataFrame(data)

#%%
def print_results(results: list[pd.DataFrame]):
    for i, df in enumerate(results):
        print(f"\nResults for {config['dataset'][i]}:")
        if config['modalities'] == 'numerical':
            print (df.groupby(['num_harmonics', 'num_modalities']).agg(recall=('recall', 'mean')).reset_index())
            print (df.groupby('num_harmonics').agg(recall=('recall', 'mean')).reset_index())
        else:
            print (df.groupby('num_modalities').agg(recall=('recall', 'mean')).reset_index())
        # print(mean_recall_df)
        # print(f"Average total recall: {mean_recall_df['recall'].mean()}")

#%%
def plot_results(results: list[pd.DataFrame]):
    df = pd.concat(results, axis=0, ignore_index=True)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='num_harmonics', y='recall', hue='dataset', style='dataset', markers=True, dashes=False,
                 markersize=8, estimator='mean')
    plt.xticks(df['num_harmonics'])  # Ensure x-ticks match num_harmonics
    plt.xlabel('Number of Harmonics', fontsize=14)  # Increase font size for x-axis
    plt.ylabel('Recall (mean)', fontsize=14)      # Increase font size for y-axis
    plt.title('Recall for numerical filters', fontsize=16)  # Increase font size for title
    plt.grid()
    plt.legend(fontsize='14')
    plt.show(block=True)

#%%
results = []
for dataset_name in config['dataset']:
    dataset, text_modality, aux_modalities = setup_data(dataset_name)
    df = evaluate(dataset, text_modality, aux_modalities)
    results.append(df)
    # plt.plot(recall_mean['num_harmonics'], recall_mean['recall'], marker='o')
    # plt.errorbar(recall_mean['num_harmonics'], recall_mean['recall'],
    #              yerr=recall_mean['std_dev'], fmt='o', capsize=5, label='Mean Recall ± Std Dev')

    # plt.xticks(recall_mean['num_harmonics'])  # Ensure x-ticks match num_harmonics
print_results(results)
if config['modalities'] == 'numerical':
    plot_results(results)

#%%
# # Plot recall vs number of harmonics
# # recall_mean = df.groupby('num_harmonics')['recall'].mean().reset_index()
# # recall_mean = df.groupby('num_harmonics').agg({'recall': 'mean', 'std_dev': 'mean'}).reset_index()
#
# # Create the plot
# plt.figure(figsize=(10, 6))
# # plt.plot(recall_mean['num_harmonics'], recall_mean['recall'], marker='o')
# # plt.errorbar(recall_mean['num_harmonics'], recall_mean['recall'],
# #              yerr=recall_mean['std_dev'], fmt='o', capsize=5, label='Mean Recall ± Std Dev')
# sns.lineplot(data=df, x='num_harmonics', y='recall', ci=95, marker='o', estimator='mean')
#
# plt.xlabel('Number of Harmonics', fontsize=14)  # Increase font size for x-axis
# plt.ylabel('Recall (mean)', fontsize=14)      # Increase font size for y-axis
# plt.title('Recall vs. Number of Harmonics', fontsize=16)  # Increase font size for title
# # plt.xticks(recall_mean['num_harmonics'])  # Ensure x-ticks match num_harmonics
# plt.xticks(df['num_harmonics'])  # Ensure x-ticks match num_harmonics
# plt.grid()
# # plt.legend
# plt.show(block=True)

#%%
# Plot recall vs filter size
# plt.figure(figsize=(10, 6))
# for num_harmonics in config['num_harmonics']:
#     recall_values = df[df['num_harmonics'] == num_harmonics]['recall'].values
#     filter_sizes = np.arange(4)+1
#     plt.plot(filter_sizes, recall_values, label=f'num_harmonics = {num_harmonics}')
# plt.title('Recall per filter size', fontsize=20)
# plt.xlabel('Number of Modalities', fontsize=14)
# plt.ylabel('Recall', fontsize=14)
# plt.xticks(filter_sizes)
# plt.legend()
# plt.grid()
# plt.show(block=True)

#%%
# Save rankings and runtimes to disk
# dataset_name = dataset.__class__.__name__.lower()
# engine_name = exp.__class__.__name__.split('Experiment')[0].lower()
# folder = f'{dataset_name}'
# if not os.path.exists(f'results/{folder}'):
#     os.makedirs(f'results/{folder}')
# with open(f'results/{folder}/rankings_{engine_name}_{config["num_harmonics"]}.pkl', 'wb') as pickle_file:
#     pickle.dump(rankings, pickle_file)
# with open(f'results/{folder}/runtimes_{engine_name}_{config["num_harmonics"]}.pkl', 'wb') as pickle_file:
#     pickle.dump(runtimes, pickle_file)

#%%
# Experiment loop
# res_faiss = defaultdict(list)
# runtimes_faiss = defaultdict(list)
#
# exp = FaissExperiment(dataset, model, text_modality, aux_modalities)
#
# for num_modalities in range(1, num_modalities + 1):  # range(1, len(aux_modalities) + 1):
#     print(f"Number of modalities: {num_modalities}")
#     for i in tqdm(range(num_repetitions), desc="Repetitions"):
#         random_id = dataset.df.sample().index[0]
#         exp.pick_random_filters(random_id, num_modalities)
#
#         print(f"Filters: {exp.selected_filters}")
#         res, runtime = exp.run_experiment(random_id)
#         if len(res) > 0:
#             res_faiss[num_modalities].append(res)
#             runtimes_faiss[num_modalities].append(runtime)
#         else:
#             res_faiss[num_modalities].append(np.nan)
#             runtimes_faiss[num_modalities].append(np.nan)
#
# # exp.analyze(runtimes_native, runtimes_fuserank)

#%%
# # Experiment loop
# res_native = defaultdict(list)
# runtimes_native = defaultdict(list)
# res_fuserank = defaultdict(list)
# runtimes_fuserank = defaultdict(list)
#
# exp = MilvusExperiment(dataset, model, text_modality, aux_modalities)
#
# for num_modalities in range(1, num_modalities + 1):  # range(1, len(aux_modalities) + 1):
#     print(f"Number of modalities: {num_modalities}")
#     for i in tqdm(range(num_repetitions), desc="Repetitions"):
#         # selected_modalities = np.random.choice(aux_modalities, size=num_modalities, replace=False)
#         # print(f"\nSelected modalities: {selected_modalities}")
#
#         # Milvus fails when there are null values in filters (at least in numerical ones)
#         has_nulls = True
#         while has_nulls:
#             random_id = dataset.df.sample().index[0]
#             has_nulls = dataset.df[aux_modalities].loc[random_id, :].isnull().any().any()
#         exp.pick_random_filters(random_id, num_modalities)
#
#         print("Native filtering...")
#         print(f"Filters: {exp.selected_filters[True]}")
#         res, runtime = exp.run_experiment(random_id, is_native=True)
#         if len(res) > 0:
#             res_native[num_modalities].append(res)
#             runtimes_native[num_modalities].append(runtime)
#
#             print("FuseRank filtering...")
#             print(f"Filters: {exp.selected_filters[False]}")
#             res, runtime = exp.run_experiment(random_id, is_native=False)
#             res_fuserank[num_modalities].append(res)
#             runtimes_fuserank[num_modalities].append(runtime)
#         else:
#             res_native[num_modalities].append(np.nan)
#             res_fuserank[num_modalities].append(np.nan)
#             runtimes_native[num_modalities].append(np.nan)
#             runtimes_fuserank[num_modalities].append(np.nan)
#
# exp.analyze(runtimes_native, runtimes_fuserank)
# exp.cleanup()