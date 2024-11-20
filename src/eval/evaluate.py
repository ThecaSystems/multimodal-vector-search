import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from time import time
from typing import Any
from tqdm import tqdm
from functools import reduce
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, Collection, connections
from collections import defaultdict
from src.load_restaurants import Restaurants
from src.load_flipkart import Flipkart
from src.load_bygghemma import Bygghemma
from src.embed import TextEmbedder
from src import encode
from src.eval.milvus_experiment import MilvusExperiment
from src.eval.faiss_experiment import FaissExperiment
from src.eval.ground_truth_experiment import GroundTruthExperiment

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
dataset = Restaurants()
model = 'mixedbread-ai/mxbai-embed-large-v1'
# aux_modalities = dataset.df.columns[1:-1].tolist()
# aux_modalities = ['City', 'Cuisines', 'Rating text']
# aux_modalities = ['Has Table booking', 'Has Online delivery']
aux_modalities = ['Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']
text_modality = dataset.df.columns[0]

#%%s
# dataset = Bygghemma()
# model = '/model'
# excluded_columns = ['aggregateRatingValue', 'aggregateRatingCount', 'name', 'description', 'itemCondition', 'lowPrice', 'highPrice', 'geolocation', 'id', 'priceCurrency']
# # aux_modalities = list(set(dataset.df.columns) - {'aggregateRatingValue', 'aggregateRatingCount', 'name', 'description', 'itemCondition', 'lowPrice', 'highPrice', 'geolocation', 'id', 'priceCurrency'})
# aux_modalities = [col for col in dataset.df.columns if col not in excluded_columns]
# text_modality = 'name'

#%%
# Experiment setup
# num_filters = np.random.randint(low=1, high=len(filterable_cols))
# selected_cols = np.random.choice(filterable_cols, size=num_filters, replace=False)

num_modalities = len(aux_modalities)
num_repetitions = 10
# coverage = 0.2

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
# Experiment loop
# exp = FaissExperiment(dataset, model, text_modality, aux_modalities)
exp = MilvusExperiment(dataset, model, text_modality, aux_modalities)
# exp = GroundTruthExperiment(dataset, model, text_modality, aux_modalities)

# experiments = [
#     # GroundTruthExperiment(dataset, model, text_modality, aux_modalities),
#     FaissExperiment(dataset, model, text_modality, aux_modalities),
#     MilvusExperiment(dataset, model, text_modality, aux_modalities)
# ]

# rankings_faiss = defaultdict(list)
# runtimes_faiss = defaultdict(list)
# rankings_milvus = defaultdict(list)
# runtimes_milvus = defaultdict(list)

# for exp in experiments:
rankings = defaultdict(list)
runtimes = defaultdict(list)
for num_modalities in range(0, num_modalities + 1):
# for num_modalities in range(0, 2):
    print(f"Number of modalities: {num_modalities}")
    for i in tqdm(range(num_repetitions), desc="Repetitions"):
        # exp = FaissExperiment(dataset, model, text_modality, aux_modalities)
        # exp = MilvusExperiment(dataset, model, text_modality, aux_modalities)
        seed = int(str(num_modalities) + str(i))
        rng = np.random.default_rng(seed=seed)
        random_id = dataset.df.sample(random_state=seed).index[0]
        random_mods = rng.choice(aux_modalities, size=num_modalities if num_modalities > 0 else 1, replace=False)
        print(f"\nSelected modalities: {random_mods}")
        ranking, runtime = exp.run_experiment(random_id, random_mods.tolist())
        if num_modalities > 0:
            rankings[num_modalities].append(ranking)
            runtimes[num_modalities].append(runtime)
# engine_name = exp.__class__.__name__.split('Experiment')[0].lower()
# save_results(rankings, runtimes, engine_name)

# exp.analyze(runtimes_native, runtimes_fuserank)

#%%
# Save rankings and runtimes to disk
dataset_name = dataset.__class__.__name__.lower()
engine_name = exp.__class__.__name__.split('Experiment')[0].lower()
folder = f'{dataset_name}'
if not os.path.exists(f'results/{folder}'):
    os.makedirs(f'results/{folder}')
with open(f'results/{folder}/rankings_{engine_name}.pkl', 'wb') as pickle_file:
    pickle.dump(rankings, pickle_file)
with open(f'results/{folder}/runtimes_{engine_name}.pkl', 'wb') as pickle_file:
    pickle.dump(runtimes, pickle_file)

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