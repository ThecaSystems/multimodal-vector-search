This is **FuseRank** - a framework for multimodal vector search in tabular data with filter support.

The code was developed as part of the study published in ECML PKDD 2024: https://link.springer.com/chapter/10.1007/978-3-031-70371-3_29

It comes bundled with 2 public datasets:

- [Flipkart](https://www.kaggle.com/datasets/atharvjairath/flipkart-ecommerce-dataset), containing 20K products from an Indian e-commerce platform
- [Restaurants](https://www.kaggle.com/datasets/mohdshahnawazaadil/restaurant-dataset), containing data about more than 8K restaurants in India

#### Setup instructions:

1. Clone the repo 
2. Create and activate a virtual environment using your preferred method
3. Install dependencies:

``pip install -r requirements.txt``

4. Install the project in editable mode: 

``pip install -e .``

5. Run the demo app with the specified dataset and language model: 

``streamlit run src/app.py -- --dataset=flipkart --model=mixedbread-ai/mxbai-embed-large-v1``

If no dataset is specified in the command line, the user will be prompted to choose a dataset upon app launch:

``streamlit run src/app.py -- --model=mixedbread-ai/mxbai-embed-large-v1``

#### Evaluation:

The code for the experiments reported in the FuseRank paper resides in ``src/eval`` folder. The experiments are controlled
by the ``config.yaml`` file. The evaluation procedure can be launched as follows:

``python src/eval/evaluate.py``
