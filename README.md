This is **FuseRank** - a framework for multimodal vector search in tabular data.

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

More detailed information on how to configure FuseRank is coming soon...