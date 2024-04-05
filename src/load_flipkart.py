import re
from src.load import DataLoader
from dataclasses import dataclass
from src.transform import LogTransform


@dataclass
class Flipkart(DataLoader):
    file_path = "flipkart/flipkart.csv"

    def preprocess_data(self) -> None:
        def format_specs(row):
            matches = re.findall(r'=>"(.*?)"', row)
            result = [
                f"{val}:" if i % 2 == 0 else f"{val};" for i, val in enumerate(matches)
            ]
            return " ".join(result)

        self.df["brand"] = self.df["brand"].fillna("n/a")
        self.df["description"] = self.df["description"].fillna("n/a")
        category_splits = self.df["product_category_tree"].str.split(">>", expand=True)
        category_splits = category_splits.replace('[\["\]]', "", regex=True)
        category_splits.columns = [f"product_category_{i + 1}" for i in range(category_splits.shape[1])]
        self.df = self.df.join(category_splits[category_splits.columns[:3]])  # retain top 3 categories
        self.df["product_specifications"] = self.df["product_specifications"].apply(lambda x: format_specs(str(x)))
        # Remove columns that can't be treated as modalities or that have too many nulls
        columns_to_drop = [
            "pid",
            "uniq_id",
            "image",
            "product_rating",
            "overall_rating",
            "product_category_tree",
            "product_url",
            "crawl_timestamp",
        ]
        self.df.drop(columns=columns_to_drop, inplace=True)

    def create_schemas(self) -> None:
        self.text_encoding_schema = {
            "product_name": 0.4,
            "description": 0.3,
            "product_specifications": 0.3,
        }
        self.transformation_schema = {
            "retail_price": LogTransform(self.df["retail_price"]),
            "discounted_price": LogTransform(self.df["discounted_price"]),
        }
