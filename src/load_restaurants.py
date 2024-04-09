from src.load import DataLoader
from dataclasses import dataclass
from src.transform import LogTransform


@dataclass
class Restaurants(DataLoader):
    file_path = "restaurants/restaurants.csv"

    def preprocess_data(self) -> None:
        self.df.dropna(subset=['Cuisines'], inplace=True)
        self.df = self.df[self.df['Latitude'] != 0]
        self.df = self.df[self.df['Country Code'] == 1]  # India
        self.df.drop(index=[8620, 3513], inplace=True)  # Rows with mislabeled country/city
        self.df['Location'] = list(zip(self.df['Longitude'], self.df['Latitude']))
        columns_to_drop = [
            "Locality",
            "Locality Verbose",
            "Address",
            "Country Code",
            "Restaurant ID",
            "Is delivering now",
            "Switch to order menu",
            "Currency",
            "Rating color",
            "Latitude",
            "Longitude"
        ]
        self.df.drop(columns=columns_to_drop, inplace=True)

    def create_schemas(self) -> None:
        self.text_encoding_schema = {
            "Restaurant Name": 0.4,
            "Cuisines": 0.6,
        }
        self.transformation_schema = {
            "Average Cost for two": LogTransform(self.df["Average Cost for two"]),
            "Price range": LogTransform(self.df["Price range"]),
        }
