import json
import pandas as pd

class SchemaValidator:
    def __init__(self, filters, schema_path='data/schema.json', cities_path = 'data/processed/distinct_cities.csv', stats_path = "data/processed/stats.json"):
        with open(schema_path) as f:
            self.schema = json.load(f)
        self.valid_cities = self._load_valid_cities(cities_path)
        self.stats = self._load_db_stats(stats_path) # loads the valide values from db
        self.errors = []
        self.filters = filters
        self.keys = ['price', 'sqft', 'bedrooms', 'bathrooms']
    
    def _load_valid_cities(self, cities_path):
        cities_df = pd.read_csv(cities_path)
        cities_list = cities_df['L_City'].to_list()
        return cities_list
    
    def _load_db_stats(self, stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        return stats

    def validate_value_range(self):
        for cat in self.keys:
            max_key = f'{cat}_max'
            min_key = f'{cat}_min'
            if max_key in self.filters and max_key in self.stats and self.filters[max_key] > self.stats[max_key]:
                self.errors.append(f"Max {cat}: {self.filters[max_key]} exceed max {cat} value in db: {self.stats[max_key]}")
            if min_key in self.filters and min_key in self.stats and self.filters[min_key] < self.stats[min_key]:
                self.errors.append(f"Min {cat}: {self.filters[min_key]} is lower than min {cat} value in db: {self.stats[min_key]}")

    def validate_max_gt_min(self):
        for cat in self.keys:
            max_key = f'{cat}_max'
            min_key = f'{cat}_min'
            if max_key in self.filters and min_key in self.filters:
                if self.filters[max_key] < self.filters[min_key]:
                    self.errors.append(f"Max {cat} {self.filters[max_key]} can't be lower than Min {cat} {self.filters[min_key]}")

    def validate_query(self):

        # Check city exists in database
        if 'city' in self.filters:
            if self.filters['city'] not in self.valid_cities:
                self.errors.append(f"City '{self.filters['city']}' not found in database")
        
        self.validate_max_gt_min()
        self.validate_value_range()


        return len(self.errors) == 0, self.errors