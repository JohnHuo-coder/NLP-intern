import json
import pandas as pd

class SchemaValidator:
    def __init__(self, filters, cities_path = 'data/processed/distinct_cities.csv', stats_path = "data/processed/stats.json"):
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
    
    def validate_negated(self):
        # an amenity or feature can't be in both negated and wanted
        for key in ["amenities", "features"]:
            negated_key = f"negated_{key}"
            negated_set = set(self.filters[negated_key])
            for i in self.filters[key]:
                if i in negated_set:
                    self.errors.append(f"{i} can't be both wanted and negated")

    def validate_query(self):

        # Check city exists in database
        if 'city' in self.filters:
            if self.filters['city'] not in self.valid_cities:
                self.errors.append(f"City '{self.filters['city']}' not found in database")
        
        self.validate_max_gt_min()
        self.validate_value_range()


        return len(self.errors) == 0, self.errors