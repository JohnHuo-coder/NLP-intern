from openai import OpenAI
import pandas as pd

client = OpenAI()
df = pd.read_csv("data/processed/listing_sample.csv")

