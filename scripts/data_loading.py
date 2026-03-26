import mysql.connector
import pandas as pd
import json

conn = mysql.connector.connect(
    host = 'localhost', user = 'root', password = 'root', database='real_estate'
)

query = """
SELECT L_ListingID, L_Address, L_City, L_Keyword2 as beds,
        LM_Dec_3 as bath, L_SystemPrice as price, L_Remarks as remarks
FROM rets_property
WHERE L_Remarks IS NOT NULL AND LENGTH(L_Remarks) > 50
ORDER BY RAND() LIMIT 1000
"""
df = pd.read_sql(query, conn)
df.to_csv('data/processed/listing_sample.csv', index = False)

extract_distinct_query = """
SELECT DISTINCT L_City 
FROM rets_property
WHERE L_City IS NOT NULL AND L_City != ''
"""
df_city = pd.read_sql(extract_distinct_query, conn)
df_city.to_csv('data/processed/distinct_cities.csv', index = False)

load_numeric_value_query = """
SELECT 
    L_Keyword2 as bedrooms,
    LM_Dec_3 as bathrooms,
    L_SystemPrice as price,
    L_Keyword1 as sqft
FROM rets_property;   
"""
# keyword 1 is lot size area in sqft, use LM_Int2_3 for living area

df_numeric = pd.read_sql(load_numeric_value_query, conn)
df_numeric.to_csv('data/processed/numeric_values.csv', index = False)

conn.close

