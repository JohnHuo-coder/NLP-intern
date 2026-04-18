import json
from collections import defaultdict
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# conver plural to singular
def normalize(text):
    return " ".join(lemmatizer.lemmatize(w) for w in text.split())

finance_opt_manul = set([
    "down payment assistance",
    "subject to existing financing",
    "assumable mortgage",
    "assumable loan",
    "conventional financing",
    "conventional loan",
    "lease purchase option",
    "third party financing",
    "financing available",
    "mortgage available",
    "investor friendly",
    "rent to own",
    "lease option",
    "land contract",
    "contract for deed",
    "jumbo loan",
    "cash buyers only",
    "bridge loan",
    "hard money",
    "assumable",
    "short sale",
    "foreclosure",
    "foreclosed",
    "bank owned",
    "reo property",
    "cash only",
    "cash sale",
    "all cash",
    "probate sale",
    "estate sale",
    "reo",
])


with open("data/processed/text_features_from_db.json", "r") as f:
    text_features = json.load(f)

fha_set = set([
    "f h a",
    "fha203k",
    "fha203b",
])
va_set = set([
    "va no no loan",
    "va no loan",
    "va loan"
])
owner_carry_set = set([
    "owner will carry",
    "owner may carry"
])
normalize_finance = {
    "fha": ["fha", "fha loan", "fha approved", "fha eligible", "fha financing", "federal housing administration loan"],
    "va": ["va", "va loan", "va approved", "va financing", "veterans affairs loan"],
    "owner carry": ["owner carry",
                    "owner will carry",
                    "owner may carry",
                    "seller carry",
                    "seller will carry",
                    "seller may carry",
                    "seller financing",
                    "owner financing",
                    "carry back",
                    "seller carryback",
                    "owner carryback"],

    "usda loan": ["usda loan", "usda financing", "usda eligible", "united states department of agriculture loan"],
    "cal vet loan": ["calvet", "california veterans loan", "cal vet loan"],
    "exchange1031": ["exchange 1031", "1031 exchange", "exchange1031", "1031exchange"],
    "private financing available": ["private financing"],
    "lease back": ["lease back", "rent back", "seller leaseback"],
    "lease option": ["lease to own", "rent to own", "lease option", "lease purchase"],
    "conventional": ["normal loan", "regular mortgage", "traditional loan", "conventional loan"]

}
excluded_terms = set([
    "owner survey",
    "lien release",
    "land use fee",
    "court approval",
    "trust deed",
    "trust conveyance",
    "owner pay points",
    "relocation property",
    "submit",
    "contract",
    "subordinate",
    "trade",
    "subject to other",
])

finance_set = set()
processed_to_raw_dict = defaultdict(list)
financing_list = text_features["financing"]
for term in financing_list:
    if term in excluded_terms:
        continue
    if term in fha_set:
        normalized_terms = normalize_finance.get("fha")
    elif term in va_set:
        normalized_terms = normalize_finance.get("va")
    elif term in owner_carry_set:
        normalized_terms = normalize_finance.get("owner carry")
    else: 
        normalized_terms = normalize_finance.get(term, [term])
    for i in normalized_terms:
        key = normalize(i)
        finance_set.add(key)
        processed_to_raw_dict[key].append(term)


finance_comb_set = ( finance_opt_manul | finance_set)

finance_list = list(finance_comb_set)
finance_list = sorted(finance_list)

finance_dict = {"finance": finance_list,
                "set_from_db": list(finance_set),
                "finance2raw": processed_to_raw_dict}
                
with open("data/processed/finance.json", "w") as f:
    json.dump(finance_dict, f, indent = 2)