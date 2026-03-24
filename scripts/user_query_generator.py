import json
import random
from pathlib import Path


def build_intent_templates():
    """Intent -> query templates with placeholders."""
    return {
        "search_buy": [
            "Find me a {bed} bedroom home in {city} under ${price}.",
            "Show listings in {city} with at least {bath} bathrooms and a {amenity}.",
            "I want to buy a {bed} bed, {bath} bath house near {city}.",
            "Any homes for sale in {city} below ${price} with {amenity}?",
            "Please find properties in {city} around {sqft} square feet.",
        ],
        "search_rent": [
            "Find rentals in {city} with {bed} bedrooms.",
            "Show apartments for rent in {city} under ${price} per month.",
            "I need a rental near {city} with {amenity}.",
            "Any {bed} bed rental options in {city} with parking?",
            "Looking for a pet-friendly rental in {city} with {bath} bathrooms.",
        ],
        "compare_options": [
            "Compare the top 3 listings in {city} by price and square feet.",
            "Which is better value: a home with {amenity} or one with larger square footage in {city}?",
            "Compare properties in {city} with {bed} bedrooms under ${price}.",
            "Can you compare nearby listings by commute and amenities?",
            "I need a side-by-side comparison of two homes in {city}.",
        ],
        "schedule_visit": [
            "Schedule a viewing for the {city} listing with {amenity}.",
            "Book a house tour this weekend for a {bed} bedroom in {city}.",
            "Can I visit the property in {city} tomorrow evening?",
            "Help me arrange a virtual tour for a home under ${price} in {city}.",
            "Set up an in-person showing for the listing with {bath} bathrooms.",
        ],
        "neighborhood_info": [
            "How is the neighborhood safety in {city}?",
            "Tell me about schools near this listing in {city}.",
            "Is {city} a good area for families?",
            "What are nearby restaurants and parks around this property?",
            "How far is this home from public transit in {city}?",
        ],
        "negotiation_offer": [
            "Draft an offer at ${price} for the home in {city}.",
            "What is a reasonable bid for a {bed} bed home in {city}?",
            "Help me write a negotiation message for a listing with {amenity}.",
            "Should I offer below asking for this property?",
            "Create a counter-offer strategy for this listing in {city}.",
        ],
        "mortgage_finance": [
            "Estimate monthly mortgage for a ${price} home with 20% down.",
            "How much house can I afford if my budget is ${price}?",
            "Calculate payment for a {bed} bedroom home in {city}.",
            "What are current rates for first-time buyers?",
            "Should I choose fixed or adjustable mortgage for this purchase?",
        ],
        "property_details": [
            "Does this listing include {amenity}?",
            "How old is the roof and HVAC in this property?",
            "Is there an HOA fee for this home in {city}?",
            "What is the lot size and interior square footage?",
            "Any recent renovation records for this house?",
        ],
        "investment_analysis": [
            "Analyze rental yield for this property in {city}.",
            "Is this listing a good flip opportunity?",
            "Estimate 5-year appreciation potential in {city}.",
            "Compare cap rate with similar homes nearby.",
            "Would this home cash flow if rented at market rate?",
        ],
        "follow_up_support": [
            "Save this listing and alert me if price drops.",
            "Notify me when similar homes in {city} are listed.",
            "Can you summarize the pros and cons of this property?",
            "Remind me to follow up with the agent tomorrow.",
            "Create a shortlist of my favorite homes in {city}.",
        ],
    }


def generate_queries(n=50, seed=42):
    rng = random.Random(seed)
    templates = build_intent_templates()

    cities = [
        "Los Angeles",
        "Irvine",
        "San Diego",
        "San Jose",
        "Sacramento",
        "Austin",
        "Seattle",
        "Phoenix",
        "Denver",
        "Miami",
    ]
    amenities = [
        "pool",
        "garage",
        "balcony",
        "gym",
        "parking",
        "garden",
        "private patio",
        "fireplace",
        "walk-in closet",
        "central air conditioning",
    ]
    beds = [1, 2, 3, 4, 5]
    baths = [1, 1.5, 2, 2.5, 3, 3.5, 4]
    prices = [250000, 350000, 450000, 550000, 650000, 800000, 950000, 1200000]
    sqfts = [900, 1200, 1500, 1800, 2200, 2800, 3400]

    all_candidates = []
    for intent, intent_templates in templates.items():
        for t in intent_templates:
            query = t.format(
                city=rng.choice(cities),
                amenity=rng.choice(amenities),
                bed=rng.choice(beds),
                bath=rng.choice(baths),
                price=rng.choice(prices),
                sqft=rng.choice(sqfts),
            )
            all_candidates.append({"intent": intent, "query": query})

    # Ensure diversity: at least one per intent, then fill to n from shuffled pool.
    selected = []
    seen = set()
    for intent in templates:
        intent_items = [x for x in all_candidates if x["intent"] == intent]
        rng.shuffle(intent_items)
        for item in intent_items:
            if item["query"] not in seen:
                seen.add(item["query"])
                selected.append(item)
                break

    rng.shuffle(all_candidates)
    for item in all_candidates:
        if len(selected) >= n:
            break
        if item["query"] in seen:
            continue
        seen.add(item["query"])
        selected.append(item)

    # If still short (very unlikely), mutate a few templates for guaranteed n.
    i = 0
    while len(selected) < n:
        intent = list(templates.keys())[i % len(templates)]
        t = rng.choice(templates[intent])
        query = t.format(
            city=rng.choice(cities),
            amenity=rng.choice(amenities),
            bed=rng.choice(beds),
            bath=rng.choice(baths),
            price=rng.choice(prices) + 1000 * i,
            sqft=rng.choice(sqfts),
        )
        if query not in seen:
            seen.add(query)
            selected.append({"intent": intent, "query": query})
        i += 1

    return selected[:n]


def main():
    root = Path(__file__).resolve().parents[1]
    output_path = root / "data" / "processed" / "user_queries_50.json"
    rows = generate_queries(n=50, seed=42)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"count": len(rows), "queries": rows}, f, indent=2)
    print(f"Wrote {len(rows)} queries to {output_path}")


if __name__ == "__main__":
    main()


