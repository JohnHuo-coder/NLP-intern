import json
import random
from pathlib import Path

# User journey stages (every generated query belongs to exactly one).
JOURNEYS = ("browsing", "researching", "ready-to-buy")


def build_journey_templates():
    """Journey stage -> list of (intent, template with placeholders).

    browsing: casual exploration, no commitment
    researching: listing search with clearer filters than browsing, still exploratory (not compare/advice)
    ready-to-buy: concrete criteria, next-step actions
    """
    return {
        "browsing": [
            (
                "browse",
                "I'm just looking around {city} — anything under ${price}?",
            ),
            (
                "browse",
                "Show me what's out there in {city} with around {bed} bedrooms.",
            ),
            (
                "browse",
                "Curious what homes near {city} might have a {amenity}.",
            ),
            (
                "browse",
                "Just browsing rentals in {city} with {bed} bedrooms.",
            ),
            (
                "browse",
                "Any pet-friendly rental ideas in {city}? Not in a rush.",
            ),
        ],
        "researching": [
            (
                "search_research",
                "I'm narrowing down in {city} — show homes around ${price}, still flexible.",
            ),
            (
                "search_research",
                "What listings in {city} have about {bed} bedrooms? Just researching, not in a hurry.",
            ),
            (
                "search_research",
                "Search {city} for places with {amenity}; I'm shortlisting before I get serious.",
            ),
            (
                "search_research",
                "Show me what's on the market in {city} with at least {bath} bathrooms.",
            ),
            (
                "search_research",
                "I'd like to see rentals in {city} near {sqft} sq ft — exploring, not applying yet.",
            ),
            (
                "search_research",
                "Pull up homes in {city} under roughly ${price} with {amenity}; still exploring the market.",
            ),
            (
                "search_research",
                "Look for options in {city}: around {bed} beds and {amenity}, no rush.",
            ),
        ],
        "ready-to-buy": [
            (
                "search_buy",
                "Find me a {bed} bedroom home in {city} under ${price}.",
            ),
            (
                "search_buy",
                "Show listings in {city} with at least {bath} bathrooms and a {amenity}.",
            ),
            (
                "search_buy",
                "I want to buy a {bed} bed, {bath} bath house near {city}.",
            ),
            (
                "search_buy",
                "Any homes for sale in {city} below ${price} with {amenity}?",
            ),
            (
                "search_buy",
                "Please find properties in {city} around {sqft} square feet.",
            ),
            (
                "search_buy",
                "Need a {bed} bed {bath} bath in {city} under ${price} with {amenity}.",
            ),
            (
                "search_buy",
                "Looking to buy in {city}: at least {bed} bedrooms, max ${price}.",
            ),
            (
                "search_buy",
                "Show single-family homes in {city} around {sqft} sq ft below ${price}.",
            ),
            (
                "search_buy",
                "I want listings in {city} with {amenity}, {bed} BR or more, budget ${price}.",
            ),
        ],
    }


def generate_queries(n=50, seed=42):
    rng = random.Random(seed)
    templates_by_journey = build_journey_templates()

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
    with open("data/processed/amenities.json", "r") as f:
        data = json.load(f)
    amenities = data["amenities"]
    beds = [1, 2, 3, 4, 5]
    baths = [1, 1.5, 2, 2.5, 3, 3.5, 4]
    prices = [250000, 350000, 450000, 550000, 650000, 800000, 950000, 1200000]
    sqfts = [900, 1200, 1500, 1800, 2200, 2800, 3400]

    def format_item(journey, intent, template):
        query = template.format(
            city=rng.choice(cities),
            amenity=rng.choice(amenities),
            bed=rng.choice(beds),
            bath=rng.choice(baths),
            price=rng.choice(prices),
            sqft=rng.choice(sqfts),
        )
        return {
            "journey": journey,
            "intent": intent,
            "query": query,
        }

    all_candidates = []
    for journey, pairs in templates_by_journey.items():
        for intent, t in pairs:
            all_candidates.append(format_item(journey, intent, t))

    # Ensure diversity: at least one per journey, then per intent within journey.
    selected = []
    seen = set()

    for journey in JOURNEYS:
        journey_items = [x for x in all_candidates if x["journey"] == journey]
        rng.shuffle(journey_items)
        for item in journey_items:
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

    i = 0
    while len(selected) < n:
        journey = JOURNEYS[i % len(JOURNEYS)]
        intent, t = rng.choice(templates_by_journey[journey])
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
            selected.append(
                {"journey": journey, "intent": intent, "query": query}
            )
        i += 1

    return selected[:n]


def main():
    root = Path(__file__).resolve().parents[2]
    output_path = root / "data" / "processed" / "user_queries_50.json"
    rows = generate_queries(n=50, seed=42)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"count": len(rows), "queries": rows}, f, indent=2)
    print(f"Wrote {len(rows)} queries to {output_path}")


if __name__ == "__main__":
    main()
