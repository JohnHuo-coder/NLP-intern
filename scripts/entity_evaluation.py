import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from entity_extractor import EntityExtractor

extractor = EntityExtractor()

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "processed" / "synthetic_remarks_labeled.json"
DEFAULT_REPORT = ROOT / "data" / "processed" / "entity_evaluation_report.txt"


def compare_numeric_result(extracted, real):
    if extracted is None and real == "none":
        return 1
    return int(extracted == real)


def compare_price(extracted, real):
    """extract_price returns int or None; label price is int or the string 'none'."""
    if real == "none":
        return int(extracted is None)
    if extracted is None:
        return 0
    return int(int(extracted) == int(real))


def gold_amenity_spans(labels_amenities):
    """Labels use a list of dicts or the string 'none' when no amenities."""
    if labels_amenities == "none" or not labels_amenities:
        return []
    return [(item["start"], item["end"]) for item in labels_amenities]


def gold_amenity_named(labels_amenities):
    if labels_amenities == "none" or not labels_amenities:
        return []
    return [
        (d["start"], d["end"], d.get("name", ""))
        for d in labels_amenities
    ]


def fmt_scalar(v):
    if v is None:
        return "None"
    if v == "none":
        return "none"
    return repr(v)


def run_evaluation(data_path: Path):
    with data_path.open(encoding="utf-8") as f:
        test_data = json.load(f)
    remarks = test_data["remarks"]
    n = len(remarks)

    bd_result = []
    ba_result = []
    price_result = []
    sqft_result = []
    tp = fp = fn = 0

    err_bedrooms = []
    err_bathrooms = []
    err_price = []
    err_sqft = []
    err_amenities = []

    for item in remarks:
        text = item["text"]
        labels = item["labels"]
        rid = item.get("id", "?")
        result = extractor.extract_all(text)

        ok_bd = compare_numeric_result(result["bedrooms"], labels["bedrooms"])
        bd_result.append(ok_bd)
        if not ok_bd:
            err_bedrooms.append(
                {
                    "id": rid,
                    "extracted": result["bedrooms"],
                    "ground_truth": labels["bedrooms"],
                }
            )

        ok_ba = compare_numeric_result(result["bathrooms"], labels["bathrooms"])
        ba_result.append(ok_ba)
        if not ok_ba:
            err_bathrooms.append(
                {
                    "id": rid,
                    "extracted": result["bathrooms"],
                    "ground_truth": labels["bathrooms"],
                }
            )

        ok_price = compare_price(result["price"], labels["price"])
        price_result.append(ok_price)
        if not ok_price:
            err_price.append(
                {
                    "id": rid,
                    "extracted": result["price"],
                    "ground_truth": labels["price"],
                }
            )

        ok_sqft = compare_numeric_result(result["sqft"], labels["sqft"])
        sqft_result.append(ok_sqft)
        if not ok_sqft:
            err_sqft.append(
                {
                    "id": rid,
                    "extracted": result["sqft"],
                    "ground_truth": labels["sqft"],
                }
            )

        gold = gold_amenity_spans(labels["amenities"])
        pred = [(t[0], t[1]) for t in result["amenities tuple"]]
        pred_set = set(pred)
        gold_set = set(gold)
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

        if pred_set != gold_set:
            pred_named = [
                (s, e, phrase) for s, e, phrase in result["amenities tuple"]
            ]
            err_amenities.append(
                {
                    "id": rid,
                    "text_preview": text if len(text) <= 280 else text[:277] + "...",
                    "ground_truth_spans": gold_amenity_named(labels["amenities"]),
                    "extracted_spans": pred_named,
                }
            )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    summary = {
        "n_remarks": n,
        "amenity_precision": precision,
        "amenity_recall": recall,
        "amenity_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "bedroom_accuracy": sum(bd_result) / n if n else 0.0,
        "bathroom_accuracy": sum(ba_result) / n if n else 0.0,
        "price_accuracy": sum(price_result) / n if n else 0.0,
        "sqft_accuracy": sum(sqft_result) / n if n else 0.0,
    }

    return summary, err_bedrooms, err_bathrooms, err_price, err_sqft, err_amenities


def write_report(
    path: Path,
    summary: dict,
    err_bedrooms,
    err_bathrooms,
    err_price,
    err_sqft,
    err_amenities,
) -> None:
    lines = []
    lines.append("Entity extraction evaluation report")
    lines.append("=" * 50)
    lines.append("")
    lines.append("Summary")
    lines.append("-" * 20)
    lines.append(f"  Remarks evaluated:     {summary['n_remarks']}")
    lines.append(f"  Amenity precision:     {summary['amenity_precision']:.6f}")
    lines.append(f"  Amenity recall:        {summary['amenity_recall']:.6f}")
    lines.append(f"  Amenity F1:              {summary['amenity_f1']:.6f}")
    lines.append(f"  (tp={summary['tp']}, fp={summary['fp']}, fn={summary['fn']})")
    lines.append(f"  Bedroom accuracy:        {summary['bedroom_accuracy']:.6f}")
    lines.append(f"  Bathroom accuracy:       {summary['bathroom_accuracy']:.6f}")
    lines.append(f"  Price accuracy:          {summary['price_accuracy']:.6f}")
    lines.append(f"  Sqft accuracy:           {summary['sqft_accuracy']:.6f}")
    lines.append("")

    lines.append("Misclassified bedrooms")
    lines.append("-" * 20)
    if not err_bedrooms:
        lines.append("  (none)")
    else:
        for row in err_bedrooms:
            lines.append(
                f"  id={row['id']}: extracted={fmt_scalar(row['extracted'])}, "
                f"ground_truth={fmt_scalar(row['ground_truth'])}"
            )
    lines.append("")

    lines.append("Misclassified bathrooms")
    lines.append("-" * 20)
    if not err_bathrooms:
        lines.append("  (none)")
    else:
        for row in err_bathrooms:
            lines.append(
                f"  id={row['id']}: extracted={fmt_scalar(row['extracted'])}, "
                f"ground_truth={fmt_scalar(row['ground_truth'])}"
            )
    lines.append("")

    lines.append("Misclassified price")
    lines.append("-" * 20)
    if not err_price:
        lines.append("  (none)")
    else:
        for row in err_price:
            lines.append(
                f"  id={row['id']}: extracted={fmt_scalar(row['extracted'])}, "
                f"ground_truth={fmt_scalar(row['ground_truth'])}"
            )
    lines.append("")

    lines.append("Misclassified sqft")
    lines.append("-" * 20)
    if not err_sqft:
        lines.append("  (none)")
    else:
        for row in err_sqft:
            lines.append(
                f"  id={row['id']}: extracted={fmt_scalar(row['extracted'])}, "
                f"ground_truth={fmt_scalar(row['ground_truth'])}"
            )
    lines.append("")

    lines.append("Amenity span mismatches (ground truth vs extracted)")
    lines.append("-" * 20)
    if not err_amenities:
        lines.append("  (none)")
    else:
        for block in err_amenities:
            lines.append(f"  id={block['id']}")
            lines.append(f"    text: {block['text_preview']}")
            lines.append("    ground_truth (start, end, name):")
            for s, e, name in block["ground_truth_spans"]:
                lines.append(f"      ({s}, {e})  {name!r}")
            if not block["ground_truth_spans"]:
                lines.append("      (none)")
            lines.append("    extracted (start, end, name):")
            for s, e, name in block["extracted_spans"]:
                lines.append(f"      ({s}, {e})  {name!r}")
            if not block["extracted_spans"]:
                lines.append("      (none)")
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
    report_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_REPORT

    summary, eb, eba, ep, es, ea = run_evaluation(data_path)
    write_report(report_path, summary, eb, eba, ep, es, ea)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
