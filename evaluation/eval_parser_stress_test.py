"""Score GPT stress-test queries against intended structured slots.

Catalog / leftover phrases are reported separately and do not affect
query-level structured pass. Extra structured keys (price leak, dual
min+max from overlapping cues) count as a miss.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from query_parser import QueryParser  # noqa: E402

sys.path.insert(0, str(ROOT / "evaluation"))
from stress_intended import intended_gold  # noqa: E402

QUERIES = ROOT / "evaluation" / "queries" / "stress_test_100.json"
OUT = ROOT / "evaluation" / "results" / "parser_stress_test_100.json"
HTML_OUT = ROOT / "evaluation" / "results" / "parser_stress_test_fails.html"
N_FAMILIES = 20

STRUCT_KEYS = (
    "city",
    "bedrooms", "bedrooms_min", "bedrooms_max",
    "bathrooms", "bathrooms_min", "bathrooms_max", "bathroom_half",
    "price", "price_min", "price_max",
    "sqft", "sqft_min", "sqft_max",
)
BUNDLES = {
    "city": ("city",),
    "beds": ("bedrooms", "bedrooms_min", "bedrooms_max"),
    "baths": ("bathrooms", "bathrooms_min", "bathrooms_max", "bathroom_half"),
    "sqft": ("sqft", "sqft_min", "sqft_max"),
    "price": ("price", "price_min", "price_max"),
}


def _slot(filters: dict, key: str):
    return filters.get(key)


def bundle_ok(expect: dict, got: dict, keys: tuple[str, ...]) -> bool:
    return all(_slot(expect, k) == _slot(got, k) for k in keys)


def structured_diffs(expect: dict, got: dict) -> list[str]:
    diffs = []
    for k in STRUCT_KEYS:
        ev, gv = _slot(expect, k), _slot(got, k)
        if ev != gv:
            diffs.append(f"{k}: expect {ev!r} got {gv!r}")
    return diffs


def _html_esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt_val(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    if isinstance(v, list):
        return ", ".join(_fmt_val(x) for x in v)
    return str(v)


def write_fail_html(payload: dict, path: Path) -> None:
    summary = payload["summary"]
    fails = [r for r in payload["queries"] if not r["structured_pass"]]
    by_fam: dict[int, list] = {}
    for row in fails:
        by_fam.setdefault(row["family"], []).append(row)

    cards = []
    for fid in sorted(by_fam):
        group = by_fam[fid]
        cards.append(f'<h2 id="f{fid}">Family {fid} · {len(group)} misses</h2>')
        for row in group:
            exp, got = row["expect"], row["got_structured"]
            keys = [k for k in STRUCT_KEYS if k in exp or k in got]
            rows_html = []
            for k in keys:
                ev, gv = exp.get(k), got.get(k)
                if ev == gv:
                    cls, tag = "ok", "match"
                elif ev is not None and gv is None:
                    cls, tag = "miss", "missing"
                elif ev is None and gv is not None:
                    cls, tag = "extra", "extra"
                else:
                    cls, tag = "wrong", "mismatch"
                rows_html.append(
                    "<tr class='{cls}'><td>{k}</td><td>{ev}</td><td>{gv}</td>"
                    "<td>{tag}</td></tr>".format(
                        cls=cls,
                        k=_html_esc(k),
                        ev=_html_esc(_fmt_val(ev)),
                        gv=_html_esc(_fmt_val(gv)),
                        tag=tag,
                    )
                )
            cat = row.get("got_catalog") or {}
            cat_bits = []
            for ck in (
                "amenities", "features", "finance",
                "negated_amenities", "negated_features", "negated_finance",
            ):
                if cat.get(ck):
                    cat_bits.append(
                        f"<span class='cat'><b>{_html_esc(ck)}</b> {_html_esc(_fmt_val(cat[ck]))}</span>"
                    )
            cat_html = " ".join(cat_bits) if cat_bits else "<span class='muted'>none</span>"
            cards.append(
                """
<article class="card">
  <div class="meta">{qid} · F{fid}</div>
  <p class="query">{query}</p>
  <table>
    <thead><tr><th>slot</th><th>expect</th><th>extracted</th><th></th></tr></thead>
    <tbody>{slot_rows}</tbody>
  </table>
  <div class="catalog">catalog: {cat}</div>
</article>
""".format(
                    qid=_html_esc(row["id"]),
                    fid=fid,
                    query=_html_esc(row["query"]),
                    slot_rows="".join(rows_html),
                    cat=cat_html,
                )
            )

    toc = " ".join(f'<a href="#f{fid}">F{fid}</a>' for fid in sorted(by_fam))
    html = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>Stress-test parser misses</title>
<style>
body {{ font-family: Segoe UI, sans-serif; margin: 24px; background: #f6f7f9; color: #1b1f24; }}
h1 {{ font-size: 20px; margin: 0 0 8px; }}
h2 {{ font-size: 16px; margin: 28px 0 12px; }}
.note {{ color: #555; max-width: 960px; line-height: 1.45; }}
.stats {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }}
.stat {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 10px 14px; min-width: 140px; }}
.k {{ font-size: 12px; color: #666; margin-bottom: 4px; }}
.toc a {{ margin-right: 8px; font-size: 13px; }}
.card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 14px 16px; margin: 0 0 12px; }}
.meta {{ font-size: 12px; color: #666; margin-bottom: 6px; }}
.query {{ font-size: 15px; line-height: 1.45; margin: 0 0 12px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #e3e6ea; padding: 5px 8px; text-align: left; }}
th {{ background: #eef2f6; }}
tr.ok td {{ color: #2d6a4f; }}
tr.miss td {{ background: #fde2e1; }}
tr.extra td {{ background: #fff3cd; }}
tr.wrong td {{ background: #fde2e1; }}
.catalog {{ margin-top: 10px; font-size: 13px; color: #333; }}
.cat {{ display: inline-block; background: #eef2f6; border-radius: 4px; padding: 2px 8px; margin: 2px 6px 2px 0; }}
.muted {{ color: #888; }}
</style></head><body>
<h1>GPT stress-test misses</h1>
<p class="note">Only queries whose intended structured slots do not exactly match
<code>QueryParser.parse()</code>. Red = missing or wrong value. Yellow = extra slot
the gold did not want. Catalog is shown for context and is not in the exact-match score.</p>
<div class="stats">
<div class="stat"><div class="k">exact match</div><div>{summary['structured_exact']}/{summary['n']} ({summary['structured_exact_pct']}%)</div></div>
<div class="stat"><div class="k">shown here</div><div>{len(fails)} misses · {len(by_fam)} families</div></div>
<div class="stat"><div class="k">city / beds / baths</div><div>{summary['field_bundle_pct']['city']}% / {summary['field_bundle_pct']['beds']}% / {summary['field_bundle_pct']['baths']}%</div></div>
<div class="stat"><div class="k">sqft / price</div><div>{summary['field_bundle_pct']['sqft']}% / {summary['field_bundle_pct']['price']}%</div></div>
</div>
<p class="toc">{toc}</p>
{''.join(cards)}
</body></html>
"""
    path.write_text(html, encoding="utf-8")


def catalog_view(got: dict) -> dict:
    out = {}
    for k in (
        "amenities", "features", "finance",
        "negated_amenities", "negated_features", "negated_finance",
        "amenity_groups", "feature_groups", "finance_groups",
    ):
        if got.get(k):
            out[k] = got[k]
    return out


def uniqueness_report(queries: list[dict], golds: list[dict]) -> dict:
    texts = [q["query"] for q in queries]
    cities = [g.get("city") for g in golds]
    tuples = [tuple(g.get(k) for k in STRUCT_KEYS) for g in golds]
    return {
        "unique_queries": len(set(texts)),
        "unique_cities": len(set(cities)),
        "unique_structured_tuples": len(set(tuples)),
        "dup_cities": sorted({c for c in cities if cities.count(c) > 1}),
        "ok": (
            len(set(texts)) == len(queries)
            and len(set(cities)) == len(queries)
            and len(set(tuples)) == len(queries)
        ),
    }


def main() -> int:
    data = json.loads(QUERIES.read_text(encoding="utf-8"))
    queries = data["queries"]
    assert len(queries) == 100

    golds = [intended_gold(item["id"], item["query"]) for item in queries]
    uniq = uniqueness_report(queries, golds)
    print(
        f"uniqueness: queries {uniq['unique_queries']}/100, "
        f"cities {uniq['unique_cities']}/100, "
        f"structured tuples {uniq['unique_structured_tuples']}/100"
        + ("" if uniq["ok"] else f"  DUP CITIES {uniq['dup_cities']}")
    )
    if not uniq["ok"]:
        print("STOP: structured fields are not unique; not running parser eval.")
        return 1

    parser = QueryParser()
    rows = []
    bundle_hits = {name: 0 for name in BUNDLES}
    n_struct = 0
    family_pass = [True] * N_FAMILIES
    family_hits = [0] * N_FAMILIES

    for item, expect in zip(queries, golds):
        n = int(item["id"][1:])
        fam = (n - 1) % N_FAMILIES
        got = parser.parse(item["query"])
        diffs = structured_diffs(expect, got)
        ok = not diffs
        if ok:
            n_struct += 1
            family_hits[fam] += 1
        else:
            family_pass[fam] = False
        for name, keys in BUNDLES.items():
            if bundle_ok(expect, got, keys):
                bundle_hits[name] += 1
        rows.append({
            "id": item["id"],
            "family": fam,
            "query": item["query"],
            "structured_pass": ok,
            "diffs": diffs,
            "expect": expect,
            "got_structured": {k: got[k] for k in STRUCT_KEYS if k in got},
            "got_catalog": catalog_view(got),
        })

    n = len(rows)
    summary = {
        "n": n,
        "uniqueness": uniq,
        "structured_exact": n_struct,
        "structured_exact_pct": round(100.0 * n_struct / n, 1),
        "families_exact": sum(family_pass),
        "families_exact_pct": round(100.0 * sum(family_pass) / N_FAMILIES, 1),
        "field_bundle_pct": {
            name: round(100.0 * hits / n, 1) for name, hits in bundle_hits.items()
        },
        "failed_families": [i for i, ok in enumerate(family_pass) if not ok],
        "passed_families": [i for i, ok in enumerate(family_pass) if ok],
    }
    payload = {"summary": summary, "queries": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_fail_html(payload, HTML_OUT)

    print(f"structured exact {n_struct}/{n} = {summary['structured_exact_pct']}%")
    print(f"wording families  {sum(family_pass)}/{N_FAMILIES} = {summary['families_exact_pct']}%")
    print("field bundles:")
    for name, pct in summary["field_bundle_pct"].items():
        print(f"  {name:5s} {bundle_hits[name]:3d}/{n}  {pct}%")
    print("passed families:", summary["passed_families"])
    print("failed families:", summary["failed_families"])
    print("family handle:")
    for i in range(N_FAMILIES):
        ids = ", ".join(f"Q{n:03d}" for n in range(i + 1, 101, 20))
        print(f"  F{i:02d} {family_hits[i]}/5  {ids}")
    print()
    seen = set()
    for row in rows:
        if row["structured_pass"] or row["family"] in seen:
            continue
        seen.add(row["family"])
        print(f"F{row['family']:02d} {row['id']} {row['query'][:110]}...")
        for d in row["diffs"]:
            print(f"    {d}")
        print()
    print(f"wrote {OUT}")
    print(f"wrote {HTML_OUT} ({len([r for r in rows if not r['structured_pass']])} misses)")
    return 0


def main_from_json() -> int:
    payload = json.loads(OUT.read_text(encoding="utf-8"))
    write_fail_html(payload, HTML_OUT)
    n_fail = sum(1 for r in payload["queries"] if not r["structured_pass"])
    print(f"wrote {HTML_OUT} ({n_fail} misses)")
    return 0


if __name__ == "__main__":
    if "--from-json" in sys.argv:
        raise SystemExit(main_from_json())
    raise SystemExit(main())
