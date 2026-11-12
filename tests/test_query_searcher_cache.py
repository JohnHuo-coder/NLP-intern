import json
from pathlib import Path
from unittest.mock import MagicMock

from scripts.query_searcher import (
    MAX_SEMANTIC_WINDOW,
    QuerySearcher,
    _cache_key,
)


def test_result_window_is_one_hundred():
    assert MAX_SEMANTIC_WINDOW == 100


def test_cache_key_collapses_case_and_space():
    assert _cache_key("  3 Bedroom in Irvine  ") == "3 bedroom in irvine"


class _FakeCursor:
    def __init__(self, store: dict):
        self._store = store
        self._rows: list = []

    def execute(self, sql, params=None):
        sql_l = " ".join(str(sql).lower().split())
        if sql_l.startswith("select id from"):
            self._rows = [{"id": i} for i in self._store["candidate_ids"]]
        elif sql_l.startswith("select count(*)"):
            self._rows = [{"total": len(self._store["candidate_ids"])}]
        elif "id in" in sql_l:
            ids = [int(i) for i in (params or [])]
            self._rows = [
                dict(self._store["rows_by_id"][i])
                for i in ids
                if i in self._store["rows_by_id"]
            ]
        else:
            self._rows = []

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def close(self):
        pass


class _FakeConn:
    def __init__(self, store: dict):
        self._store = store

    def cursor(self, dictionary=True):
        return _FakeCursor(self._store)

    def close(self):
        pass


class _FakePool:
    def __init__(self, candidate_ids: list[int]):
        self.store = {
            "candidate_ids": list(candidate_ids),
            "rows_by_id": {
                i: {"id": i, "city": "Irvine", "photos": None} for i in candidate_ids
            },
        }

    def get_connection(self):
        return _FakeConn(self.store)


class _FakeSemantic:
    def __init__(self):
        self.calls: list[dict] = []

    def search(self, query_text, k=5, listing_ids=None, **kwargs):
        self.calls.append({"text": query_text, "k": k, "listing_ids": listing_ids})
        source = list(listing_ids) if listing_ids is not None else list(range(1, k + 1))
        return [(int(lid), "remark", 1.0) for lid in source[:k]]


class _StubParser:
    def __init__(self, filters: dict):
        self.filters = filters

    def parse(self, query: str) -> dict:
        return dict(self.filters)

    def to_sql(self, filters, table, limit, offset, order_by):
        sql = (
            f"SELECT id FROM {table} WHERE L_City = %s "
            f"ORDER BY {order_by} LIMIT {limit} OFFSET {offset}"
        )
        return sql, ["Irvine"]


def _paths(tmp_path: Path) -> tuple[str, str]:
    cities = tmp_path / "cities.csv"
    cities.write_text("L_City\nIrvine\n", encoding="utf-8")
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({}), encoding="utf-8")
    return str(cities), str(stats)


def _hybrid_searcher(tmp_path: Path, *, window: int = 4, ttl_s: float = 3600):
    cities, stats = _paths(tmp_path)
    leftover = MagicMock()
    leftover.extract_if_unmatched.return_value = "quiet street"
    semantic = _FakeSemantic()
    ids = [10, 20, 30, 40, 50]
    searcher = QuerySearcher(
        _FakePool(ids),
        parser=_StubParser({"city": "Irvine", "bedrooms": 3}),
        semantic=semantic,
        leftover_llm=leftover,
        cities_path=cities,
        stats_path=stats,
        max_semantic_window=window,
        result_cache_ttl_s=ttl_s,
        result_cache_maxsize=32,
    )
    return searcher, leftover, semantic


def test_hybrid_miss_ranks_full_window_and_pages_from_cache(tmp_path: Path):
    searcher, leftover, semantic = _hybrid_searcher(tmp_path, window=4)
    query = "3 bedroom in Irvine with a quiet street"

    page1 = searcher.search(query, limit=2, offset=0)
    assert "error" not in page1
    assert page1["mode"] == "hybrid"
    assert [row["id"] for row in page1["listings"]] == [10, 20]
    assert page1["pagination"]["total"] == 4
    leftover.extract_if_unmatched.assert_called_once()
    assert len(semantic.calls) == 1
    assert semantic.calls[0]["k"] == 4
    assert semantic.calls[0]["text"] == "quiet street"

    page2 = searcher.search(query, limit=2, offset=2)
    assert [row["id"] for row in page2["listings"]] == [30, 40]
    leftover.extract_if_unmatched.assert_called_once()
    assert len(semantic.calls) == 1

    past_end = searcher.search(query, limit=2, offset=4)
    assert past_end["error"]["code"] == "not_found"
    leftover.extract_if_unmatched.assert_called_once()
    assert len(semantic.calls) == 1


def test_same_query_different_case_hits_cache(tmp_path: Path):
    searcher, leftover, semantic = _hybrid_searcher(tmp_path, window=4)
    searcher.search("3 Bedroom in Irvine", limit=2, offset=0)
    searcher.search("  3 bedroom in irvine ", limit=2, offset=2)
    leftover.extract_if_unmatched.assert_called_once()
    assert len(semantic.calls) == 1


def test_semantic_path_caches_id_list(tmp_path: Path):
    cities, stats = _paths(tmp_path)
    leftover = MagicMock()
    leftover.extract_if_unmatched.return_value = "cozy near parks"
    semantic = _FakeSemantic()
    searcher = QuerySearcher(
        _FakePool([1, 2, 3]),
        parser=_StubParser({}),
        semantic=semantic,
        leftover_llm=leftover,
        cities_path=cities,
        stats_path=stats,
        max_semantic_window=3,
        result_cache_ttl_s=3600,
        result_cache_maxsize=8,
    )
    q = "cozy near parks"
    page1 = searcher.search(q, limit=1, offset=0)
    assert page1["mode"] == "semantic"
    assert [row["id"] for row in page1["listings"]] == [1]
    page2 = searcher.search(q, limit=1, offset=1)
    assert [row["id"] for row in page2["listings"]] == [2]
    leftover.extract_if_unmatched.assert_called_once()
    assert len(semantic.calls) == 1
    assert semantic.calls[0]["k"] == 3
    assert semantic.calls[0]["listing_ids"] is None
