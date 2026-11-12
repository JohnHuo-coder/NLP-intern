from unittest.mock import MagicMock

from scripts.semantic_text_llm import (
    SemanticTextLLM,
    has_catalog_terms,
    ranking_query,
)
from scripts.text_cleaning import TextCleaner


def _llm(content: str | None) -> SemanticTextLLM:
    llm = SemanticTextLLM(api_key="test")
    completion = MagicMock()
    completion.choices[0].message.content = content
    llm.client.chat.completions.create = MagicMock(return_value=completion)
    return llm


def test_extract_skips_blank_query():
    llm = _llm('{"semantic_text": "quiet street"}')
    assert llm.extract("   ") == ""
    llm.client.chat.completions.create.assert_not_called()


def test_extract_cleans_query_and_collapses_leftover():
    llm = _llm('{"semantic_text": "  quiet   street  "}')
    query = "3 br near a quiet street"
    assert llm.extract(query) == "quiet street"

    call = llm.client.chat.completions.create.call_args.kwargs
    assert call["model"] == "gpt-4o-mini"
    assert call["temperature"] == 0
    assert call["messages"][1]["content"] == TextCleaner().clean_text(query)
    assert "bedroom" in call["messages"][1]["content"]
    assert call["response_format"]["json_schema"]["name"] == "semantic_text_extract"


def test_extract_empty_model_output():
    assert _llm(None).extract("quiet street") == ""
    assert _llm('{"semantic_text": ""}').extract("quiet street") == ""
    assert _llm('{"semantic_text": 12}').extract("quiet street") == "12"


def test_extract_if_unmatched_runs_when_catalog_terms_exist():
    llm = _llm('{"semantic_text": "quiet and cozy near the beach"}')
    filters = {"amenities": ["pool"], "features": ["gym"]}
    assert llm.extract_if_unmatched(
        "pool and gym, quiet and cozy near the beach",
        filters,
    ) == "quiet and cozy near the beach"
    llm.client.chat.completions.create.assert_called_once()


def test_has_catalog_terms():
    assert has_catalog_terms(None) is False
    assert has_catalog_terms({}) is False
    assert has_catalog_terms({"amenities": [], "features": []}) is False
    assert has_catalog_terms({"negated_finance": ["fha"]}) is True


def test_ranking_query_uses_leftover_beside_catalog_terms():
    filters = {"amenities": ["pool"]}
    assert ranking_query("pool, quiet street", "quiet street", filters) == "quiet street"
    assert ranking_query("pool and gym", "", filters) == ""
    assert ranking_query("pool and gym", "   ", filters) == ""


def test_ranking_query_falls_back_to_full_query_without_catalog_terms():
    assert ranking_query("  cozy home near parks  ", "", {}) == "cozy home near parks"
    assert ranking_query("browse", "", {}) == ""
    assert ranking_query("  Browse  ", None, None) == ""
    assert ranking_query("", "", None) == ""
