"""Turn a parsed filter dict into listing SQL (column routing for catalog terms)."""

from __future__ import annotations


LISTING_SELECT = """
            SELECT id,
                L_Address as address,
                L_Zip as zipcode,
                L_City as city,
                L_State as state,
                L_Keyword2 as bedrooms,
                LM_Dec_3 as bathrooms,
                L_SystemPrice as price,
                LM_Int2_3 as living_area,
                L_Remarks as remark,
                L_Photos as photos,
                Flooring as flooring,
                ViewYN,
                PoolPrivateYN,
                AttachedGarageYN,
                FireplaceYN,
                HeatingYN,
                Appliances,
                CoolingYN,
                GarageYN,
                SpaYN,
                BathroomsHalf as half_bathrooms,
                AssociationAmenities,
                StructureType,
                ArchitecturalStyle,
                Cooling,
                Heating,
                View,
                FireplaceFeatures,
                InteriorFeatures,
                PoolFeatures,
                CommunityFeatures,
                SecurityFeatures,
                SpaFeatures
            FROM {table} WHERE {where_clause}
            ORDER BY {order_by}
            LIMIT {limit} OFFSET {offset}"""

_NUMERIC_CLAUSES = (
    ("price_max", "L_SystemPrice <= %s"),
    ("price_min", "L_SystemPrice >= %s"),
    ("price", "L_SystemPrice = %s"),
    ("bedrooms", "L_Keyword2 = %s"),
    ("bedrooms_min", "L_Keyword2 >= %s"),
    ("bedrooms_max", "L_Keyword2 <= %s"),
    ("bathrooms", "LM_Dec_3 = %s"),
    ("bathrooms_min", "LM_Dec_3 >= %s"),
    ("bathrooms_max", "LM_Dec_3 <= %s"),
    ("bathroom_half", "BathroomsHalf = %s"),
    ("sqft_min", "LM_Int2_3 >= %s"),
    ("sqft_max", "LM_Int2_3 <= %s"),
    ("sqft", "LM_Int2_3 = %s"),
    ("city", "L_City = %s"),
)


def _nospace(s: str) -> str:
    """MLS catalog tokens often drop spaces. 'walk in closet' -> 'walkincloset'."""
    return s.replace(" ", "")


def _or_like(conditions: list, params: list, columns: list[str], needles: list[str]) -> None:
    """Match if any column ILIKE-hits. One needle is reused per column when lengths match.

    Example: columns=[AssociationAmenities, L_Remarks], needles=['%gym%', '%gym%']
    -> (LOWER(AssociationAmenities) LIKE %s OR LOWER(L_Remarks) LIKE %s)
    """
    parts = [f"LOWER({c}) LIKE %s" for c in columns]
    conditions.append("(" + " OR ".join(parts) + ")")
    params.extend(needles)


def _col_or_remarks(conditions: list, params: list, column: str, raw: str, term: str) -> None:
    """Hit the structured MLS column *or* free-text remarks.

    Example: pool feature 'infinity' -> PoolFeatures LIKE '%infinity%' OR L_Remarks LIKE '%infinity%'
    ``raw`` is the DB spelling (spaces stripped); ``term`` is the query phrase.
    """
    _or_like(
        conditions,
        params,
        [column, "L_Remarks"],
        [f"%{_nospace(raw)}%", f"%{term}%"],
    )


def _not_like_or_null_parts(column: str, needle: str) -> tuple[str, str]:
    """Negation fragment: column must not contain needle; NULL counts as 'does not have'.

    Example: no carpet -> (LOWER(Flooring) NOT LIKE '%carpet%' OR Flooring IS NULL)
    """
    return f"(LOWER({column}) NOT LIKE %s OR {column} IS NULL)", f"%{needle}%"


def _taxonomy_maps(blob: dict, set_key: str, raw_key: str | None = None):
    """Pull one catalog slice: the phrase set plus phrase -> DB raw-value map.

    Example: _taxonomy_maps(amenities.json blob, 'pool') ->
    ({'infinity', 'lap', ...}, {'infinity': 'Infinity', ...})
    """
    raw_key = raw_key or f"{set_key}2raw"
    return set(blob["set_from_db"][set_key]), blob[raw_key]


def add_numeric_conditions(conditions: list, params: list, filters: dict) -> None:
    """Append price/beds/baths/sqft/city equalities from the parser dict, in clause order."""
    for key, clause in _NUMERIC_CLAUSES:
        if key in filters:
            conditions.append(clause)
            params.append(filters[key])


def _and_bundle(clauses: list[str]) -> str | None:
    """AND the SQL fragments that together represent *one* catalog term.

    Example: garage -> (GarageYN = 1 OR AttachedGarageYN = 1) AND PoolFeatures LIKE ...
    would be two clauses joined with AND. Empty list means the term has no SQL.
    """
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return "(" + " AND ".join(clauses) + ")"


def _append_or_groups(conditions: list, params: list, groups: list, clause_fn) -> None:
    """AND across groups; OR the per-term bundles inside a group.

    ``gym and parking`` -> [['gym'], ['parking']] -> gym AND parking
    ``pool or spa`` / ``garage/driveway`` -> [['pool', 'spa']] -> (pool OR spa)
    """
    for group in groups:
        bundles = []
        bundle_params = []
        for term in group:
            clauses, term_params = clause_fn(term)
            bundled = _and_bundle(clauses)
            if bundled:
                bundles.append(bundled)
                bundle_params.extend(term_params)
        if not bundles:
            continue
        if len(bundles) == 1:
            conditions.append(bundles[0])
        else:
            conditions.append("(" + " OR ".join(bundles) + ")")
        params.extend(bundle_params)


def _append_negated_terms(conditions: list, params: list, terms: list, clause_fn) -> None:
    """Each rejected term is AND'd (``no pool, no HOA`` → NOT pool AND NOT HOA)."""
    for term in terms:
        clauses, term_params = clause_fn(term)
        bundled = _and_bundle(clauses)
        if bundled:
            conditions.append(bundled)
            params.extend(term_params)


def _positive_amenity_clauses(amenity: str, ctx: dict) -> tuple[list[str], list]:
    conditions: list[str] = []
    params: list = []
    if amenity == "private pool":
        return ["PoolPrivateYN = 1"], []
    if amenity == "attached garage":
        return ["AttachedGarageYN = 1"], []
    if amenity == "spa":
        return ["SpaYN = 1"], []
    if amenity in ("pool", "swimming pool"):
        conditions.append(
            "(LOWER(AssociationAmenities) LIKE %s OR LOWER(CommunityFeatures) LIKE %s "
            "OR (LOWER(PoolFeatures) != 'none' AND LOWER(PoolFeatures) IS NOT NULL) "
            "OR LOWER(L_Remarks) LIKE %s)"
        )
        needle = f"%{_nospace(amenity)}%"
        params.extend([needle, needle, needle])
        return conditions, params
    if amenity == "guard":
        _or_like(
            conditions, params,
            ["AssociationAmenities", "Security", "L_Remarks"],
            [f"%{amenity}%"] * 3,
        )
        return conditions, params
    if amenity == "gated":
        _or_like(
            conditions, params,
            ["CommunityFeatures", "Security", "L_Remarks"],
            [f"%{amenity}%"] * 3,
        )
        # no return: original fall-through
    if amenity in ("horse trail", "golf", "golf course", "dog", "dog park", "park"):
        needle = f"%{_nospace(amenity)}%"
        _or_like(
            conditions, params,
            ["AssociationAmenities", "CommunityFeatures", "L_Remarks"],
            [needle, needle, needle],
        )
        return conditions, params
    if "garage" in amenity:
        conditions.append("(GarageYN = 1 OR AttachedGarageYN = 1)")
    if "parking" in amenity:
        conditions.append("(OpenParkingSpaces >= 1 OR GarageYN = 1 OR AttachedGarageYN = 1)")
    if amenity in ctx["pool_set"]:
        _col_or_remarks(conditions, params, "PoolFeatures", ctx["pool2raw"][amenity], amenity)
    elif amenity in ctx["spa_set"]:
        conditions.append("SpaYN = 1")
        _col_or_remarks(conditions, params, "SpaFeatures", ctx["spa2raw"][amenity], amenity)
    elif amenity in ctx["view_set"]:
        conditions.append("ViewYN = 1")
        _col_or_remarks(conditions, params, "View", ctx["view2raw"][amenity], amenity)
    elif amenity in ctx["security_set"]:
        _col_or_remarks(
            conditions, params, "SecurityFeatures", ctx["security2raw"][amenity], amenity
        )
    elif amenity in ctx["association_set"]:
        _col_or_remarks(
            conditions, params, "AssociationAmenities", ctx["association2raw"][amenity], amenity
        )
    elif amenity in ctx["community_set"]:
        _col_or_remarks(
            conditions, params, "CommunityFeatures", ctx["community2raw"][amenity], amenity
        )
    else:
        conditions.append("LOWER(L_Remarks) LIKE %s")
        params.append(f"%{amenity}%")
    return conditions, params


def _negated_amenity_clauses(amenity: str, ctx: dict) -> tuple[list[str], list]:
    """SQL for one rejected amenity. Same column routing as the positive path.

    ``no parking`` also zeros garage/open-parking flags, then still applies
    the taxonomy/remarks NOT LIKE when the phrase maps to a catalog set.
    """
    conditions: list[str] = []
    params: list = []
    if amenity == "private pool":
        return ["PoolPrivateYN = 0"], []
    if amenity == "attached garage":
        return ["AttachedGarageYN = 0"], []
    if amenity == "garage":
        return ["(GarageYN = 0 AND AttachedGarageYN = 0)"], []
    if amenity == "spa":
        return ["SpaYN = 0"], []
    if "parking" in amenity:
        conditions.append("(OpenParkingSpaces = 0 AND GarageYN = 0 AND AttachedGarageYN = 0)")
    if amenity in ctx["pool_set"]:
        clause, param = _not_like_or_null_parts("PoolFeatures", _nospace(ctx["pool2raw"][amenity]))
        conditions.append(clause)
        params.append(param)
    elif amenity in ctx["spa_set"]:
        conditions.append("SpaYN = 0")
        clause, param = _not_like_or_null_parts("SpaFeatures", _nospace(ctx["spa2raw"][amenity]))
        conditions.append(clause)
        params.append(param)
    elif amenity in ctx["view_set"]:
        conditions.append("ViewYN = 0")
        clause, param = _not_like_or_null_parts("View", _nospace(ctx["view2raw"][amenity]))
        conditions.append(clause)
        params.append(param)
    elif amenity in ctx["security_set"]:
        clause, param = _not_like_or_null_parts(
            "SecurityFeatures", _nospace(ctx["security2raw"][amenity])
        )
        conditions.append(clause)
        params.append(param)
    elif amenity in ctx["association_set"]:
        clause, param = _not_like_or_null_parts(
            "AssociationAmenities", _nospace(ctx["association2raw"][amenity])
        )
        conditions.append(clause)
        params.append(param)
    elif amenity in ctx["community_set"]:
        clause, param = _not_like_or_null_parts(
            "CommunityFeatures", _nospace(ctx["community2raw"][amenity])
        )
        conditions.append(clause)
        params.append(param)
    else:
        clause, param = _not_like_or_null_parts("L_Remarks", amenity.lower())
        conditions.append(clause)
        params.append(param)
    return conditions, params


def add_amenity_conditions(conditions: list, params: list, filters: dict, extractor) -> None:
    blob = extractor.amenity_full
    pool_set, pool2raw = _taxonomy_maps(blob, "pool")
    spa_set, spa2raw = _taxonomy_maps(blob, "spa")
    view_set, view2raw = _taxonomy_maps(blob, "view")
    security_set, security2raw = _taxonomy_maps(blob, "security")
    association_set, association2raw = _taxonomy_maps(blob, "association")
    community_set, community2raw = _taxonomy_maps(blob, "community")
    ctx = {
        "pool_set": pool_set, "pool2raw": pool2raw,
        "spa_set": spa_set, "spa2raw": spa2raw,
        "view_set": view_set, "view2raw": view2raw,
        "security_set": security_set, "security2raw": security2raw,
        "association_set": association_set, "association2raw": association2raw,
        "community_set": community_set, "community2raw": community2raw,
    }

    groups = filters.get("amenity_groups") or [
        [t] for t in filters.get("amenities", [])
    ]
    _append_or_groups(
        conditions, params, groups, lambda term: _positive_amenity_clauses(term, ctx)
    )
    _append_negated_terms(
        conditions,
        params,
        filters.get("negated_amenities", []),
        lambda term: _negated_amenity_clauses(term, ctx),
    )


def _positive_feature_clauses(feature: str, ctx: dict) -> tuple[list[str], list]:
    conditions: list[str] = []
    params: list = []
    if "fireplace" in feature:
        return ["FireplaceYN = 1"], []
    if feature in ctx["interior_set"]:
        parts = []
        for raw in ctx["interior2raw"][feature]:
            parts.append("LOWER(InteriorFeatures) LIKE %s")
            params.append(f"%{_nospace(raw)}%")
        parts.append("LOWER(L_Remarks) LIKE %s")
        params.append(f"%{feature}%")
        conditions.append("(" + " OR ".join(parts) + ")")
        return conditions, params
    if feature in ctx["appl_set"]:
        _col_or_remarks(conditions, params, "Appliances", ctx["appl2raw"][feature], feature)
    elif feature in ctx["cooling_set"]:
        conditions.append("CoolingYN = 1")
        _col_or_remarks(conditions, params, "Cooling", ctx["cool2raw"][feature], feature)
    elif feature in ctx["floor_set"]:
        _col_or_remarks(conditions, params, "Flooring", ctx["floor2raw"][feature], feature)
    elif feature in ctx["heating_set"]:
        conditions.append("HeatingYN = 1")
        _col_or_remarks(conditions, params, "Heating", ctx["heat2raw"][feature], feature)
    else:
        conditions.append("LOWER(L_Remarks) LIKE %s")
        params.append(f"%{feature}%")
    return conditions, params


def _negated_feature_clauses(feature: str, ctx: dict) -> tuple[list[str], list]:
    """SQL for one rejected interior feature / appliance / HVAC / flooring term."""
    conditions: list[str] = []
    params: list = []
    if "fireplace" in feature:
        return ["FireplaceYN = 0"], []
    if feature in ctx["interior_set"]:
        parts = []
        for raw in ctx["interior2raw"][feature]:
            parts.append("(LOWER(InteriorFeatures) NOT LIKE %s OR InteriorFeatures IS NULL)")
            params.append(f"%{_nospace(raw).lower()}%")
        conditions.append("(" + " AND ".join(parts) + ")")
        return conditions, params
    if feature in ctx["appl_set"]:
        clause, param = _not_like_or_null_parts("Appliances", _nospace(ctx["appl2raw"][feature]))
        return [clause], [param]
    if feature in ctx["cooling_set"]:
        clause, param = _not_like_or_null_parts("Cooling", _nospace(ctx["cool2raw"][feature]))
        return ["CoolingYN = 0", clause], [param]
    if feature in ctx["floor_set"]:
        clause, param = _not_like_or_null_parts("Flooring", _nospace(ctx["floor2raw"][feature]))
        return [clause], [param]
    if feature in ctx["heating_set"]:
        clause, param = _not_like_or_null_parts("Heating", _nospace(ctx["heat2raw"][feature]))
        return ["HeatingYN = 0", clause], [param]
    clause, param = _not_like_or_null_parts("L_Remarks", feature.lower())
    return [clause], [param]


def add_feature_conditions(conditions: list, params: list, filters: dict, extractor) -> None:
    blob = extractor.feature_full
    interior_set, interior2raw = _taxonomy_maps(blob, "interior")
    floor_set, floor2raw = _taxonomy_maps(blob, "floor")
    appl_set, appl2raw = _taxonomy_maps(blob, "appl")
    cooling_set, cool2raw = _taxonomy_maps(blob, "cooling", "cool2raw")
    heating_set, heat2raw = _taxonomy_maps(blob, "heating", "heat2raw")
    ctx = {
        "interior_set": interior_set, "interior2raw": interior2raw,
        "floor_set": floor_set, "floor2raw": floor2raw,
        "appl_set": appl_set, "appl2raw": appl2raw,
        "cooling_set": cooling_set, "cool2raw": cool2raw,
        "heating_set": heating_set, "heat2raw": heat2raw,
    }

    groups = filters.get("feature_groups") or [
        [t] for t in filters.get("features", [])
    ]
    _append_or_groups(
        conditions, params, groups, lambda term: _positive_feature_clauses(term, ctx)
    )
    _append_negated_terms(
        conditions,
        params,
        filters.get("negated_features", []),
        lambda term: _negated_feature_clauses(term, ctx),
    )


def _positive_finance_clauses(finance: str, ctx: dict) -> tuple[list[str], list]:
    """One wanted term: any ListingTerms spelling *or* remarks mention.

    Example: FHA -> ListingTerms LIKE '%fha203k%' OR ... OR L_Remarks LIKE '%fha%'
    """
    if finance in ctx["finance_set"]:
        parts = []
        params = []
        for raw in ctx["finance2raw"][finance]:
            parts.append("LOWER(ListingTerms) LIKE %s")
            params.append(f"%{_nospace(raw)}%")
        parts.append("LOWER(L_Remarks) LIKE %s")
        params.append(f"%{finance}%")
        return ["(" + " OR ".join(parts) + ")"], params
    return ["LOWER(L_Remarks) LIKE %s"], [f"%{finance}%"]


def _negated_finance_clauses(finance: str, ctx: dict) -> tuple[list[str], list]:
    """One rejected term: none of the ListingTerms spellings and not in remarks.

    Example: no VA -> ListingTerms NOT LIKE '%valoan%' (or NULL) AND remarks NOT LIKE '%va%'
    """
    conditions: list[str] = []
    params: list = []
    if finance in ctx["finance_set"]:
        for raw in ctx["finance2raw"][finance]:
            clause, param = _not_like_or_null_parts("ListingTerms", _nospace(raw))
            conditions.append(clause)
            params.append(param)
    clause, param = _not_like_or_null_parts("L_Remarks", finance.lower())
    conditions.append(clause)
    params.append(param)
    return conditions, params


def add_finance_conditions(conditions: list, params: list, filters: dict, extractor) -> None:
    blob = extractor.finance_full
    ctx = {
        "finance_set": set(blob["set_from_db"]),
        "finance2raw": blob["finance2raw"],
    }
    groups = filters.get("finance_groups") or [
        [t] for t in filters.get("finance", [])
    ]
    _append_or_groups(
        conditions, params, groups, lambda term: _positive_finance_clauses(term, ctx)
    )
    _append_negated_terms(
        conditions,
        params,
        filters.get("negated_finance", []),
        lambda term: _negated_finance_clauses(term, ctx),
    )


def build_listing_sql(
    extractor,
    filters: dict,
    *,
    table: str = "rets_property",
    order_by: str = "id DESC",
    limit: int = 50,
    offset: int = 0,
) -> tuple[str, list]:
    conditions: list[str] = []
    params: list = []
    add_numeric_conditions(conditions, params, filters)
    add_amenity_conditions(conditions, params, filters, extractor)
    add_feature_conditions(conditions, params, filters, extractor)
    add_finance_conditions(conditions, params, filters, extractor)

    where_clause = " AND ".join(conditions) if conditions else "1=1"
    sql = LISTING_SELECT.format(
        table=table,
        where_clause=where_clause,
        order_by=order_by,
        limit=max(1, int(limit)),
        offset=max(0, int(offset)),
    )
    return sql, params
