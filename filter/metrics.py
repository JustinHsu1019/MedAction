"""
metrics.py — DTC / RAC metric computation.

DTC (Disease Trajectory Consistency):
  Shortest-path distance (hops) in PrimeKG from this turn's top-1 diagnosis to GT.
  Smaller is better; 0 = exact match.

RAC (Reasoning-Action Consistency):
  Shortest-path distance in ClinGraph from the previous turn's ordered tests
  to the *change* in DDx list (delta). Smaller = the new diagnoses are tightly
  linked to what was tested.
"""
import re
import numpy as np
import networkx as nx


# ── Shared utilities ─────────────────────────────────────────

def resolve_node(text, nodes_df, index, id_col, name_col, encoder, threshold=0.0):
    """Map free text to a KG node ID via FAISS semantic search."""
    if not text or (isinstance(text, float) and np.isnan(text)):
        return None
    q_vec = encoder.encode([str(text)])
    scores, idxs = index.search(q_vec.astype("float32"), 1)
    if scores[0][0] < threshold:
        return None
    return nodes_df.iloc[idxs[0][0]][id_col]


def resolve_cached(text, nodes_df, index, id_col, name_col, text2vec):
    """Cached version of resolve_node (uses a pre-encoded text2vec dict)."""
    if not text or (isinstance(text, float) and np.isnan(text)):
        return None
    vec = text2vec.get(str(text))
    if vec is None:
        return None
    scores, idxs = index.search(vec.reshape(1, -1).astype("float32"), 1)
    return nodes_df.iloc[idxs[0][0]][id_col]


def safe_hop(G, s, t):
    """Safe shortest-path; returns None if unreachable."""
    if not s or not t:
        return None
    if s not in G or t not in G:
        return None
    if s == t:
        return 0
    try:
        return nx.shortest_path_length(G, s, t)
    except nx.NetworkXNoPath:
        return None


# ── DDx list parser ──────────────────────────────────────────

def parse_ddx_list(raw_response):
    """
    Parse disease names from the DDx List inside raw_response.

    Example input:
      ### DDx List:
      1. Ectopic pregnancy - must be ruled out...
      2. Appendicitis (pelvic variant) - hypogastric tenderness...

    Returns:
      ["Ectopic pregnancy", "Appendicitis (pelvic variant)", ...]
      or None if parsing fails.
    """
    if not raw_response or (isinstance(raw_response, float) and np.isnan(raw_response)):
        return None

    text = str(raw_response)

    ddx_match = re.search(r"###\s*DDx\s*List\s*:", text, re.IGNORECASE)
    if not ddx_match:
        return None

    after_header = text[ddx_match.end():]
    next_section = re.search(r"\n\s*###\s", after_header)
    if next_section:
        ddx_block = after_header[:next_section.start()]
    else:
        ddx_block = after_header

    items = re.findall(r"\d+\.\s*(.+?)(?:\n|$)", ddx_block)
    if not items:
        return None

    diseases = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        parts = re.split(r"\s*[-–—]\s+", item, maxsplit=1)
        name = parts[0].strip().rstrip(".")
        if name:
            diseases.append(name)

    return diseases if diseases else None


# ── DTC: Disease Trajectory Consistency ──────────────────────

def compute_dtc(pred_ddx_list, gt_diagnosis, nodes_df, index, id_col, name_col, G, encoder):
    """Returns {"dtc_hop": int or None, "best_ddx": str or None}."""
    if not pred_ddx_list or not gt_diagnosis:
        return {"dtc_hop": None, "best_ddx": None}

    gt_id = resolve_node(gt_diagnosis, nodes_df, index, id_col, name_col, encoder)
    best_hop, best_ddx = None, None
    for ddx in pred_ddx_list:
        ddx_id = resolve_node(ddx, nodes_df, index, id_col, name_col, encoder)
        h = safe_hop(G, ddx_id, gt_id)
        if h is not None and (best_hop is None or h < best_hop):
            best_hop, best_ddx = h, ddx
    return {"dtc_hop": best_hop, "best_ddx": best_ddx}


def compute_dtc_cached(pred_ddx_list, gt_diagnosis, nodes_df, index, id_col, name_col, G, text2vec):
    """Cached version of compute_dtc."""
    if not pred_ddx_list or not gt_diagnosis:
        return {"dtc_hop": None, "best_ddx": None}

    gt_id = resolve_cached(gt_diagnosis, nodes_df, index, id_col, name_col, text2vec)
    best_hop, best_ddx = None, None
    for ddx in pred_ddx_list:
        ddx_id = resolve_cached(ddx, nodes_df, index, id_col, name_col, text2vec)
        h = safe_hop(G, ddx_id, gt_id)
        if h is not None and (best_hop is None or h < best_hop):
            best_hop, best_ddx = h, ddx
    return {"dtc_hop": best_hop, "best_ddx": best_ddx}


# ── RAC: Reasoning-Action Consistency ────────────────────────

def compute_delta_ddx(ddx_prev, ddx_curr, nodes_df, index, id_col, name_col, text2vec):
    """
    Compare two turns' DDx lists and return the delta (added + removed),
    matching by node ID (not string match).

    Returns:
      delta_ddx_texts: list of disease-name strings that changed (original text)
      or None if either input is empty.
    """
    if not ddx_prev or not ddx_curr:
        return None

    prev_map = {}
    for d in ddx_prev:
        nid = resolve_cached(d, nodes_df, index, id_col, name_col, text2vec)
        if nid is not None:
            prev_map[nid] = d

    curr_map = {}
    for d in ddx_curr:
        nid = resolve_cached(d, nodes_df, index, id_col, name_col, text2vec)
        if nid is not None:
            curr_map[nid] = d

    prev_ids = set(prev_map.keys())
    curr_ids = set(curr_map.keys())

    added_ids   = curr_ids - prev_ids
    removed_ids = prev_ids - curr_ids

    delta_texts = []
    for nid in added_ids:
        delta_texts.append(curr_map[nid])
    for nid in removed_ids:
        delta_texts.append(prev_map[nid])

    return delta_texts if delta_texts else None


def compute_rac_cached(delta_ddx_texts, actions_prev,
                       nodes_df, index, id_col, name_col, G, text2vec):
    """
    RAC: mean shortest-path distance from each disease in
    delta_ddx to the *nearest* action from the previous turn. Hops returned
    directly (no inverse); smaller is better.

    Returns:
      {"rac_hop": float or None, "n_delta": int, "n_valid": int}
    """
    if not delta_ddx_texts or not actions_prev:
        return {"rac_hop": None, "n_delta": 0, "n_valid": 0}

    action_ids = []
    for act in actions_prev:
        aid = resolve_cached(act, nodes_df, index, id_col, name_col, text2vec)
        if aid is not None:
            action_ids.append(aid)

    if not action_ids:
        return {"rac_hop": None, "n_delta": len(delta_ddx_texts), "n_valid": 0}

    hops = []
    for ddx_text in delta_ddx_texts:
        ddx_id = resolve_cached(ddx_text, nodes_df, index, id_col, name_col, text2vec)
        if ddx_id is None:
            continue

        min_hop = None
        for aid in action_ids:
            h = safe_hop(G, ddx_id, aid)
            if h is not None and (min_hop is None or h < min_hop):
                min_hop = h

        if min_hop is not None:
            hops.append(min_hop)

    return {
        "rac_hop":  float(np.mean(hops)) if hops else None,
        "n_delta":  len(delta_ddx_texts),
        "n_valid":  len(hops),
    }
