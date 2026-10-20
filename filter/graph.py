"""
graph.py — Load ClinGraph / PrimeKG knowledge graphs into NetworkX.
"""
import pandas as pd
import networkx as nx


def robust_load_graph(csv_path):
    """Auto-detect CSV format and load the edge list into a NetworkX Graph."""
    print(f"Loading: {csv_path.split('/')[-1]} ...")

    df = None
    for encoding in ['utf-8-sig', 'utf-8', 'latin1', 'cp950']:
        for sep in [',', '\t']:
            try:
                temp_df = pd.read_csv(csv_path, sep=sep, encoding=encoding,
                                      nrows=10, on_bad_lines='skip')
                if temp_df.shape[1] > 1:
                    df = pd.read_csv(csv_path, sep=sep, encoding=encoding,
                                     dtype=str, on_bad_lines='skip')
                    break
            except Exception:
                continue
        if df is not None:
            break

    if df is None:
        raise ValueError(f"Failed to parse file: {csv_path}")

    df.columns = [str(c).strip().lower() for c in df.columns]

    src_candidates = ['src', 'x_id', 'source', 'node_id_x', 'id_1', 'node_1']
    dst_candidates = ['dst', 'y_id', 'target', 'node_id_y', 'id_2', 'node_2']

    src_col = next((c for c in df.columns if any(k in c for k in src_candidates)), df.columns[0])
    dst_col = next((c for c in df.columns if any(k in c for k in dst_candidates) and c != src_col), df.columns[1])

    print(f"   detected columns: {list(df.columns)}")
    print(f"   edge mapping: [{src_col}] --(edge)--> [{dst_col}]")

    G = nx.from_pandas_edgelist(df, src_col, dst_col)
    print(f"   nodes: {G.number_of_nodes():,}  edges: {G.number_of_edges():,}\n")
    return G
