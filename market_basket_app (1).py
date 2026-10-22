import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations

st.set_page_config(page_title="Market Basket Analysis", page_icon="🛒", layout="wide")

st.title("🛒 Market Basket Analysis")
st.caption("Frequent itemsets, association rules, product cleanup, and a product relationship graph.")

st.markdown(
    """
Upload a CSV containing either **one row per purchased item** (with transaction/order ID and item/product name),
or **one row per transaction** with product columns. The app can also create a sample dataset for testing.
"""
)

SAMPLE_DATA = pd.DataFrame(
    [
        ["T001", "Milk"], ["T001", "Bread"], ["T001", "Eggs"],
        ["T002", "Milk"], ["T002", "Bread"],
        ["T003", "Bread"], ["T003", "Butter"],
        ["T004", "Milk"], ["T004", "Bread"], ["T004", "Butter"],
        ["T005", "Eggs"], ["T005", "Bread"],
        ["T006", "Milk"], ["T006", "Eggs"],
        ["T007", "Bread"], ["T007", "Butter"], ["T007", "Jam"],
        ["T008", "Milk"], ["T008", "Bread"], ["T008", "Butter"],
        ["T009", "Eggs"], ["T009", "Bread"], ["T009", "Jam"],
        ["T010", "Milk"], ["T010", "Bread"], ["T010", "Eggs"],
        ["T011", "Coffee"], ["T011", "Sugar"], ["T011", "Milk"],
        ["T012", "Coffee"], ["T012", "Sugar"], ["T012", "Bread"],
        ["T013", "Tea"], ["T013", "Sugar"], ["T013", "Biscuits"],
        ["T014", "Tea"], ["T014", "Biscuits"], ["T014", "Milk"],
        ["T015", "Coffee"], ["T015", "Milk"], ["T015", "Bread"],
    ],
    columns=["TransactionID", "Item"]
)

uploaded = st.file_uploader("Upload your transaction CSV", type=["csv"])
use_sample = st.checkbox("Use built-in sample data (good for testing the graph)", value=uploaded is None)

@st.cache_data
def load_csv(file_bytes):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))

if uploaded is not None and not use_sample:
    try:
        raw_df = load_csv(uploaded.getvalue())
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        st.stop()
elif use_sample:
    raw_df = SAMPLE_DATA.copy()
else:
    st.info("Upload a CSV or select the sample data checkbox to begin.")
    st.stop()

if raw_df.empty:
    st.error("The uploaded CSV is empty.")
    st.stop()

with st.expander("Preview uploaded data", expanded=True):
    st.dataframe(raw_df.head(20), use_container_width=True)
    st.write(f"Rows: **{len(raw_df):,}** · Columns: **{len(raw_df.columns)}**")

st.sidebar.header("Analysis settings")
data_mode = st.sidebar.radio(
    "CSV format",
    ["Transaction ID + item column", "One row per transaction (wide format)"],
    help="Choose the first option when each row represents one purchased product."
)

def clean_item(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())

def make_basket_long(df, transaction_col, item_col):
    tmp = df[[transaction_col, item_col]].copy()
    tmp = tmp.dropna(subset=[transaction_col, item_col])
    tmp[transaction_col] = tmp[transaction_col].astype(str).str.strip()
    tmp[item_col] = tmp[item_col].map(clean_item)
    tmp = tmp[(tmp[transaction_col] != "") & (tmp[item_col] != "")]
    tmp = tmp.drop_duplicates([transaction_col, item_col])
    if tmp.empty:
        return pd.DataFrame()
    basket = pd.crosstab(tmp[transaction_col], tmp[item_col]).gt(0)
    return basket.astype(bool)

if data_mode == "Transaction ID + item column":
    cols = list(raw_df.columns)
    guessed_tx = next((c for c in cols if any(k in str(c).lower() for k in ["transaction", "invoice", "order", "basket", "receipt", "bill"])), cols[0])
    guessed_item = next((c for c in cols if any(k in str(c).lower() for k in ["item", "product", "description", "goods", "name", "sku"])), cols[1] if len(cols) > 1 else cols[0])
    c1, c2 = st.columns(2)
    with c1:
        transaction_col = st.selectbox("Transaction / Order ID column", cols, index=cols.index(guessed_tx))
    with c2:
        item_col = st.selectbox("Product / Item column", cols, index=cols.index(guessed_item))
    if transaction_col == item_col:
        st.error("Choose two different columns for transaction ID and item.")
        st.stop()
    basket = make_basket_long(raw_df, transaction_col, item_col)
else:
    wide = raw_df.copy()
    transaction_col = st.selectbox("Optional transaction ID column (not treated as a product)", ["(No ID column)"] + list(wide.columns))
    if transaction_col != "(No ID column)":
        wide = wide.drop(columns=[transaction_col])
    wide.columns = [str(c).strip() for c in wide.columns]
    for col in wide.columns:
        # Accept 1/0, true/false, yes/no, or positive quantities.
        vals = wide[col]
        if pd.api.types.is_numeric_dtype(vals):
            wide[col] = vals.fillna(0).gt(0)
        else:
            normalized = vals.astype(str).str.strip().str.lower()
            wide[col] = normalized.isin(["1", "true", "yes", "y", "present"])
    basket = wide.fillna(False).astype(bool)
    basket.columns = [clean_item(c) for c in basket.columns]
    basket = basket.loc[:, [c != "" for c in basket.columns]]
    basket = basket.loc[:, ~pd.Index(basket.columns).duplicated()]

if basket.empty or basket.shape[1] == 0:
    st.error("No transaction matrix could be created. Check the selected columns and CSV format.")
    st.stop()

# Remove products that never occur.
basket = basket.loc[:, basket.any(axis=0)]
if basket.empty or basket.shape[1] == 0:
    st.error("No products with positive quantities were found.")
    st.stop()

st.subheader("Transaction matrix")
st.write(f"**{basket.shape[0]:,} transactions** · **{basket.shape[1]:,} distinct products**")
st.dataframe(basket.astype(int).head(20), use_container_width=True)

# Optional near-duplicate product-name review using RapidFuzz if installed.
st.subheader("Product name cleanup")
products = sorted(map(str, basket.columns))
with st.expander("Find possible duplicate / similar product names"):
    threshold = st.slider("Similarity threshold", 60, 100, 88)
    pairs = []
    try:
        from rapidfuzz import fuzz
        for i, a in enumerate(products):
            for b in products[i + 1:]:
                score = fuzz.ratio(a, b)
                if score >= threshold:
                    pairs.append({"Product A": a, "Product B": b, "Similarity (%)": score})
    except Exception:
        st.info("Install rapidfuzz to enable fuzzy matching. Exact duplicate names are already combined.")
    if pairs:
        st.dataframe(pd.DataFrame(pairs).sort_values("Similarity (%)", ascending=False), use_container_width=True)
        st.caption("These are suggestions only. Review them before merging names.")
    elif products:
        st.write("No similar product names found at this threshold.")

st.sidebar.subheader("Frequent pattern settings")
min_support_pct = st.sidebar.slider("Minimum support (%)", 1, 80, 10) / 100
min_confidence = st.sidebar.slider("Minimum confidence (%)", 10, 100, 30) / 100
max_itemset_size = st.sidebar.slider("Maximum itemset size", 2, 5, 3)
max_products_graph = st.sidebar.slider("Maximum products shown in graph", 5, 40, 20)

# Calculate frequent itemsets without requiring mlxtend.
@st.cache_data
def calculate_itemsets(matrix_as_int, columns, min_support, max_size):
    matrix = pd.DataFrame(matrix_as_int, columns=columns).astype(bool)
    n = len(matrix)
    if n == 0:
        return pd.DataFrame(columns=["itemsets", "support", "count"])
    single_counts = matrix.sum(axis=0)
    frequent = {}
    for item, count in single_counts.items():
        support = float(count) / n
        if support >= min_support:
            frequent[frozenset([item])] = (support, int(count))
    all_found = dict(frequent)
    previous = [x for x in frequent.keys()]
    for size in range(2, max_size + 1):
        candidates = set()
        for a, b in combinations(previous, 2):
            union = a | b
            if len(union) == size:
                candidates.add(union)
        next_level = []
        for candidate in candidates:
            # Apriori property: all subsets must be frequent.
            if any(frozenset(subset) not in all_found for subset in combinations(candidate, size - 1)):
                continue
            mask = matrix[list(candidate)].all(axis=1)
            count = int(mask.sum())
            support = count / n
            if support >= min_support:
                all_found[candidate] = (support, count)
                next_level.append(candidate)
        previous = next_level
        if not previous:
            break
    rows = [{"itemsets": tuple(sorted(fs)), "support": s, "count": c} for fs, (s, c) in all_found.items()]
    return pd.DataFrame(rows, columns=["itemsets", "support", "count"]).sort_values(
        ["support", "count"], ascending=False
    ).reset_index(drop=True)

with st.spinner("Mining frequent itemsets..."):
    itemsets = calculate_itemsets(
        basket.astype(int).values.tolist(),
        list(basket.columns),
        min_support_pct,
        max_itemset_size
    )

if itemsets.empty:
    st.warning("No frequent itemsets matched this minimum support. Lower the support percentage and try again.")
    st.stop()

def mine_rules(itemsets_df, min_conf):
    support_map = {frozenset(row.itemsets): float(row.support) for row in itemsets_df.itertuples()}
    rows = []
    for itemset, support in support_map.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for r in range(1, len(items)):
            for antecedent_tuple in combinations(items, r):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent
                antecedent_support = support_map.get(antecedent)
                consequent_support = support_map.get(consequent)
                if not antecedent_support or not consequent_support:
                    continue
                confidence = support / antecedent_support
                lift = confidence / consequent_support if consequent_support else 0
                if confidence >= min_conf:
                    rows.append({
                        "Antecedents": ", ".join(sorted(antecedent)),
                        "Consequents": ", ".join(sorted(consequent)),
                        "Support": support,
                        "Confidence": confidence,
                        "Lift": lift,
                        "itemset": itemset
                    })
    if not rows:
        return pd.DataFrame(columns=["Antecedents", "Consequents", "Support", "Confidence", "Lift", "itemset"])
    return pd.DataFrame(rows).sort_values(["Lift", "Confidence"], ascending=False).reset_index(drop=True)

rules = mine_rules(itemsets, min_confidence)

m1, m2, m3 = st.columns(3)
m1.metric("Transactions", f"{len(basket):,}")
m2.metric("Frequent itemsets", f"{len(itemsets):,}")
m3.metric("Association rules", f"{len(rules):,}")

tab1, tab2, tab3 = st.tabs(["Frequent itemsets", "Association rules", "Product graph"])

with tab1:
    show_items = itemsets[["itemsets", "support", "count"]].copy()
    show_items["itemsets"] = show_items["itemsets"].apply(lambda x: " + ".join(x))
    show_items["support"] = (show_items["support"] * 100).round(2).astype(str) + "%"
    show_items.columns = ["Frequent itemset", "Support", "Transaction count"]
    st.dataframe(show_items, use_container_width=True)
    st.download_button(
        "Download frequent itemsets CSV",
        itemsets.drop(columns=[], errors="ignore").assign(itemsets=itemsets["itemsets"].apply(lambda x: " + ".join(x))).to_csv(index=False).encode("utf-8"),
        file_name="frequent_itemsets.csv",
        mime="text/csv"
    )

with tab2:
    if rules.empty:
        st.info("No rules meet the current confidence setting. Try lowering minimum confidence or minimum support.")
    else:
        display_rules = rules[["Antecedents", "Consequents", "Support", "Confidence", "Lift"]].copy()
        for col in ["Support", "Confidence"]:
            display_rules[col] = (display_rules[col] * 100).round(2).astype(str) + "%"
        display_rules["Lift"] = display_rules["Lift"].round(3)
        st.dataframe(display_rules, use_container_width=True)
        st.download_button(
            "Download association rules CSV",
            rules[["Antecedents", "Consequents", "Support", "Confidence", "Lift"]].to_csv(index=False).encode("utf-8"),
            file_name="association_rules.csv",
            mime="text/csv"
        )
        st.caption("Support = fraction of all transactions containing the itemset. Confidence = chance of the consequent when the antecedent appears. Lift > 1 indicates a positive association.")

with tab3:
    st.write("Each node is a product. Edges connect products that appear together in frequent pairs; thicker edges mean stronger co-occurrence.")
    pair_counts = {}
    # Pair co-occurrence based on transaction matrix.
    cols = list(basket.columns)
    for a, b in combinations(cols, 2):
        count = int((basket[a] & basket[b]).sum())
        support = count / len(basket)
        if count > 0 and support >= min_support_pct:
            pair_counts[(a, b)] = {"count": count, "support": support}
    if not pair_counts:
        st.info("No product pairs meet the support threshold. Lower minimum support to show graph connections.")
    else:
        try:
            import networkx as nx
            import plotly.graph_objects as go
            graph = nx.Graph()
            # Prioritize strongest pairs, then limit graph size for readability.
            selected_pairs = sorted(pair_counts.items(), key=lambda x: (x[1]["count"], x[1]["support"]), reverse=True)
            graph_products = set()
            chosen = []
            for pair, details in selected_pairs:
                if len(graph_products) < max_products_graph or (pair[0] in graph_products and pair[1] in graph_products):
                    chosen.append((pair, details))
                    graph_products.update(pair)
                if len(graph_products) >= max_products_graph and len(chosen) >= max_products_graph * 2:
                    break
            for (a, b), details in chosen:
                graph.add_edge(a, b, weight=details["count"], support=details["support"])
            if graph.number_of_nodes() == 0:
                st.info("No graph nodes to display.")
            else:
                pos = nx.spring_layout(graph, seed=42, k=0.9)
                edge_x, edge_y = [], []
                for a, b in graph.edges():
                    x0, y0 = pos[a]
                    x1, y1 = pos[b]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y, mode="lines",
                    line=dict(width=1.5), hoverinfo="none"
                )
                node_x, node_y, node_text, node_size = [], [], [], []
                for node in graph.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    degree = graph.degree(node)
                    node_size.append(16 + degree * 4)
                    node_text.append(f"{node}<br>Connections: {degree}<br>Transactions: {int(basket[node].sum())}")
                node_trace = go.Scatter(
                    x=node_x, y=node_y, mode="markers+text",
                    text=list(graph.nodes()), textposition="top center",
                    hovertext=node_text, hoverinfo="text",
                    marker=dict(size=node_size, line=dict(width=1))
                )
                fig = go.Figure(data=[edge_trace, node_trace])
                fig.update_layout(
                    showlegend=False, margin=dict(l=10, r=10, t=25, b=10),
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    height=600, title="Product Co-occurrence Network"
                )
                st.plotly_chart(fig, use_container_width=True)
                edge_table = pd.DataFrame([
                    {"Product A": a, "Product B": b, "Together count": d["count"], "Support (%)": round(d["support"] * 100, 2)}
                    for (a, b), d in chosen
                ]).sort_values("Together count", ascending=False)
                st.dataframe(edge_table, use_container_width=True)
        except ImportError:
            st.error("Graph libraries are missing. Add networkx and plotly to requirements.txt and redeploy.")

st.markdown("---")
st.caption("Tip: For a meaningful analysis, upload real transaction-level data. Product-name similarity matches are suggestions and should be reviewed before merging.")
