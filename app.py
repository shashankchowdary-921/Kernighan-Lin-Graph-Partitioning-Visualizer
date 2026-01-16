import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
from io import StringIO
from datetime import datetime

# ============================================================
# Page Config + Minimal CSS for clean UI
# ============================================================
st.set_page_config(page_title="KL Graph Partitioning Visualizer", layout="wide")

st.markdown(
    """
<style>
/* Reduce big default paddings a bit */
.block-container { padding-top: 1.0rem; padding-bottom: 1.5rem; }

/* Make sidebar sections nicer */
section[data-testid="stSidebar"] { padding-top: 0.5rem; }

/* Make metric cards a bit smaller */
div[data-testid="stMetricValue"] { font-size: 1.6rem; }

/* Reduce header spacing */
h1 { margin-bottom: 0.2rem; }
</style>
""",
    unsafe_allow_html=True
)

st.title("🔀 Kernighan–Lin Graph Partitioning Visualizer")
st.caption("Manual / Random / Predefined • Auto-run • Cut analysis • Step-by-step KL swaps • Logs")


# ============================================================
# Session State
# ============================================================
def init_state():
    if "edges_text" not in st.session_state:
        st.session_state.edges_text = ""

    if "log" not in st.session_state:
        st.session_state.log = []

    if "best_cut" not in st.session_state:
        st.session_state.best_cut = None

    if "prev_cut" not in st.session_state:
        st.session_state.prev_cut = None

    if "prev_edges_hash" not in st.session_state:
        st.session_state.prev_edges_hash = ""

    if "predefined_choice" not in st.session_state:
        st.session_state.predefined_choice = "Example 1 (Classic small graph)"

init_state()


# ============================================================
# Logging
# ============================================================
def add_log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.log.append(f"[{timestamp}] {msg}")

def clear_log():
    st.session_state.log = []


# ============================================================
# Graph Helpers
# ============================================================
def parse_edges(text: str):
    edges = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) == 2:
            u, v = parts
            w = 1.0
        elif len(parts) == 3:
            u, v, w = parts
            w = float(w)
        else:
            raise ValueError(f"Line {i} invalid: '{line}'. Use: u v [weight]")
        edges.append((str(u), str(v), float(w)))
    return edges

def build_graph(edges):
    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=float(w))
    return G

def w(G, u, v):
    if G.has_edge(u, v):
        return float(G[u][v].get("weight", 1.0))
    return 0.0

def cut_size(G, A, B):
    total = 0.0
    for u, v, data in G.edges(data=True):
        if (u in A and v in B) or (u in B and v in A):
            total += float(data.get("weight", 1.0))
    return total

def get_cut_edges(G, A, B):
    cut_edges = []
    for u, v, data in G.edges(data=True):
        if (u in A and v in B) or (u in B and v in A):
            cut_edges.append((u, v, float(data.get("weight", 1.0))))
    return cut_edges

def compute_D_values(G, A, B):
    D = {}
    for v in G.nodes():
        if v in A:
            ext = sum(w(G, v, x) for x in B)
            inte = sum(w(G, v, x) for x in A)
        else:
            ext = sum(w(G, v, x) for x in A)
            inte = sum(w(G, v, x) for x in B)
        D[v] = ext - inte
    return D

def initial_partition(nodes, mode="Automatic", seed=42):
    nodes = list(nodes)
    if mode == "Random":
        random.seed(int(seed))
        random.shuffle(nodes)
    else:
        nodes = sorted(nodes)
    mid = len(nodes) // 2
    return set(nodes[:mid]), set(nodes[mid:])

def kernighan_lin_with_steps(G, init_A, init_B, max_passes=10):
    """
    KL with step-by-step table.
    """
    A = set(init_A)
    B = set(init_B)

    all_steps = []
    pass_num = 0

    while pass_num < max_passes:
        pass_num += 1

        locked_A = set()
        locked_B = set()

        chosen_pairs = []
        gains = []

        D = compute_D_values(G, A, B)
        m = min(len(A), len(B))
        if m == 0:
            break

        A_work = set(A)
        B_work = set(B)

        for _ in range(m):
            best_pair = None
            best_gain = -1e18

            for a in A_work - locked_A:
                for b in B_work - locked_B:
                    g = D[a] + D[b] - 2 * w(G, a, b)
                    if g > best_gain:
                        best_gain = g
                        best_pair = (a, b)

            if best_pair is None:
                break

            a, b = best_pair
            locked_A.add(a)
            locked_B.add(b)
            chosen_pairs.append((a, b))
            gains.append(best_gain)

            # Update D values after virtual swap
            for v in (A_work - locked_A):
                D[v] = D[v] + 2 * w(G, v, a) - 2 * w(G, v, b)
            for v in (B_work - locked_B):
                D[v] = D[v] + 2 * w(G, v, b) - 2 * w(G, v, a)

            # virtual swap
            A_work.remove(a)
            B_work.remove(b)
            A_work.add(b)
            B_work.add(a)

        if not gains:
            break

        # cumulative gain
        running = 0.0
        best_cum = -1e18
        k_star = -1
        cum = []

        for i, g in enumerate(gains):
            running += g
            cum.append(running)
            if running > best_cum:
                best_cum = running
                k_star = i

        for i, ((a, b), g, c) in enumerate(zip(chosen_pairs, gains, cum), start=1):
            all_steps.append({
                "Pass": pass_num,
                "Swap #": i,
                "Swap": f"{a} ↔ {b}",
                "Gain": round(float(g), 4),
                "Cumulative Gain": round(float(c), 4),
                "Applied?": "✅" if i <= (k_star + 1) and best_cum > 0 else "❌"
            })

        # commit swaps only if improvement
        if best_cum > 0 and k_star >= 0:
            for i in range(k_star + 1):
                a, b = chosen_pairs[i]
                A.remove(a)
                B.remove(b)
                A.add(b)
                B.add(a)
        else:
            break

    return A, B, pd.DataFrame(all_steps)


def draw_partition_graph(G, A=None, B=None, layout="Spring", show_weights=True):
    fig, ax = plt.subplots(figsize=(8.8, 5.6))

    if layout == "Circular":
        pos = nx.circular_layout(G)
    elif layout == "Shell":
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    node_colors = []
    for n in G.nodes():
        if A is not None and n in A:
            node_colors.append("tab:blue")
        elif B is not None and n in B:
            node_colors.append("tab:orange")
        else:
            node_colors.append("tab:gray")

    cut_edges = []
    normal_edges = list(G.edges())

    if A is not None and B is not None:
        cut_edges = [(u, v) for (u, v, _) in get_cut_edges(G, A, B)]
        normal_edges = [e for e in G.edges() if e not in cut_edges and (e[1], e[0]) not in cut_edges]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=820, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, width=2, alpha=0.55, ax=ax)

    if cut_edges:
        nx.draw_networkx_edges(G, pos, edgelist=cut_edges, width=4, alpha=0.95, edge_color="red", ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", ax=ax)

    if show_weights:
        edge_labels = {(u, v): f"{d.get('weight', 1):.1f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax)

    ax.set_axis_off()
    return fig


def generate_random_graph(n_nodes, edge_prob, max_weight, seed=42):
    random.seed(seed)
    nodes = [chr(ord("A") + i) for i in range(n_nodes)]
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if random.random() <= edge_prob:
                edges.append((nodes[i], nodes[j], random.randint(1, max_weight)))
    return edges

def edges_to_text(edges):
    return "\n".join([f"{u} {v} {w}" for u, v, w in edges])

def make_summary_txt(A, B, cur_cut, best_cut, delta_abs, delta_pct, cut_edges):
    buf = StringIO()
    buf.write("Kernighan–Lin Graph Partitioning Visualizer\n")
    buf.write("=========================================\n\n")
    buf.write(f"Current Cut: {cur_cut:.2f}\n")
    buf.write(f"Best Cut:    {best_cut:.2f}\n")
    buf.write(f"Delta abs:   {delta_abs:.2f}\n")
    buf.write(f"Delta %:     {delta_pct:.2f}%\n\n")

    buf.write(f"Partition A (Blue): {', '.join(sorted(A))}\n")
    buf.write(f"Partition B (Orange): {', '.join(sorted(B))}\n\n")

    buf.write("Cut Edges (u - v : w)\n")
    buf.write("---------------------\n")
    if not cut_edges:
        buf.write("None\n")
    else:
        for u, v, wgt in cut_edges:
            buf.write(f"{u} - {v} : {wgt}\n")

    return buf.getvalue()


# ============================================================
# Predefined examples
# ============================================================
PREDEFINED = {
    "Example 1 (Classic small graph)": """A B 1
A C 2
B C 3
B D 4
C D 2
C E 1
D E 3
""",
    "Example 2 (Balanced 6 nodes)": """A B 2
A C 1
B D 3
C D 2
C E 3
D F 2
E F 4
B E 1
""",
    "Example 3 (8 nodes)": """A B 3
A C 2
B D 2
C D 4
C E 2
D F 3
E F 1
E G 2
F H 2
G H 3
B E 2
"""
}


# ============================================================
# Sidebar (Cleaner)
# ============================================================
st.sidebar.header("⚙️ Controls")

with st.sidebar.expander("🧩 Graph Input Mode", expanded=True):
    mode = st.radio("Mode", ["Manual", "Random", "Predefined"], horizontal=True)

with st.sidebar.expander("🎛️ Visualization Settings", expanded=True):
    layout = st.selectbox("Graph Layout", ["Spring", "Circular", "Shell"])
    show_weights = st.checkbox("Show edge weights", value=True)

with st.sidebar.expander("🧠 KL Settings", expanded=True):
    init_mode = st.selectbox("Initial Partition Mode", ["Automatic", "Random"])
    seed = st.number_input("Partition seed", 0, 999999, 42)
    max_passes = st.slider("Max KL passes", 1, 25, 10)
    single_pass = st.checkbox("Run Single Pass Only (Pass 1)", value=False)

with st.sidebar.expander("🛠️ Actions", expanded=True):
    colA, colB = st.columns(2)
    with colA:
        if st.button("🧾 Clear Log", use_container_width=True):
            clear_log()
    with colB:
        if st.button("🧹 Clear Edges", use_container_width=True):
            st.session_state.edges_text = ""
            st.session_state.best_cut = None
            st.session_state.prev_cut = None
            add_log("Edges cleared.")

# ---- Mode content
if mode == "Manual":
    with st.sidebar.expander("➕ Click-to-Add Edge", expanded=True):
        total_nodes = st.number_input("Total Nodes (even recommended)", 2, 200, 6, step=1)
        if total_nodes % 2 != 0:
            st.warning("⚠️ For KL, even number of nodes is recommended (balanced bipartition).")

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            u = st.text_input("Node 1", value="A")
        with c2:
            v = st.text_input("Node 2", value="B")
        with c3:
            wt = st.number_input("Weight", min_value=1.0, value=1.0, step=1.0)

        if st.button("➕ Add Edge", use_container_width=True):
            line = f"{u.strip()} {v.strip()} {float(wt)}"
            st.session_state.edges_text = (st.session_state.edges_text.strip() + "\n" + line).strip()
            add_log(f"Edge added: {line}")

elif mode == "Random":
    with st.sidebar.expander("🎲 Random Graph Generator", expanded=True):
        rn = st.slider("Number of nodes", 4, 30, 8)
        rp = st.slider("Edge probability", 0.05, 1.0, 0.35, 0.05)
        rw = st.slider("Max weight", 1, 30, 5)
        rseed = st.number_input("Generator seed", min_value=0, max_value=999999, value=42, step=1)

        if st.button("✨ Generate Random Graph", use_container_width=True):
            rand_edges = generate_random_graph(rn, rp, rw, seed=int(rseed))
            st.session_state.edges_text = edges_to_text(rand_edges)
            st.session_state.best_cut = None
            st.session_state.prev_cut = None
            add_log(f"Random graph generated: nodes={rn}, p={rp}, maxW={rw}, seed={rseed}")

elif mode == "Predefined":
    with st.sidebar.expander("📦 Predefined Graph", expanded=True):
        choice = st.selectbox("Select example", list(PREDEFINED.keys()), key="predefined_choice")

        # ✅ PATCH: Auto-load predefined graph on dropdown change
        st.session_state.edges_text = PREDEFINED[choice]
        st.session_state.best_cut = None
        st.session_state.prev_cut = None
        add_log(f"Loaded predefined graph: {choice} (auto)")

# Edge editor always shown (but collapsed)
with st.sidebar.expander("✏️ Edge List Editor", expanded=False):
    edges_text = st.text_area(
        "Edges (node1 node2 weight) per line",
        value=st.session_state.edges_text,
        height=220
    )
    st.session_state.edges_text = edges_text

st.sidebar.caption("✅ Auto-run enabled: changes immediately recompute results.")


# ============================================================
# Main: Clean Layout using Tabs
# ============================================================
tab_overview, tab_analysis, tab_steps, tab_log = st.tabs(["📌 Overview", "✂️ Cut Analysis", "🧾 Steps", "🪵 Log"])

# Auto-run computation
edges_text = st.session_state.edges_text

try:
    edges = parse_edges(edges_text) if edges_text.strip() else []
    G = build_graph(edges)

    if G.number_of_nodes() < 2:
        with tab_overview:
            st.info("Enter edges to start (at least 2 nodes required).")
        st.stop()

    init_A, init_B = initial_partition(G.nodes(), mode=init_mode, seed=int(seed))
    passes_to_use = 1 if single_pass else max_passes

    A, B, steps_df = kernighan_lin_with_steps(G, init_A, init_B, max_passes=passes_to_use)

    cur_cut = cut_size(G, A, B)
    cut_edges = get_cut_edges(G, A, B)

    # best cut update
    if st.session_state.best_cut is None or cur_cut < st.session_state.best_cut:
        st.session_state.best_cut = cur_cut
    best_cut = st.session_state.best_cut

    delta_abs = cur_cut - best_cut
    delta_pct = (delta_abs / best_cut * 100.0) if best_cut and best_cut != 0 else 0.0

    # log on any change
    edges_hash = str(hash(edges_text + init_mode + str(seed) + str(max_passes) + str(single_pass) + layout + str(show_weights)))
    if edges_hash != st.session_state.prev_edges_hash:
        st.session_state.prev_edges_hash = edges_hash
        add_log(f"Recomputed KL (passes={passes_to_use}). Current cut={cur_cut:.2f}, best={best_cut:.2f}")

    # ----------------- Overview TAB -----------------
    with tab_overview:
        top1, top2, top3 = st.columns([1, 1, 1])
        with top1:
            st.metric("Current Cut", f"{cur_cut:.2f}")
        with top2:
            st.metric("Best Cut", f"{best_cut:.2f}")
        with top3:
            st.metric("Δ (abs)", f"{delta_abs:.2f}")

        a1, a2 = st.columns([1.2, 1])
        with a1:
            st.subheader("📌 Graph Partition State")
            st.success(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
            st.pyplot(draw_partition_graph(G, A=A, B=B, layout=layout, show_weights=show_weights))
            st.info("🔴 Red edges = cut edges (crossing partitions).")

        with a2:
            st.subheader("🧠 Partitions")
            st.markdown("**Partition A (Blue)**")
            st.code(", ".join(sorted(A)), language="text")
            st.markdown("**Partition B (Orange)**")
            st.code(", ".join(sorted(B)), language="text")

            st.markdown("**Cut Edges (u - v : w)**")
            if not cut_edges:
                st.write("No cut edges.")
            else:
                st.code("\n".join([f"{u} - {v} : {wgt}" for u, v, wgt in cut_edges]), language="text")

            # ✅ PATCH: Download summary
            summary_txt = make_summary_txt(A, B, cur_cut, best_cut, delta_abs, delta_pct, cut_edges)
            st.download_button(
                "⬇️ Download Summary (TXT)",
                data=summary_txt,
                file_name="kl_summary.txt",
                mime="text/plain",
                use_container_width=True
            )

    # ----------------- Cut Analysis TAB -----------------
    with tab_analysis:
        st.subheader("✂️ Cut Size Analysis")

        st.write(f"**Current Cut Size:** `{cur_cut:.2f}`")
        st.write(f"**Δ (%):** `{delta_pct:.2f}%`")

        if cut_edges:
            df_cut = pd.DataFrame(cut_edges, columns=["u", "v", "w"])
            df_cut["edge"] = df_cut["u"] + "-" + df_cut["v"]

            st.markdown("### 📈 Cut Edge Weights Distribution")
            fig2, ax2 = plt.subplots(figsize=(9, 4))
            ax2.bar(df_cut["edge"], df_cut["w"])
            ax2.set_ylabel("Weight")
            ax2.set_xlabel("Cut Edges")
            ax2.set_title("Cut Edges Weight Distribution")
            plt.xticks(rotation=45)
            st.pyplot(fig2)

            st.markdown("### Cut edges table")
            st.dataframe(df_cut[["edge", "w"]], use_container_width=True)
        else:
            st.info("No cut edges found for this partition.")

    # ----------------- Steps TAB -----------------
    with tab_steps:
        st.subheader("🧾 Step-by-step Kernighan–Lin Swaps (per pass)")
        st.caption("Each pass selects swap pairs with gains; best prefix with positive cumulative gain is applied.")

        if steps_df.empty:
            st.warning("No improving swaps found (or graph too small).")
        else:
            st.dataframe(steps_df, use_container_width=True)

            # ✅ PATCH: Download swaps as CSV
            csv = steps_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Steps (CSV)",
                data=csv,
                file_name="kl_steps.csv",
                mime="text/csv",
                use_container_width=True
            )

            # ✅ PATCH: Download swaps as TXT
            txt_buf = StringIO()
            txt_buf.write("KL Step-by-step Swaps\n=====================\n\n")
            txt_buf.write(steps_df.to_string(index=False))
            st.download_button(
                "⬇️ Download Steps (TXT)",
                data=txt_buf.getvalue(),
                file_name="kl_steps.txt",
                mime="text/plain",
                use_container_width=True
            )

    # ----------------- Log TAB -----------------
    with tab_log:
        st.subheader("🪵 Log")
        log_text = "\n".join(st.session_state.log[-400:]) if st.session_state.log else "(empty)"
        st.text_area("Log output", value=log_text, height=420)

except Exception as e:
    st.error(f"Error: {e}")
