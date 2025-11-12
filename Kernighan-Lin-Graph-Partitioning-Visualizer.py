
#
# Kernighan-Lin Visualizer (v6.4 - kl35.py)
# Fixes:
# - Removed stray "{e}") fragments causing SyntaxError
# - Deduped and corrected show_metrics_popup; now a KL_GUI_App method
# - Ensured button color keys exist
# - Kept metrics popup + auto-open CSV behavior
#
import networkx as nx
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

# --- GUI Libraries ---
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font
import sys
import io
import os
import webbrowser

# ### GRAPH COLORS ###
COLOR_A = '#a9d0f5' # Light Blue
COLOR_B = '#fde1b0' # Light Orange

#
# ======================================================================
# 1. KERNIGHAN-LIN VISUALIZATION CLASS
# ======================================================================
#
class KernighanLinVisualizer:
    def __init__(self, graph, fig, ax_graph, ax_plot, colors):
        if len(graph.nodes) % 2 != 0:
            # Hard guard — algorithm assumes equal partition sizes
            raise ValueError("The Kernighan-Lin algorithm requires an even number of vertices.")
        
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.num_nodes = len(self.nodes)
        self.colors = colors
        
        self.fig = fig
        self.ax_graph = ax_graph
        self.ax_plot = ax_plot
        
        # Style the figure and axes for light mode
        self.fig.patch.set_facecolor(self.colors['bg'])
        
        for ax in [self.ax_graph, self.ax_plot]:
            ax.set_facecolor(self.colors['plot_bg'])
            for spine in ax.spines.values():
                spine.set_edgecolor(self.colors['border'])
            
        try:
            self.pos = nx.kamada_kawai_layout(self.graph) 
        except ImportError:
            print("Scipy not found. 'pip install scipy' for a cleaner layout.")
            print("Falling back to spring_layout.")
            self.pos = nx.spring_layout(self.graph, seed=42)
            
        self.partition_a = set()
        self.partition_b = set()
        self.cut_history = []
        self.metrics = []  # per-pass metrics dicts

    def _volume(self, part):
        """Weighted degree sum (volume) of a partition."""
        vol = 0.0
        for u in part:
            for v, data in self.graph[u].items():
                vol += data.get('weight', 1.0)
        return vol

    def conductance(self, part_a, part_b):
        """Phi(S) = cut(S,~S)/min(vol(S), vol(~S)) (weighted)."""
        cut_w = self.calculate_cut_size(part_a, part_b)
        va = self._volume(part_a)
        vb = self._volume(part_b)
        denom = min(va, vb)
        return (cut_w / denom) if denom > 0 else 0.0

    def balance_ratio(self, part_a, part_b):
        """|A|-|B| balance ratio: closer to 1.0 is better (perfectly equal)."""
        a, b = len(part_a), len(part_b)
        if a == 0 or b == 0: return 0.0
        return min(a, b) / max(a, b)

    def calculate_cut_size(self, part_a, part_b):
        cut_size = 0.0
        for u, v, data in self.graph.edges(data=True):
            weight = data.get('weight', 1.0) 
            if (u in part_a and v in part_b) or \
               (u in part_b and v in part_a):
                cut_size += weight
        return cut_size

    def draw(self, title, swapped_pair=None, pass_step_gains=None, pass_start_cut=None, 
             current_iteration=None, show_partition_a=None, show_partition_b=None):
        self.ax_graph.clear()
        self.ax_plot.clear()

        # Re-apply light mode face colors
        self.ax_graph.set_facecolor(self.colors['plot_bg'])
        self.ax_plot.set_facecolor(self.colors['plot_bg'])

        current_part_a = show_partition_a if show_partition_a is not None else self.partition_a
        current_part_b = show_partition_b if show_partition_b is not None else self.partition_b

        # --- Draw Graph ---
        # --- Badge: current cut → best cut (Δ, %) ---
        try:
            current_cut_size = self.calculate_cut_size(current_part_a, current_part_b)
        except Exception:
            current_cut_size = None
        best_cut = None
        if self.cut_history:
            try:
                best_cut = min(self.cut_history)
            except Exception:
                best_cut = None
        if current_cut_size is not None and best_cut is not None:
            improvement = max(0.0, current_cut_size - best_cut)  # how much better best is vs current
            pct = (improvement / current_cut_size * 100.0) if abs(current_cut_size) > 1e-12 else 0.0
            badge = f"Cut: {current_cut_size:.2f}\\n→ {best_cut:.2f} (Δ -{improvement:.2f}, {pct:.1f}%)"
            self.ax_graph.text(0.98, 0.02, badge, transform=self.ax_graph.transAxes,
                               va='bottom', ha='right', fontsize=10, color=self.colors['text'],
                               bbox=dict(facecolor=self.colors['bg'], edgecolor=self.colors['border'], boxstyle='round,pad=0.3'))

        if hasattr(self, "on_live_metrics"):
            if current_cut_size is not None and best_cut is not None:
                delta_abs = current_cut_size - best_cut
                delta_pct = (delta_abs / current_cut_size * 100.0) if abs(current_cut_size) > 1e-12 else 0.0
                self.on_live_metrics(current_cut_size, best_cut, delta_abs, delta_pct)

        node_list_sorted = sorted(self.graph.nodes())
        color_map_sorted = [COLOR_A if node in current_part_a else COLOR_B for node in node_list_sorted]
        node_color_dict = {node: color for node, color in zip(node_list_sorted, color_map_sorted)}
        cut_edges = [(u,v) for u,v in self.graph.edges() if (u in current_part_a and v in current_part_b) or \
                     (u in current_part_b and v in current_part_a)]
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')

        nx.draw(self.graph, self.pos, nodelist=node_list_sorted, node_color=color_map_sorted, with_labels=True, 
                ax=self.ax_graph, node_size=500, font_weight='bold', font_color=self.colors['text'])
        nx.draw_networkx_edges(self.graph, self.pos, edgelist=cut_edges, edge_color=self.colors['accent_red'], 
                               style='dashed', width=1.5, ax=self.ax_graph)
        nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels=edge_labels, ax=self.ax_graph, 
                                     font_size=8, font_color=self.colors['text'])

        if swapped_pair:
            nx.draw_networkx_nodes(self.graph, self.pos, nodelist=[swapped_pair[0]], 
                                   node_color=node_color_dict[swapped_pair[0]], node_size=700, 
                                   ax=self.ax_graph, edgecolors=self.colors['accent_red'], linewidths=2.5)
            nx.draw_networkx_nodes(self.graph, self.pos, nodelist=[swapped_pair[1]], 
                                   node_color=node_color_dict[swapped_pair[1]], node_size=700, 
                                   ax=self.ax_graph, edgecolors=self.colors['accent_red'], linewidths=2.5)

        self.ax_graph.set_title(title, fontsize=12, color=self.colors['text_sec'])

        # --- Plot Logic ---
        tc = self.colors['text']
        
        if pass_step_gains is not None:
            # --- SHOWING DETAILS OF A SINGLE PASS ---
            self.ax_plot.set_title(f"Pass {current_iteration}: Step-by-Step", fontsize=12, color=self.colors['text_sec'])
            self.ax_plot.set_xlabel("Step Number (k)", color=tc)
            self.ax_plot.set_ylabel("Potential Cut Size", color=tc)
            
            cut_sizes_at_step = [pass_start_cut]
            cumulative_g = 0
            for g in pass_step_gains:
                cumulative_g += g
                cut_sizes_at_step.append(pass_start_cut - cumulative_g)
            
            steps = range(len(cut_sizes_at_step))
            current_k = len(pass_step_gains)
            current_cum_gain = sum(pass_step_gains) if pass_step_gains else 0.0
            base = pass_start_cut if (pass_start_cut is not None and abs(pass_start_cut) > 1e-12) else None
            current_impr_pct = (current_cum_gain / base * 100.0) if base else 0.0

            self.ax_plot.plot(steps, cut_sizes_at_step, '-', color=self.colors['text_sec'], linewidth=1.5, alpha=0.6)
            self.ax_plot.plot(0, pass_start_cut, 's', color=self.colors['accent_red'], markersize=9, label='Initial (k=0)', zorder=3)
            if len(steps) > 1:
                self.ax_plot.plot(steps[1:], cut_sizes_at_step[1:], 'o', color=self.colors['accent_blue'], markersize=8, label='Swaps (k>0)', zorder=3)
            best_step_cut = min(cut_sizes_at_step)
            best_step_k = cut_sizes_at_step.index(best_step_cut)
            best_impr_pct = ((pass_start_cut - best_step_cut) / base * 100.0) if base else 0.0
            self.ax_plot.plot(best_step_k, best_step_cut, '*', color=self.colors['accent_green'], 
                              markersize=18, label=f'Best: {best_step_cut:.2f} (k={best_step_k}, {best_impr_pct:.1f}%)', zorder=4)
            
            self.ax_plot.axhline(y=pass_start_cut, color=self.colors['accent_red'], linestyle=':', alpha=0.5)
            self.ax_plot.set_xticks(steps)

            box_txt = f"k={current_k} | Cum Gain={current_cum_gain:.2f} | Δ%={current_impr_pct:.1f}%"
            self.ax_plot.text(0.02, 0.98, box_txt, transform=self.ax_plot.transAxes,
                              va='top', ha='left', fontsize=10, color=self.colors['text'],
                              bbox=dict(facecolor=self.colors['bg'], edgecolor=self.colors['border'], boxstyle='round,pad=0.3'))
        
        else:
            # --- SHOWING HISTORY OF ALL PASSES ---
            self.ax_plot.set_title("History by Pass", fontsize=12, color=self.colors['text_sec'])
            self.ax_plot.set_xlabel("Pass Number (0 = Initial)", color=tc)
            self.ax_plot.set_ylabel("Cut Size (Sum of Weights)", color=tc)
        
            if self.cut_history:
                xs = list(range(len(self.cut_history)))
                ys = self.cut_history
        
                self.ax_plot.plot(xs, ys, '-', linewidth=1.2,
                                  color=self.colors['text_sec'], alpha=0.6, label='Cut after pass')
                self.ax_plot.scatter(xs, ys, s=60, zorder=3,
                                     label='Per-pass cut', color=self.colors['accent_blue'])
                for i, (x, y) in enumerate(zip(xs, ys)):
                    if i == 0:
                        lbl = f'{y:.2f}'
                    else:
                        dy = y - ys[i-1]
                        pct = (dy / ys[i-1] * 100.0) if abs(ys[i-1]) > 1e-12 else 0.0
                        lbl = f'{y:.2f}\\nΔ {dy:+.2f} ({pct:+.1f}%)'
                    self.ax_plot.annotate(lbl, (x, y),
                                          textcoords="offset points", xytext=(0, 8),
                                          ha='center', color=self.colors['text'])
                self.ax_plot.scatter([0], [ys[0]], marker='s', s=90,
                                     color=self.colors['accent_red'], zorder=4, label='Initial')
                best_idx = min(range(len(ys)), key=lambda i: ys[i])
                self.ax_plot.scatter([best_idx], [ys[best_idx]], marker='*', s=160,
                                     color=self.colors['accent_green'], zorder=5,
                                     label=f'Best: {ys[best_idx]:.2f}')
                for x in xs:
                    self.ax_plot.axvline(x=x, color=self.colors['accent_blue'], linestyle='--', alpha=0.3, linewidth=0.8)
                self.ax_plot.set_xticks(xs)
        
        self.ax_plot.grid(True, color=self.colors['border'], linestyle=':')
        self.ax_plot.tick_params(axis='x', colors=tc)
        self.ax_plot.tick_params(axis='y', colors=tc)
        legend = self.ax_plot.legend(loc='upper right', fontsize=9)
        legend.get_frame().set_facecolor(self.colors['bg'])
        legend.get_frame().set_edgecolor(self.colors['border'])
        for text in legend.get_texts():
            text.set_color(self.colors['text'])
        
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def draw_final_separated_view(self):
        print("\\nGenerating final separated graph view...")
        fig_final, ax_final = plt.subplots(figsize=(12, 8))
        fig_final.patch.set_facecolor(self.colors['bg'])
        ax_final.set_facecolor(self.colors['bg'])

        pos_final = {}
        nodes_a = list(self.partition_a)
        subgraph_a = self.graph.subgraph(nodes_a)
        pos_a = nx.spring_layout(subgraph_a, seed=42)
        for node, (x, y) in pos_a.items(): pos_final[node] = (-1 + 0.8 * x, y) 

        nodes_b = list(self.partition_b)
        subgraph_b = self.graph.subgraph(nodes_b)
        pos_b = nx.spring_layout(subgraph_b, seed=42)
        for node, (x, y) in pos_b.items(): pos_final[node] = (1 + 0.8 * x, y)
        
        final_cut_size = self.calculate_cut_size(self.partition_a, self.partition_b)
        ax_final.set_title(f"Final Partition (Separated View) | Cut Size: {final_cut_size:.2f}", 
                           fontsize=16, color=self.colors['text'])
        color_map = [COLOR_A if node in self.partition_a else COLOR_B for node in self.graph.nodes()]
        
        nx.draw_networkx_nodes(self.graph, pos_final, node_color=color_map, node_size=600, ax=ax_final)
        nx.draw_networkx_labels(self.graph, pos_final, ax=ax_final, font_weight='bold', font_color=self.colors['text'])
        nx.draw_networkx_edges(self.graph, pos_final, edge_color=self.colors['border'], width=1.0, alpha=0.7, ax=ax_final)
        cut_edges = [(u, v) for u, v in self.graph.edges() if (u in self.partition_a and v in self.partition_b) or \
                     (u in self.partition_b and v in self.partition_a)]
        nx.draw_networkx_edges(self.graph, pos_final, edgelist=cut_edges, edge_color=self.colors['accent_red'], width=1.5, style='dashed', ax=ax_final)
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos_final, edge_labels=edge_labels, ax=ax_final, 
                                     font_size=8, font_color=self.colors['text'])
        ax_final.axis('off')
        fig_final.tight_layout()
        plt.ioff()
        plt.show()

    def run_generator(self, single_pass_only=False):
        node_set = set(self.nodes)
        # FORCE PARTITION for 6-node example
        if self.num_nodes == 6 and node_set == {0, 1, 2, 3, 4, 5}:
            print("--- DETECTED 6-NODE EXAMPLE: Forcing A={0,2,4}, B={1,3,5} ---")
            self.partition_a = {0, 2, 4}
            self.partition_b = {1, 3, 5}
        else:
            # Deterministic initial partition for reproducibility
            self.partition_a = set(sorted(self.nodes)[:self.num_nodes // 2])
            self.partition_b = set(sorted(self.nodes)[self.num_nodes // 2:])

        initial_cut_size = self.calculate_cut_size(self.partition_a, self.partition_b)
        self.cut_history.append(initial_cut_size)
        self.draw(f"Initial Partition (k=0) | Cut Size: {initial_cut_size:.2f}", pass_start_cut=initial_cut_size, pass_step_gains=[], current_iteration=1)
        yield
        
        iteration = 0
        while True:
            iteration += 1
            _t0 = time.time()
            print(f"\\n--- Starting Pass {iteration} ---")
            current_pass_start_cut = self.calculate_cut_size(self.partition_a, self.partition_b)
            d_values = defaultdict(float)
            for node in self.nodes:
                for neighbor in self.graph.neighbors(node):
                    weight = self.graph[node][neighbor].get('weight', 1.0)
                    if (node in self.partition_a and neighbor in self.partition_b) or \
                       (node in self.partition_b and neighbor in self.partition_a):
                        d_values[node] += weight
                    else:
                        d_values[node] -= weight
            
            A_unlocked = list(self.partition_a)
            B_unlocked = list(self.partition_b)
            gains = []
            swapped_pairs = []
            final_sim_a, final_sim_b = None, None 
            
            for i in range(self.num_nodes // 2):
                max_gain = -float('inf')
                best_pair = (None, None)
                for a in A_unlocked:
                    for b in B_unlocked:
                        edge_data = self.graph.get_edge_data(a, b, default={'weight': 0})
                        c_ab_weight = edge_data.get('weight', 0.0)
                        gain = d_values[a] + d_values[b] - 2 * c_ab_weight
                        if gain > max_gain:
                            max_gain = gain
                            best_pair = (a, b)
                if best_pair == (None, None): break 
                a_swap, b_swap = best_pair
                cum_gain = sum(gains) + max_gain
                base = current_pass_start_cut if abs(current_pass_start_cut) > 1e-12 else None
                impr_pct_live = (cum_gain / base * 100.0) if base else 0.0
                print(f"  Step {i+1} (k={i+1}): Best pair ({a_swap}, {b_swap}) | Gain: {max_gain:.2f} | CumGain: {cum_gain:.2f} ({impr_pct_live:.1f}%)")
                gains.append(max_gain)
                swapped_pairs.append(best_pair)
                A_unlocked.remove(a_swap)
                B_unlocked.remove(b_swap)
                temp_part_a = self.partition_a.copy()
                temp_part_b = self.partition_b.copy()
                for j in range(i + 1):
                    a_s, b_s = swapped_pairs[j]
                    if a_s in temp_part_a and b_s in temp_part_b:
                        temp_part_a.remove(a_s); temp_part_a.add(b_s)
                        temp_part_b.remove(b_s); temp_part_b.add(a_s)
                    elif a_s in temp_part_b and b_s in temp_part_a:
                        temp_part_a.remove(b_s); temp_part_a.add(a_s)
                        temp_part_b.remove(a_s); temp_part_b.add(b_s)
                final_sim_a, final_sim_b = temp_part_a, temp_part_b
                title = f"Pass {iteration}, Step {i+1}: Swap {best_pair} | Gain: {max_gain:.2f} | Δ%={impr_pct_live:.1f}%"
                self.draw(title,
                          swapped_pair=best_pair, pass_step_gains=gains, 
                          pass_start_cut=current_pass_start_cut, current_iteration=iteration,
                          show_partition_a=temp_part_a, show_partition_b=temp_part_b)
                yield
                for neighbor in self.graph.neighbors(a_swap):
                    weight = self.graph[a_swap][neighbor].get('weight', 1.0)
                    if neighbor in A_unlocked: d_values[neighbor] += 2 * weight
                    if neighbor in B_unlocked: d_values[neighbor] -= 2 * weight
                for neighbor in self.graph.neighbors(b_swap):
                    weight = self.graph[b_swap][neighbor].get('weight', 1.0)
                    if neighbor in B_unlocked: d_values[neighbor] += 2 * weight
                    if neighbor in A_unlocked: d_values[neighbor] -= 2 * weight
            
            max_cumulative_gain = -float('inf')
            best_k = -1 
            cumulative_gain = 0
            print("\\nCumulative Gains for this pass:")
            for i, g in enumerate(gains):
                cumulative_gain += g
                print(f"  k={i+1}: Swap {swapped_pairs[i]}, Gain={g:.2f}, Cumulative={cumulative_gain:.2f}")
                if cumulative_gain > max_cumulative_gain:
                    max_cumulative_gain = cumulative_gain
                    best_k = i
            
            if max_cumulative_gain > 1e-5:
                new_cut_size = None
                print(f"-> Max cumulative gain is {max_cumulative_gain:.2f} at k={best_k+1}.")
                print(f"-> Swapping first {best_k+1} pairs.")
                for i in range(best_k + 1):
                    a_s, b_s = swapped_pairs[i]
                    self.partition_a.remove(a_s)
                    self.partition_b.add(a_s)
                    self.partition_b.remove(b_s)
                    self.partition_a.add(b_s)
                new_cut_size = self.calculate_cut_size(self.partition_a, self.partition_b)
                self.cut_history.append(new_cut_size)

                base = current_pass_start_cut if abs(current_pass_start_cut) > 1e-12 else None
                impr_pct = ((current_pass_start_cut - new_cut_size) / base * 100.0) if base else 0.0

                # --- METRICS: record improving pass ---
                cut_edges_count = sum(1 for u, v in self.graph.edges()
                                      if (u in self.partition_a and v in self.partition_b) or (u in self.partition_b and v in self.partition_a))
                bal = self.balance_ratio(self.partition_a, self.partition_b)
                phi = self.conductance(self.partition_a, self.partition_b)
                runtime_ms = (time.time() - _t0) * 1000.0
                delta_val = float(new_cut_size - current_pass_start_cut)
                delta_pct_val = float((delta_val / current_pass_start_cut * 100.0) if abs(current_pass_start_cut) > 1e-12 else 0.0)
                self.metrics.append({
                    "pass": iteration,
                    "start_cut": float(current_pass_start_cut),
                    "end_cut": float(new_cut_size),
                    "delta": delta_val,
                    "delta_pct": delta_pct_val,
                    "swaps_tried": int(len(gains)),
                    "swaps_applied": int(best_k + 1),
                    "cut_edges": int(cut_edges_count),
                    "balance_ratio": float(bal),
                    "conductance": float(phi),
                    "runtime_ms": float(runtime_ms)
                })
                if single_pass_only and iteration == 1:
                    print("\\n--- STOPPING AFTER PASS 1 (USER REQUESTED) ---")
                    self.draw(f"Pass 1 Finished | Final Cut: {new_cut_size:.2f} | Δ%={impr_pct:.1f}%",
                              pass_step_gains=gains, pass_start_cut=current_pass_start_cut,
                              current_iteration=iteration)
                    yield
                    break 
                else:
                    self.draw(f"End of Pass {iteration} | New Cut Size: {new_cut_size:.2f} | Δ%={impr_pct:.1f}%")
                    yield
            else:
                final_cut_size = self.calculate_cut_size(self.partition_a, self.partition_b)
                # Record this pass even with no improvement so history marks ALL iterations
                self.cut_history.append(final_cut_size)
                # METRICS for a non-improving pass
                cut_edges_count = sum(1 for u, v in self.graph.edges() if (u in self.partition_a and v in self.partition_b) or (u in self.partition_b and v in self.partition_a))
                bal = self.balance_ratio(self.partition_a, self.partition_b)
                phi = self.conductance(self.partition_a, self.partition_b)
                runtime_ms = (time.time() - _t0) * 1000.0
                self.metrics.append({
                    "pass": iteration,
                    "start_cut": float(current_pass_start_cut),
                    "end_cut": float(final_cut_size),
                    "delta": float(final_cut_size - current_pass_start_cut),
                    "delta_pct": float(0.0),
                    "swaps_tried": int(len(gains)),
                    "swaps_applied": int(0),
                    "cut_edges": int(cut_edges_count),
                    "balance_ratio": float(bal),
                    "conductance": float(phi),
                    "runtime_ms": float(runtime_ms)
                })
                print(f"[METRICS] pass={iteration} start={current_pass_start_cut:.2f} end={final_cut_size:.2f} Δ={final_cut_size-current_pass_start_cut:+.2f} (0.0%) swaps=0/{len(gains)} bal={bal:.2f} phi={phi:.3f} time={runtime_ms:.1f}ms")
                if single_pass_only and iteration == 1:
                    self.draw(f"Algorithm Converged (Pass 1) | Final Cut: {final_cut_size:.2f}",
                              pass_step_gains=gains, pass_start_cut=current_pass_start_cut,
                              current_iteration=iteration)
                else:
                    self.draw(f"Algorithm Converged! | Final Cut Size: {final_cut_size:.2f}")
                print("No further improvement found (Max cumulative gain <= 0). Terminating.")
                yield
                break

        print("\\n--- Algorithm Finished ---")
        print(f"Final Partition A (Blue): {sorted(list(self.partition_a))}")
        print(f"Final Partition B (Orange): {sorted(list(self.partition_b))}")

        print(f"Final Cut Size: {self.calculate_cut_size(self.partition_a, self.partition_b):.2f}")
        # --- Export metrics to CSV ---
        try:
            import csv, os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            ts = time.strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(script_dir, f"kl_metrics_{ts}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["pass","start_cut","end_cut","delta","delta_pct","swaps_tried","swaps_applied","cut_edges","balance_ratio","conductance","runtime_ms"])
                writer.writeheader()
                for row in self.metrics:
                    writer.writerow(row)
            self.metrics_csv_path = csv_path
            print(f"[METRICS] Exported per-pass metrics to: {csv_path}")
        except Exception as e:
            print(f"[METRICS] CSV export failed: {e}")
        yield

#
# ======================================================================
# 2. TKINTER GUI APPLICATION CLASS
# ======================================================================
#

class TextRedirector(io.StringIO):
    def __init__(self, widget, colors):
        super().__init__()
        self.widget = widget
        self.colors = colors
    def write(self, s):
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, s, 'stdout')
        self.widget.see(tk.END)
        self.widget.configure(state='disabled')
    def flush(self): pass

class KL_GUI_App:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("KERNIGHAN-LIN Visualizer")
        self.root.geometry("1100x750")
        self.root.minsize(800, 600)
        
        self.colors = {
            'bg': '#FFFFFF', 'controls_bg': '#F8F9FA', 'plot_bg': '#F1F3F5', 'log_bg': '#FAFAFA',
            'text': '#212529', 'text_sec': '#495057', 'border': '#DEE2E6', 'entry_bg': '#FFFFFF',
            'accent_blue': '#0D6EFD', 'accent_red': '#DC3545', 'accent_green': '#198754',
            'btn_run': '#198754', 'btn_run_text': '#FFFFFF',
            'btn_sec': '#0D6EFD', 'btn_sec_text': '#FFFFFF',
            'btn_quit': '#DC3545', 'btn_quit_text': '#FFFFFF',
        }
        self.FONT_TITLE = font.Font(family="Helvetica", size=20, weight="bold", underline=True)
        self.FONT_HEADING = font.Font(family="Helvetica", size=12, weight="bold")
        self.FONT_LABEL = font.Font(family="Helvetica", size=11)
        self.FONT_BUTTON = font.Font(family="Helvetica", size=11, weight="bold")
        self.FONT_LOG = font.Font(family="Courier", size=10)

        self.root.configure(bg=self.colors['bg'])
        
        self.single_pass_var = tk.BooleanVar(value=True)

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('.', background=self.colors['controls_bg'], foreground=self.colors['text'],
                             fieldbackground=self.colors['entry_bg'], font=self.FONT_LABEL, borderwidth=0)
        self.style.configure('TPanedWindow', background=self.colors['bg'], sashwidth=8)
        self.style.configure('TLabelFrame', background=self.colors['controls_bg'], borderwidth=1, relief=tk.SOLID)
        self.style.configure('TLabelFrame.Label', background=self.colors['controls_bg'], foreground=self.colors['text'], font=self.FONT_HEADING)
        self.style.configure('TNotebook', background=self.colors['controls_bg'], tabposition='n', borderwidth=0)
        self.style.configure('TNotebook.Tab', font=self.FONT_HEADING, padding=(10, 5), background=self.colors['plot_bg'],
                             foreground=self.colors['text_sec'], borderwidth=1)
        self.style.map('TNotebook.Tab', background=[('selected', self.colors['controls_bg'])],
                       foreground=[('selected', self.colors['accent_blue'])], expand=[('selected', (1,1,1,0))])
        self.style.configure('TFrame', background=self.colors['controls_bg'])
        self.style.map('TEntry', fieldbackground=[('!focus', self.colors['entry_bg'])], foreground=[('!focus', self.colors['text'])],
                       insertcolor=[('', self.colors['text'])], bordercolor=[('focus', self.colors['accent_blue'])],
                       lightcolor=[('focus', self.colors['accent_blue'])], borderwidth=[('focus', 1)])
        self.style.configure('TCheckbutton', background=self.colors['controls_bg'], foreground=self.colors['text'], font=self.FONT_LABEL)
        self.style.map('TCheckbutton', background=[('active', self.colors['controls_bg'])])

        self.main_paned_window = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        self.main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.top_pane = ttk.PanedWindow(self.main_paned_window, orient=tk.HORIZONTAL)
        self.main_paned_window.add(self.top_pane, weight=4)
        self.log_frame = ttk.LabelFrame(self.main_paned_window, text="Log", height=150)
        self.main_paned_window.add(self.log_frame, weight=1)

        self.control_frame = ttk.Frame(self.top_pane, width=350, style='TFrame', borderwidth=1, relief="solid")
        self.top_pane.add(self.control_frame, weight=1)
        self.viz_frame = ttk.Frame(self.top_pane, style='TFrame', borderwidth=1, relief="solid")
        self.top_pane.add(self.viz_frame, weight=3)
        self.create_control_widgets()
        self.viz_frame = ttk.Frame(self.top_pane, style='TFrame', borderwidth=1, relief="solid")
        self.top_pane.add(self.viz_frame, weight=3)
        self.create_viz_widgets()
        self.create_log_widgets()
        
        self.graph = None
        self.run_gen = None
        self.stdout_redirector = TextRedirector(self.log_text, self.colors)
        self.original_stdout = sys.stdout
        sys.stdout = self.stdout_redirector
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Live metrics updater hook
        self.live_metrics = self.metrics_vars if hasattr(self, 'metrics_vars') else None

        self._last_input_key = None                                     

    def create_control_widgets(self):
        self.control_frame.pack_propagate(False)
        self.control_frame.columnconfigure(0, weight=1)
        self.control_frame.rowconfigure(1, weight=1)
        title_label = ttk.Label(self.control_frame, text="KERNIGHAN-LIN", font=self.FONT_TITLE,
                                background=self.colors['controls_bg'], foreground=self.colors['accent_blue'],
                                padding=(0, 10, 0, 15), anchor=tk.CENTER)
        title_label.grid(row=0, column=0, sticky="ew", padx=15, pady=(10,0))
        self.notebook = ttk.Notebook(self.control_frame)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.create_manual_tab()
        self.create_random_tab()
        self.create_predefined_tab()
        
        options_frame = ttk.Frame(self.control_frame)
        options_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=(5,0))
        pass_check = ttk.Checkbutton(options_frame, text="Run Single Pass Only (Pass 1)", 
                                     variable=self.single_pass_var, style='TCheckbutton')
        pass_check.pack(side=tk.LEFT)

        button_bar = ttk.Frame(self.control_frame, style='TFrame')
        button_bar.grid(row=3, column=0, sticky="ew", padx=10, pady=15)
        # --- Live Metrics Box ---
        self.metrics_vars = {
            "current": tk.StringVar(value="—"),
            "best": tk.StringVar(value="—"),
            "delta": tk.StringVar(value="—"),
            "impr": tk.StringVar(value="—")
        }
        metrics_box = ttk.LabelFrame(self.control_frame, text="Metrics (Live)", padding=10)
        metrics_box.grid(row=4, column=0, sticky="ew", padx=10, pady=(0,10))
        ttk.Label(metrics_box, text="Current Cut:").grid(row=0, column=0, sticky="w")
        ttk.Label(metrics_box, textvariable=self.metrics_vars["current"]).grid(row=0, column=1, sticky="e")
        ttk.Label(metrics_box, text="Best Cut:").grid(row=1, column=0, sticky="w")
        ttk.Label(metrics_box, textvariable=self.metrics_vars["best"]).grid(row=1, column=1, sticky="e")
        ttk.Label(metrics_box, text="Δ (abs):").grid(row=2, column=0, sticky="w")
        ttk.Label(metrics_box, textvariable=self.metrics_vars["delta"]).grid(row=2, column=1, sticky="e")
        ttk.Label(metrics_box, text="Δ (%):").grid(row=3, column=0, sticky="w")
        ttk.Label(metrics_box, textvariable=self.metrics_vars["impr"]).grid(row=3, column=1, sticky="e")

        button_bar.columnconfigure(1, weight=1)
        self.help_button = tk.Button(button_bar, text="Help", command=self.show_help, font=self.FONT_BUTTON,
                                     bg=self.colors['btn_sec'], fg=self.colors['btn_sec_text'], relief=tk.FLAT, padx=10, pady=5, width=6)
        self.help_button.grid(row=0, column=0, padx=(0, 5))
        self.run_button = tk.Button(button_bar, text="Run Visualization", command=self.run_app, font=self.FONT_BUTTON,
                                    bg=self.colors['btn_run'], fg=self.colors['btn_run_text'], relief=tk.FLAT, padx=10, pady=5)
        self.run_button.grid(row=0, column=1, sticky="ew", padx=5)
        self.quit_button = tk.Button(button_bar, text="Quit", command=self.on_closing, font=self.FONT_BUTTON,
                                     bg=self.colors['btn_quit'], fg=self.colors['btn_quit_text'], relief=tk.FLAT, padx=10, pady=5, width=6)
        self.quit_button.grid(row=0, column=2, padx=(5, 0))

    def create_manual_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Manual")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)
        n_frame = ttk.Frame(tab)
        n_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        ttk.Label(n_frame, text="Total Nodes (even):", font=self.FONT_LABEL).pack(side=tk.LEFT)
        self.manual_nodes = ttk.Entry(n_frame, width=10, font=self.FONT_LABEL)
        self.manual_nodes.pack(side=tk.LEFT, padx=10)
        ttk.Label(tab, text="Edges ('node1 node2 weight' per line):", font=self.FONT_LABEL).grid(row=1, column=0, columnspan=2, sticky='w')
        self.manual_edges_text = scrolledtext.ScrolledText(tab, height=10, width=30, wrap=tk.WORD, 
                                                           relief=tk.SOLID, bd=1, font=self.FONT_LABEL,
                                                           bg=self.colors['entry_bg'], fg=self.colors['text'],
                                                           insertbackground=self.colors['text'],
                                                           highlightcolor=self.colors['accent_blue'],
                                                           highlightbackground=self.colors['border'],
                                                           highlightthickness=1)
        self.manual_edges_text.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(5, 10))
        self.clear_button = tk.Button(tab, text="Clear", font=self.FONT_BUTTON,
                                      bg=self.colors['btn_sec'], fg=self.colors['btn_sec_text'],
                                      command=lambda: [self.manual_nodes.delete(0, tk.END), self.manual_edges_text.delete("1.0", tk.END)],
                                      relief=tk.FLAT, padx=10, pady=5)
        self.clear_button.grid(row=3, column=0, columnspan=2, sticky='e', pady=(0, 5))

    def create_random_tab(self):
        tab = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(tab, text="Random")
        self.random_vars = {
            "Nodes": ttk.Entry(tab, width=10, font=self.FONT_LABEL),
            "P Intra": ttk.Entry(tab, width=10, font=self.FONT_LABEL),
            "P Inter": ttk.Entry(tab, width=10, font=self.FONT_LABEL),
            "Seed": ttk.Entry(tab, width=10, font=self.FONT_LABEL)
        }
        ttk.Label(tab, text="Total Nodes (even):", font=self.FONT_LABEL).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.random_vars["Nodes"].grid(row=0, column=1, sticky=tk.W, padx=10)
        ttk.Label(tab, text="P (Intra-Cluster, e.g., 0.7):", font=self.FONT_LABEL).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.random_vars["P Intra"].grid(row=1, column=1, sticky=tk.W, padx=10)
        ttk.Label(tab, text="P (Inter-Cluster, e.g., 0.05):", font=self.FONT_LABEL).grid(row=2, column=0, sticky=tk.W, pady=5)
        self.random_vars["P Inter"].grid(row=2, column=1, sticky=tk.W, padx=10)
        ttk.Label(tab, text="Seed (e.g., 42):", font=self.FONT_LABEL).grid(row=3, column=0, sticky=tk.W, pady=5)
        self.random_vars["Seed"].grid(row=3, column=1, sticky=tk.W, padx=10)                                     

    def create_predefined_tab(self):
        tab = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(tab, text="Predefined")
        self.predefined_type = tk.StringVar(value="Cycle")
        self.predefined_nodes = ttk.Entry(tab, width=10, font=self.FONT_LABEL)
        ttk.Label(tab, text="Graph Type:", font=self.FONT_LABEL).grid(row=0, column=0, sticky=tk.W, pady=5)
        opt_menu = ttk.OptionMenu(tab, self.predefined_type, "Cycle", "Complete", "Path", "Star", "Grid")
        opt_menu.grid(row=0, column=1, sticky=tk.EW, padx=10)
        ttk.Label(tab, text="Total Nodes (even):", font=self.FONT_LABEL).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.predefined_nodes.grid(row=1, column=1, sticky=tk.W, padx=10)

    def create_viz_widgets(self):
        self.viz_frame.pack_propagate(False)
        self.viz_frame.columnconfigure(0, weight=1)
        self.viz_frame.rowconfigure(1, weight=1) # Header row
        self.viz_frame.rowconfigure(2, weight=10) # Canvas row

        # --- HEADERS ABOVE GRAPHS ---
        header_frame = ttk.Frame(self.viz_frame, style='TFrame')
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        header_frame.columnconfigure(0, weight=1)
        header_frame.columnconfigure(1, weight=1)

        lbl_left = tk.Label(header_frame, text="GRAPH PARTITION STATE", 
                            font=self.FONT_HEADING, bg=self.colors['controls_bg'], fg=self.colors['accent_blue'])
        lbl_left.grid(row=0, column=0)

        lbl_right = tk.Label(header_frame, text="CUT SIZE ANALYSIS", 
                             font=self.FONT_HEADING, bg=self.colors['controls_bg'], fg=self.colors['accent_blue'])
        lbl_right.grid(row=0, column=1)
        # --------------------------------

        self.fig, (self.ax_graph, self.ax_plot) = plt.subplots(1, 2, figsize=(10, 6))
        self.fig.patch.set_facecolor(self.colors['bg'])
        
        for ax in [self.ax_graph, self.ax_plot]:
            ax.set_facecolor(self.colors['plot_bg'])
            for spine in ax.spines.values(): spine.set_edgecolor(self.colors['border'])
            ax.tick_params(axis='x', colors=self.colors['text'])
            ax.tick_params(axis='y', colors=self.colors['text'])
            ax.xaxis.label.set_color(self.colors['text'])
            ax.yaxis.label.set_color(self.colors['text'])
            
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.98])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().configure(bg=self.colors['bg'], highlightthickness=0)
        self.canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew", padx=10, pady=(5, 10))

    def create_log_widgets(self):
        self.log_frame.rowconfigure(0, weight=1)
        self.log_frame.columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=10, wrap=tk.WORD, font=self.FONT_LOG,
                                                  state='disabled', relief=tk.FLAT, bd=0, bg=self.colors['log_bg'], fg=self.colors['text'], insertbackground=self.colors['text'])
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        clear_log_btn = tk.Button(self.log_frame, text="Clear Log", command=self.clear_log, font=self.FONT_BUTTON,
                                  bg=self.colors['btn_sec'], fg=self.colors['btn_sec_text'], relief=tk.FLAT, padx=10, pady=2)
        clear_log_btn.grid(row=0, column=1, sticky="se", padx=(0,5), pady=5)
    
    def update_live_metrics(self, current, best, delta_abs, delta_pct):
        if hasattr(self, "metrics_vars") and isinstance(self.metrics_vars, dict):
            self.metrics_vars["current"].set(f"{current:.2f}")
            self.metrics_vars["best"].set(f"{best:.2f}")
            self.metrics_vars["delta"].set(f"{-delta_abs:.2f}")  # show improvement as positive
            self.metrics_vars["impr"].set(f"{(delta_abs/current*100.0 if abs(current)>1e-12 else 0.0):.1f}%")

    def show_metrics_popup(self, csv_path):
        import csv, os, webbrowser
        top = tk.Toplevel(self.root)
        top.title("KL Metrics Summary")
        top.geometry("720x360")
        frm = ttk.Frame(top, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        lbl = ttk.Label(frm, text=f"Metrics exported to: {csv_path}", font=self.FONT_LABEL)
        lbl.pack(anchor="w", pady=(0,6))

        cols = ["pass","start_cut","end_cut","delta","delta_pct","swaps_tried","swaps_applied","cut_edges","balance_ratio","conductance","runtime_ms"]
        tree = ttk.Treeview(frm, columns=cols, show="headings", height=10)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=110, anchor="center")
        tree.pack(fill=tk.BOTH, expand=True)

        rows_loaded = 0
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= 100:
                        break
                    values = [row.get(c, "") for c in cols]
                    tree.insert("", "end", values=values)
                    rows_loaded += 1

            if rows_loaded == 0:
                messagebox.showwarning("Metrics", "No metrics rows recorded for this run.")
            else:
                ttk.Label(frm, text=f"Loaded {rows_loaded} rows", font=self.FONT_LABEL).pack(anchor="w", pady=(6,0))

        except FileNotFoundError:
            messagebox.showerror("Metrics", "Metrics CSV not found.")
        except Exception as e:
            messagebox.showerror("Metrics", f"Could not read metrics CSV: {e}")

        btn_bar = ttk.Frame(frm)
        btn_bar.pack(fill=tk.X, pady=(8,0))

        def open_csv():
            try:
                if hasattr(os, "startfile"):
                    os.startfile(csv_path)  # Windows
                else:
                    webbrowser.open(f"file://{csv_path}")
            except Exception as e:
                messagebox.showerror("Open CSV", f"Could not open CSV: {e}")

        open_btn = tk.Button(btn_bar, text="Open CSV", command=open_csv, font=self.FONT_BUTTON,
                             bg=self.colors['btn_sec'], fg=self.colors['btn_sec_text'],
                             relief=tk.FLAT, padx=10, pady=4)
        open_btn.pack(side=tk.LEFT)

        close_btn = tk.Button(btn_bar, text="Close", command=top.destroy, font=self.FONT_BUTTON,
                              bg=self.colors['btn_quit'], fg=self.colors['btn_quit_text'],
                              relief=tk.FLAT, padx=10, pady=4)
        close_btn.pack(side=tk.RIGHT)

        top.transient(self.root)
        top.grab_set()
        self.root.wait_window(top)

    def clear_log(self):
        self.log_text.configure(state='normal')
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state='disabled')
        
    def show_help(self):
        message = (
            "Kernighan–Lin Visualizer — How to Run\n"
            "\n"
            "General:\n"
            "• KL requires an even number of nodes so the graph can be split equally into two partitions.\n"
            "• Edges can have weights; cut size is the sum of weights of edges crossing between partitions.\n"
            "• Use the checkbox to run only Pass 1 (single pass) for teaching/demo.\n"
            "\n"
            "Modes:\n"
            "1) Manual\n"
            "   - Total Nodes: Enter a single even, positive integer (e.g., 6).\n"
            "   - Edges: Enter one edge per line in the format 'u v w'. Example: 0 1 3.5\n"
            "   - Nodes are 0..(n-1).\n"
            "\n"
            "2) Random\n"
            "   - Total Nodes: Even integer.\n"
            "   - P (Intra-Cluster): Probability of edges within halves (e.g., 0.7).\n"
            "   - P (Inter-Cluster): Probability of edges across halves (e.g., 0.05).\n"
            "   - Seed: Optional integer to reproduce graphs.\n"
            "\n"
            "3) Predefined\n"
            "   - Choose graph type. Enter an even node count.\n"
            "   - Star uses n-1 leaves. Grid requires a perfect square (4, 16, 36...).\n"
            "\n"
            "Reading the UI:\n"
            "• Left: Graph partition state. Blue=Partition A, Orange=Partition B. Dashed red edges are cut edges.\n"
            "• Right (Step-by-step): Shows potential cut size after each tentative swap (k). Legend marks Initial, Swaps, and Best step.\n"
            "  - The status box (top-left) shows cumulative Gain and Δ% improvement vs the start of the pass.\n"
            "  - Δ% = (CumulativeGain / Cut_at_pass_start) × 100.\n"
            "• Right (History by Pass): Every pass is marked with value and Δ from the previous pass (absolute and %).\n"
            "\n"
            "Tips:\n"
            "• For a 6-node demo (0..5), the app auto-starts with A={0,2,4}, B={1,3,5} for clarity.\n"
            "• After convergence, you can open a separated-view window to see the final partitions side-by-side.\n"
        )
        messagebox.showinfo("Help — How to Run KL", message)

    def _build_input_key(self):
        """Create a tuple key that represents the current inputs to decide if graph must rebuild."""
        mode = self.notebook.tab(self.notebook.select(), "text")
        if mode == "Manual":
            return ("Manual",
                    self.manual_nodes.get(),
                    self.manual_edges_text.get("1.0", tk.END).strip())
        elif mode == "Random":
            return ("Random",
                    self.random_vars["Nodes"].get(),
                    self.random_vars["P Intra"].get(),
                    self.random_vars["P Inter"].get(),
                    self.random_vars["Seed"].get())
        elif mode == "Predefined":
            return ("Predefined",
                    self.predefined_type.get(),
                    self.predefined_nodes.get())
        return ("Unknown",)

    def run_app(self):
        try:
            self.clear_log()
            print("Parsing graph input...")
            mode = self.notebook.tab(self.notebook.select(), "text")

            input_key = self._build_input_key()
            rebuild = (self._last_input_key != input_key)

            if rebuild:
                if mode == "Manual": self.parse_manual_input()
                elif mode == "Random": self.parse_random_input()
                elif mode == "Predefined": self.parse_predefined_input()
                self._last_input_key = input_key
            else:
                print("Reusing existing graph for fair comparison (inputs unchanged).")

            if self.graph:
                print(f"Graph ready with {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges.")
                self.run_button.config(state='disabled', text='Running...', bg=self.colors['text_sec'])
                self.visualizer = KernighanLinVisualizer(self.graph, self.fig, self.ax_graph, self.ax_plot, self.colors)
                # attach live metrics callback
                self.visualizer.on_live_metrics = self.update_live_metrics

                if hasattr(self, "current_seed"):
                    self.visualizer.seed_for_partition = self.current_seed + 2

                self.run_gen = self.visualizer.run_generator(single_pass_only=self.single_pass_var.get())
                self.animate_run()
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
            print(f"Error: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
            print(f"Error: {e}")

    def animate_run(self):
        try:
            next(self.run_gen)
            self.root.after(300, self.animate_run)
        except StopIteration:
            print("\\nAnimation finished.")
            self.run_button.config(state='normal', text='Run Visualization', bg=self.colors['btn_run'])
            # Auto-show metrics popup (wait briefly for CSV if needed)
            try:
                import os
                csv_path = getattr(self.visualizer, "metrics_csv_path", None)
                if not csv_path:
                    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kl_metrics.csv")
                def _try_popup(attempt=0):
                    if os.path.exists(csv_path) or attempt >= 5:
                        try:
                            self.show_metrics_popup(csv_path)
                        except Exception as e2:
                            print(f"Could not show metrics popup: {e2}")
                    else:
                        # wait 150ms then try again
                        self.root.after(150, lambda: _try_popup(attempt+1))
                _try_popup(0)
            except Exception as e:
                print(f"Could not show metrics popup: {e}")
            if messagebox.askyesno("Finished", "Algorithm has converged.\\n\\nShow final separated partition view in a new window?"):
                self.visualizer.draw_final_separated_view()
        except Exception as e:
            messagebox.showerror("Runtime Error", f"An error occurred during visualization: {e}")
            print(f"Runtime Error: {e}")
            self.run_button.config(state='normal', text='Run Visualization', bg=self.colors['btn_run'])

    # -----------------
    # Input parsers
    # -----------------
    def _warn_odd_nodes(self):
        messagebox.showwarning("Odd Number of Nodes",
                               "An even node count is required for Kernighan–Lin.\\n"
                               "Please enter an EVEN number (e.g., 6, 8, 10...).")
        
    def parse_manual_input(self):
        n_str = self.manual_nodes.get()
        if not n_str: raise ValueError("Node count must be provided.")
        n = int(n_str)
        if n <= 0:
            raise ValueError("Node count must be a positive even number.")
        if n % 2 != 0:
            self._warn_odd_nodes()
            raise ValueError("Node count must be even.")
        G = nx.Graph()
        G.add_nodes_from(range(n))
        edge_data = self.manual_edges_text.get("1.0", tk.END).strip()
        if not edge_data: raise ValueError("No edge data provided.")
        for line in edge_data.split('\n'):
            line = line.strip()
            if not line: continue
            if line.lower() == 'done': continue
            parts = line.split()
            if len(parts) != 3: raise ValueError(f"Invalid edge format: '{line}'. Must be 'u v w'.")
            u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
            if 0 <= u < n and 0 <= v < n: G.add_edge(u, v, weight=w)
            else: raise ValueError(f"Nodes in edge '{line}' are out of range [0, {n-1}].")
        self.graph = G

    def parse_random_input(self):
        n_str = self.random_vars["Nodes"].get()
        p_intra_str = self.random_vars["P Intra"].get()
        p_inter_str = self.random_vars["P Inter"].get()
        seed_str = self.random_vars["Seed"].get()

        if not all([n_str, p_intra_str, p_inter_str]):
            raise ValueError("All fields for Random Partition must be filled.")
        num_nodes = int(n_str)
        if num_nodes <= 0:
            raise ValueError("Node count must be a positive even number.")
        if num_nodes % 2 != 0:
            self._warn_odd_nodes()
            raise ValueError("Node count must be even.")

        p_intra = float(p_intra_str)
        p_inter = float(p_inter_str)
        seed = int(seed_str) if seed_str else 42
        
        sizes = [num_nodes // 2, num_nodes // 2]
        G = nx.random_partition_graph(sizes, p_intra, p_inter, seed=seed)
        rng = random.Random(seed + 1)
        self.graph = self.add_random_weights(G, rng=rng)
        self.current_seed = seed

    def parse_predefined_input(self):
        n_str = self.predefined_nodes.get()
        if not n_str: raise ValueError("Node count must be provided.")
        n = int(n_str)
        if n <= 0:
            raise ValueError("Node count must be a positive even number.")
        if n % 2 != 0:
            self._warn_odd_nodes()
            raise ValueError("Node count must be even.")
        graph_type = self.predefined_type.get()
        G = None
        if graph_type == 'Complete': G = nx.complete_graph(n)
        elif graph_type == 'Path': G = nx.path_graph(n)
        elif graph_type == 'Cycle': G = nx.cycle_graph(n)
        elif graph_type == 'Star': G = nx.star_graph(n - 1)
        elif graph_type == 'Grid':
            if int(n**0.5)**2 == n: G = nx.grid_2d_graph(int(n**0.5), int(n**0.5))
            else: raise ValueError("Grid must have a square number of nodes (e.g., 4, 16, 36).")
        if G:
            if not all(isinstance(node, int) for node in G.nodes()): G = nx.convert_node_labels_to_integers(G, first_label=0)
            self.graph = self.add_random_weights(G)
        else: raise ValueError("Could not create predefined graph.")

    def add_random_weights(self, graph, min_w=1.0, max_w=10.0, rng=None):
        rng = rng or random
        for u, v in graph.edges():
            graph[u][v]['weight'] = round(rng.uniform(min_w, max_w), 1)
        return graph

    def on_closing(self):
        sys.stdout = self.original_stdout
        self.root.destroy()

#
# ======================================================================
# 3. MAIN EXECUTION
# ======================================================================
#
if __name__ == "__main__":
    plt.ioff() # Turn off matplotlib's interactive mode
    root = tk.Tk()
    app = KL_GUI_App(root)
    root.mainloop()
