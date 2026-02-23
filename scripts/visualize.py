#!/usr/bin/env python3
"""Generate all visualizations for the literature review."""

import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import json

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Calm, clearly distinguishable palette for up to 8 categories
CALM_PALETTE = [
    '#3A7DC9',  # steel blue
    '#56A86E',  # sage green
    '#C96B3A',  # terracotta
    '#8A6FBF',  # soft purple
    '#BFA830',  # warm olive
    '#3AAEC9',  # sky teal
    '#C93A6B',  # dusty rose
    '#6BA8BF',  # muted cyan
]


class BenchmarkVisualizer:
    def __init__(self, csv_path: str = "data/papers.csv"):
        self.df = pd.read_csv(csv_path)
        # Normalize column names to match script expectations
        self.df['Year'] = self.df['Paper (Author, Year)'].str.extract(r'(\d{4})')
        self.df['Year'] = pd.to_numeric(self.df['Year'])
        self.df['Category'] = self.df['Type']
        self.df['Dialect_Coverage'] = self.df['A3\nDialect Coverage']
        self.df['Synthetic_Method'] = self.df['C2\nSynthetic Gen. Method'].fillna('N/A')
        self.df['Stability_Reported'] = self.df['E4\nStability Reported']
        self.df['Bias_Mitigation'] = self.df['E5\nBias Mitigation']
        self.output_dir = Path("visualizations")
        self.output_dir.mkdir(exist_ok=True)

        metadata_file = Path("data/metadata/all_papers_metadata.json")
        if metadata_file.exists():
            with open(metadata_file) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _save_fig(self, fig, stem: str):
        """Save a matplotlib figure as both PNG and PDF at 300 dpi."""
        fig.savefig(self.output_dir / f'{stem}.png', dpi=300, bbox_inches='tight')
        fig.savefig(self.output_dir / f'{stem}.pdf', dpi=300, bbox_inches='tight')
        print(f"Saved: {stem}.png + {stem}.pdf")

    def _short_name(self, cell: str, maxlen: int = 20) -> str:
        """Extract short paper name from 'TITLE (Author, Year)' cell."""
        clean = str(cell).replace('\n', ' ').replace('*', '').replace('\u2605', '').strip()
        m = re.match(r'^([^\(]+)', clean)
        name = m.group(1).strip() if m else clean
        return name[:maxlen].strip()

    def _add_bar_labels(self, ax, horizontal: bool = False):
        """Add value labels on bar charts."""
        for bar in ax.patches:
            val = int(bar.get_width() if horizontal else bar.get_height())
            if val == 0:
                continue
            if horizontal:
                ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                        str(val), va='center', fontsize=9, fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        str(val), ha='center', fontsize=9, fontweight='bold')

    # ── Plot 1 (existing) ─────────────────────────────────────────────────────

    def plot_temporal_distribution(self):
        """Papers by year and category."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        year_counts = self.df['Year'].value_counts().sort_index()
        ax1.bar(year_counts.index, year_counts.values, color=CALM_PALETTE[0], alpha=0.85)
        ax1.set_xlabel('Year', fontweight='bold')
        ax1.set_ylabel('Number of Papers', fontweight='bold')
        ax1.set_title('Arabic LLM Benchmarking Papers by Year', fontweight='bold')

        category_year = self.df.groupby(['Year', 'Category']).size().unstack(fill_value=0)
        n_cats = len(category_year.columns)
        category_year.plot(kind='bar', stacked=True, ax=ax2,
                           color=CALM_PALETTE[:n_cats], alpha=0.88)
        ax2.set_xlabel('Year', fontweight='bold')
        ax2.set_title('Papers by Category', fontweight='bold')
        ax2.legend(title='Category', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)

        plt.tight_layout()
        self._save_fig(fig, 'temporal_distribution')
        plt.close()

    # ── Plot 2 (existing) ─────────────────────────────────────────────────────

    def plot_gap_dashboard(self):
        """Gap analysis for RQs."""
        arabic_only = self.df[self.df['Category'].str.startswith('Arabic', na=False)]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'RQ1: Dialect Stratification', 'RQ2: Synthetic Methods',
                'RQ3: Stability Reporting', 'RQ3: Bias Mitigation'
            )
        )

        gaps = [
            (arabic_only['Dialect_Coverage'].apply(
                lambda x: 'Stratified' if 'stratified' in str(x).lower() else 'Not stratified'
            ).value_counts(), '#FFC000', 1, 1),
            (arabic_only['Synthetic_Method'].apply(
                lambda x: 'Used' if str(x) != 'N/A' else 'Not used'
            ).value_counts(), '#70AD47', 1, 2),
            (arabic_only['Stability_Reported'].apply(
                lambda x: 'Yes' if str(x).startswith('Yes') else 'No'
            ).value_counts(), '#5B9BD5', 2, 1),
            (arabic_only['Bias_Mitigation'].apply(
                lambda x: 'Yes' if str(x) != 'No' else 'No'
            ).value_counts(), '#ED7D31', 2, 2)
        ]

        for data, color, row, col in gaps:
            fig.add_trace(go.Bar(x=data.index, y=data.values, marker_color=color),
                         row=row, col=col)

        fig.update_layout(title_text="Gap Analysis Dashboard", showlegend=False, height=700)
        fig.write_html(str(self.output_dir / 'gap_dashboard.html'))
        print("Saved: gap_dashboard.html")
        try:
            fig.write_image(str(self.output_dir / 'gap_dashboard.pdf'), scale=3)
            print("Saved: gap_dashboard.pdf")
        except Exception:
            print("Note: gap_dashboard.pdf skipped (install kaleido: pip install kaleido)")

    # ── Plot 3: Data Quality (A1 + A5) ───────────────────────────────────────

    def plot_data_quality(self):
        """Annotation Type (A1) and Language Scope (A5)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        def annot_cat(v):
            v = str(v).lower()
            if 'human' in v and 'synthetic' not in v:
                return 'Human-Annotated'
            if 'semi' in v:
                return 'Semi-Synthetic'
            if 'synthetic' in v or 'llm' in v:
                return 'Synthetic (LLM)'
            return 'N/A'

        annot = self.df['A1\nAnnotation Type'].apply(annot_cat).value_counts()
        ax1.barh(annot.index, annot.values, color=CALM_PALETTE[:len(annot)], alpha=0.85)
        ax1.set_xlabel('Number of Papers', fontweight='bold')
        ax1.set_title('Annotation Type (A1)', fontweight='bold')
        self._add_bar_labels(ax1, horizontal=True)

        def lang_cat(v):
            v = str(v).lower()
            if 'arabic' in v:
                return 'Arabic-Focused'
            if 'agnostic' in v:
                return 'Language-Agnostic'
            return 'English-Focused'

        lang = self.df['A5\nLanguage Scope'].apply(lang_cat).value_counts()
        ax2.barh(lang.index, lang.values, color=CALM_PALETTE[2:2 + len(lang)], alpha=0.85)
        ax2.set_xlabel('Number of Papers', fontweight='bold')
        ax2.set_title('Language Scope (A5)', fontweight='bold')
        self._add_bar_labels(ax2, horizontal=True)

        fig.suptitle('Data Characteristics — Annotation & Language', fontweight='bold', fontsize=13)
        plt.tight_layout()
        self._save_fig(fig, 'data_quality')
        plt.close()

    # ── Plot 4: Task Design (B3 + B4) ────────────────────────────────────────

    def plot_task_design(self):
        """Task Design Origin (B3) and Reasoning Type (B4)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        def b3_cat(v):
            v = str(v).lower()
            if 'native' in v and 'arabic' in v:
                return 'Native Arabic'
            if 'mixed' in v:
                return 'Mixed'
            if 'translated' in v:
                return 'Translated (EN→AR)'
            if 'synthetic' in v:
                return 'Synthetic'
            if 'english' in v or 'native' in v:
                return 'Native English'
            return 'Other'

        b3 = self.df['B3\nTask Design Origin'].apply(b3_cat).value_counts()
        ax1.barh(b3.index, b3.values, color=CALM_PALETTE[:len(b3)], alpha=0.85)
        ax1.set_xlabel('Number of Papers', fontweight='bold')
        ax1.set_title('Task Design Origin (B3)', fontweight='bold')
        self._add_bar_labels(ax1, horizontal=True)

        def b4_cat(v):
            v = str(v).lower()
            if 'not applicable' in v or v == 'nan':
                return 'N/A'
            if 'logical' in v and 'math' in v:
                return 'Logical + Math'
            if 'various' in v or 'multiple' in v or 'temporal' in v or 'causal' in v:
                return 'Various / Multi-type'
            if 'logical' in v:
                return 'Logical'
            if 'math' in v:
                return 'Mathematical'
            return 'Other'

        b4 = self.df['B4\nReasoning Type'].apply(b4_cat).value_counts()
        ax2.barh(b4.index, b4.values, color=CALM_PALETTE[3:3 + len(b4)], alpha=0.85)
        ax2.set_xlabel('Number of Papers', fontweight='bold')
        ax2.set_title('Reasoning Type Coverage (B4)', fontweight='bold')
        self._add_bar_labels(ax2, horizontal=True)

        fig.suptitle('Task Characteristics — Design Origin & Reasoning', fontweight='bold', fontsize=13)
        plt.tight_layout()
        self._save_fig(fig, 'task_design')
        plt.close()

    # ── Plot 5: Construction Quality (C4 + C5) ───────────────────────────────

    def plot_construction_quality(self):
        """Human Intervention (C4) and Contamination Detection (C5)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        def c4_cat(v):
            v = str(v).lower()
            if 'high' in v:
                return 'High'
            if 'medium' in v:
                return 'Medium'
            if 'low' in v:
                return 'Low'
            return 'N/A'

        order_c4 = ['High', 'Medium', 'Low', 'N/A']
        c4 = self.df['C4\nHuman Intervention'].apply(c4_cat).value_counts()
        c4 = c4.reindex([x for x in order_c4 if x in c4.index])
        bars1 = ax1.bar(c4.index, c4.values,
                        color=[CALM_PALETTE[0], CALM_PALETTE[5], CALM_PALETTE[2], CALM_PALETTE[7]][:len(c4)],
                        alpha=0.85)
        ax1.set_ylabel('Number of Papers', fontweight='bold')
        ax1.set_title('Human Intervention Level (C4)', fontweight='bold')
        self._add_bar_labels(ax1, horizontal=False)

        def c5_cat(v):
            v = str(v).lower()
            if 'not specified' in v:
                return 'Not Specified'
            if 'yes' in v or 'blind' in v or 'proprietary' in v or 'by design' in v:
                return 'Yes'
            if 'partial' in v:
                return 'Partial'
            if 'no' in v:
                return 'No'
            return 'Other'

        c5 = self.df['C5\nContamination Detect.'].apply(c5_cat).value_counts()
        ax2.bar(c5.index, c5.values, color=CALM_PALETTE[1:1 + len(c5)], alpha=0.85)
        ax2.set_ylabel('Number of Papers', fontweight='bold')
        ax2.set_title('Contamination Detection (C5)', fontweight='bold')
        self._add_bar_labels(ax2, horizontal=False)

        fig.suptitle('Construction Quality — Human Oversight & Contamination', fontweight='bold', fontsize=13)
        plt.tight_layout()
        self._save_fig(fig, 'construction_quality')
        plt.close()

    # ── Plot 6: Openness & Temporality (C6 + D1) ─────────────────────────────

    def plot_openness_and_temporality(self):
        """Open Code/Data (C6) by paper type, and Benchmark Temporality (D1)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        def c6_cat(v):
            v = str(v).lower()
            if v.startswith('yes') or v == 'yes':
                return 'Open'
            if 'partial' in v:
                return 'Partial'
            return 'Closed / N/A'

        arabic = self.df[self.df['Category'].str.startswith('Arabic', na=False)]
        general = self.df[~self.df['Category'].str.startswith('Arabic', na=False)]
        cats = ['Open', 'Partial', 'Closed / N/A']
        ar_vals = [arabic['C6\nOpen Code/Data'].apply(c6_cat).value_counts().get(c, 0) for c in cats]
        gn_vals = [general['C6\nOpen Code/Data'].apply(c6_cat).value_counts().get(c, 0) for c in cats]
        x = np.arange(len(cats))
        w = 0.35
        ax1.bar(x - w / 2, ar_vals, w, label='Arabic Papers',  color=CALM_PALETTE[0], alpha=0.85)
        ax1.bar(x + w / 2, gn_vals, w, label='General Papers', color=CALM_PALETTE[2], alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(cats)
        ax1.set_ylabel('Number of Papers', fontweight='bold')
        ax1.set_title('Open Code / Data (C6) by Paper Type', fontweight='bold')
        ax1.legend()

        def d1_cat(v):
            v = str(v).lower()
            if 'dynamic' in v:
                return 'Dynamic'
            if 'static' in v and 'audit' in v:
                return 'Static + Audit'
            if 'static' in v:
                return 'Static'
            return 'Other'

        d1 = self.df['D1\nBenchmark Temporality'].apply(d1_cat).value_counts()
        ax2.bar(d1.index, d1.values, color=CALM_PALETTE[4:4 + len(d1)], alpha=0.85)
        ax2.set_ylabel('Number of Papers', fontweight='bold')
        ax2.set_title('Benchmark Temporality (D1)', fontweight='bold')
        self._add_bar_labels(ax2, horizontal=False)

        fig.suptitle('Openness & Temporality of Benchmarks', fontweight='bold', fontsize=13)
        plt.tight_layout()
        self._save_fig(fig, 'openness_temporality')
        plt.close()

    # ── Plot 7: Evaluation Landscape (E1 + D5) ───────────────────────────────

    def plot_eval_landscape(self):
        """Evaluation Protocol (E1) and Prompting Strategy (D5)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        def e1_cat(v):
            v = str(v).lower()
            if 'hybrid' in v:
                return 'Hybrid'
            if 'llm' in v and 'judge' in v:
                return 'LLM-as-Judge'
            if 'human' in v:
                return 'Human-Eval'
            if 'rule' in v:
                return 'Rule-Based'
            if 'statistical' in v:
                return 'Statistical'
            return 'Other'

        e1 = self.df['E1\nEval Protocol'].apply(e1_cat).value_counts()
        ax1.barh(e1.index, e1.values, color=CALM_PALETTE[:len(e1)], alpha=0.85)
        ax1.set_xlabel('Number of Papers', fontweight='bold')
        ax1.set_title('Evaluation Protocol (E1)', fontweight='bold')
        self._add_bar_labels(ax1, horizontal=True)

        def d5_cat(v):
            v = str(v).lower()
            has_zero = 'zero' in v
            has_few  = 'few'  in v
            has_cot  = 'cot'  in v or 'chain' in v
            if has_zero and has_few and has_cot:
                return 'Zero + Few + CoT'
            if has_zero and has_few:
                return 'Zero + Few-shot'
            if has_cot:
                return 'CoT (incl.)'
            if has_zero:
                return 'Zero-shot Only'
            if has_few:
                return 'Few-shot Only'
            return 'Other / N/A'

        d5 = self.df['D5\nPrompting Strategy'].apply(d5_cat).value_counts()
        ax2.barh(d5.index, d5.values, color=CALM_PALETTE[3:3 + len(d5)], alpha=0.85)
        ax2.set_xlabel('Number of Papers', fontweight='bold')
        ax2.set_title('Prompting Strategy (D5)', fontweight='bold')
        self._add_bar_labels(ax2, horizontal=True)

        fig.suptitle('Evaluation Landscape — Protocol & Prompting', fontweight='bold', fontsize=13)
        plt.tight_layout()
        self._save_fig(fig, 'eval_landscape')
        plt.close()

    # ── Plot 8: Feature Completeness Heatmap ─────────────────────────────────

    def plot_feature_heatmap(self):
        """Binary feature matrix: all 17 papers x 10 key quality features."""

        names = self.df['Paper (Author, Year)'].apply(
            lambda x: self._short_name(x, maxlen=22)
        ).tolist()

        # Scoring functions: return 0 (No), 0.5 (Partial), 1 (Yes)
        def sc_open(v):
            v = str(v).lower()
            if v.startswith('yes') or v == 'yes': return 1.0
            if 'partial' in v: return 0.5
            return 0.0

        def sc_leaderboard(v):
            v = str(v).lower()
            return 1.0 if 'yes' in v else 0.0

        def sc_contamination(v):
            v = str(v).lower()
            if 'yes' in v or 'blind' in v or 'proprietary' in v or 'by design' in v: return 1.0
            if 'partial' in v: return 0.5
            return 0.0

        def sc_stability(v):
            return 1.0 if 'yes' in str(v).lower() else 0.0

        def sc_bias(v):
            v = str(v).lower()
            if v.strip() == 'no': return 0.0
            if 'partial' in v: return 0.5
            if 'yes' in v: return 1.0
            return 0.0

        def sc_dialect(v):
            v = str(v).lower()
            if 'stratif' in v: return 1.0
            if 'msa-only' in v or 'n/a' in v or 'english' in v: return 0.0
            return 0.5

        def sc_synthetic(v):
            v = str(v)
            return 0.0 if (v == 'nan' or 'N/A' in v) else 1.0

        def sc_human(v):
            v = str(v).lower()
            if 'high' in v: return 1.0
            if 'medium' in v: return 0.5
            return 0.0

        def sc_native(v):
            v = str(v).lower()
            if 'native' in v and 'arabic' in v: return 1.0
            if 'mixed' in v: return 0.5
            return 0.0

        def sc_dynamic(v):
            return 1.0 if 'dynamic' in str(v).lower() else 0.0

        features = [
            ('Open Data\n(C6)',          'C6\nOpen Code/Data',          sc_open),
            ('Leaderboard\n(D6)',         'D6\nLeaderboard',             sc_leaderboard),
            ('Contamination\nCheck (C5)', 'C5\nContamination Detect.',   sc_contamination),
            ('Stability\nReported (E4)', 'E4\nStability Reported',       sc_stability),
            ('Bias\nMitigation (E5)',    'E5\nBias Mitigation',          sc_bias),
            ('Dialect\nCoverage (A3)',   'A3\nDialect Coverage',         sc_dialect),
            ('Synthetic\nMethod (C2)',   'C2\nSynthetic Gen. Method',    sc_synthetic),
            ('Human\nIntervention (C4)','C4\nHuman Intervention',        sc_human),
            ('Native Arabic\nTasks (B3)','B3\nTask Design Origin',       sc_native),
            ('Dynamic\nBenchmark (D1)', 'D1\nBenchmark Temporality',     sc_dynamic),
        ]

        matrix = np.array([
            [fn(row[col]) for _, col, fn in features]
            for _, row in self.df.iterrows()
        ])

        feat_labels = [lbl for lbl, _, _ in features]

        fig, ax = plt.subplots(figsize=(16, 9))
        im = ax.imshow(matrix, cmap='YlGn', aspect='auto', vmin=0, vmax=1)

        ax.set_xticks(range(len(feat_labels)))
        ax.set_xticklabels(feat_labels, fontsize=8.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title('Feature Completeness Heatmap — All Papers x Key Quality Criteria',
                     fontweight='bold', fontsize=12, pad=12)

        def val_label(v):
            if v >= 0.75: return 'Yes'
            if v >= 0.25: return 'Part.'
            return 'No'

        for i in range(len(names)):
            for j in range(len(feat_labels)):
                val  = matrix[i, j]
                text = val_label(val)
                col  = 'white' if val >= 0.75 else '#333333'
                ax.text(j, i, text, ha='center', va='center', fontsize=7.5, color=col)

        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['No', 'Partial', 'Yes'])
        cbar.set_label('Feature Score', fontsize=9)

        plt.tight_layout()
        self._save_fig(fig, 'feature_heatmap')
        plt.close()

    # ── Plot 9: Radar Comparison (Arabic papers + Proposal) ──────────────────

    def plot_radar_comparison(self):
        """Radar/spider chart: Arabic papers + This Proposal across 5 dimensions."""

        # Include Arabic papers and the proposal row
        mask = (self.df['Category'].str.startswith('Arabic', na=False) |
                self.df['Paper (Author, Year)'].str.contains('AbuHweidi', na=False))
        subset = self.df[mask].copy().reset_index(drop=True)
        names  = subset['Paper (Author, Year)'].apply(
            lambda x: self._short_name(x, maxlen=18)
        ).tolist()

        # Dimension scoring (0–3 per dimension)
        def dim_data_quality(row):
            s = 0
            if 'msa-only' not in str(row['A3\nDialect Coverage']).lower(): s += 1
            if 'human' in str(row['A1\nAnnotation Type']).lower(): s += 1
            if 'arabic' in str(row['A5\nLanguage Scope']).lower(): s += 1
            return s

        def dim_task_rigor(row):
            s = 0
            b3 = str(row['B3\nTask Design Origin']).lower()
            if 'native' in b3 and 'arabic' in b3: s += 1
            if str(row['B4\nReasoning Type']).lower() not in ['not applicable', 'nan']: s += 1
            if len(str(row['B1\nTask Types Covered'])) > 30: s += 1
            return s

        def dim_construction(row):
            s = 0
            c5 = str(row['C5\nContamination Detect.']).lower()
            if 'yes' in c5 or 'blind' in c5 or 'by design' in c5: s += 1
            if 'high' in str(row['C4\nHuman Intervention']).lower(): s += 1
            if str(row['C6\nOpen Code/Data']).lower().startswith('yes'): s += 1
            return s

        def dim_eval_method(row):
            s = 0
            if 'yes' in str(row['D6\nLeaderboard']).lower(): s += 1
            e1 = str(row['E1\nEval Protocol']).lower()
            if 'hybrid' in e1 or 'llm' in e1: s += 1
            e3 = str(row['E3\nScoring Granularity']).lower()
            if 'per-task' in e3 or 'multi' in e3 or 'uncertainty' in e3: s += 1
            return s

        def dim_stability_fairness(row):
            s = 0
            if 'yes' in str(row['E4\nStability Reported']).lower(): s += 1
            if str(row['E5\nBias Mitigation']).lower().strip() != 'no': s += 1
            d5 = str(row['D5\nPrompting Strategy']).lower()
            if 'cot' in d5 or ('zero' in d5 and 'few' in d5): s += 1
            return s

        score_fns   = [dim_data_quality, dim_task_rigor, dim_construction,
                       dim_eval_method, dim_stability_fairness]
        dim_labels  = ['Data\nQuality', 'Task\nRigor', 'Construction\nQuality',
                       'Eval\nMethodology', 'Stability\n& Fairness']
        N = len(dim_labels)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        for idx, (_, row) in enumerate(subset.iterrows()):
            scores = [fn(row) for fn in score_fns]
            scores += scores[:1]
            is_proposal = 'AbuHweidi' in str(row['Paper (Author, Year)'])
            color = '#C93A6B' if is_proposal else CALM_PALETTE[idx % len(CALM_PALETTE)]
            lw    = 3 if is_proposal else 1.5
            alpha = 0.18 if is_proposal else 0.06
            label = names[idx] + (' (Proposal)' if is_proposal else '')
            ax.plot(angles, scores, 'o-', linewidth=lw, color=color, label=label,
                    zorder=5 if is_proposal else 3)
            ax.fill(angles, scores, alpha=alpha, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dim_labels, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 3)
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['1', '2', '3'], fontsize=8, color='gray')
        ax.grid(True, alpha=0.35)
        ax.set_title('Arabic Benchmarks: Multi-Dimensional Profile\n(vs. This Proposal)',
                     fontweight='bold', fontsize=13, pad=25)
        ax.legend(loc='upper right', bbox_to_anchor=(1.42, 1.18),
                  fontsize=8.5, framealpha=0.8)

        plt.tight_layout()
        self._save_fig(fig, 'radar_comparison')
        plt.close()

    # ── Master runner ─────────────────────────────────────────────────────────

    def generate_all(self):
        print("\nGenerating visualizations...\n")
        self.plot_temporal_distribution()        # 1 — temporal
        self.plot_gap_dashboard()                # 2 — gap dashboard (HTML + optional PDF)
        self.plot_data_quality()                 # 3 — A1 + A5
        self.plot_task_design()                  # 4 — B3 + B4
        self.plot_construction_quality()         # 5 — C4 + C5
        self.plot_openness_and_temporality()     # 6 — C6 + D1
        self.plot_eval_landscape()               # 7 — E1 + D5
        self.plot_feature_heatmap()              # 8 — full feature matrix
        self.plot_radar_comparison()             # 9 — radar / spider
        print(f"\nDone. All outputs saved to {self.output_dir}/")


if __name__ == "__main__":
    visualizer = BenchmarkVisualizer()
    visualizer.generate_all()
