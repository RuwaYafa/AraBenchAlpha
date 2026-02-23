#!/usr/bin/env python3
"""
Fetch BibTeX entries for all papers in papers.csv via Semantic Scholar API.
Output: data/references.bib  (ready for LaTeX \bibliography{references})

Usage:
    python scripts/fetch_bib.py
"""

import re
import sys
import time
import requests
import pandas as pd
from pathlib import Path

# Fix Windows terminal encoding (handles Arabic, emoji, special chars)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH    = "data/papers.csv"
OUT_BIB     = "data/references.bib"
API_BASE    = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS      = "title,authors,year,venue,externalIds,citationStyles"
SLEEP_SEC   = 8      # polite delay between requests (avoids 429 rate limit)
MAX_RETRIES = 5      # retry on 429 with exponential backoff

# Full titles for papers whose short CSV names cause wrong API matches.
# Key: first word of the CSV title hint (lowercase).
# Value: exact title to search on Semantic Scholar, or None to skip API entirely.
# Add / fix entries here as needed.
KNOWN_TITLES = {
    'helm':             'Holistic Evaluation of Language Models',
    'dynabench':        'Dynabench: Rethinking Benchmarking in NLP',
    'orca':             'ORCA: A Challenging Benchmark for Arabic Language Understanding',
    # Open Arabic LLM Leaderboard: no single arXiv paper; fill manually from HuggingFace
    'open arabic llm lb v1': None,
    'open arabic llm lb v2': None,
    'judging':          'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
    'length-controlled': 'Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators',
    'style':            'Style Over Substance: Evaluation Biases for Large Language Models',
    'error':            'Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations',
    'llm-driven':       'LLM-Driven Synthetic Data Generation Curation and Evaluation for Mathematical Reasoning',
    'self-instruct':    'Self-Instruct: Aligning Language Models with Self-Generated Instructions',
    'wizardlm':         'WizardLM: Empowering Large Language Models to Follow Complex Instructions',
    # Recent Arabic papers not yet indexed on Semantic Scholar — skip API
    'balsam':           None,
    'abbl':             None,
}
# ──────────────────────────────────────────────────────────────────────────────


def parse_paper_cell(cell: str):
    """
    Parse cells like:
        'BALSAM (Almatham et al., 2025)'
        'Judging LLM-as-a-Judge / MT-Bench\n(Zheng et al., 2023)'
    Returns (title_hint, first_author_lastname, year) or None to skip.
    """
    cell = cell.strip().replace('\n', ' ')

    # Skip the self-proposal row
    if 'This Proposal' in cell or 'AbuHweidi' in cell:
        return None

    # Extract year
    year_m = re.search(r'\b(20\d{2})\b', cell)
    year = year_m.group(1) if year_m else ''

    # Extract first author last name  e.g. "(Almatham et al., 2025)"
    author_m = re.search(r'\((\w+)\s+et al\.', cell)
    if not author_m:
        # Single author  e.g. "(Wang, 2022)"
        author_m = re.search(r'\((\w+),', cell)
    first_author = author_m.group(1) if author_m else ''

    # Title hint = everything before the parenthesis block
    title_hint = re.sub(r'\s*\(.*', '', cell).strip()
    title_hint = re.sub(r'\s+', ' ', title_hint)

    return title_hint, first_author, year


def build_query(title_hint: str, author: str, year: str) -> str:
    """Combine title acronym / short name with author + year for a tight query."""
    parts = [title_hint, author, year]
    return ' '.join(p for p in parts if p)


def search_semantic_scholar(query: str):
    """Return the top-1 paper JSON from Semantic Scholar, or None.
    Retries automatically on HTTP 429 with exponential backoff."""
    params = {'query': query, 'fields': FIELDS, 'limit': 1}
    wait = 5  # initial backoff seconds for 429
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(API_BASE, params=params, timeout=15)
            if r.status_code == 429:
                print(f"  [rate limit] waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait)
                wait = min(wait * 2, 120)  # cap at 2 minutes
                continue
            r.raise_for_status()
            data = r.json()
            hits = data.get('data', [])
            return hits[0] if hits else None
        except requests.HTTPError as e:
            print(f"  [HTTP error] {e}")
            return None
        except Exception as e:
            print(f"  [error] {e}")
            return None
    print(f"  [failed after {MAX_RETRIES} retries]")
    return None


def paper_to_bibtex(paper: dict, cite_key: str) -> str:
    """
    Use the citationStyles.bibtex field if available.
    Otherwise, build a minimal @article entry from available fields.
    """
    bib = (paper.get('citationStyles') or {}).get('bibtex', '')
    if bib:
        # Replace the auto-generated cite key with our own
        bib = re.sub(r'@(\w+)\{[^,]+,', f'@\\1{{{cite_key},', bib, count=1)
        return bib.strip()

    # Manual fallback
    authors = ' and '.join(
        a.get('name', '') for a in paper.get('authors', [])
    )
    title  = paper.get('title', 'Unknown Title')
    year   = paper.get('year', '')
    venue  = paper.get('venue', '')
    doi    = (paper.get('externalIds') or {}).get('DOI', '')
    arxiv  = (paper.get('externalIds') or {}).get('ArXiv', '')

    lines = [f'@article{{{cite_key},']
    lines.append(f'  author  = {{{authors}}},')
    lines.append(f'  title   = {{{{{title}}}}},')
    if year:   lines.append(f'  year    = {{{year}}},')
    if venue:  lines.append(f'  journal = {{{venue}}},')
    if doi:    lines.append(f'  doi     = {{{doi}}},')
    if arxiv:  lines.append(f'  url     = {{https://arxiv.org/abs/{arxiv}}},')
    lines.append('}')
    return '\n'.join(lines)


def make_cite_key(title_hint: str, author: str, year: str) -> str:
    """e.g. 'BALSAM', 'Almatham', '2025'  ->  'BALSAM_Almatham2025'"""
    safe_title = re.sub(r'[^A-Za-z0-9]', '', title_hint.split()[0])
    return f'{safe_title}_{author}{year}'


def main():
    df = pd.read_csv(CSV_PATH)
    entries = []
    not_found = []

    print(f"\nFetching BibTeX for {len(df)} rows...")
    print("Waiting 20s for Semantic Scholar rate-limit window to reset...\n")
    time.sleep(20)

    for _, row in df.iterrows():
        cell = str(row['Paper (Author, Year)'])
        parsed = parse_paper_cell(cell)
        if parsed is None:
            print(f"  [skip] {cell[:50]}")
            continue

        title_hint, author, year = parsed
        cite_key   = make_cite_key(title_hint, author, year)
        words      = title_hint.lower().split()
        # Try longest prefix match first (up to 5 words), then shorter
        title_key  = next(
            (' '.join(words[:n]) for n in range(min(5, len(words)), 0, -1)
             if ' '.join(words[:n]) in KNOWN_TITLES),
            None
        )

        if title_key is not None:
            override = KNOWN_TITLES[title_key]  # may be str or None
            if override is None:
                # Explicitly skipped (not yet indexed)
                print(f"  [skip-api] {title_hint} — not yet indexed")
                placeholder = (
                    f'@misc{{{cite_key},\n'
                    f'  author = {{{author}}},\n'
                    f'  title  = {{{{{title_hint}}}}},\n'
                    f'  year   = {{{year}}},\n'
                    f'  note   = {{NOT FOUND — fill manually}},\n'
                    f'}}'
                )
                entries.append(placeholder)
                not_found.append(cite_key)
                continue
            query = override
        else:
            query = build_query(title_hint, author, year)

        print(f"  Searching: {query!r}")
        paper = search_semantic_scholar(query)
        time.sleep(SLEEP_SEC)

        if paper:
            bib = paper_to_bibtex(paper, cite_key)
            entries.append(bib)
            found_title = paper.get('title', '')[:60]
            print(f"    -> Found: {found_title}")
        else:
            # Write a placeholder so the .bib compiles without errors
            placeholder = (
                f'@misc{{{cite_key},\n'
                f'  author = {{{author}}},\n'
                f'  title  = {{{{{title_hint}}}}},\n'
                f'  year   = {{{year}}},\n'
                f'  note   = {{NOT FOUND — fill manually}},\n'
                f'}}'
            )
            entries.append(placeholder)
            not_found.append(cite_key)
            print(f"    -> Not found — placeholder written")

    # Write .bib file
    out = Path(OUT_BIB)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n\n'.join(entries) + '\n', encoding='utf-8')

    print(f"\nDone.  {len(entries)} entries -> {OUT_BIB}")
    if not_found:
        print(f"Placeholders (fill manually): {not_found}")


if __name__ == '__main__':
    main()
