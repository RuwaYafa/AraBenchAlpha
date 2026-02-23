#!/usr/bin/env python3
"""
AraBenchAlpha - Paper Metadata Fetcher
Automatically fetches metadata from Arxiv, ACL Anthology, and Semantic Scholar
for papers in the literature review.

Usage:
    python scripts/fetch_papers.py --input data/papers.csv --output data/papers_enriched.csv
"""

import pandas as pd
import requests
from typing import Optional, Dict, List
import time
import re
from pathlib import Path
import argparse

# ═══════════════════════════════════════════════════════════════════════════
# ARXIV FETCHER
# ═══════════════════════════════════════════════════════════════════════════

def fetch_arxiv_metadata(arxiv_id: str) -> Optional[Dict]:
    """
    Fetch paper metadata from Arxiv API.
    
    Args:
        arxiv_id: Arxiv ID (e.g., "2507.22603")
    
    Returns:
        Dictionary with title, authors, abstract, pdf_url, published_date
    """
    try:
        import arxiv
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(client.results(search), None)
        
        if paper:
            return {
                'title': paper.title,
                'authors': ', '.join([a.name for a in paper.authors]),
                'abstract': paper.summary,
                'pdf_url': paper.pdf_url,
                'published': paper.published.strftime('%Y-%m-%d'),
                'arxiv_id': arxiv_id,
            }
    except ImportError:
        print("Warning: arxiv package not installed. Install with: pip install arxiv")
    except Exception as e:
        print(f"Error fetching arxiv {arxiv_id}: {e}")
    
    return None


# ═══════════════════════════════════════════════════════════════════════════
# ACL ANTHOLOGY FETCHER
# ═══════════════════════════════════════════════════════════════════════════

def fetch_acl_metadata(acl_id: str) -> Optional[Dict]:
    """
    Fetch paper metadata from ACL Anthology.
    
    Args:
        acl_id: ACL Anthology ID (e.g., "2025.arabicnlp-main.21")
    
    Returns:
        Dictionary with title, authors, venue, year, pdf_url
    """
    base_url = "https://aclanthology.org"
    paper_url = f"{base_url}/{acl_id}/"
    
    try:
        response = requests.get(paper_url, timeout=10)
        if response.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title_tag = soup.find('h2', {'id': 'title'})
            authors_div = soup.find('p', {'class': 'lead'})
            
            return {
                'title': title_tag.get_text(strip=True) if title_tag else None,
                'authors': authors_div.get_text(strip=True) if authors_div else None,
                'acl_id': acl_id,
                'pdf_url': f"{base_url}/{acl_id}.pdf",
                'venue': acl_id.split('.')[0],  # e.g., "2025"
            }
    except ImportError:
        print("Warning: beautifulsoup4 not installed. Install with: pip install beautifulsoup4")
    except Exception as e:
        print(f"Error fetching ACL {acl_id}: {e}")
    
    return None


# ═══════════════════════════════════════════════════════════════════════════
# SEMANTIC SCHOLAR FETCHER
# ═══════════════════════════════════════════════════════════════════════════

def fetch_semantic_scholar(title: str) -> Optional[Dict]:
    """
    Fetch paper metadata from Semantic Scholar API.
    
    Args:
        title: Paper title
    
    Returns:
        Dictionary with citations, influential_citations, year, venue
    """
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        'query': title,
        'limit': 1,
        'fields': 'title,year,citationCount,influentialCitationCount,venue,authors'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('data') and len(data['data']) > 0:
                paper = data['data'][0]
                return {
                    'citations': paper.get('citationCount', 0),
                    'influential_citations': paper.get('influentialCitationCount', 0),
                    'year': paper.get('year'),
                    'venue': paper.get('venue'),
                }
    except Exception as e:
        print(f"Error fetching Semantic Scholar for '{title[:50]}...': {e}")
    
    return None


# ═══════════════════════════════════════════════════════════════════════════
# GITHUB CODE AVAILABILITY CHECKER
# ═══════════════════════════════════════════════════════════════════════════

def check_github_code(paper_title: str, author_name: Optional[str] = None) -> Optional[str]:
    """
    Search GitHub for code repositories related to the paper.
    
    Args:
        paper_title: Paper title
        author_name: Optional first author name
    
    Returns:
        GitHub repo URL if found, None otherwise
    """
    # Clean title for search
    clean_title = re.sub(r'[^\w\s]', '', paper_title).strip()
    query = f"{clean_title} {author_name}" if author_name else clean_title
    
    api_url = "https://api.github.com/search/repositories"
    params = {
        'q': query,
        'sort': 'stars',
        'order': 'desc',
        'per_page': 1
    }
    
    try:
        response = requests.get(api_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('items') and len(data['items']) > 0:
                return data['items'][0]['html_url']
    except Exception as e:
        print(f"Error checking GitHub for '{paper_title[:50]}...': {e}")
    
    return None


# ═══════════════════════════════════════════════════════════════════════════
# PAPERS WITH CODE CHECKER
# ═══════════════════════════════════════════════════════════════════════════

def check_papers_with_code(arxiv_id: Optional[str] = None, title: Optional[str] = None) -> Optional[Dict]:
    """
    Check if paper has code on Papers With Code.
    
    Args:
        arxiv_id: Arxiv ID (preferred)
        title: Paper title (fallback)
    
    Returns:
        Dictionary with code_url, dataset_url if found
    """
    if arxiv_id:
        # Papers With Code uses arxiv IDs without version
        clean_id = arxiv_id.split('v')[0]
        url = f"https://paperswithcode.com/api/v1/papers/arxiv:{clean_id}"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    'code_available': True,
                    'pwc_url': f"https://paperswithcode.com/paper/{data.get('url_abs', '')}",
                }
        except Exception as e:
            print(f"Error checking PapersWithCode for {arxiv_id}: {e}")
    
    return {'code_available': False}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENRICHMENT FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def enrich_paper_metadata(row: pd.Series, rate_limit_delay: float = 1.0) -> pd.Series:
    """
    Enrich a single paper row with metadata from multiple sources.
    
    Args:
        row: Pandas Series with at least 'Paper' and optionally 'arxiv_id', 'acl_id'
        rate_limit_delay: Seconds to wait between API calls
    
    Returns:
        Enhanced Series with additional metadata columns
    """
    enriched = row.copy()
    
    # Extract IDs from the row if available
    arxiv_id = row.get('arxiv_id') or row.get('Arxiv_ID')
    acl_id = row.get('acl_id') or row.get('ACL_ID')
    paper_title = row.get('Paper') or row.get('Title')
    
    # Fetch from Arxiv
    if arxiv_id and pd.notna(arxiv_id):
        print(f"  Fetching Arxiv: {arxiv_id}")
        arxiv_data = fetch_arxiv_metadata(str(arxiv_id))
        if arxiv_data:
            enriched['Arxiv_Title'] = arxiv_data.get('title')
            enriched['Arxiv_Authors'] = arxiv_data.get('authors')
            enriched['Arxiv_PDF'] = arxiv_data.get('pdf_url')
            enriched['Published_Date'] = arxiv_data.get('published')
        time.sleep(rate_limit_delay)
    
    # Fetch from ACL Anthology
    if acl_id and pd.notna(acl_id):
        print(f"  Fetching ACL: {acl_id}")
        acl_data = fetch_acl_metadata(str(acl_id))
        if acl_data:
            enriched['ACL_Title'] = acl_data.get('title')
            enriched['ACL_Authors'] = acl_data.get('authors')
            enriched['ACL_PDF'] = acl_data.get('pdf_url')
            enriched['Venue'] = acl_data.get('venue')
        time.sleep(rate_limit_delay)
    
    # Fetch from Semantic Scholar
    if paper_title and pd.notna(paper_title):
        print(f"  Fetching Semantic Scholar: {paper_title[:50]}...")
        ss_data = fetch_semantic_scholar(str(paper_title))
        if ss_data:
            enriched['Citations'] = ss_data.get('citations', 0)
            enriched['Influential_Citations'] = ss_data.get('influential_citations', 0)
            if not enriched.get('Venue'):
                enriched['Venue'] = ss_data.get('venue')
        time.sleep(rate_limit_delay)
    
    # Check GitHub
    author = enriched.get('Arxiv_Authors') or enriched.get('ACL_Authors')
    if author:
        first_author = str(author).split(',')[0].strip()
    else:
        first_author = None
    
    if paper_title:
        github_url = check_github_code(str(paper_title), first_author)
        if github_url:
            enriched['GitHub_URL'] = github_url
        time.sleep(rate_limit_delay)
    
    # Check Papers With Code
    if arxiv_id and pd.notna(arxiv_id):
        pwc_data = check_papers_with_code(arxiv_id=str(arxiv_id))
        if pwc_data:
            enriched['Code_Available'] = pwc_data.get('code_available', False)
            enriched['PapersWithCode_URL'] = pwc_data.get('pwc_url')
        time.sleep(rate_limit_delay)
    
    return enriched


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Fetch and enrich paper metadata')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file with papers')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file with enriched metadata')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between API calls (seconds)')
    
    args = parser.parse_args()
    
    # Load papers
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {args.input} not found")
        return
    
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} papers from {args.input}")
    
    # Enrich each paper
    enriched_rows = []
    for idx, row in df.iterrows():
        print(f"\n[{idx+1}/{len(df)}] Processing: {row.get('Paper', 'Unknown')[:60]}...")
        enriched = enrich_paper_metadata(row, rate_limit_delay=args.delay)
        enriched_rows.append(enriched)
    
    # Save enriched dataset
    enriched_df = pd.DataFrame(enriched_rows)
    enriched_df.to_csv(args.output, index=False)
    print(f"\n✓ Saved enriched dataset to {args.output}")
    print(f"  Added columns: {list(set(enriched_df.columns) - set(df.columns))}")


if __name__ == "__main__":
    main()
