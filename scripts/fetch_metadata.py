#!/usr/bin/env python3
"""
Fetch metadata for papers in the literature review.

Supports:
- Arxiv papers (via arxiv API)
- ACL Anthology papers (via requests + BeautifulSoup)
- Semantic Scholar (for citation counts)
- GitHub (for code availability check)
"""

import pandas as pd
import arxiv
import requests
from bs4 import BeautifulSoup
import time
import json
from pathlib import Path
from typing import Optional, Dict
import re

class PaperMetadataFetcher:
    def __init__(self, csv_path: str = "data/papers.csv"):
        self.df = pd.read_csv(csv_path)
        self.metadata_dir = Path("data/metadata")
        self.metadata_dir.mkdir(exist_ok=True, parents=True)
        
        # Cache to avoid re-fetching
        self.cache_file = self.metadata_dir / "fetch_cache.json"
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def extract_arxiv_id(self, url: str) -> Optional[str]:
        """Extract arxiv ID from URL."""
        if not url or pd.isna(url):
            return None
        match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)', url)
        return match.group(1) if match else None
    
    def extract_acl_id(self, url: str) -> Optional[str]:
        """Extract ACL anthology ID from URL."""
        if not url or pd.isna(url):
            return None
        match = re.search(r'aclanthology\.org/([^/]+)/', url)
        return match.group(1) if match else None
    
    def fetch_arxiv_metadata(self, arxiv_id: str) -> Dict:
        """Fetch metadata from Arxiv."""
        if arxiv_id in self.cache:
            return self.cache[arxiv_id]
        
        try:
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(client.results(search))
            
            metadata = {
                'title': paper.title,
                'authors': [a.name for a in paper.authors],
                'abstract': paper.summary,
                'published': paper.published.strftime('%Y-%m-%d'),
                'updated': paper.updated.strftime('%Y-%m-%d'),
                'categories': paper.categories,
                'pdf_url': paper.pdf_url,
                'source': 'arxiv'
            }
            
            self.cache[arxiv_id] = metadata
            self._save_cache()
            time.sleep(1)  # Rate limiting
            return metadata
        except Exception as e:
            print(f"Error fetching arxiv {arxiv_id}: {e}")
            return {}
    
    def fetch_acl_metadata(self, acl_id: str) -> Dict:
        """Fetch metadata from ACL Anthology."""
        if acl_id in self.cache:
            return self.cache[acl_id]
        
        try:
            url = f"https://aclanthology.org/{acl_id}/"
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h2', id='title')
            title = title_elem.text.strip() if title_elem else None
            
            # Extract authors
            authors = []
            for author in soup.find_all('a', href=re.compile(r'/people/')):
                authors.append(author.text.strip())
            
            # Extract abstract
            abstract_elem = soup.find('div', class_='acl-abstract')
            abstract = abstract_elem.text.strip() if abstract_elem else None
            
            # Extract PDF URL
            pdf_elem = soup.find('a', class_='badge-primary', string=re.compile('PDF'))
            pdf_url = pdf_elem['href'] if pdf_elem else None
            
            metadata = {
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'pdf_url': pdf_url,
                'source': 'acl'
            }
            
            self.cache[acl_id] = metadata
            self._save_cache()
            time.sleep(1)
            return metadata
        except Exception as e:
            print(f"Error fetching ACL {acl_id}: {e}")
            return {}
    
    def fetch_semantic_scholar(self, title: str) -> Dict:
        """Fetch citation count from Semantic Scholar."""
        cache_key = f"s2_{title}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {'query': title, 'limit': 1}
            response = requests.get(api_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    paper_id = data['data'][0]['paperId']
                    
                    # Get full paper details
                    detail_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
                    params = {'fields': 'citationCount,year,venue'}
                    detail_response = requests.get(detail_url, params=params, timeout=10)
                    
                    if detail_response.status_code == 200:
                        metadata = detail_response.json()
                        self.cache[cache_key] = metadata
                        self._save_cache()
                        time.sleep(1)
                        return metadata
            
            return {}
        except Exception as e:
            print(f"Error fetching S2 for {title}: {e}")
            return {}
    
    def check_github_code(self, repo_url: str) -> Dict:
        """Check if GitHub repo exists and get metadata."""
        if not repo_url or pd.isna(repo_url):
            return {'exists': False}
        
        try:
            # Extract owner/repo from URL
            match = re.search(r'github\.com/([^/]+/[^/]+)', repo_url)
            if not match:
                return {'exists': False}
            
            repo_path = match.group(1).rstrip('/')
            api_url = f"https://api.github.com/repos/{repo_path}"
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'exists': True,
                    'stars': data.get('stargazers_count', 0),
                    'forks': data.get('forks_count', 0),
                    'last_updated': data.get('updated_at'),
                    'language': data.get('language')
                }
            return {'exists': False}
        except Exception as e:
            print(f"Error checking GitHub {repo_url}: {e}")
            return {'exists': False}
    
    def process_all_papers(self, save_individual: bool = True):
        """Process all papers in the CSV."""
        results = []
        
        for idx, row in self.df.iterrows():
            paper_id = row['Paper']
            print(f"\nProcessing {idx+1}/{len(self.df)}: {paper_id}")
            
            metadata = {
                'paper_id': paper_id,
                'year': row['Year'],
                'category': row['Category'],
            }
            
            # Fetch from appropriate source
            arxiv_id = self.extract_arxiv_id(row.get('Paper_URL', ''))
            acl_id = self.extract_acl_id(row.get('Paper_URL', ''))
            
            if arxiv_id:
                print(f"  Fetching from Arxiv: {arxiv_id}")
                arxiv_meta = self.fetch_arxiv_metadata(arxiv_id)
                metadata.update(arxiv_meta)
            elif acl_id:
                print(f"  Fetching from ACL: {acl_id}")
                acl_meta = self.fetch_acl_metadata(acl_id)
                metadata.update(acl_meta)
            
            # Get citation count
            if metadata.get('title'):
                print(f"  Fetching citations from Semantic Scholar")
                s2_meta = self.fetch_semantic_scholar(metadata['title'])
                if s2_meta:
                    metadata['citation_count'] = s2_meta.get('citationCount', 0)
                    metadata['s2_venue'] = s2_meta.get('venue')
            
            # Check code availability
            if row.get('Code_URL') and not pd.isna(row['Code_URL']):
                print(f"  Checking GitHub repo")
                github_meta = self.check_github_code(row['Code_URL'])
                metadata['github'] = github_meta
            
            results.append(metadata)
            
            # Save individual JSON
            if save_individual:
                output_file = self.metadata_dir / f"{paper_id.replace(' ', '_')}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Save combined results
        combined_file = self.metadata_dir / "all_papers_metadata.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Processed {len(results)} papers")
        print(f"✓ Metadata saved to {combined_file}")
        
        return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch paper metadata")
    parser.add_argument('--csv', default='data/papers.csv', help='Input CSV file')
    parser.add_argument('--skip-cache', action='store_true', help='Ignore cache')
    args = parser.parse_args()
    
    fetcher = PaperMetadataFetcher(args.csv)
    
    if args.skip_cache:
        fetcher.cache = {}
    
    fetcher.process_all_papers()
