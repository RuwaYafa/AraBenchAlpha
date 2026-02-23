#!/usr/bin/env python3
"""Check code and data availability for all papers."""

import pandas as pd
import requests
import time
from pathlib import Path
import json

class ArtifactChecker:
    def __init__(self, csv_path: str = "data/papers.csv"):
        self.df = pd.read_csv(csv_path)
        self.results = []
    
    def check_url_accessible(self, url: str, timeout: int = 5) -> bool:
        """Check if URL is accessible."""
        if not url or pd.isna(url) or url == '':
            return False
        try:
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            return response.status_code < 400
        except:
            return False
    
    def check_github_repo(self, url: str) -> dict:
        """Check GitHub repo existence and stats."""
        if not url or pd.isna(url):
            return {'exists': False}
        
        try:
            import re
            match = re.search(r'github\.com/([^/]+/[^/]+)', url)
            if not match:
                return {'exists': False}
            
            repo = match.group(1).rstrip('/')
            api_url = f"https://api.github.com/repos/{repo}"
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'exists': True,
                    'stars': data.get('stargazers_count', 0),
                    'forks': data.get('forks_count', 0),
                    'last_updated': data.get('updated_at', 'N/A')
                }
            return {'exists': False}
        except Exception as e:
            print(f"  Error checking {url}: {e}")
            return {'exists': False}
    
    def check_all(self):
        """Check all papers."""
        print("\n🔍 Checking artifact availability...\n")
        
        for idx, row in self.df.iterrows():
            paper = row['Paper']
            print(f"[{idx+1}/{len(self.df)}] {paper}")
            
            result = {
                'paper': paper,
                'year': row['Year'],
                'category': row['Category'],
                'paper_url_accessible': self.check_url_accessible(row.get('Paper_URL')),
                'code_available': False,
                'code_stats': {},
                'data_available': self.check_url_accessible(row.get('Data_URL'))
            }
            
            if row.get('Code_URL') and not pd.isna(row['Code_URL']):
                github_info = self.check_github_repo(row['Code_URL'])
                result['code_available'] = github_info['exists']
                result['code_stats'] = github_info
                if github_info['exists']:
                    print(f"  ✓ Code: ⭐{github_info['stars']} forks:{github_info['forks']}")
            
            self.results.append(result)
            time.sleep(0.5)  # Rate limiting
        
        # Save results
        output_file = Path("data/artifact_availability.json")
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Summary
        total = len(self.results)
        code_available = sum(1 for r in self.results if r['code_available'])
        data_available = sum(1 for r in self.results if r['data_available'])
        
        print(f"\n📊 Summary:")
        print(f"  Total papers: {total}")
        print(f"  Code available: {code_available} ({code_available/total*100:.1f}%)")
        print(f"  Data available: {data_available} ({data_available/total*100:.1f}%)")
        print(f"\n✓ Results saved to {output_file}")

if __name__ == "__main__":
    checker = ArtifactChecker()
    checker.check_all()
