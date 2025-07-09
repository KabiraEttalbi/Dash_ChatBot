#!/usr/bin/env python3
"""
Data Cleaning and Processing Script

This script cleans and processes the extracted arXiv data,
preparing it for semantic indexing and chatbot training.
"""

import json
import pandas as pd
import re
import os
from typing import List, Dict, Any
from datetime import datetime
import argparse

class DataCleaner:
    """Class to clean and process arXiv data"""
    
    def __init__(self):
        """Initialize the DataCleaner"""
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
        }
    
    def load_data(self, file_path: str) -> List[Dict]:
        """Load data from JSON or CSV file"""
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        else:
            raise ValueError("Unsupported file format. Use JSON or CSV.")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text or pd.isna(text):
            return ""
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-$$$$]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Extract keywords from text"""
        if not text:
            return []
        
        # Convert to lowercase and split
        words = text.lower().split()
        
        # Filter out stop words and short words
        keywords = [
            word.strip('.,!?;:()[]{}') 
            for word in words 
            if len(word) >= min_length and word.lower() not in self.stop_words
        ]
        
        return list(set(keywords))  # Remove duplicates
    
    def process_authors(self, authors_data: Any) -> Dict:
        """Process authors information"""
        if isinstance(authors_data, str):
            # If authors is a string (from CSV), split it
            author_names = [name.strip() for name in authors_data.split(';') if name.strip()]
            return {
                'names': author_names,
                'count': len(author_names),
                'affiliations': []
            }
        elif isinstance(authors_data, list):
            # If authors is a list (from JSON)
            names = [author.get('name', '') for author in authors_data if author.get('name')]
            affiliations = [author.get('affiliation', '') for author in authors_data if author.get('affiliation')]
            return {
                'names': names,
                'count': len(names),
                'affiliations': list(set(filter(None, affiliations)))
            }
        else:
            return {'names': [], 'count': 0, 'affiliations': []}
    
    def process_categories(self, categories_data: Any) -> Dict:
        """Process categories information"""
        if isinstance(categories_data, str):
            # If categories is a string (from CSV)
            category_terms = [cat.strip() for cat in categories_data.split(';') if cat.strip()]
            return {
                'terms': category_terms,
                'primary': category_terms[0] if category_terms else '',
                'count': len(category_terms)
            }
        elif isinstance(categories_data, list):
            # If categories is a list (from JSON)
            terms = [cat.get('term', '') for cat in categories_data if cat.get('term')]
            primary = next((cat.get('term', '') for cat in categories_data if cat.get('primary')), '')
            return {
                'terms': terms,
                'primary': primary,
                'count': len(terms)
            }
        else:
            return {'terms': [], 'primary': '', 'count': 0}
    
    def clean_paper(self, paper: Dict) -> Dict:
        """Clean a single paper's data"""
        cleaned_paper = {}
        
        # Basic information
        cleaned_paper['arxiv_id'] = paper.get('arxiv_id', '')
        cleaned_paper['title'] = self.clean_text(paper.get('title', ''))
        cleaned_paper['summary'] = self.clean_text(paper.get('summary', ''))
        cleaned_paper['published'] = paper.get('published', '')
        cleaned_paper['updated'] = paper.get('updated', '')
        
        # Process authors
        authors_info = self.process_authors(paper.get('authors', []))
        cleaned_paper['authors'] = authors_info
        
        # Process categories
        categories_info = self.process_categories(paper.get('categories', []))
        cleaned_paper['categories'] = categories_info
        
        # Extract keywords from title and summary
        title_keywords = self.extract_keywords(cleaned_paper['title'])
        summary_keywords = self.extract_keywords(cleaned_paper['summary'])
        cleaned_paper['keywords'] = list(set(title_keywords + summary_keywords))
        
        # Additional metadata
        cleaned_paper['doi'] = paper.get('doi', '')
        cleaned_paper['journal_ref'] = paper.get('journal_ref', '')
        cleaned_paper['pdf_url'] = paper.get('pdf_url', '')
        cleaned_paper['abstract_url'] = paper.get('abstract_url', '')
        
        # Add processing timestamp
        cleaned_paper['processed_at'] = datetime.now().isoformat()
        
        # Create combined text for semantic search
        combined_text = f"{cleaned_paper['title']} {cleaned_paper['summary']}"
        cleaned_paper['combined_text'] = combined_text
        
        return cleaned_paper
    
    def clean_dataset(self, papers: List[Dict]) -> List[Dict]:
        """Clean the entire dataset"""
        print(f"Cleaning {len(papers)} papers...")
        
        cleaned_papers = []
        for i, paper in enumerate(papers):
            try:
                cleaned_paper = self.clean_paper(paper)
                
                # Validate required fields
                if cleaned_paper['title'] and cleaned_paper['summary']:
                    cleaned_papers.append(cleaned_paper)
                else:
                    print(f"Skipping paper {i+1}: Missing title or summary")
                    
            except Exception as e:
                print(f"Error cleaning paper {i+1}: {e}")
                continue
        
        print(f"✅ Cleaned {len(cleaned_papers)} papers successfully")
        return cleaned_papers
    
    def save_cleaned_data(self, papers: List[Dict], output_file: str) -> None:
        """Save cleaned data"""
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        if output_file.endswith('.json'):
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(papers, f, ensure_ascii=False, indent=2)
        elif output_file.endswith('.csv'):
            # Flatten for CSV
            flattened_papers = []
            for paper in papers:
                flat_paper = {
                    'arxiv_id': paper['arxiv_id'],
                    'title': paper['title'],
                    'summary': paper['summary'],
                    'published': paper['published'],
                    'updated': paper['updated'],
                    'authors_names': '; '.join(paper['authors']['names']),
                    'authors_count': paper['authors']['count'],
                    'affiliations': '; '.join(paper['authors']['affiliations']),
                    'categories': '; '.join(paper['categories']['terms']),
                    'primary_category': paper['categories']['primary'],
                    'keywords': '; '.join(paper['keywords']),
                    'doi': paper['doi'],
                    'journal_ref': paper['journal_ref'],
                    'pdf_url': paper['pdf_url'],
                    'abstract_url': paper['abstract_url'],
                    'combined_text': paper['combined_text'],
                    'processed_at': paper['processed_at']
                }
                flattened_papers.append(flat_paper)
            
            df = pd.DataFrame(flattened_papers)
            df.to_csv(output_file, index=False)
        
        print(f"✅ Saved cleaned data to {output_file}")
    
    def generate_statistics(self, papers: List[Dict]) -> Dict:
        """Generate statistics about the cleaned dataset"""
        if not papers:
            return {}
        
        stats = {
            'total_papers': len(papers),
            'date_range': {
                'earliest': min(paper['published'] for paper in papers if paper['published']),
                'latest': max(paper['published'] for paper in papers if paper['published'])
            },
            'categories': {},
            'authors_stats': {
                'total_unique_authors': len(set(
                    author for paper in papers 
                    for author in paper['authors']['names']
                )),
                'avg_authors_per_paper': sum(paper['authors']['count'] for paper in papers) / len(papers)
            },
            'text_stats': {
                'avg_title_length': sum(len(paper['title']) for paper in papers) / len(papers),
                'avg_summary_length': sum(len(paper['summary']) for paper in papers) / len(papers),
                'avg_keywords_per_paper': sum(len(paper['keywords']) for paper in papers) / len(papers)
            }
        }
        
        # Category statistics
        category_counts = {}
        for paper in papers:
            for category in paper['categories']['terms']:
                category_counts[category] = category_counts.get(category, 0) + 1
        
        stats['categories'] = dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return stats

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Clean arXiv data')
    parser.add_argument('--input', type=str, required=True, help='Input file path')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--stats', action='store_true', help='Generate statistics')
    
    args = parser.parse_args()
    
    cleaner = DataCleaner()
    
    # Load data
    print(f"Loading data from {args.input}...")
    papers = cleaner.load_data(args.input)
    
    # Clean data
    cleaned_papers = cleaner.clean_dataset(papers)
    
    # Save cleaned data
    cleaner.save_cleaned_data(cleaned_papers, args.output)
    
    # Generate statistics if requested
    if args.stats:
        stats = cleaner.generate_statistics(cleaned_papers)
        stats_file = args.output.replace('.json', '_stats.json').replace('.csv', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"✅ Statistics saved to {stats_file}")

if __name__ == "__main__":
    main()
