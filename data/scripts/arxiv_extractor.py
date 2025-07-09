#!/usr/bin/env python3
"""
arXiv Data Extractor

This script extracts scientific papers from the arXiv API based on search queries,
processes the data, and saves it in structured formats (JSON/CSV).

Usage:
    python arxiv_extractor.py --query "quantum computing" --max_results 100 --output ../data_source/quantum_papers.json
"""

import os
import time
import json
import argparse
import csv
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import pandas as pd
import re
from collections import Counter
import sys

# Load environment variables
load_dotenv()

class ArxivExtractor:
    """Class to extract data from arXiv API with large-scale processing capabilities"""
    
    def __init__(self):
        """Initialize the ArxivExtractor with base URL and namespaces"""
        self.base_url = "http://export.arxiv.org/api/query"
        self.namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        self.session = requests.Session()  # Reuse connection
        
    def extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract potential keywords from text using simple NLP techniques"""
        if not text:
            return []
        
        # Common stop words to filter out
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'we', 'they', 'them', 'their', 'our', 'your', 'his', 'her', 'its'
        }
        
        # Extract words, filter by length and stop words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [word for word in words if word not in stop_words]
        
        # Get unique keywords and sort by frequency
        keyword_counts = Counter(keywords)
        
        # Return top keywords (limit to avoid too many)
        return [word for word, count in keyword_counts.most_common(20)]
        
    def search_papers(self, query: str, start: int = 0, max_results: int = 10) -> str:
        """
        Search for papers on arXiv based on the query
        
        Args:
            query: Search query string
            start: Starting index for results
            max_results: Maximum number of results to return (max 100 per query)
            
        Returns:
            XML response string
        """
        params = {
            'search_query': query,
            'start': start,
            'max_results': min(max_results, 100)  # API limit is 100 per request
        }
        
        try:
            print(f"Fetching results {start} to {start + min(max_results, 100)} for query: {query}")
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from arXiv API: {e}")
            return None
    
    def parse_response(self, response_text: str) -> List[Dict]:
        """Parse the XML response from arXiv API with enhanced data extraction"""
        if not response_text:
            return []
    
        root = ET.fromstring(response_text)
        papers = []
    
        for entry in root.findall('.//atom:entry', self.namespaces):
            paper = {}
        
            # Extract basic metadata
            paper['id'] = self._get_text(entry, './atom:id')
            paper['arxiv_id'] = paper['id'].split('/')[-1] if paper['id'] else None
        
            # Clean and extract title and summary
            title = self._get_text(entry, './atom:title').replace('\n', ' ').strip()
            summary = self._get_text(entry, './atom:summary').replace('\n', ' ').strip()
        
            paper['title'] = title
            paper['summary'] = summary
            paper['published'] = self._get_text(entry, './atom:published')
            paper['updated'] = self._get_text(entry, './atom:updated')
        
            # Extract keywords from title and summary
            title_keywords = self.extract_keywords_from_text(title)
            summary_keywords = self.extract_keywords_from_text(summary)
        
            # Combine and deduplicate keywords
            all_keywords = list(set(title_keywords + summary_keywords))
            paper['extracted_keywords'] = all_keywords
        
            # Extract authors with more details
            authors = []
            for author in entry.findall('./atom:author', self.namespaces):
                name = self._get_text(author, './atom:name')
                affiliation = self._get_text(author, './arxiv:affiliation', default='')
                authors.append({
                    'name': name,
                    'affiliation': affiliation
                })
            paper['authors'] = authors
        
            # Extract categories with more structure
            categories = []
            primary_category = entry.find('./arxiv:primary_category', self.namespaces)
            if primary_category is not None:
                primary = primary_category.get('term')
                if primary:
                    categories.append({'term': primary, 'primary': True})
        
            for category in entry.findall('./atom:category', self.namespaces):
                term = category.get('term')
                if term and term not in [c['term'] for c in categories]:
                    categories.append({'term': term, 'primary': False})
        
            paper['categories'] = categories
        
            # Extract all links
            links = []
            pdf_url = None
            abstract_url = None
        
            for link in entry.findall('./atom:link', self.namespaces):
                href = link.get('href')
                rel = link.get('rel')
                link_type = link.get('type')
            
                links.append({
                    'href': href,
                    'rel': rel,
                    'type': link_type
                })
            
                # Extract specific URLs for easy access
                if link_type == 'application/pdf':
                    pdf_url = href
                elif rel == 'alternate' and link_type == 'text/html':
                    abstract_url = href
        
            paper['links'] = links
            paper['pdf_url'] = pdf_url
            paper['abstract_url'] = abstract_url
        
            # Extract additional metadata
            paper['doi'] = self._get_text(entry, './arxiv:doi', default=None)
            paper['journal_ref'] = self._get_text(entry, './arxiv:journal_ref', default=None)
            paper['comment'] = self._get_text(entry, './arxiv:comment', default=None)
        
            # Add extraction timestamp and text length metrics
            paper['extracted_at'] = datetime.now().isoformat()
            paper['title_length'] = len(title)
            paper['summary_length'] = len(summary)
            paper['total_text_length'] = len(title) + len(summary)
        
            # Create combined searchable text
            combined_text = f"{title} {summary}"
            if paper['comment']:
                combined_text += f" {paper['comment']}"
            paper['combined_text'] = combined_text
        
            papers.append(paper)
    
        return papers
    
    def _get_text(self, element: ET.Element, xpath: str, default: str = '') -> str:
        """Helper method to extract text from XML element"""
        result = element.find(xpath, self.namespaces)
        return result.text.strip() if result is not None and result.text else default
    
    def extract_papers(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Extract papers from arXiv API with pagination
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of dictionaries containing paper information
        """
        all_papers = []
        start = 0
        
        print(f"Starting extraction for query: '{query}' (max {max_results} papers)")
        
        while len(all_papers) < max_results:
            remaining = max_results - len(all_papers)
            batch_size = min(100, remaining)  # API limit is 100 per request
            
            response_text = self.search_papers(query, start, batch_size)
            if not response_text:
                break
                
            papers = self.parse_response(response_text)
            if not papers:
                print("No more papers found")
                break
                
            all_papers.extend(papers)
            print(f"Extracted {len(all_papers)} papers so far...")
            
            # Check if we've reached the end of results
            if len(papers) < batch_size:
                break
                
            # Respect arXiv API rate limits (3 seconds between requests)
            print("Waiting 3 seconds (API rate limit)...")
            time.sleep(3)
            start += batch_size
        
        print(f"Extraction complete! Total papers extracted: {len(all_papers)}")
        return all_papers[:max_results]
    
    def save_to_json(self, papers: List[Dict], output_file: str) -> None:
        """Save papers to JSON file"""
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(papers, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved {len(papers)} papers to {output_file}")
    
    def save_to_csv(self, papers: List[Dict], output_file: str) -> None:
        """Save papers to CSV file"""
        if not papers:
            print("No papers to save")
            return
            
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Flatten the data structure for CSV
        flattened_papers = []
        for paper in papers:
            flat_paper = {
                'arxiv_id': paper['arxiv_id'],
                'title': paper['title'],
                'summary': paper['summary'],
                'published': paper['published'],
                'updated': paper['updated'],
                'authors': '; '.join([a['name'] for a in paper['authors']]),
                'affiliations': '; '.join([a['affiliation'] for a in paper['authors'] if a['affiliation']]),
                'primary_category': next((c['term'] for c in paper['categories'] if c.get('primary')), ''),
                'categories': '; '.join([c['term'] for c in paper['categories']]),
                'pdf_url': paper.get('pdf_url', ''),
                'abstract_url': paper.get('abstract_url', ''),
                'doi': paper['doi'],
                'journal_ref': paper['journal_ref'],
                'extracted_at': paper['extracted_at'],
                'extracted_keywords': '; '.join(paper['extracted_keywords']),
                'comment': paper['comment'],
                'title_length': paper['title_length'],
                'summary_length': paper['summary_length'],
                'total_text_length': paper['total_text_length'],
                'combined_text': paper['combined_text']
            }
            flattened_papers.append(flat_paper)
        
        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(flattened_papers)
        df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
        
        print(f"✅ Saved {len(papers)} papers to {output_file}")

    def extract_large_dataset(self, target_size_gb: float = 1.5, output_dir: str = "data/data_source") -> Dict:
        """
        Extract a large dataset targeting specific size
        
        Args:
            target_size_gb: Target dataset size in GB
            output_dir: Output directory for data files
            
        Returns:
            Dictionary with extraction statistics
        """
        print(f"=== Starting Large-Scale Data Extraction ===")
        print(f"Target size: {target_size_gb} GB")
        
        # Define diverse queries to get varied data
        query_categories = [
            # Computer Science categories
            "cat:cs.AI",           # Artificial Intelligence
            "cat:cs.LG",           # Machine Learning  
            "cat:cs.CV",           # Computer Vision
            "cat:cs.CL",           # Computation and Language
            "cat:cs.NE",           # Neural and Evolutionary Computing
            "cat:cs.IR",           # Information Retrieval
            "cat:cs.RO",           # Robotics
            "cat:cs.CR",           # Cryptography and Security
            
            # Physics categories
            "cat:physics.comp-ph", # Computational Physics
            "cat:physics.data-an", # Data Analysis
            "cat:quant-ph",        # Quantum Physics
            
            # Mathematics categories  
            "cat:math.ST",         # Statistics Theory
            "cat:math.OC",         # Optimization and Control
            "cat:math.PR",         # Probability
            
            # Biology categories
            "cat:q-bio.QM",        # Quantitative Methods
            "cat:q-bio.GN",        # Genomics
            
            # Popular keyword searches
            "machine learning",
            "deep learning", 
            "neural network",
            "artificial intelligence",
            "quantum computing",
            "computer vision",
            "natural language processing",
            "data science",
            "reinforcement learning",
            "transformer",
        ]
        
        all_papers = []
        total_size_mb = 0
        target_size_mb = target_size_gb * 1024
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, query in enumerate(query_categories):
            if total_size_mb >= target_size_mb:
                print(f"✅ Target size reached: {total_size_mb:.1f} MB")
                break
                
            print(f"\n--- Processing Query {i+1}/{len(query_categories)}: '{query}' ---")
            
            # Calculate how many papers to extract for this query
            remaining_mb = target_size_mb - total_size_mb
            # Estimate ~3KB per paper on average
            papers_needed = min(2000, int(remaining_mb / 0.003))  # Conservative estimate
            
            if papers_needed < 50:
                papers_needed = 50  # Minimum batch size
                
            print(f"Extracting ~{papers_needed} papers for this query...")
            
            try:
                papers = self.extract_papers(query, papers_needed)
                
                if papers:
                    # Save this batch
                    batch_filename = f"batch_{i+1:02d}_{query.replace(':', '_').replace(' ', '_')}.json"
                    batch_path = os.path.join(output_dir, batch_filename)
                    
                    self.save_to_json(papers, batch_path)
                    
                    # Calculate size
                    batch_size_mb = os.path.getsize(batch_path) / (1024 * 1024)
                    total_size_mb += batch_size_mb
                    
                    all_papers.extend(papers)
                    
                    print(f"✅ Batch saved: {len(papers)} papers, {batch_size_mb:.1f} MB")
                    print(f"📊 Total progress: {len(all_papers)} papers, {total_size_mb:.1f} MB / {target_size_mb:.1f} MB")
                    
                else:
                    print(f"⚠️ No papers found for query: {query}")
                    
            except Exception as e:
                print(f"❌ Error processing query '{query}': {e}")
                continue
        
        # Save combined dataset
        if all_papers:
            combined_path = os.path.join(output_dir, "combined_large_dataset.json")
            self.save_to_json(all_papers, combined_path)
            
            # Also save as CSV for easier analysis
            combined_csv_path = os.path.join(output_dir, "combined_large_dataset.csv") 
            self.save_to_csv(all_papers, combined_csv_path)
        
        # Generate statistics
        stats = {
            'total_papers': len(all_papers),
            'total_size_mb': total_size_mb,
            'total_size_gb': total_size_mb / 1024,
            'queries_processed': i + 1,
            'avg_papers_per_query': len(all_papers) / (i + 1) if i > 0 else 0,
            'avg_size_per_paper_kb': (total_size_mb * 1024) / len(all_papers) if all_papers else 0,
            'extraction_completed_at': datetime.now().isoformat()
        }
        
        # Save statistics
        stats_path = os.path.join(output_dir, "extraction_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== Extraction Complete ===")
        print(f"📊 Final Statistics:")
        print(f"   Total papers: {stats['total_papers']:,}")
        print(f"   Total size: {stats['total_size_gb']:.2f} GB ({stats['total_size_mb']:.1f} MB)")
        print(f"   Average per paper: {stats['avg_size_per_paper_kb']:.1f} KB")
        print(f"   Queries processed: {stats['queries_processed']}")
        
        return stats

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description='Extract papers from arXiv API')
    parser.add_argument('--query', type=str, help='Search query (for single query extraction)')
    parser.add_argument('--max_results', type=int, default=100, help='Maximum number of results')
    parser.add_argument('--output', type=str, help='Output file path (JSON or CSV)')
    parser.add_argument('--format', type=str, choices=['json', 'csv', 'both'], default='json', help='Output format')
    
    # New arguments for large-scale extraction
    parser.add_argument('--large_scale', action='store_true', help='Extract large dataset (~1.5GB)')
    parser.add_argument('--target_size', type=float, default=1.5, help='Target dataset size in GB (default: 1.5)')
    parser.add_argument('--output_dir', type=str, default='data/data_source', help='Output directory for large-scale extraction')
    
    args = parser.parse_args()
    
    extractor = ArxivExtractor()
    
    # If no arguments provided, run default large-scale extraction
    if len(sys.argv) == 1:
        print("🚀 No arguments provided. Starting default large-scale extraction...")
        print("📁 Output files: data/data_source/raw_data.json and data/data_source/raw_data.csv")
        
        # Run large-scale extraction
        stats = extractor.extract_large_dataset(
            target_size_gb=1.5, 
            output_dir='data/data_source'
        )
        
        # Save the combined dataset with specific names
        if stats['total_papers'] > 0:
            # Load all batch files and combine
            import glob
            all_papers = []
            
            batch_files = glob.glob('data/data_source/batch_*.json')
            for batch_file in batch_files:
                try:
                    with open(batch_file, 'r', encoding='utf-8') as f:
                        batch_papers = json.load(f)
                        all_papers.extend(batch_papers)
                except Exception as e:
                    print(f"Warning: Could not load {batch_file}: {e}")
            
            # Save as raw_data.json and raw_data.csv
            if all_papers:
                extractor.save_to_json(all_papers, 'data/data_source/raw_data.json')
                extractor.save_to_csv(all_papers, 'data/data_source/raw_data.csv')
                
                print(f"\n✅ Data extraction completed!")
                print(f"📊 Total papers: {len(all_papers):,}")
                print(f"📁 Files created:")
                print(f"   - data/data_source/raw_data.json")
                print(f"   - data/data_source/raw_data.csv")
                print(f"\n🔄 Next step: Clean the data with:")
                print(f"   python data/scripts/data_cleaner.py --input data/data_source/raw_data.json --output data/processed/clean_data.json --stats")
        
        return
    
    if args.large_scale:
        # Large-scale extraction mode
        print("🚀 Starting large-scale extraction mode...")
        stats = extractor.extract_large_dataset(args.target_size, args.output_dir)
        
        print(f"\n✅ Large-scale extraction completed!")
        print(f"Check the '{args.output_dir}' directory for your data files.")
        
    else:
        # Single query extraction mode (original functionality)
        if not args.query or not args.output:
            parser.error("For single query extraction, both --query and --output are required")
            
        papers = extractor.extract_papers(args.query, args.max_results)
        
        if not papers:
            print("No papers extracted. Exiting.")
            return
        
        # Save based on format argument or file extension
        if args.format == 'both':
            base_name = os.path.splitext(args.output)[0]
            extractor.save_to_json(papers, f"{base_name}.json")
            extractor.save_to_csv(papers, f"{base_name}.csv")
        elif args.format == 'json' or args.output.endswith('.json'):
            extractor.save_to_json(papers, args.output)
        elif args.format == 'csv' or args.output.endswith('.csv'):
            extractor.save_to_csv(papers, args.output)
        else:
            print("Unsupported output format. Please use .json or .csv extension, or specify --format.")

if __name__ == "__main__":
    main()
