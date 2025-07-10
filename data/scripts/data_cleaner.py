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
import sys
from collections import Counter  # Add this line

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
    
    def advanced_text_cleaning(self, text: str) -> str:
        """Advanced text cleaning with better normalization"""
        if not text or pd.isna(text):
            return ""
        
        # Remove LaTeX commands and mathematical expressions
        text = re.sub(r'\$[^$]*\$', ' ', text)  # Remove inline math
        text = re.sub(r'\$\$[^$]*\$\$', ' ', text)  # Remove display math
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', ' ', text)  # Remove LaTeX commands
        
        # Remove URLs and email addresses
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\$$\$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-$$$$]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very short words (less than 2 characters)
        words = text.split()
        words = [word for word in words if len(word) >= 2]
        text = ' '.join(words)
        
        return text.strip()

    def validate_paper_quality(self, paper: Dict) -> bool:
        """Validate if a paper meets quality standards"""
        # Check required fields
        if not paper.get('title') or not paper.get('summary'):
            return False
        
        # Check minimum text length
        if len(paper['title']) < 10 or len(paper['summary']) < 50:
            return False
        
        # Check if text is mostly non-English characters
        title_ascii_ratio = sum(1 for c in paper['title'] if ord(c) < 128) / len(paper['title'])
        summary_ascii_ratio = sum(1 for c in paper['summary'] if ord(c) < 128) / len(paper['summary'])
        
        if title_ascii_ratio < 0.7 or summary_ascii_ratio < 0.7:
            return False
        
        return True

    def extract_enhanced_keywords(self, text: str, min_length: int = 3, max_keywords: int = 30) -> List[str]:
        """Enhanced keyword extraction with better filtering"""
        if not text:
            return []
        
        # Extended stop words list
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'we', 'they', 'them', 'their', 'our', 'your', 'his', 'her', 'its',
            'also', 'such', 'which', 'where', 'when', 'how', 'why', 'what', 'who',
            'more', 'most', 'some', 'many', 'much', 'very', 'well', 'good', 'new',
            'first', 'last', 'long', 'great', 'little', 'own', 'other', 'old', 'right',
            'big', 'high', 'different', 'small', 'large', 'next', 'early', 'young',
            'important', 'few', 'public', 'bad', 'same', 'able', 'show', 'shows',
            'paper', 'study', 'research', 'work', 'method', 'approach', 'results',
            'conclusion', 'abstract', 'introduction', 'discussion', 'analysis'
        }
        
        # Extract words with better pattern matching
        # Include hyphenated words and technical terms
        words = re.findall(r'\b[a-zA-Z](?:[a-zA-Z\-]*[a-zA-Z])?\b', text.lower())
        
        # Filter words
        filtered_words = []
        for word in words:
            # Skip if too short or in stop words
            if len(word) < min_length or word in stop_words:
                continue
            
            # Skip if all digits or mostly punctuation
            if word.isdigit() or len(re.sub(r'[^a-zA-Z]', '', word)) < min_length:
                continue
            
            filtered_words.append(word)
        
        # Count frequency and return top keywords
        from collections import Counter
        word_counts = Counter(filtered_words)
        
        # Return top keywords
        return [word for word, count in word_counts.most_common(max_keywords)]

    def clean_paper(self, paper: Dict) -> Dict:
        """Enhanced paper cleaning with better data processing"""
        cleaned_paper = {}
        
        # Basic information with enhanced cleaning
        cleaned_paper['arxiv_id'] = paper.get('arxiv_id', '')
        cleaned_paper['title'] = self.advanced_text_cleaning(paper.get('title', ''))
        cleaned_paper['summary'] = self.advanced_text_cleaning(paper.get('summary', ''))
        
        # Validate paper quality
        temp_paper = {
            'title': cleaned_paper['title'],
            'summary': cleaned_paper['summary']
        }
        if not self.validate_paper_quality(temp_paper):
            return None  # Return None for invalid papers
        
        # Date processing
        cleaned_paper['published'] = paper.get('published', '')
        cleaned_paper['updated'] = paper.get('updated', '')
        
        # Extract publication year for easier filtering
        if cleaned_paper['published']:
            try:
                pub_year = cleaned_paper['published'][:4]
                cleaned_paper['publication_year'] = int(pub_year) if pub_year.isdigit() else None
            except:
                cleaned_paper['publication_year'] = None
        else:
            cleaned_paper['publication_year'] = None
        
        # Process authors with enhanced structure
        authors_info = self.process_authors(paper.get('authors', []))
        cleaned_paper['authors'] = authors_info
        
        # Process categories with enhanced structure
        categories_info = self.process_categories(paper.get('categories', []))
        cleaned_paper['categories'] = categories_info
        
        # Enhanced keyword extraction
        title_keywords = self.extract_enhanced_keywords(cleaned_paper['title'])
        summary_keywords = self.extract_enhanced_keywords(cleaned_paper['summary'])
        
        # Combine original extracted keywords if available
        original_keywords = paper.get('extracted_keywords', [])
        if isinstance(original_keywords, list):
            all_keywords = list(set(title_keywords + summary_keywords + original_keywords))
        else:
            all_keywords = list(set(title_keywords + summary_keywords))
        
        cleaned_paper['keywords'] = all_keywords[:50]  # Limit to top 50 keywords
        
        # Additional metadata
        cleaned_paper['doi'] = paper.get('doi', '') or ''
        cleaned_paper['journal_ref'] = paper.get('journal_ref', '') or ''
        cleaned_paper['comment'] = self.clean_text(paper.get('comment', '')) or ''
        cleaned_paper['pdf_url'] = paper.get('pdf_url', '') or ''
        cleaned_paper['abstract_url'] = paper.get('abstract_url', '') or ''
        
        # Text metrics
        cleaned_paper['title_length'] = len(cleaned_paper['title'])
        cleaned_paper['summary_length'] = len(cleaned_paper['summary'])
        cleaned_paper['keyword_count'] = len(cleaned_paper['keywords'])
        
        # Add processing timestamp
        cleaned_paper['processed_at'] = datetime.now().isoformat()
        
        # Create enhanced combined text for semantic search
        combined_parts = [cleaned_paper['title'], cleaned_paper['summary']]
        if cleaned_paper['comment']:
            combined_parts.append(cleaned_paper['comment'])
        
        cleaned_paper['combined_text'] = ' '.join(combined_parts)
        cleaned_paper['combined_text_length'] = len(cleaned_paper['combined_text'])
        
        return cleaned_paper
    
    def clean_dataset(self, papers: List[Dict]) -> List[Dict]:
        """Enhanced dataset cleaning with better validation"""
        print(f"Cleaning {len(papers)} papers...")
        
        cleaned_papers = []
        skipped_count = 0
        
        for i, paper in enumerate(papers):
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(papers)} papers...")
            
            try:
                cleaned_paper = self.clean_paper(paper)
                
                if cleaned_paper is not None:
                    cleaned_papers.append(cleaned_paper)
                else:
                    skipped_count += 1
                    
            except Exception as e:
                print(f"Error cleaning paper {i+1}: {e}")
                skipped_count += 1
                continue
        
        print(f"✅ Cleaned {len(cleaned_papers)} papers successfully")
        print(f"⚠️ Skipped {skipped_count} papers due to quality issues")
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
    
    def generate_enhanced_statistics(self, papers: List[Dict]) -> Dict:
        """Generate comprehensive statistics about the cleaned dataset"""
        if not papers:
            return {}
        
        # Basic statistics
        stats = {
            'total_papers': len(papers),
            'processing_date': datetime.now().isoformat(),
        }
        
        # Date range analysis
        years = [p['publication_year'] for p in papers if p.get('publication_year')]
        if years:
            stats['publication_years'] = {
                'earliest': min(years),
                'latest': max(years),
                'range': max(years) - min(years),
                'distribution': dict(Counter(years))
            }
        
        # Category analysis
        all_categories = []
        primary_categories = []
        for paper in papers:
            categories = paper.get('categories', {}).get('terms', [])
            all_categories.extend(categories)
            
            primary = paper.get('categories', {}).get('primary', '')
            if primary:
                primary_categories.append(primary)
        
        stats['categories'] = {
            'total_unique': len(set(all_categories)),
            'most_common': dict(Counter(all_categories).most_common(20)),
            'primary_categories': dict(Counter(primary_categories).most_common(10))
        }
        
        # Author statistics
        all_authors = []
        author_counts = []
        for paper in papers:
            authors = paper.get('authors', {}).get('names', [])
            all_authors.extend(authors)
            author_counts.append(len(authors))
        
        stats['authors'] = {
            'total_unique_authors': len(set(all_authors)),
            'avg_authors_per_paper': sum(author_counts) / len(author_counts) if author_counts else 0,
            'max_authors_per_paper': max(author_counts) if author_counts else 0,
            'min_authors_per_paper': min(author_counts) if author_counts else 0,
            'most_prolific_authors': dict(Counter(all_authors).most_common(10))
        }
        
        # Text statistics
        title_lengths = [p['title_length'] for p in papers]
        summary_lengths = [p['summary_length'] for p in papers]
        keyword_counts = [p['keyword_count'] for p in papers]
        
        stats['text_metrics'] = {
            'title_length': {
                'avg': sum(title_lengths) / len(title_lengths),
                'min': min(title_lengths),
                'max': max(title_lengths)
            },
            'summary_length': {
                'avg': sum(summary_lengths) / len(summary_lengths),
                'min': min(summary_lengths),
                'max': max(summary_lengths)
            },
            'keywords_per_paper': {
                'avg': sum(keyword_counts) / len(keyword_counts),
                'min': min(keyword_counts),
                'max': max(keyword_counts)
            }
        }
        
        # Keyword analysis
        all_keywords = []
        for paper in papers:
            all_keywords.extend(paper.get('keywords', []))
        
        stats['keywords'] = {
            'total_unique_keywords': len(set(all_keywords)),
            'most_common_keywords': dict(Counter(all_keywords).most_common(50))
        }
        
        return stats

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
    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--stats', action='store_true', help='Generate statistics')
    parser.add_argument('--auto', action='store_true', help='Auto-clean raw_data files')
    
    args = parser.parse_args()
    
    cleaner = DataCleaner()
    
    # If no arguments or --auto flag, use default files
    if len(sys.argv) == 1 or args.auto:
        print("🧹 Starting automatic data cleaning...")
        print("📁 Input: data/data_source/raw_data.json")
        print("📁 Output: data/processed/clean_data.json")
        
        input_file = 'data/data_source/raw_data.json'
        output_file = 'data/processed/clean_data.json'
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"❌ Input file not found: {input_file}")
            print("Please run the data extraction first:")
            print("   python data/scripts/arxiv_extractor.py")
            return
        
        # Load and clean data
        print(f"Loading data from {input_file}...")
        papers = cleaner.load_data(input_file)
        
        # Clean data
        cleaned_papers = cleaner.clean_dataset(papers)
        
        # Save cleaned data
        cleaner.save_cleaned_data(cleaned_papers, output_file)
        
        # Also save as CSV
        csv_output = output_file.replace('.json', '.csv')
        cleaner.save_cleaned_data(cleaned_papers, csv_output)
        
        # Generate statistics
        stats = cleaner.generate_enhanced_statistics(cleaned_papers)
        stats_file = 'data/processed/cleaning_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"✅ Statistics saved to {stats_file}")
        
        print(f"\n✅ Data cleaning completed!")
        print(f"📊 Cleaned {len(cleaned_papers):,} papers")
        print(f"📁 Files created:")
        print(f"   - {output_file}")
        print(f"   - {csv_output}")
        print(f"   - {stats_file}")
        print(f"\n🔄 Next step: Build semantic index with:")
        print(f"   python data/scripts/semantic_indexer.py --input {output_file} --output data/search_index --test")
        
        return
    
    # Manual mode with specified files
    if not args.input or not args.output:
        parser.error("For manual cleaning, both --input and --output are required")
    
    # Load data
    print(f"Loading data from {args.input}...")
    papers = cleaner.load_data(args.input)
    
    # Clean data
    cleaned_papers = cleaner.clean_dataset(papers)
    
    # Save cleaned data
    cleaner.save_cleaned_data(cleaned_papers, args.output)
    
    # Generate statistics if requested
    if args.stats:
        stats = cleaner.generate_enhanced_statistics(cleaned_papers)
        stats_file = args.output.replace('.json', '_stats.json').replace('.csv', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"✅ Statistics saved to {stats_file}")

if __name__ == "__main__":
    main()
