#!/usr/bin/env python3
"""
Semantic Indexing Script

This script creates semantic embeddings for the cleaned arXiv papers
and builds a FAISS index for fast similarity search.
"""

import json
import os
import pickle
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime
import argparse

class SemanticIndexer:
    """Class to create semantic embeddings and FAISS index"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic indexer
        
        Args:
            model_name: Name of the sentence transformer model
        """
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.index = None
        self.paper_ids = []
        self.embeddings = None
        
    def load_papers(self, file_path: str) -> List[Dict]:
        """Load cleaned papers from JSON file"""
        print(f"Loading papers from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        print(f"Loaded {len(papers)} papers")
        return papers
    
    def create_embeddings(self, papers: List[Dict]) -> np.ndarray:
        """
        Create embeddings for all papers
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Numpy array of embeddings
        """
        print("Creating embeddings...")
        
        # Extract text for embedding (title + summary)
        texts = []
        valid_papers = []
        
        for paper in papers:
            combined_text = paper.get('combined_text', '')
            if combined_text.strip():
                texts.append(combined_text)
                valid_papers.append(paper)
            else:
                print(f"Skipping paper {paper.get('arxiv_id', 'unknown')}: No text content")
        
        if not texts:
            raise ValueError("No valid texts found for embedding")
        
        print(f"Creating embeddings for {len(texts)} papers...")
        
        # Create embeddings in batches to manage memory
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)
        
        # Store paper IDs for later reference
        self.paper_ids = [paper['arxiv_id'] for paper in valid_papers]
        
        print(f"✅ Created embeddings with shape: {embeddings.shape}")
        return embeddings, valid_papers
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build FAISS index for fast similarity search
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            FAISS index
        """
        print("Building FAISS index...")
        
        dimension = embeddings.shape[1]
        n_vectors = embeddings.shape[0]
        
        # Choose index type based on dataset size
        if n_vectors < 1000:
            # Simple flat index for small datasets
            index = faiss.IndexFlatIP(dimension)
            print("Using IndexFlatIP (exact search)")
        elif n_vectors < 10000:
            # IVF index for medium datasets
            nlist = min(100, n_vectors // 10)
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            print(f"Using IndexIVFFlat with {nlist} clusters")
        else:
            # IVF-PQ index for large datasets
            nlist = min(1000, n_vectors // 50)
            m = 8  # number of subquantizers
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
            print(f"Using IndexIVFPQ with {nlist} clusters")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Train index if necessary
        if hasattr(index, 'train'):
            print("Training index...")
            index.train(embeddings)
        
        # Add embeddings to index
        print("Adding embeddings to index...")
        index.add(embeddings)
        
        # Set search parameters for IVF indices
        if hasattr(index, 'nprobe'):
            index.nprobe = min(10, index.nlist)
        
        print(f"✅ Built FAISS index with {index.ntotal} vectors")
        return index
    
    def save_index(self, index: faiss.Index, papers: List[Dict], output_dir: str):
        """
        Save the FAISS index and metadata
        
        Args:
            index: FAISS index
            papers: List of paper dictionaries
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(output_dir, "faiss_index.bin")
        faiss.write_index(index, index_path)
        print(f"✅ Saved FAISS index to {index_path}")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'embedding_dimension': self.model.get_sentence_embedding_dimension(),
            'total_papers': len(papers),
            'paper_ids': self.paper_ids,
            'created_at': datetime.now().isoformat(),
            'index_type': type(index).__name__
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved metadata to {metadata_path}")
        
        # Save paper data for search results
        papers_path = os.path.join(output_dir, "papers_data.json")
        with open(papers_path, 'w', encoding='utf-8') as f:
            json.dump(papers, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved papers data to {papers_path}")
        
        # Save embeddings for potential future use
        embeddings_path = os.path.join(output_dir, "embeddings.npy")
        np.save(embeddings_path, self.embeddings)
        print(f"✅ Saved embeddings to {embeddings_path}")
    
    def build_complete_index(self, input_file: str, output_dir: str):
        """
        Complete pipeline to build semantic index
        
        Args:
            input_file: Path to cleaned papers JSON file
            output_dir: Output directory for index files
        """
        print("=== Starting Semantic Indexing Pipeline ===")
        
        # Load papers
        papers = self.load_papers(input_file)
        
        # Create embeddings
        self.embeddings, valid_papers = self.create_embeddings(papers)
        
        # Build FAISS index
        self.index = self.build_faiss_index(self.embeddings)
        
        # Save everything
        self.save_index(self.index, valid_papers, output_dir)
        
        print("=== Semantic Indexing Complete ===")
        
        return {
            'total_papers': len(valid_papers),
            'embedding_dimension': self.embeddings.shape[1],
            'index_type': type(self.index).__name__
        }

class SemanticSearcher:
    """Class to perform semantic search using the built index"""
    
    def __init__(self, index_dir: str):
        """
        Initialize the semantic searcher
        
        Args:
            index_dir: Directory containing the index files
        """
        self.index_dir = index_dir
        self.load_index()
    
    def load_index(self):
        """Load the FAISS index and metadata"""
        # Load metadata
        metadata_path = os.path.join(self.index_dir, "metadata.json")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Load model
        print(f"Loading model: {self.metadata['model_name']}")
        self.model = SentenceTransformer(self.metadata['model_name'])
        
        # Load FAISS index
        index_path = os.path.join(self.index_dir, "faiss_index.bin")
        self.index = faiss.read_index(index_path)
        
        # Load papers data
        papers_path = os.path.join(self.index_dir, "papers_data.json")
        with open(papers_path, 'r', encoding='utf-8') as f:
            self.papers = json.load(f)
        
        # Create paper lookup dictionary
        self.papers_dict = {paper['arxiv_id']: paper for paper in self.papers}
        
        print(f"✅ Loaded index with {len(self.papers)} papers")
    
    def search(self, query: str, k: int = 10, threshold: float = 0.0) -> List[Dict]:
        """
        Perform semantic search
        
        Args:
            query: Search query string
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results with similarity scores
        """
        # Create query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search in index
        scores, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx < len(self.metadata['paper_ids']):
                paper_id = self.metadata['paper_ids'][idx]
                if paper_id in self.papers_dict:
                    result = self.papers_dict[paper_id].copy()
                    result['similarity_score'] = float(score)
                    results.append(result)
        
        return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create semantic index for arXiv papers')
    parser.add_argument('--input', type=str, required=True, help='Input cleaned papers JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output directory for index')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2', help='Sentence transformer model')
    parser.add_argument('--test', action='store_true', help='Test the index after creation')
    
    args = parser.parse_args()
    
    # Build index
    indexer = SemanticIndexer(args.model)
    stats = indexer.build_complete_index(args.input, args.output)
    
    print(f"\n📊 Index Statistics:")
    print(f"   Total papers: {stats['total_papers']}")
    print(f"   Embedding dimension: {stats['embedding_dimension']}")
    print(f"   Index type: {stats['index_type']}")
    
    # Test search if requested
    if args.test:
        print("\n🔍 Testing search functionality...")
        searcher = SemanticSearcher(args.output)
        
        test_queries = [
            "quantum computing",
            "machine learning neural networks",
            "natural language processing"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = searcher.search(query, k=3)
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['title'][:80]}... (score: {result['similarity_score']:.3f})")

if __name__ == "__main__":
    main()
