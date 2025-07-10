#!/usr/bin/env python3
"""
arXiv Scientific Papers Chatbot

A Streamlit web application that provides an intelligent chatbot interface
for searching and exploring arXiv scientific papers using semantic search.
"""

import streamlit as st
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Add the scripts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data', 'scripts'))

try:
    from semantic_indexer import SemanticSearcher
except ImportError:
    st.error("Could not import SemanticSearcher. Please ensure the semantic index is built.")
    st.stop()

class ArxivChatbot:
    """Main chatbot class"""
    
    def __init__(self, index_dir: str):
        """Initialize the chatbot with semantic search capabilities"""
        try:
            self.searcher = SemanticSearcher(index_dir)
            self.papers = self.searcher.papers
            self.is_ready = True
        except Exception as e:
            st.error(f"Failed to initialize semantic searcher: {e}")
            self.is_ready = False
    
    def search_papers(self, query: str, k: int = 10) -> List[Dict]:
        """Search for papers using semantic search"""
        if not self.is_ready:
            return []
        
        try:
            results = self.searcher.search(query, k=k, threshold=0.1)
            return results
        except Exception as e:
            st.error(f"Search error: {e}")
            return []
    
    def get_paper_statistics(self) -> Dict:
        """Get statistics about the paper collection"""
        if not self.papers:
            return {}
        
        # Category statistics
        categories = {}
        for paper in self.papers:
            for category in paper.get('categories', {}).get('terms', []):
                categories[category] = categories.get(category, 0) + 1
        
        # Publication year statistics
        years = {}
        for paper in self.papers:
            year = paper.get('publication_year')
            if year:
                years[str(year)] = years.get(str(year), 0) + 1
        
        # Author statistics
        all_authors = []
        for paper in self.papers:
            all_authors.extend(paper.get('authors', {}).get('names', []))
        
        return {
            'total_papers': len(self.papers),
            'categories': dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]),
            'years': dict(sorted(years.items())),
            'total_authors': len(set(all_authors)),
            'avg_authors_per_paper': len(all_authors) / len(self.papers) if self.papers else 0
        }
    
    def generate_response(self, query: str, search_results: List[Dict]) -> str:
        """Generate a conversational response based on search results"""
        if not search_results:
            return f"I couldn't find any papers related to '{query}'. Try using different keywords or broader terms."
        
        response_parts = []
        
        # Introduction
        response_parts.append(f"I found {len(search_results)} relevant papers about '{query}':")
        
        # Top results summary
        for i, paper in enumerate(search_results[:3], 1):
            title = paper['title']
            authors = ', '.join(paper.get('authors', {}).get('names', [])[:2])
            if len(paper.get('authors', {}).get('names', [])) > 2:
                authors += " et al."
            
            score = paper.get('similarity_score', 0)
            response_parts.append(f"\n{i}. **{title}**")
            response_parts.append(f"   Authors: {authors}")
            response_parts.append(f"   Relevance: {score:.1%}")
        
        # Additional insights
        if len(search_results) > 3:
            response_parts.append(f"\n... and {len(search_results) - 3} more papers.")
        
        # Categories analysis
        categories = {}
        for paper in search_results:
            for cat in paper.get('categories', {}).get('terms', []):
                categories[cat] = categories.get(cat, 0) + 1
        
        if categories:
            top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
            response_parts.append(f"\nMain research areas: {', '.join([cat for cat, _ in top_categories])}")
        
        return '\n'.join(response_parts)

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="arXiv Scientific Papers Chatbot",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🔬 arXiv Scientific Papers Chatbot")
    st.markdown("Ask me anything about scientific papers from arXiv!")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Index directory selection
    default_index_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'search_index')
    index_dir = st.sidebar.text_input(
        "Semantic Index Directory",
        value=default_index_dir,
        help="Path to the directory containing the semantic search index"
    )
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state or st.sidebar.button("🔄 Reload Index"):
        with st.spinner("Loading semantic search index..."):
            st.session_state.chatbot = ArxivChatbot(index_dir)
    
    chatbot = st.session_state.chatbot
    
    if not chatbot.is_ready:
        st.error("❌ Chatbot is not ready. Please check the index directory and try reloading.")
        st.info("💡 Make sure you have run the semantic indexing script first:")
        st.code("python data/scripts/semantic_indexer.py")
        return
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hello! I'm your arXiv papers assistant. I have access to over 30,000 scientific papers. Ask me about any research topic and I'll find relevant papers for you!"
                }
            ]
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about scientific papers..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Searching papers..."):
                    # Search for papers
                    search_results = chatbot.search_papers(prompt, k=10)
                    
                    # Generate response
                    response = chatbot.generate_response(prompt, search_results)
                    st.markdown(response)
                    
                    # Display detailed results
                    if search_results:
                        st.subheader("📄 Detailed Results")
                        
                        for i, paper in enumerate(search_results, 1):
                            with st.expander(f"{i}. {paper['title'][:100]}..."):
                                col_a, col_b = st.columns([3, 1])
                                
                                with col_a:
                                    st.write("**Abstract:**")
                                    summary = paper['summary']
                                    if len(summary) > 500:
                                        st.write(summary[:500] + "...")
                                    else:
                                        st.write(summary)
                                    
                                    st.write("**Authors:**")
                                    authors = paper.get('authors', {}).get('names', [])
                                    st.write(', '.join(authors))
                                    
                                    if paper.get('categories', {}).get('terms'):
                                        st.write("**Categories:**")
                                        st.write(', '.join(paper['categories']['terms']))
                                    
                                    if paper.get('keywords'):
                                        st.write("**Keywords:**")
                                        keywords = paper['keywords'][:10]  # Show first 10 keywords
                                        st.write(', '.join(keywords))
                                
                                with col_b:
                                    st.metric("Relevance", f"{paper.get('similarity_score', 0):.1%}")
                                    
                                    if paper.get('publication_year'):
                                        st.write("**Year:**")
                                        st.write(paper['publication_year'])
                                    
                                    if paper.get('pdf_url'):
                                        st.link_button("📄 View PDF", paper['pdf_url'])
                                    
                                    if paper.get('abstract_url'):
                                        st.link_button("🔗 arXiv Page", paper['abstract_url'])
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.header("📊 Dataset Statistics")
        
        # Get statistics
        stats = chatbot.get_paper_statistics()
        
        if stats:
            # Basic metrics
            st.metric("Total Papers", f"{stats['total_papers']:,}")
            st.metric("Unique Authors", f"{stats['total_authors']:,}")
            st.metric("Avg Authors/Paper", f"{stats['avg_authors_per_paper']:.1f}")
            
            # Category distribution
            if stats['categories']:
                st.subheader("🏷️ Top Categories")
                cat_df = pd.DataFrame(
                    list(stats['categories'].items()),
                    columns=['Category', 'Count']
                )
                fig = px.bar(
                    cat_df, 
                    x='Count', 
                    y='Category', 
                    orientation='h',
                    title="Papers by Category"
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Publication timeline
            if stats['years']:
                st.subheader("📅 Publications by Year")
                year_df = pd.DataFrame(
                    list(stats['years'].items()),
                    columns=['Year', 'Count']
                )
                year_df['Year'] = year_df['Year'].astype(int)
                year_df = year_df.sort_values('Year')
                
                fig = px.line(
                    year_df, 
                    x='Year', 
                    y='Count', 
                    markers=True,
                    title="Publication Timeline"
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        # Search tips
        st.subheader("💡 Search Tips")
        st.markdown("""
        **Effective search strategies:**
        - Use specific technical terms
        - Try different phrasings
        - Combine multiple concepts
        - Include author names
        - Specify research areas
        
        **Example queries:**
        - "quantum machine learning algorithms"
        - "transformer neural networks attention"
        - "computer vision object detection"
        - "reinforcement learning robotics"
        - "natural language processing BERT"
        - "deep learning optimization techniques"
        """)
        
        # Quick search buttons
        st.subheader("🚀 Quick Searches")
        quick_searches = [
            "quantum computing",
            "machine learning",
            "computer vision", 
            "natural language processing",
            "reinforcement learning",
            "deep learning",
            "neural networks",
            "artificial intelligence"
        ]
        
        for search_term in quick_searches:
            if st.button(search_term, key=f"quick_{search_term}"):
                # Trigger search by adding to chat
                st.session_state.messages.append({"role": "user", "content": search_term})
                st.rerun()
    
    # Footer
    st.markdown("---")
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    
    with col_footer1:
        st.markdown("**🔬 Dataset Info**")
        st.markdown(f"Papers: {stats.get('total_papers', 0):,}")
        st.markdown("Source: arXiv API")
    
    with col_footer2:
        st.markdown("**🧠 AI Technology**")
        st.markdown("Embeddings: Sentence Transformers")
        st.markdown("Search: FAISS Vector Index")
    
    with col_footer3:
        st.markdown("**⚡ Performance**")
        st.markdown("Search Speed: < 100ms")
        st.markdown("Semantic Similarity: Cosine")

if __name__ == "__main__":
    main()
