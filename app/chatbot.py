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
from collections import Counter # Added for enhanced statistics and response generation
import time # Added for streaming effect in testing_chatbot logic

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
        """Get comprehensive statistics about the paper collection"""
        if not self.papers:
            return {}
        
        categories_count = Counter()
        years_count = Counter()
        all_authors = []
        all_affiliations = []
        all_keywords = Counter()
        publication_names = Counter()

        for paper in self.papers:
            # Categories
            for category in paper.get('categories', {}).get('terms', []):
                categories_count[category] += 1
            
            # Publication year
            year = paper.get('publication_year')
            if year:
                years_count[str(year)] += 1
            
            # Authors and Affiliations
            authors_info = paper.get('authors', {})
            all_authors.extend(authors_info.get('names', []))
            all_affiliations.extend(authors_info.get('affiliations', [])) # Assuming affiliations are processed by data_cleaner
            
            # Keywords
            for keyword in paper.get('keywords', []):
                all_keywords[keyword] += 1
            
            # Publication Names (using journal_ref)
            journal_ref = paper.get('journal_ref')
            if journal_ref:
                publication_names[journal_ref] += 1

        total_papers = len(self.papers)
        total_unique_authors = len(set(all_authors))
        total_unique_affiliations = len(set(all_affiliations))
        
        return {
            'total_articles': total_papers, # Renamed to match Dash app's expectation
            'categories': dict(categories_count.most_common(20)),
            'articles_by_year': dict(sorted(years_count.items())), # Renamed to match Dash app
            'total_authors': total_unique_authors,
            'avg_authors_per_paper': sum(paper.get('authors', {}).get('count', 0) for paper in self.papers) / total_papers if total_papers else 0,
            'recent_papers': sum(1 for paper in self.papers if paper.get('publication_year', 0) >= 2020),
            'top_authors': dict(Counter(all_authors).most_common(15)),
            'top_keywords': dict(all_keywords.most_common(15)), # Top 15 keywords
            'top_publications': dict(publication_names.most_common(10)), # Top 10 publications
            'total_affiliations': total_unique_affiliations
        }
    
    def generate_response(self, query: str, search_results: List[Dict]) -> Dict:
        """Generate a comprehensive, educational ChatGPT-like response based on search results"""
        if not search_results:
            response_text = f"""I couldn't find any papers related to '{query}' in my database. This could be because:

- The topic might be very new or highly specialized
- Try using different keywords or broader terms
- Consider alternative phrasings of your question

Would you like me to suggest some related topics or help you refine your search?"""
            return {
                'response': response_text,
                'articles': [],
                'overall_relevance_score': 0.0,
                'total_articles_found': 0,
                'analysis_data': {}
            }
        
        # Analyze search results to extract comprehensive information
        top_papers = search_results[:10]  # Use top 10 papers for comprehensive response
        
        # Extract detailed information
        all_keywords = []
        categories = Counter()
        recent_findings = []
        key_authors = []
        methodologies = []
        applications = []
        challenges = []
        
        for paper in top_papers:
            # Collect keywords and categories
            all_keywords.extend(paper.get('keywords', [])[:10])
            for cat in paper.get('categories', {}).get('terms', []):
                categories[cat] += 1
        
            # Collect author information
            authors = paper.get('authors', {}).get('names', [])
            key_authors.extend(authors[:3])
        
            # Extract methodologies, applications, and challenges from abstracts
            summary = paper.get('summary', '').lower()
            title = paper.get('title', '').lower()
        
            # Look for methodological terms
            method_keywords = ['algorithm', 'method', 'approach', 'technique', 'framework', 'model', 'system']
            for keyword in method_keywords:
                if keyword in summary or keyword in title:
                    sentences = paper.get('summary', '').split('. ')
                    for sentence in sentences:
                        if keyword in sentence.lower() and len(sentence.strip()) > 30:
                            methodologies.append(sentence.strip())
                            break
            
            # Look for applications
            app_keywords = ['application', 'applied', 'used for', 'implementation', 'real-world', 'practical']
            for keyword in app_keywords:
                if keyword in summary:
                    sentences = paper.get('summary', '').split('. ')
                    for sentence in sentences:
                        if keyword in sentence.lower() and len(sentence.strip()) > 30:
                            applications.append(sentence.strip())
                            break
            
            # Look for challenges and limitations
            challenge_keywords = ['challenge', 'limitation', 'problem', 'difficulty', 'issue', 'bottleneck']
            for keyword in challenge_keywords:
                if keyword in summary:
                    sentences = paper.get('summary', '').split('. ')
                    for sentence in sentences:
                        if keyword in sentence.lower() and len(sentence.strip()) > 30:
                            challenges.append(sentence.strip())
                            break
            
            # Extract key findings (first meaningful sentence)
            sentences = paper.get('summary', '').split('. ')
            for sentence in sentences[:3]:
                if len(sentence.strip()) > 50 and any(word in sentence.lower() for word in ['show', 'demonstrate', 'find', 'result', 'achieve', 'improve']):
                    recent_findings.append(sentence.strip())
                    break
    
        # Count frequencies for insights
        keyword_counts = Counter(all_keywords)
        author_counts = Counter(key_authors)
        
        # Generate comprehensive response
        response_parts = []
        
        # Determine query type and generate appropriate introduction
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'explain', 'define', 'introduction to']):
            # Educational/explanatory response
            response_parts.append(f"# 🎓 Understanding {query.title()}")
            response_parts.append("")
            response_parts.append(f"Great question! Let me break down **{query}** based on the latest research findings:")
            response_parts.append("")
            
            # Core definition and concepts
            if keyword_counts:
                top_concepts = [concept for concept, _ in keyword_counts.most_common(8)]
                response_parts.append("## 🔍 **Core Concepts & Components:**")
                for i, concept in enumerate(top_concepts, 1):
                    response_parts.append(f"- **{concept.title()}** - A fundamental element in this field")
                response_parts.append("")
            
            # Key methodologies
            if methodologies:
                response_parts.append("## ⚙️ **Key Methodologies & Approaches:**")
                for i, method in enumerate(set(methodologies[:5]), 1):
                    response_parts.append(f"- {method}")
                response_parts.append("")
        
        elif any(word in query_lower for word in ['how', 'method', 'approach', 'technique', 'implement']):
            # Methodological response
            response_parts.append(f"# 🛠️ How to Approach {query.title()}")
            response_parts.append("")
            response_parts.append(f"Here's a comprehensive guide on **{query}** based on current research:")
            response_parts.append("")
            
            if methodologies:
                response_parts.append("## 📋 **Step-by-Step Methodologies:**")
                for i, method in enumerate(set(methodologies[:6]), 1):
                    response_parts.append(f"- **Step {i}:** {method}")
                response_parts.append("")
        
        elif any(word in query_lower for word in ['latest', 'recent', 'new', 'current', 'trend', 'state of art']):
            # Trend-focused response
            recent_papers = [p for p in top_papers if p.get('publication_year', 0) >= 2020]
            response_parts.append(f"# 📈 Latest Developments in {query.title()}")
            response_parts.append("")
            response_parts.append(f"Here are the **cutting-edge developments** in **{query}**:")
            response_parts.append("")
            
            if recent_papers:
                response_parts.append("## 🚀 **Recent Breakthroughs:**")
                for i, paper in enumerate(recent_papers[:4], 1):
                    title = paper['title']
                    year = paper.get('publication_year', 'N/A')
                    authors = ', '.join(paper.get('authors', {}).get('names', [])[:2])
                    response_parts.append(f"- **{title}** ({year}) by {authors}")
                response_parts.append("")
        
        elif any(word in query_lower for word in ['application', 'use case', 'example', 'practical']):
            # Application-focused response
            response_parts.append(f"# 💼 Practical Applications of {query.title()}")
            response_parts.append("")
            response_parts.append(f"Here are the **real-world applications** of **{query}**:")
            response_parts.append("")
            
            if applications:
                response_parts.append("## 🌍 **Real-World Use Cases:**")
                for i, app in enumerate(set(applications[:5]), 1):
                    response_parts.append(f"- {app}")
                response_parts.append("")
        
        else:
            # General comprehensive response
            response_parts.append(f"# 🧠 Complete Guide to {query.title()}")
            response_parts.append("")
            response_parts.append(f"Based on my analysis of current research, here's everything you need to know about **{query}**:")
            response_parts.append("")
        
        # Research landscape overview
        if categories:
            top_categories = categories.most_common(4)
            response_parts.append("## 🏗️ **Research Landscape:**")
            response_parts.append("This field spans multiple disciplines:")
            for cat, count in top_categories:
                # Convert category codes to readable names
                cat_name = cat.replace('cs.', 'Computer Science - ').replace('stat.', 'Statistics - ').replace('math.', 'Mathematics - ')
                response_parts.append(f"- **{cat_name}** ({count} papers)")
            response_parts.append("")
        
        # Key findings and insights
        if recent_findings:
            response_parts.append("## 🔬 **Key Research Findings:**")
            for i, finding in enumerate(set(recent_findings[:5]), 1):
                response_parts.append(f"- {finding}")
            response_parts.append("")
        
        # Challenges and limitations
        if challenges:
            response_parts.append("## ⚠️ **Current Challenges & Limitations:**")
            for i, challenge in enumerate(set(challenges[:4]), 1):
                response_parts.append(f"- {challenge}")
            response_parts.append("")
        
        # Statistical insights
        years = [p.get('publication_year', 0) for p in top_papers if p.get('publication_year')]
        if years:
            year_range = f"{min(years)}-{max(years)}"
            recent_count = sum(1 for year in years if year >= 2020)
            total_authors_in_search = len(set(key_authors))
            avg_relevance = sum(p.get('similarity_score', 0) for p in top_papers) / len(top_papers) if top_papers else 0
            
            response_parts.append("## 📊 **Research Analytics:**")
            response_parts.append(f"- **Timeline**: Active research from {year_range}")
            response_parts.append(f"- **Recent Activity**: {recent_count} papers published since 2020")
            response_parts.append(f"- **Research Community**: {total_authors_in_search} active researchers")
            response_parts.append(f"- **Content Relevance**: {avg_relevance:.1%} average match to your query")
            response_parts.append("")
        
        # Leading researchers and institutions
        if author_counts:
            leading_authors = [author for author, _ in author_counts.most_common(5)]
            response_parts.append(f"## 👨‍🔬 **Leading Researchers**: {', '.join(leading_authors)}")
            response_parts.append("")
        
        response_parts.append("## 🎯 **Key Takeaways & Future Directions:**")
        response_parts.append("- This field shows strong interdisciplinary collaboration")
        response_parts.append("- Multiple research methodologies are being explored")
        
        if years and recent_count > len(top_papers) // 2:
            response_parts.append("- Rapid growth in recent publications indicates high research interest")
        
        response_parts.append("- Practical applications are driving theoretical advances")
        response_parts.append("- Cross-domain knowledge transfer is accelerating innovation")
        response_parts.append("")
        
        # Learning path suggestion (for educational queries)
        if any(word in query_lower for word in ['learn', 'study', 'understand', 'beginner', 'start']):
            response_parts.append("## 📚 **Suggested Learning Path:**")
            response_parts.append("- **Foundation**: Start with basic concepts and terminology")
            response_parts.append("- **Methodology**: Study the core approaches and techniques")
            response_parts.append("- **Applications**: Explore real-world use cases")
            response_parts.append("- **Advanced Topics**: Dive into recent research developments")
            response_parts.append("- **Hands-on Practice**: Implement solutions and experiment")
            response_parts.append("")
        
        response_parts.append("💡 *Want to dive deeper into any specific aspect? Just ask!*")
        
        full_response_text = '\n'.join(response_parts)

        # Prepare articles for Dash app (add 'score' key)
        dash_articles = []
        for paper in search_results:
            article_data = paper.copy()
            article_data['score'] = paper.get('similarity_score', 0.0)
            # Ensure 'keywords' is a string for the Dash app's split(';')
            if isinstance(article_data.get('keywords'), list):
                article_data['keywords'] = '; '.join(article_data['keywords'])
            dash_articles.append(article_data)

        # Prepare analysis_data for Dash app
        analysis_data = {
            'total_unique_keywords': len(set(all_keywords)) # Keywords from search results
        }

        overall_relevance_score = sum(p.get('similarity_score', 0) for p in search_results) / len(search_results) if search_results else 0.0
        total_articles_found = len(search_results)

        return {
            'response': full_response_text,
            'articles': dash_articles,
            'overall_relevance_score': overall_relevance_score,
            'total_articles_found': total_articles_found,
            'analysis_data': analysis_data
        }

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
                    # The original Streamlit app expects a string, so we extract it from the dict
                    response_dict = chatbot.generate_response(prompt, search_results)
                    response = response_dict['response']
                    st.markdown(response)
                    
                    # Display detailed results (this part is from the original Streamlit app)
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
            st.metric("Total Papers", f"{stats['total_articles']:,}")
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
            if stats['articles_by_year']:
                st.subheader("📅 Publications by Year")
                year_df = pd.DataFrame(
                    list(stats['articles_by_year'].items()),
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
        st.markdown(f"Papers: {stats.get('total_articles', 0):,}")
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
