import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import json
import os
import sys

# Add the app directory to the path to import ArxivChatbot
sys.path.append(os.path.join(os.path.dirname(__file__)))

from chatbot import ArxivChatbot # Changed from ScopusChatbot

# Initialisation de l'application Dash avec suppression des exceptions de callback
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "arXiv AI Research Assistant" # Changed title

# Vérification de l'initialisation du chatbot
def initialize_chatbot():
    """Initialise le chatbot en vérifiant les prérequis"""
    try:
        # Adjust paths to match the existing project structure
        index_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'search_index')
        index_path = os.path.join(index_dir, "faiss_index.bin")
        metadata_path = os.path.join(index_dir, "metadata.json")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            return None, "Index sémantique non trouvé. Veuillez d'abord créer l'index avec data/scripts/semantic_indexer.py"
        
        # Instantiate ArxivChatbot with the correct index directory
        chatbot = ArxivChatbot(index_dir=index_dir)
        if not chatbot.is_ready:
            return None, "Chatbot non initialisé. Vérifiez les fichiers de données."
        
        return chatbot, "Chatbot initialisé avec succès"
        
    except Exception as e:
        return None, f"Erreur lors de l'initialisation: {str(e)}"
    
# Initialisation du chatbot
chatbot, init_message = initialize_chatbot()

# Layout de l'application
def create_layout():
  if chatbot is None:
      # Interface d'erreur si le chatbot n'est pas initialisé
      return html.Div([
          html.Div([
              html.Div([
                  html.Div("🤖", className="logo-icon"),
                  html.Div([
                      html.H1("arXiv AI Research Assistant", className="header-title"), # Changed title
                      html.P("Intelligence artificielle pour la recherche scientifique", className="header-subtitle")
                  ], className="header-text")
              ], className="header-content")
          ], className="header"),
          
          html.Div([
              html.Div([
                  html.Div([
                      html.Div("⚠️", className="error-icon"),
                      html.H2("Configuration requise", className="error-title"),
                      html.P(init_message, className="error-message"),
                  ], className="error-header"),
                  
                  html.Div([
                      html.H3("🚀 Démarrage rapide", className="setup-title"),
                      html.Div([
                          html.Div([
                              html.Div("1", className="step-number"),
                              html.Div([
                                  html.H4("Extraire les données"),
                                  html.P("Lancez la commande d'extraction pour obtenir les données arXiv"),
                                  html.Code("python data/scripts/arxiv_extractor.py", className="code-block")
                              ], className="step-content")
                          ], className="setup-step"),
                          
                          html.Div([
                              html.Div("2", className="step-number"),
                              html.Div([
                                  html.H4("Nettoyer et indexer les données"),
                                  html.P("Lancez la commande de nettoyage et d'indexation pour préparer les données"),
                                  html.Code("python data/scripts/data_cleaner.py", className="code-block"),
                                  html.Code("python data/scripts/semantic_indexer.py", className="code-block")
                              ], className="step-content")
                          ], className="setup-step"),
                          
                          html.Div([
                              html.Div("3", className="step-number"),
                              html.Div([
                                  html.H4("Lancer l'assistant"),
                                  html.P("Redémarrez l'application une fois l'extraction et l'indexation terminées"),
                                  html.Code("python app/dash_app.py", className="code-block") # Changed to dash_app.py
                              ], className="step-content")
                          ], className="setup-step")
                      ], className="setup-steps")
                  ], className="setup-section")
              ], className="error-container")
          ], className="error-page")
      ], className="app-container error-state")
  
  # Interface normale si le chatbot est initialisé
  return html.Div([
      # En-tête avec navigation
      html.Div([
          html.Div([
              html.Div("🤖", className="logo-icon"),
              html.Div([
                  html.H1("arXiv AI", className="header-title"), # Changed title
                  html.P("Research Assistant", className="header-subtitle")
              ], className="header-text")
          ], className="header-brand"),
          
          # Navigation principale
          html.Div([
              html.Button([
                  html.Span("💬", className="nav-icon"),
                  html.Span("Assistant", className="nav-text")
              ], id="btn-chatbot", className="nav-item active"),
              html.Button([
                  html.Span("📄", className="nav-icon"),
                  html.Span("Articles", className="nav-text")
              ], id="btn-articles", className="nav-item"),
              html.Button([
                  html.Span("📊", className="nav-icon"),
                  html.Span("Analyses", className="nav-text")
              ], id="btn-visualizations", className="nav-item"),
              html.Button([
                  html.Span("📈", className="nav-icon"),
                  html.Span("Statistiques", className="nav-text")
              ], id="btn-statistics", className="nav-item")
          ], className="header-nav")
      ], className="header-content"),
      
      # Contenu principal
      html.Div(id="main-content", className="main-content"),
      
      # Stores pour sauvegarder les données
      dcc.Store(id="chat-data", data=[]),
      dcc.Store(id="current-results", data=[]),
      dcc.Store(id="active-view", data="chatbot"),
      dcc.Store(id="current-analysis-data", data={}), # Nouveau store pour les données d'analyse
      dcc.Store(id="total-articles-found", data=0), # Nouveau store pour le total
      
      # Notifications toast
      html.Div(id="toast-container", className="toast-container")
  ], className="app-container")

app.layout = create_layout()

# Callbacks (seulement si le chatbot est initialisé)
if chatbot is not None:
  @app.callback(
      [Output("active-view", "data"),
       Output("btn-chatbot", "className"),
       Output("btn-articles", "className"),
       Output("btn-visualizations", "className"),
       Output("btn-statistics", "className")],
      [Input("btn-chatbot", "n_clicks"),
       Input("btn-articles", "n_clicks"),
       Input("btn-visualizations", "n_clicks"),
       Input("btn-statistics", "n_clicks")],
      [State("active-view", "data")]
  )
  def update_active_view(n_clicks_chatbot, n_clicks_articles, n_clicks_viz, n_clicks_stats, current_active_view):
      ctx = callback_context
      if not ctx.triggered:
          chatbot_class = "nav-item active" if current_active_view == "chatbot" else "nav-item"
          articles_class = "nav-item active" if current_active_view == "articles" else "nav-item"
          viz_class = "nav-item active" if current_active_view == "visualizations" else "nav-item"
          stats_class = "nav-item active" if current_active_view == "statistics" else "nav-item"
          return current_active_view, chatbot_class, articles_class, viz_class, stats_class
      
      button_id = ctx.triggered_id
      new_active_view = current_active_view
      
      if button_id == "btn-chatbot":
          new_active_view = "chatbot"
      elif button_id == "btn-articles":
          new_active_view = "articles"
      elif button_id == "btn-visualizations":
          new_active_view = "visualizations"
      elif button_id == "btn-statistics":
          new_active_view = "statistics"
      
      chatbot_class = "nav-item active" if new_active_view == "chatbot" else "nav-item"
      articles_class = "nav-item active" if new_active_view == "articles" else "nav-item"
      viz_class = "nav-item active" if new_active_view == "visualizations" else "nav-item"
      stats_class = "nav-item active" if new_active_view == "statistics" else "nav-item"

      return new_active_view, chatbot_class, articles_class, viz_class, stats_class

  @app.callback(
      Output("main-content", "children"),
      [Input("active-view", "data"),
       Input("chat-data", "data"),
       Input("current-results", "data"),
       Input("current-analysis-data", "data"), # Ajout
       Input("total-articles-found", "data")] # Ajout
  )
  def render_main_content(active_view, chat_data, results_data, analysis_data, total_articles_found):
      if active_view == "chatbot":
          return create_chatbot_view(chat_data)
      elif active_view == "articles":
          return create_articles_view(results_data, total_articles_found) # Passe le total
      elif active_view == "visualizations":
          return create_visualizations_view(results_data, analysis_data, total_articles_found) # Passe l'analyse et le total
      elif active_view == "statistics":
          # Use the global chatbot instance to get statistics
          return create_statistics_view(chatbot) # Pass the chatbot instance
      
      return html.Div()

  @app.callback(
      [Output("chat-data", "data"),
       Output("current-results", "data"),
       Output("current-analysis-data", "data"), # Ajout
       Output("total-articles-found", "data"), # Ajout
       Output("user-input", "value")],
      [Input("send-button", "n_clicks"),
       Input("user-input", "n_submit")],
      [State("user-input", "value"),
       State("chat-data", "data")],
      prevent_initial_call=True
  )
  def handle_chat_interaction(send_n_clicks, submit_n_clicks, user_input, chat_data):
      ctx = callback_context
      if not ctx.triggered:
          return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
      
      trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
      
      if not user_input or user_input.strip() == "":
          return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
      
      if trigger_id not in ["send-button", "user-input"]:
          return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
      
      chat_data = chat_data or []
      
      # Search for papers first
      search_results = chatbot.search_papers(user_input.strip(), k=15) # k=15 as in testing_chatbot.py
      
      # Generate response using the modified generate_response
      response_data = chatbot.generate_response(user_input.strip(), search_results)
      
      chat_data.append({
          "type": "user",
          "content": user_input.strip(),
          "timestamp": datetime.now().isoformat()
      })
      
      chat_data.append({
          "type": "bot",
          "content": response_data['response'],
          "timestamp": datetime.now().isoformat(),
          "articles": response_data['articles'], # Articles limités pour l'affichage
          "overall_relevance_score": response_data.get('overall_relevance_score', 0.0),
          "total_articles_found": response_data.get('total_articles_found', 0) # Total des articles trouvés
      })
      
      return chat_data, response_data['articles'], response_data['analysis_data'], response_data['total_articles_found'], ""

  # Callback pour les suggestions rapides
  @app.callback(
      Output("user-input", "value", allow_duplicate=True),
      [Input("suggestion-1", "n_clicks"),
       Input("suggestion-2", "n_clicks"),
       Input("suggestion-3", "n_clicks"),
       Input("suggestion-4", "n_clicks")],
      prevent_initial_call=True
  )
  def handle_quick_suggestions(n1, n2, n3, n4):
      ctx = callback_context
      if not ctx.triggered:
          return dash.no_update
      
      button_id = ctx.triggered_id
      
      suggestions = {
          "suggestion-1": "Qu'est-ce que le machine learning ?", # Nouvelle suggestion
          "suggestion-2": "Articles récents sur l'intelligence artificielle", 
          "suggestion-3": "Statistiques des publications",
          "suggestion-4": "Quelles sont les tendances actuelles ?"
      }
      
      return suggestions.get(button_id, dash.no_update)

  # Callback pour la navigation depuis les états vides
  @app.callback(
      Output("active-view", "data", allow_duplicate=True),
      [Input("empty-action-btn", "n_clicks")],
      prevent_initial_call=True
  )
  def handle_empty_action(n_clicks):
      if n_clicks:
          return "chatbot"
      return dash.no_update

def create_chatbot_view(chat_data):
  """Crée la vue du chatbot avec une UX optimisée"""
  
  # Messages de chat
  chat_elements = []
  if not chat_data:
      # Message de bienvenue
      chat_elements.append(
          html.Div([
              html.Div([
                  html.Div("🤖", className="bot-avatar"),
                  html.Div([
                      html.Div("Assistant IA", className="bot-name"),
                      html.Div("Bonjour ! Je suis votre assistant de recherche scientifique. Posez-moi vos questions sur les publications académiques.", className="bot-message welcome-message")
                  ], className="message-content")
              ], className="message-wrapper bot-wrapper")
          ], className="message-container welcome-container")
      )
  else:
      for msg in chat_data:
          if msg["type"] == "user":
              chat_elements.append(
                  html.Div([
                      html.Div([
                          html.Div(msg["content"], className="user-message"),
                          html.Div(datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M"), className="message-time")
                      ], className="message-content"),
                      html.Div("👤", className="user-avatar")
                  ], className="message-wrapper user-wrapper")
              )
          else:
              bot_content = [
                  html.Div([
                      html.Div("🤖", className="bot-avatar"),
                      html.Div([
                          html.Div("Assistant IA", className="bot-name"),
                          html.Div(msg["content"], className="bot-message"),
                          html.Div([
                              html.Div(datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M"), className="message-time"),
                              html.Div(f"Pertinence: {msg.get('overall_relevance_score', 0):.0%}", className="relevance-score") if msg.get('overall_relevance_score', 0) > 0 else None
                          ], className="message-meta")
                      ], className="message-content")
                  ], className="message-wrapper bot-wrapper")
              ]
              
              chat_elements.append(
                  html.Div(bot_content, className="message-container bot-container")
              )
  
  return html.Div([
      # Zone de chat
      html.Div([
          html.Div(chat_elements, id="chat-messages", className="chat-messages"),
      ], className="chat-area"),
      
      # Zone de saisie améliorée
      html.Div([
          html.Div([
              dcc.Input(
                  id="user-input",
                  type="text",
                  placeholder="Posez votre question sur la recherche scientifique...",
                  className="message-input",
                  maxLength=1000,
                  debounce=True
              ),
              html.Button([
                  html.Span("✈️", className="send-icon")
              ], id="send-button", className="send-button", title="Envoyer (Ctrl+Enter)")
          ], className="input-group"),
          
          # Suggestions rapides
          html.Div([
              html.Button("❓ Qu'est-ce que l'IA ?", id="suggestion-1", className="quick-suggestion"), # Nouvelle suggestion
              html.Button("🧠 Intelligence Artificielle", id="suggestion-2", className="quick-suggestion"), 
              html.Button("📊 Statistiques", id="suggestion-3", className="quick-suggestion"),
              html.Button("📈 Tendances", id="suggestion-4", className="quick-suggestion")
          ], className="quick-suggestions")
      ], className="input-area"),
      
      # Indicateur de frappe
      html.Div(id="typing-indicator", className="typing-indicator", style={"display": "none"})
  ], className="chatbot-view")

def create_articles_view(results_data, total_articles_found):
  """Vue des articles avec design amélioré"""
  if not results_data:
      return html.Div([
          html.Div([
              html.Div("📄", className="empty-icon"),
              html.H3("Aucun article trouvé", className="empty-title"),
              html.P("Commencez une recherche dans l'assistant pour voir les résultats ici.", className="empty-description"),
              html.Button("Aller à l'assistant →", id="empty-action-btn", className="empty-action")
          ], className="articles-view")
      ], className="articles-view")
  
  # Filtres et tri
  filter_section = html.Div([
      html.Div([
          html.H3(f"{total_articles_found} articles pertinents trouvés", className="results-title"), # Affiche le total
          html.Div([
              dcc.Dropdown(
                  options=[
                      {"label": "Plus pertinent", "value": "relevance"},
                      {"label": "Plus récent", "value": "date"},
                      {"label": "Alphabétique", "value": "title"}
                  ],
                  value="relevance",
                  className="sort-dropdown",
                  placeholder="Trier par..."
              )
          ], className="sort-controls")
      ], className="results-header")
  ], className="filter-section")
  
  # Liste des articles
  articles_list = []
  for i, article in enumerate(results_data, 1):
      # Score de pertinence avec couleur
      score = article['score']
      if score >= 0.8:
          score_class = "score-high"
      elif score >= 0.6:
          score_class = "score-medium"
      else:
          score_class = "score-low"
      
      article_card = html.Div([
          html.Div([
              html.Div([
                  html.Span(f"#{i}", className="article-rank"),
                  html.Div([
                      html.Div(f"{score:.1%}", className=f"relevance-badge {score_class}"),
                      html.Button("🔖", className="bookmark-btn", title="Sauvegarder")
                  ], className="article-actions")
              ], className="article-header"),
              
              html.H4(article['title'], className="article-title"),
              
              html.Div([
                  html.Span("📅", className="meta-icon"),
                  html.Span(article.get('published', 'Date inconnue')[:10], className="meta-text"), # Changed to 'published'
                  html.Span("•", className="meta-separator"),
                  html.Span("📖", className="meta-icon"),
                  html.Span(article.get('journal_ref', 'Journal inconnu')[:50] + "..." if len(article.get('journal_ref', '')) > 50 else article.get('journal_ref', 'Journal inconnu'), className="meta-text") # Changed to 'journal_ref'
              ], className="article-meta"),
              
              html.P(
                  article.get('summary', 'Pas de résumé disponible')[:300] + "..." if len(article.get('summary', '')) > 300 
                  else article.get('summary', 'Pas de résumé disponible'), 
                  className="article-abstract"
              ),
              
              html.Div([
                  html.Div([
                      html.Span("🏷️", className="tag-icon"),
                      *[html.Span(kw.strip(), className="keyword-tag") for kw in (article.get('keywords', '') or '').split(';')[:5] if kw.strip()]
                  ], className="article-keywords") if article.get('keywords') else None,
                  
                  html.Div([
                      html.Button("Lire plus", className="read-more-btn"),
                      html.Button("📋", className="copy-btn", title="Copier la référence"),
                      html.A("🔗", href=article.get('abstract_url'), target="_blank", className="link-btn", title="Ouvrir l'article arXiv") if article.get('abstract_url') else None # Changed to abstract_url
                  ], className="article-footer")
              ], className="article-bottom")
          ], className="article-content")
      ], className="article-card")
      
      articles_list.append(article_card)
  
  return html.Div([
      filter_section,
      html.Div(articles_list, className="articles-grid")
  ], className="articles-view")

def create_visualizations_view(results_data, analysis_data, total_articles_found):
  """Vue des visualisations avec design moderne"""
  if not results_data:
      return html.Div([
          html.Div([
              html.Div("📊", className="empty-icon"),
              html.H3("Aucune donnée à visualiser", className="empty-title"),
              html.P("Les graphiques apparaîtront ici après une recherche.", className="empty-description")
          ], className="empty-state")
      ], className="viz-view")
  
  # Métriques rapides
  overall_relevance_score = sum(a['score'] for a in results_data) / len(results_data) if results_data else 0
  total_unique_keywords = analysis_data.get('total_unique_keywords', 0)

  metrics = html.Div([
      html.Div([
          html.Div("📄", className="metric-icon"),
          html.Div([
              html.Div(str(total_articles_found), className="metric-value"), # Total articles trouvés
              html.Div("Articles pertinents", className="metric-label")
          ], className="metric-content")
      ], className="metric-card"),
      
      html.Div([
          html.Div("⭐", className="metric-icon"),
          html.Div([
              html.Div(f"{overall_relevance_score:.1%}", className="metric-value"),
              html.Div("Pertinence moy.", className="metric-label")
          ], className="metric-content")
      ], className="metric-card"),
      
      html.Div([
          html.Div("🏷️", className="metric-icon"),
          html.Div([
              html.Div(str(total_unique_keywords), className="metric-value"), # Total mots-clés uniques
              html.Div("Mots-clés uniques", className="metric-label")
          ], className="metric-content")
      ], className="metric-card")
  ], className="metrics-row")
  
  # Graphiques
  charts = create_enhanced_charts(results_data)
  
  return html.Div([
      metrics,
      html.Div(charts, className="charts-grid")
  ], className="viz-view")

def create_statistics_view(chatbot_instance): # Changed parameter name
  """Vue des statistiques globales de la base de données"""
  if chatbot_instance is None or not chatbot_instance.is_ready:
      return html.Div([
          html.Div([
              html.Div("📈", className="empty-icon"),
              html.H3("Base de données non disponible", className="empty-title"),
              html.P("Veuillez vous assurer que le chatbot est initialisé et que les données ont été extraites.", className="empty-description")
          ], className="stats-view")
      ], className="stats-view")

  # Récupérer les statistiques globales de la base de données
  global_stats = chatbot_instance.get_paper_statistics() # Call method on chatbot_instance
  
  if not global_stats:
      return html.Div([
          html.Div([
              html.Div("📈", className="empty-icon"),
              html.H3("Aucune statistique globale disponible", className="empty-title"),
              html.P("La base de données est vide ou inaccessible. Veuillez extraire des données.", className="empty-description")
          ], className="stats-view")
      ], className="stats-view")

  total_articles_db = global_stats.get('total_articles', 0)
  articles_by_year = global_stats.get('articles_by_year', {})
  top_publications = global_stats.get('top_publications', {}) # Changed to dict
  top_keywords_db = global_stats.get('top_keywords', {}) # Changed to dict
  total_authors_db = global_stats.get('total_authors', 0)
  total_affiliations_db = global_stats.get('total_affiliations', 0)

  return html.Div([
      html.H2("📊 Statistiques Globales de la Base de Données", className="section-title"),
      
      html.Div([
          html.Div([
              html.Div("Total Articles", className="metric-label"),
              html.Div(str(total_articles_db), className="metric-value")
          ], className="metric-card"),
          html.Div([
              html.Div("Total Auteurs Uniques", className="metric-label"),
              html.Div(str(total_authors_db), className="metric-value")
          ], className="metric-card"),
          html.Div([
              html.Div("Total Affiliations Uniques", className="metric-label"),
              html.Div(str(total_affiliations_db), className="metric-value")
          ], className="metric-card"),
      ], className="metrics-row"),

      html.Div([
          html.H3("📅 Articles par Année", className="section-title"),
          dcc.Graph(
              figure=px.bar(
                  x=list(articles_by_year.keys()),
                  y=list(articles_by_year.values()),
                  title="Répartition des publications par année",
                  labels={'x': 'Année', 'y': 'Nombre d\'articles'},
                  color_discrete_sequence=px.colors.qualitative.Pastel
              ).update_layout(
                  plot_bgcolor='rgba(0,0,0,0)',
                  paper_bgcolor='rgba(0,0,0,0)',
                  font=dict(color='#f0f6fc', size=12),
                  title=dict(font=dict(size=16, color='#f0f6fc')),
                  margin=dict(l=20, r=20, t=40, b=20)
              ),
              config={'displayModeBar': False},
              className="chart-container"
          )
      ], className="stats-section"),

      html.Div([
          html.H3("📚 Top 10 Publications", className="section-title"),
          html.Div([
              html.Div([
                  html.Div("Rang", className="table-header"),
                  html.Div("Publication", className="table-header"),
                  html.Div("Articles", className="table-header")
              ], className="table-row table-header-row"),
              *[html.Div([
                  html.Div(f"#{i+1}", className="table-cell rank-cell"),
                  html.Div(pub[0], className="table-cell keyword-cell"),
                  html.Div(str(pub[1]), className="table-cell count-cell")
              ], className="table-row") for i, pub in enumerate(list(top_publications.items()))] # Iterate over items
          ], className="data-table")
      ], className="stats-section"),

      html.Div([
          html.H3("🔑 Top 15 Mots-clés (Base de Données)", className="section-title"),
          html.Div([
              html.Div([
                  html.Div("Rang", className="table-header"),
                  html.Div("Mot-clé", className="table-header"),
                  html.Div("Fréquence", className="table-header")
              ], className="table-row table-header-row"),
              *[html.Div([
                  html.Div(f"#{i+1}", className="table-cell rank-cell"),
                  html.Div(kw[0], className="table-cell keyword-cell"),
                  html.Div(str(kw[1]), className="table-cell count-cell")
              ], className="table-row") for i, kw in enumerate(list(top_keywords_db.items()))] # Iterate over items
          ], className="data-table")
      ], className="stats-section")

  ], className="stats-view")

def create_enhanced_charts(results_data):
  """Crée des graphiques améliorés"""
  charts = []
  
  # Graphique des scores de pertinence
  scores = [article['score'] for article in results_data]
  titles = [article['title'][:40] + "..." if len(article['title']) > 40 
            else article['title'] for article in results_data]
  
  fig_scores = px.bar(
      x=scores[:10],  # Top 10 seulement
      y=titles[:10],
      orientation='h',
      title="Top 10 - Scores de Pertinence",
      color=scores[:10],
      color_continuous_scale="Viridis"
  )
  
  fig_scores.update_layout(
      height=500,
      plot_bgcolor='rgba(0,0,0,0)',
      paper_bgcolor='rgba(0,0,0,0)',
      font=dict(color='#f0f6fc', size=12),
      title=dict(font=dict(size=16, color='#f0f6fc')),
      showlegend=False,
      margin=dict(l=20, r=20, t=40, b=20)
  )
  
  charts.append(
      html.Div([
          dcc.Graph(figure=fig_scores, config={'displayModeBar': False})
      ], className="chart-container")
  )
  
  return charts

# CSS et JavaScript personnalisés pour une UX parfaite
app.index_string = '''
<!DOCTYPE html>
<html>
  <head>
      {%metas%}
      <title>{%title%}</title>
      {%favicon%}
      {%css%}
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
      <style>
          :root {
              /* Couleurs principales */
              --bg-primary: #0a0e1a;
              --bg-secondary: #111827;
              --bg-tertiary: #1f2937;
              --bg-quaternary: #374151;
              --bg-glass: rgba(17, 24, 39, 0.8);
              
              /* Texte */
              --text-primary: #f9fafb;
              --text-secondary: #d1d5db;
              --text-muted: #9ca3af;
              --text-inverse: #111827;
              
              /* Accents */
              --accent-primary: #3b82f6;
              --accent-secondary: #8b5cf6;
              --accent-success: #10b981;
              --accent-warning: #f59e0b;
              --accent-danger: #ef4444;
              
              /* Gradients */
              --gradient-primary: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
              --gradient-success: linear-gradient(135deg, #10b981 0%, #059669 100%);
              --gradient-warm: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
              
              /* Ombres */
              --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
              --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
              --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
              --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
              --shadow-glow: 0 0 20px rgba(59, 130, 246, 0.3);
              
              /* Bordures */
              --border-primary: #374151;
              --border-secondary: #4b5563;
              --border-accent: #3b82f6;
              
              /* Rayons */
              --radius-sm: 0.375rem;
              --radius-md: 0.5rem;
              --radius-lg: 0.75rem;
              --radius-xl: 1rem;
              --radius-2xl: 1.5rem;
              
              /* Transitions */
              --transition-fast: 0.15s ease;
              --transition-normal: 0.3s ease;
              --transition-slow: 0.5s ease;
          }

          * {
              box-sizing: border-box;
              margin: 0;
              padding: 0;
          }

          body {
              font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
              background: var(--bg-primary);
              color: var(--text-primary);
              line-height: 1.6;
              font-size: 14px;
              overflow-x: hidden;
              -webkit-font-smoothing: antialiased;
              -moz-osx-font-smoothing: grayscale;
          }

          /* Layout principal */
          .app-container {
              min-height: 100vh;
              display: flex;
              flex-direction: column;
              background: radial-gradient(ellipse at top, rgba(59, 130, 246, 0.1) 0%, transparent 50%);
          }

          /* Header amélioré */
          .header {
              background: var(--bg-glass);
              backdrop-filter: blur(20px);
              border-bottom: 1px solid var(--border-primary);
              position: sticky;
              top: 0;
              z-index: 1000;
              padding: 0;
          }

          .header-content {
              max-width: 1400px;
              margin: 0 auto;
              padding: 1rem 2rem;
              display: flex;
              align-items: center;
              justify-content: space-between;
              gap: 2rem;
          }

          .header-brand {
              display: flex;
              align-items: center;
              gap: 1rem;
          }

          .logo-icon {
              font-size: 2rem;
              background: var(--gradient-primary);
              -webkit-background-clip: text;
              -webkit-text-fill-color: transparent;
              background-clip: text;
              filter: drop-shadow(0 0 10px rgba(59, 130, 246, 0.5));
          }

          .header-text {
              display: flex;
              flex-direction: column;
          }

          .header-title {
              font-size: 1.5rem;
              font-weight: 700;
              color: var(--text-primary);
              margin: 0;
              line-height: 1.2;
          }

          .header-subtitle {
              font-size: 0.875rem;
              color: var(--text-muted);
              margin: 0;
              font-weight: 400;
          }

          /* Navigation */
          .header-nav {
              display: flex;
              gap: 0.5rem;
              background: var(--bg-tertiary);
              padding: 0.5rem;
              border-radius: var(--radius-xl);
              border: 1px solid var(--border-primary);
          }

          .nav-item {
              display: flex;
              align-items: center;
              gap: 0.5rem;
              padding: 0.75rem 1.25rem;
              border: none;
              background: transparent;
              color: var(--text-secondary);
              border-radius: var(--radius-lg);
              cursor: pointer;
              transition: all var(--transition-fast);
              font-weight: 500;
              font-size: 0.875rem;
              position: relative;
              overflow: hidden;
          }

          .nav-item::before {
              content: '';
              position: absolute;
              top: 0;
              left: 0;
              right: 0;
              bottom: 0;
              background: var(--gradient-primary);
              opacity: 0;
              transition: opacity var(--transition-fast);
              z-index: -1;
          }

          .nav-item:hover {
              color: var(--text-primary);
              transform: translateY(-1px);
          }

          .nav-item:hover::before {
              opacity: 0.1;
          }

          .nav-item.active {
              background: var(--gradient-primary);
              color: white;
              box-shadow: var(--shadow-glow);
              font-weight: 600;
          }

          .nav-item.active::before {
              opacity: 1;
          }

          .nav-icon {
              font-size: 1rem;
          }

          .nav-text {
              font-size: 0.875rem;
          }

          /* Contenu principal */
          .main-content {
              flex: 1;
              max-width: 1400px;
              margin: 0 auto;
              width: 100%;
              padding: 2rem;
              min-height: calc(100vh - 100px);
          }

          /* Vue Chatbot */
          .chatbot-view {
              display: flex;
              flex-direction: column;
              height: calc(100vh - 140px);
              max-width: 900px;
              margin: 0 auto;
              gap: 1rem;
          }

          .chat-area {
              flex: 1;
              background: var(--bg-secondary);
              border: 1px solid var(--border-primary);
              border-radius: var(--radius-2xl);
              overflow: hidden;
              display: flex;
              flex-direction: column;
              box-shadow: var(--shadow-xl);
          }

          .chat-messages {
              flex: 1;
              overflow-y: auto;
              padding: 2rem;
              display: flex;
              flex-direction: column;
              gap: 1.5rem;
              scroll-behavior: smooth;
          }

          /* Messages */
          .message-container {
              display: flex;
              flex-direction: column;
              animation: slideInUp 0.4s ease-out;
          }

          @keyframes slideInUp {
              from {
                  opacity: 0;
                  transform: translateY(20px);
              }
              to {
                  opacity: 1;
                  transform: translateY(0);
              }
          }

          .message-wrapper {
              display: flex;
              align-items: flex-start;
              gap: 1rem;
              max-width: 85%;
          }

          .user-wrapper {
              align-self: flex-end;
              flex-direction: row-reverse;
          }

          .bot-wrapper {
              align-self: flex-start;
          }

          .user-avatar, .bot-avatar {
              width: 2.5rem;
              height: 2.5rem;
              border-radius: 50%;
              display: flex;
              align-items: center;
              justify-content: center;
              font-size: 1.25rem;
              flex-shrink: 0;
          }

          .user-avatar {
              background: var(--gradient-warm);
              color: white;
          }

          .bot-avatar {
              background: var(--gradient-primary);
              color: white;
              box-shadow: var(--shadow-glow);
          }

          .message-content {
              flex: 1;
              display: flex;
              flex-direction: column;
              gap: 0.5rem;
          }

          .bot-name {
              font-size: 0.75rem;
              font-weight: 600;
              color: var(--text-muted);
              text-transform: uppercase;
              letter-spacing: 0.05em;
          }

          .user-message {
              background: var(--gradient-primary);
              color: white;
              padding: 1rem 1.25rem;
              border-radius: 1.5rem 1.5rem 0.5rem 1.5rem;
              font-size: 0.95rem;
              line-height: 1.5;
              box-shadow: var(--shadow-lg);
              word-wrap: break-word;
          }

          .bot-message {
              background: var(--bg-tertiary);
              color: var(--text-primary);
              padding: 1rem 1.25rem;
              border-radius: 1.5rem 1.5rem 1.5rem 0.5rem;
              font-size: 0.95rem;
              line-height: 1.6;
              border: 1px solid var(--border-primary);
              white-space: pre-wrap;
              word-wrap: break-word;
          }

          .welcome-message {
              background: var(--gradient-success);
              color: white;
              box-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
          }

          .message-meta {
              display: flex;
              align-items: center;
              gap: 1rem;
              margin-top: 0.25rem;
          }

          .message-time {
              font-size: 0.75rem;
              color: var(--text-muted);
          }

          .relevance-score {
              font-size: 0.75rem;
              color: var(--accent-success);
              font-weight: 500;
          }

          /* Zone de saisie */
          .input-area {
              padding: 1.5rem;
              background: var(--bg-secondary);
              border-top: 1px solid var(--border-primary);
          }

          .input-group {
              display: flex;
              gap: 1rem;
              align-items: center;
              background: var(--bg-tertiary);
              border: 2px solid var(--border-primary);
              border-radius: var(--radius-xl);
              padding: 1rem;
              transition: border-color var(--transition-fast);
          }

          .input-group:focus-within {
              border-color: var(--accent-primary);
              box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
          }

          .message-input {
              flex: 1;
              background: transparent;
              border: none;
              color: var(--text-primary);
              font-size: 1rem;
              line-height: 1.5;
              outline: none;
              font-family: inherit;
              padding: 0.75rem 0;
          }

          .message-input::placeholder {
              color: var(--text-muted);
          }

          .send-button {
              width: 3rem;
              height: 3rem;
              background: var(--gradient-primary);
              border: none;
              border-radius: 50%;
              color: white;
              cursor: pointer;
              display: flex;
              align-items: center;
              justify-content: center;
              transition: all var(--transition-fast);
              box-shadow: var(--shadow-lg);
              flex-shrink: 0;
          }

          .send-button:hover {
              transform: translateY(-2px) scale(1.05);
              box-shadow: var(--shadow-glow);
          }

          .send-button:active {
              transform: translateY(0) scale(0.95);
          }

          .send-icon {
              font-size: 1.25rem;
          }

          /* Suggestions rapides */
          .quick-suggestions {
              display: flex;
              gap: 0.75rem;
              margin-top: 1rem;
              flex-wrap: wrap;
          }

          .quick-suggestion {
              background: var(--bg-tertiary);
              border: 1px solid var(--border-primary);
              color: var(--text-secondary);
              padding: 0.5rem 1rem;
              border-radius: var(--radius-lg);
              font-size: 0.875rem;
              cursor: pointer;
              transition: all var(--transition-fast);
              white-space: nowrap;
          }

          .quick-suggestion:hover {
              background: var(--bg-quaternary);
              color: var(--text-primary);
              border-color: var(--accent-primary);
              transform: translateY(-1px);
          }

          /* Vue Articles */
          .articles-view {
              max-width: 1000px;
              margin: 0 auto;
          }

          .filter-section {
              background: var(--bg-secondary);
              border: 1px solid var(--border-primary);
              border-radius: var(--radius-xl);
              padding: 1.5rem;
              margin-bottom: 2rem;
              box-shadow: var(--shadow-lg);
          }

          .results-header {
              display: flex;
              justify-content: space-between;
              align-items: center;
              gap: 2rem;
          }

          .results-title {
              font-size: 1.5rem;
              font-weight: 600;
              color: var(--text-primary);
              margin: 0;
          }

          .sort-controls {
              min-width: 200px;
          }

          .articles-grid {
              display: grid;
              gap: 1.5rem;
          }

          .article-card {
              background: var(--bg-secondary);
              border: 1px solid var(--border-primary);
              border-radius: var(--radius-xl);
              overflow: hidden;
              transition: all var(--transition-normal);
              box-shadow: var(--shadow-lg);
              position: relative;
          }

          .article-card::before {
              content: '';
              position: absolute;
              top: 0;
              left: 0;
              right: 0;
              height: 3px;
              background: var(--gradient-primary);
              opacity: 0;
              transition: opacity var(--transition-fast);
          }

          .article-card:hover {
              transform: translateY(-4px);
              box-shadow: var(--shadow-xl);
              border-color: var(--accent-primary);
          }

          .article-card:hover::before {
              opacity: 1;
          }

          .article-content {
              padding: 1.5rem;
          }

          .article-header {
              display: flex;
              justify-content: space-between;
              align-items: center;
              margin-bottom: 1rem;
          }

          .article-rank {
              background: var(--bg-tertiary);
              color: var(--text-muted);
              padding: 0.25rem 0.75rem;
              border-radius: var(--radius-lg);
              font-size: 0.875rem;
              font-weight: 600;
          }

          .article-actions {
              display: flex;
              gap: 0.5rem;
              align-items: center;
          }

          .relevance-badge {
              padding: 0.25rem 0.75rem;
              border-radius: var(--radius-lg);
              font-size: 0.75rem;
              font-weight: 600;
              color: white;
          }

          .score-high {
              background: var(--accent-success);
          }

          .score-medium {
              background: var(--accent-warning);
          }

          .score-low {
              background: var(--accent-danger);
          }

          .bookmark-btn {
              background: transparent;
              border: 1px solid var(--border-primary);
              color: var(--text-muted);
              width: 2rem;
              height: 2rem;
              border-radius: 50%;
              cursor: pointer;
              transition: all var(--transition-fast);
              display: flex;
              align-items: center;
              justify-content: center;
          }

          .bookmark-btn:hover {
              background: var(--accent-warning);
              color: white;
              border-color: var(--accent-warning);
          }

          .article-title {
              font-size: 1.25rem;
              font-weight: 600;
              color: var(--text-primary);
              line-height: 1.4;
              margin-bottom: 1rem;
          }

          .article-meta {
              display: flex;
              align-items: center;
              gap: 0.5rem;
              margin-bottom: 1rem;
              flex-wrap: wrap;
          }

          .meta-icon {
              color: var(--text-muted);
              font-size: 0.875rem;
          }

          .meta-text {
              color: var(--text-secondary);
              font-size: 0.875rem;
          }

          .meta-separator {
              color: var(--text-muted);
              margin: 0 0.25rem;
          }

          .article-abstract {
              color: var(--text-secondary);
              line-height: 1.6;
              margin-bottom: 1.5rem;
          }

          .article-bottom {
              display: flex;
              flex-direction: column;
              gap: 1rem;
          }

          .article-keywords {
              display: flex;
              align-items: center;
              gap: 0.5rem;
              flex-wrap: wrap;
          }

          .tag-icon {
              color: var(--text-muted);
              font-size: 0.875rem;
          }

          .keyword-tag {
              background: var(--bg-tertiary);
              color: var(--text-secondary);
              padding: 0.25rem 0.75rem;
              border-radius: var(--radius-lg);
              font-size: 0.75rem;
              border: 1px solid var(--border-primary);
          }

          .article-footer {
              display: flex;
              gap: 0.75rem;
              align-items: center;
          }

          .read-more-btn {
              background: var(--gradient-primary);
              color: white;
              border: none;
              padding: 0.5rem 1rem;
              border-radius: var(--radius-lg);
              font-size: 0.875rem;
              font-weight: 500;
              cursor: pointer;
              transition: all var(--transition-fast);
          }

          .read-more-btn:hover {
              transform: translateY(-1px);
              box-shadow: var(--shadow-lg);
          }

          .copy-btn, .link-btn {
              background: transparent;
              border: 1px solid var(--border-primary);
              color: var(--text-muted);
              width: 2rem;
              height: 2rem;
              border-radius: 50%;
              cursor: pointer;
              transition: all var(--transition-fast);
              display: flex;
              align-items: center;
              justify-content: center;
          }

          .copy-btn:hover {
              background: var(--accent-secondary);
              color: white;
              border-color: var(--accent-secondary);
          }

          .link-btn:hover {
              background: var(--accent-primary);
              color: white;
              border-color: var(--accent-primary);
          }

          /* Vue Visualisations */
          .viz-view {
              max-width: 1200px;
              margin: 0 auto;
          }

          .metrics-row {
              display: grid;
              grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
              gap: 1.5rem;
              margin-bottom: 2rem;
          }

          .metric-card {
              background: var(--bg-secondary);
              border: 1px solid var(--border-primary);
              border-radius: var(--radius-xl);
              padding: 1.5rem;
              display: flex;
              flex-direction: column; /* Changement pour aligner le texte */
              align-items: flex-start; /* Alignement à gauche */
              gap: 0.5rem; /* Espacement réduit */
              transition: all var(--transition-normal);
              box-shadow: var(--shadow-lg);
              position: relative;
              overflow: hidden;
          }

          .metric-card::before {
              content: '';
              position: absolute;
              top: 0;
              left: 0;
              right: 0;
              height: 3px;
              background: var(--gradient-primary);
          }

          .metric-card:hover {
              transform: translateY(-2px);
              box-shadow: var(--shadow-xl);
          }

          .metric-icon {
              font-size: 2rem;
              opacity: 0.8;
          }

          .metric-content {
              flex: 1;
          }

          .metric-value {
              font-size: 2rem;
              font-weight: 700;
              color: var(--text-primary);
              line-height: 1;
          }

          .metric-label {
              font-size: 0.875rem;
              color: var(--text-muted);
              margin-top: 0.25rem;
          }

          .charts-grid {
              display: grid;
              gap: 2rem;
          }

          .chart-container {
              background: var(--bg-secondary);
              border: 1px solid var(--border-primary);
              border-radius: var(--radius-xl);
              padding: 1.5rem;
              box-shadow: var(--shadow-lg);
          }

          /* Vue Statistiques */
          .stats-view {
              max-width: 1000px;
              margin: 0 auto;
          }

          .stats-section {
              background: var(--bg-secondary);
              border: 1px solid var(--border-primary);
              border-radius: var(--radius-xl);
              padding: 2rem;
              margin-bottom: 2rem;
              box-shadow: var(--shadow-lg);
          }

          .section-title {
              font-size: 1.5rem;
              font-weight: 600;
              color: var(--text-primary);
              margin-bottom: 1.5rem;
              display: flex;
              align-items: center;
              gap: 0.5rem;
          }

          .data-table {
              border-radius: var(--radius-lg);
              overflow: hidden;
              border: 1px solid var(--border-primary);
          }

          .table-row {
              display: grid;
              grid-template-columns: 60px 1fr 100px; /* Adjusted for 3 columns */
              gap: 1rem;
              align-items: center;
          }

          .table-header-row {
              background: var(--bg-tertiary);
              border-bottom: 1px solid var(--border-primary);
          }

          .table-header {
              padding: 1rem;
              font-weight: 600;
              color: var(--text-primary);
              font-size: 0.875rem;
              text-transform: uppercase;
              letter-spacing: 0.05em;
          }

          .table-row:not(.table-header-row) {
              border-bottom: 1px solid var(--border-primary);
              transition: background-color var(--transition-fast);
          }

          .table-row:not(.table-header-row):hover {
              background: var(--bg-tertiary);
          }

          .table-row:not(.table-header-row):last-child {
              border-bottom: none;
          }

          .table-cell {
              padding: 1rem;
              color: var(--text-secondary);
          }

          .rank-cell {
              font-weight: 600;
              color: var(--text-muted);
              text-align: center;
          }

          .keyword-cell {
              font-weight: 500;
              color: var(--text-primary);
          }

          .count-cell {
              font-weight: 600;
              color: var(--accent-primary);
              text-align: center;
          }

          .percent-cell {
              font-weight: 500;
              color: var(--accent-success);
              text-align: center;
          }

          /* États vides */
          .empty-state {
              display: flex;
              flex-direction: column;
              align-items: center;
              justify-content: center;
              padding: 4rem 2rem;
              text-align: center;
              min-height: 400px;
          }

          .empty-icon {
              font-size: 4rem;
              opacity: 0.5;
              margin-bottom: 1rem;
          }

          .empty-title {
              font-size: 1.5rem;
              font-weight: 600;
              color: var(--text-primary);
              margin-bottom: 0.5rem;
          }

          .empty-description {
              color: var(--text-muted);
              margin-bottom: 2rem;
              max-width: 400px;
          }

          .empty-action {
              background: var(--gradient-primary);
              color: white;
              border: none;
              padding: 0.75rem 1.5rem;
              border-radius: var(--radius-lg);
              font-weight: 500;
              cursor: pointer;
              transition: all var(--transition-fast);
          }

          .empty-action:hover {
              transform: translateY(-2px);
              box-shadow: var(--shadow-lg);
          }

          /* Page d'erreur */
          .error-state {
              background: radial-gradient(ellipse at center, rgba(239, 68, 68, 0.1) 0%, transparent 50%);
          }

          .error-page {
              display: flex;
              align-items: center;
              justify-content: center;
              min-height: calc(100vh - 100px);
              padding: 2rem;
          }

          .error-container {
              max-width: 800px;
              width: 100%;
              background: var(--bg-secondary);
              border: 1px solid var(--border-primary);
              border-radius: var(--radius-2xl);
              padding: 3rem;
              box-shadow: var(--shadow-xl);
          }

          .error-header {
              text-align: center;
              margin-bottom: 3rem;
          }

          .error-icon {
              font-size: 4rem;
              margin-bottom: 1rem;
          }

          .error-title {
              font-size: 2rem;
              font-weight: 700;
              color: var(--text-primary);
              margin-bottom: 1rem;
          }

          .error-message {
              color: var(--accent-danger);
              font-size: 1.125rem;
              margin-bottom: 0;
          }

          .setup-section {
              background: var(--bg-tertiary);
              border: 1px solid var(--border-primary);
              border-radius: var(--radius-xl);
              padding: 2rem;
          }

          .setup-title {
              font-size: 1.5rem;
              font-weight: 600;
              color: var(--text-primary);
              margin-bottom: 2rem;
              display: flex;
              align-items: center;
              gap: 0.5rem;
          }

          .setup-steps {
              display: flex;
              flex-direction: column;
              gap: 2rem;
          }

          .setup-step {
              display: flex;
              gap: 1.5rem;
              align-items: flex-start;
          }

          .step-number {
              width: 2.5rem;
              height: 2.5rem;
              background: var(--gradient-primary);
              color: white;
              border-radius: 50%;
              display: flex;
              align-items: center;
              justify-content: center;
              font-weight: 600;
              flex-shrink: 0;
              box-shadow: var(--shadow-glow);
          }

          .step-content {
              flex: 1;
          }

          .step-content h4 {
              font-size: 1.125rem;
              font-weight: 600;
              color: var(--text-primary);
              margin-bottom: 0.5rem;
          }

          .step-content p {
              color: var(--text-secondary);
              margin-bottom: 1rem;
              line-height: 1.6;
          }

          .step-link {
              color: var(--accent-primary);
              text-decoration: none;
              font-weight: 500;
              transition: color var(--transition-fast);
          }

          .step-link:hover {
              color: var(--accent-secondary);
          }

          .code-block {
              display: block;
              background: var(--bg-primary);
              color: var(--text-primary);
              padding: 1rem;
              border-radius: var(--radius-lg);
              font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
              font-size: 0.875rem;
              line-height: 1.5;
              border: 1px solid var(--border-primary);
              overflow-x: auto;
              white-space: pre-wrap; /* Changed from 'pre' to 'pre-wrap' */
              word-break: break-word; /* Added for better word breaking */
          }

          /* Add this media query for smaller screens */
          @media (max-width: 480px) {
              .code-block {
                  font-size: 0.75rem; /* Smaller font size on very small screens */
                  padding: 0.75rem;
              }
          }

          /* Scrollbars personnalisées */
          ::-webkit-scrollbar {
              width: 8px;
              height: 8px;
          }

          ::-webkit-scrollbar-track {
              background: var(--bg-tertiary);
              border-radius: var(--radius-sm);
          }

          ::-webkit-scrollbar-thumb {
              background: var(--border-secondary);
              border-radius: var(--radius-sm);
              transition: background var(--transition-fast);
          }

          ::-webkit-scrollbar-thumb:hover {
              background: var(--text-muted);
          }

          /* Responsive Design */
          @media (max-width: 1024px) {
              .header-content {
                  padding: 1rem;
                  flex-direction: column;
                  gap: 1rem;
              }

              .header-nav {
                  width: 100%;
                  justify-content: center;
              }

              .main-content {
                  padding: 1rem;
              }

              .chatbot-view {
                  height: calc(100vh - 180px);
              }
          }

          @media (max-width: 768px) {
              .header-nav {
                  flex-wrap: wrap;
                  gap: 0.25rem;
              }

              .nav-item {
                  padding: 0.5rem 0.75rem;
                  font-size: 0.8rem;
              }

              .nav-text {
                  display: none;
              }

              .chat-messages {
                  padding: 1rem;
              }

              .message-wrapper {
                  max-width: 95%;
              }

              .input-area {
                  padding: 1rem;
              }

              .quick-suggestions {
                  gap: 0.5rem;
              }

              .quick-suggestion {
                  font-size: 0.8rem;
                  padding: 0.4rem 0.8rem;
              }

              .metrics-row {
                  grid-template-columns: 1fr;
              }

              .results-header {
                  flex-direction: column;
                  align-items: stretch;
                  gap: 1rem;
              }

              .table-row {
                  grid-template-columns: 50px 1fr 80px; /* Adjusted for 3 columns */
                  gap: 0.5rem;
              }

              .table-cell {
                  padding: 0.75rem 0.5rem;
                  font-size: 0.875rem;
              }

              .setup-step {
                  gap: 1rem;
              }

              .error-container {
                  padding: 2rem;
              }
          }

          @media (max-width: 480px) {
              .header-title {
                  font-size: 1.25rem;
              }

              .header-subtitle {
                  font-size: 0.8rem;
              }

              .logo-icon {
                  font-size: 1.5rem;
              }

              .user-avatar, .bot-avatar {
                  width: 2rem;
                  height: 2rem;
                  font-size: 1rem;
              }

              .message-input {
                  font-size: 0.9rem;
              }

              .send-button {
                  width: 2.5rem;
                  height: 2.5rem;
              }

              .article-content {
                  padding: 1rem;
              }

              .article-title {
                  font-size: 1.125rem;
              }

              .code-block {
                  font-size: 0.8rem;
                  padding: 0.75rem;
              }
          }

          /* Animations et effets */
          @keyframes pulse {
              0%, 100% {
                  opacity: 1;
              }
              50% {
                  opacity: 0.5;
              }
          }

          @keyframes fadeIn {
              from {
                  opacity: 0;
              }
              to {
                  opacity: 1;
              }
          }

          @keyframes slideInLeft {
              from {
                  opacity: 0;
                  transform: translateX(-20px);
              }
              to {
                  opacity: 1;
                  transform: translateX(0);
              }
          }

          @keyframes slideInRight {
              from {
                  opacity: 0;
                  transform: translateX(20px);
              }
              to {
                  opacity: 1;
                  transform: translateX(0);
              }
          }

          /* Indicateur de frappe */
          .typing-indicator {
              display: flex;
              align-items: center;
              gap: 0.5rem;
              padding: 1rem;
              color: var(--text-muted);
              font-size: 0.875rem;
              font-style: italic;
          }

          .typing-indicator::before {
              content: '💭';
              animation: pulse 1.5s infinite;
          }

          /* Loading states */
          ._dash-loading {
              color: var(--accent-primary);
          }

          .js-plotly-plot {
              border-radius: var(--radius-xl);
              overflow: hidden;
          }

          /* Améliorations d'accessibilité */
          .nav-item:focus,
          .send-button:focus,
          .message-input:focus,
          .quick-suggestion:focus {
              outline: 2px solid var(--accent-primary);
              outline-offset: 2px;
          }

          /* Préférences de mouvement réduit */
          @media (prefers-reduced-motion: reduce) {
              *,
              *::before,
              *::after {
                  animation-duration: 0.01ms !important;
                  animation-iteration-count: 1 !important;
                  transition-duration: 0.01ms !important;
              }
          }

          /* Mode sombre forcé */
          @media (prefers-color-scheme: dark) {
              :root {
                  color-scheme: dark;
              }
          }

          /* Impression */
          @media print {
              .header-nav,
              .input-area,
              .quick-suggestions,
              .send-button {
                  display: none !important;
              }
              
              .chat-messages {
                  background: white !important;
                  color: black !important;
              }
          }
      </style>
  </head>
  <body>
      {%app_entry%}
      <footer>
          {%config%}
          {%scripts%}
          {%renderer%}
      </footer>
      <script>
          // Amélioration de l'UX avec JavaScript
          document.addEventListener('DOMContentLoaded', function() {
              // Auto-scroll vers le bas du chat
              function scrollToBottom() {
                  const chatMessages = document.getElementById('chat-messages');
                  if (chatMessages) {
                      chatMessages.scrollTop = chatMessages.scrollHeight;
                  }
              }

              // Observer pour détecter les nouveaux messages
              const observer = new MutationObserver(function(mutations) {
                  mutations.forEach(function(mutation) {
                      if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                          setTimeout(scrollToBottom, 100);
                      }
                  });
              });

              const chatContainer = document.getElementById('chat-messages');
              if (chatContainer) {
                  observer.observe(chatContainer, { childList: true, subtree: true });
              }

              // Raccourcis clavier
              document.addEventListener('keydown', function(e) {
                  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                      const sendButton = document.getElementById('send-button');
                      if (sendButton) {
                          sendButton.click();
                      }
                  }
              });
          });
      </script>
  </body>
</html>
'''

if __name__ == "__main__":
  if chatbot is None:
      print("⚠️  Chatbot non initialisé")
      print("📋 Veuillez d'abord extraire et indexer des données avec:")
      print("   python data/scripts/arxiv_extractor.py")
      print("   python data/scripts/data_cleaner.py")
      print("   python data/scripts/semantic_indexer.py")
  else:
      print("🚀 Démarrage du arXiv AI Research Assistant...")
      print("📱 Ouvrez votre navigateur à l'adresse: http://127.0.0.1:8050")
  
  app.run_server(debug=True, host='127.0.0.1', port=8050)
