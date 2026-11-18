"""
Semantic drift detection using TF-IDF and contextual embeddings.

This module implements multiple approaches to detect semantic shift in terminology
over time periods, focusing on identifying emerging dog whistle terms.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class TFIDFDriftAnalyzer:
    """
    Analyzes semantic drift using TF-IDF vectorization across time periods.
    
    Detects terms that show significant changes in usage patterns between
    temporal segments, indicating potential emergence or evolution of meaning.
    """
    
    def __init__(self, min_df: int = 2, max_df: float = 0.8, ngram_range: Tuple[int, int] = (1, 3)):
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.vectorizers = {}
        self.tfidf_matrices = {}
        self.feature_names = {}
        
    def fit_transform_periods(self, temporal_segments: Dict[str, pd.DataFrame],
                            text_column: str = 'cleaned_text') -> Dict[str, np.ndarray]:
        """
        Fit TF-IDF vectorizers for each temporal period.
        
        Args:
            temporal_segments: Dictionary mapping periods to dataframes
            text_column: Column containing preprocessed text
            
        Returns:
            Dictionary mapping periods to TF-IDF matrices
        """
        for period, df in temporal_segments.items():
            # Combine all text from the period
            texts = df[text_column].tolist()
            
            # Fit vectorizer for this period
            vectorizer = TfidfVectorizer(
                min_df=self.min_df,
                max_df=self.max_df,
                ngram_range=self.ngram_range,
                stop_words='english'
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            self.vectorizers[period] = vectorizer
            self.tfidf_matrices[period] = tfidf_matrix
            self.feature_names[period] = vectorizer.get_feature_names_out()
            
        return self.tfidf_matrices
    
    def calculate_drift_scores(self, baseline_period: str, 
                             comparison_periods: List[str]) -> pd.DataFrame:
        """
        Calculate drift scores for terms between baseline and comparison periods.
        
        Args:
            baseline_period: Reference period for comparison
            comparison_periods: List of periods to compare against baseline
            
        Returns:
            DataFrame with drift scores for each term
        """
        if baseline_period not in self.tfidf_matrices:
            raise ValueError(f"Baseline period {baseline_period} not found")
            
        baseline_features = set(self.feature_names[baseline_period])
        drift_results = []
        
        for comparison_period in comparison_periods:
            if comparison_period not in self.tfidf_matrices:
                continue
                
            comparison_features = set(self.feature_names[comparison_period])
            
            # Find overlapping terms
            common_terms = baseline_features.intersection(comparison_features)
            
            # Calculate mean TF-IDF scores for each period
            baseline_vectorizer = self.vectorizers[baseline_period]
            comparison_vectorizer = self.vectorizers[comparison_period]
            
            baseline_mean_scores = np.mean(self.tfidf_matrices[baseline_period].toarray(), axis=0)
            comparison_mean_scores = np.mean(self.tfidf_matrices[comparison_period].toarray(), axis=0)
            
            for term in common_terms:
                try:
                    baseline_idx = list(self.feature_names[baseline_period]).index(term)
                    comparison_idx = list(self.feature_names[comparison_period]).index(term)
                    
                    baseline_score = baseline_mean_scores[baseline_idx]
                    comparison_score = comparison_mean_scores[comparison_idx]
                    
                    # Calculate drift score (relative change)
                    if baseline_score > 0:
                        drift_score = (comparison_score - baseline_score) / baseline_score
                    else:
                        drift_score = comparison_score
                    
                    drift_results.append({
                        'term': term,
                        'baseline_period': baseline_period,
                        'comparison_period': comparison_period,
                        'baseline_score': baseline_score,
                        'comparison_score': comparison_score,
                        'drift_score': drift_score,
                        'absolute_drift': abs(drift_score)
                    })
                except (ValueError, IndexError):
                    continue
        
        return pd.DataFrame(drift_results)
    
    def find_emerging_terms(self, drift_df: pd.DataFrame, 
                          threshold: float = 0.5, min_comparison_score: float = 0.01) -> pd.DataFrame:
        """
        Identify emerging terms based on drift analysis.
        
        Args:
            drift_df: DataFrame from calculate_drift_scores
            threshold: Minimum drift score for emerging terms
            min_comparison_score: Minimum TF-IDF score in comparison period
            
        Returns:
            DataFrame with emerging terms
        """
        emerging = drift_df[
            (drift_df['drift_score'] > threshold) & 
            (drift_df['comparison_score'] > min_comparison_score)
        ].copy()
        
        # Sort by drift score
        emerging = emerging.sort_values('drift_score', ascending=False)
        
        return emerging
    
    def visualize_drift_timeline(self, drift_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
        """
        Create an interactive timeline visualization of term drift.
        
        Args:
            drift_df: DataFrame from calculate_drift_scores
            top_n: Number of top drifting terms to show
            
        Returns:
            Plotly figure object
        """
        # Get top drifting terms
        top_terms = drift_df.nlargest(top_n, 'absolute_drift')['term'].unique()
        
        # Filter data for visualization
        viz_data = drift_df[drift_df['term'].isin(top_terms)]
        
        fig = px.line(
            viz_data, 
            x='comparison_period', 
            y='drift_score',
            color='term',
            title=f'Top {top_n} Terms by Semantic Drift',
            labels={'drift_score': 'Drift Score', 'comparison_period': 'Time Period'}
        )
        
        fig.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Drift Score",
            hovermode='x unified'
        )
        
        return fig


class EmbeddingDriftAnalyzer:
    """
    Analyzes semantic drift using contextual embeddings.
    
    Uses sentence transformers to create embeddings for terms in context
    and detects shifts in semantic space over time.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.term_embeddings = {}
        self.contexts = {}
        
    def extract_term_contexts(self, temporal_segments: Dict[str, pd.DataFrame],
                            terms: List[str], context_window: int = 50) -> Dict[str, Dict[str, List[str]]]:
        """
        Extract contexts for specific terms across time periods.
        
        Args:
            temporal_segments: Dictionary mapping periods to dataframes
            terms: List of terms to analyze
            context_window: Number of characters around term for context
            
        Returns:
            Nested dictionary: {period: {term: [contexts]}}
        """
        contexts = defaultdict(lambda: defaultdict(list))
        
        for period, df in temporal_segments.items():
            for term in terms:
                for text in df['cleaned_text']:
                    if isinstance(text, str) and term in text.lower():
                        # Find all occurrences of the term
                        text_lower = text.lower()
                        term_lower = term.lower()
                        
                        start = 0
                        while True:
                            pos = text_lower.find(term_lower, start)
                            if pos == -1:
                                break
                                
                            # Extract context around the term
                            context_start = max(0, pos - context_window)
                            context_end = min(len(text), pos + len(term) + context_window)
                            context = text[context_start:context_end]
                            
                            contexts[period][term].append(context)
                            start = pos + 1
        
        return dict(contexts)
    
    def compute_embeddings(self, contexts: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute embeddings for term contexts.
        
        Args:
            contexts: Nested dictionary of contexts from extract_term_contexts
            
        Returns:
            Nested dictionary: {period: {term: embedding_matrix}}
        """
        embeddings = defaultdict(dict)
        
        for period, period_contexts in contexts.items():
            for term, term_contexts in period_contexts.items():
                if term_contexts:
                    # Compute embeddings for all contexts of this term
                    term_embeddings = self.model.encode(term_contexts)
                    embeddings[period][term] = term_embeddings
        
        return dict(embeddings)
    
    def calculate_embedding_drift(self, embeddings: Dict[str, Dict[str, np.ndarray]],
                                baseline_period: str) -> pd.DataFrame:
        """
        Calculate embedding drift scores between periods.
        
        Args:
            embeddings: Embeddings from compute_embeddings
            baseline_period: Reference period for comparison
            
        Returns:
            DataFrame with embedding drift scores
        """
        if baseline_period not in embeddings:
            raise ValueError(f"Baseline period {baseline_period} not found")
            
        drift_results = []
        baseline_embeddings = embeddings[baseline_period]
        
        for period, period_embeddings in embeddings.items():
            if period == baseline_period:
                continue
                
            for term in baseline_embeddings.keys():
                if term in period_embeddings:
                    baseline_mean = np.mean(baseline_embeddings[term], axis=0)
                    comparison_mean = np.mean(period_embeddings[term], axis=0)
                    
                    # Calculate cosine similarity between period means
                    similarity = cosine_similarity([baseline_mean], [comparison_mean])[0][0]
                    drift_score = 1 - similarity  # Higher drift = lower similarity
                    
                    drift_results.append({
                        'term': term,
                        'baseline_period': baseline_period,
                        'comparison_period': period,
                        'similarity': similarity,
                        'drift_score': drift_score,
                        'baseline_contexts': len(baseline_embeddings[term]),
                        'comparison_contexts': len(period_embeddings[term])
                    })
        
        return pd.DataFrame(drift_results)
    
    def visualize_embedding_space(self, embeddings: Dict[str, Dict[str, np.ndarray]],
                                terms: List[str], max_contexts_per_term: int = 50) -> go.Figure:
        """
        Visualize term embeddings in 2D space using PCA.
        
        Args:
            embeddings: Embeddings from compute_embeddings
            terms: Terms to visualize
            max_contexts_per_term: Maximum contexts to include per term for visualization
            
        Returns:
            Plotly figure object
        """
        all_embeddings = []
        labels = []
        periods = []
        
        for period, period_embeddings in embeddings.items():
            for term in terms:
                if term in period_embeddings:
                    term_embeds = period_embeddings[term]
                    # Subsample if too many contexts
                    if len(term_embeds) > max_contexts_per_term:
                        indices = np.random.choice(len(term_embeds), max_contexts_per_term, replace=False)
                        term_embeds = term_embeds[indices]
                    
                    all_embeddings.extend(term_embeds)
                    labels.extend([term] * len(term_embeds))
                    periods.extend([period] * len(term_embeds))
        
        if not all_embeddings:
            return go.Figure()
        
        # Apply PCA for 2D visualization
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(all_embeddings)
        
        # Create DataFrame for plotting
        viz_df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'term': labels,
            'period': periods
        })
        
        fig = px.scatter(
            viz_df, 
            x='x', 
            y='y', 
            color='term',
            symbol='period',
            title='Term Embeddings in Semantic Space',
            labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                   'y': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'}
        )
        
        return fig


class CombinedDriftAnalyzer:
    """
    Combines TF-IDF and embedding drift analysis for comprehensive detection.
    """
    
    def __init__(self):
        self.tfidf_analyzer = TFIDFDriftAnalyzer()
        self.embedding_analyzer = EmbeddingDriftAnalyzer()
    
    def analyze_comprehensive_drift(self, temporal_segments: Dict[str, pd.DataFrame],
                                  baseline_period: str, 
                                  target_terms: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Run comprehensive drift analysis using both TF-IDF and embeddings.
        
        Args:
            temporal_segments: Dictionary mapping periods to dataframes
            baseline_period: Reference period for comparison
            target_terms: Specific terms to analyze (if None, uses emerging terms from TF-IDF)
            
        Returns:
            Dictionary with analysis results
        """
        results = {}
        
        # Step 1: TF-IDF drift analysis
        print("Running TF-IDF drift analysis...")
        self.tfidf_analyzer.fit_transform_periods(temporal_segments)
        comparison_periods = [p for p in temporal_segments.keys() if p != baseline_period]
        
        tfidf_drift = self.tfidf_analyzer.calculate_drift_scores(baseline_period, comparison_periods)
        emerging_tfidf = self.tfidf_analyzer.find_emerging_terms(tfidf_drift)
        
        results['tfidf_drift'] = tfidf_drift
        results['emerging_tfidf'] = emerging_tfidf
        
        # Step 2: Determine terms for embedding analysis
        if target_terms is None:
            # Use top emerging terms from TF-IDF analysis
            target_terms = emerging_tfidf.head(20)['term'].unique().tolist()
            print(f"Using top {len(target_terms)} emerging terms from TF-IDF analysis")
        
        if target_terms:
            # Step 3: Embedding drift analysis
            print("Running embedding drift analysis...")
            contexts = self.embedding_analyzer.extract_term_contexts(temporal_segments, target_terms)
            embeddings = self.embedding_analyzer.compute_embeddings(contexts)
            embedding_drift = self.embedding_analyzer.calculate_embedding_drift(embeddings, baseline_period)
            
            results['embedding_drift'] = embedding_drift
            results['embeddings'] = embeddings
            results['contexts'] = contexts
        
        # Step 4: Combine results
        if 'embedding_drift' in results:
            combined_drift = self._combine_drift_scores(tfidf_drift, embedding_drift)
            results['combined_drift'] = combined_drift
        
        return results
    
    def _combine_drift_scores(self, tfidf_drift: pd.DataFrame, 
                            embedding_drift: pd.DataFrame) -> pd.DataFrame:
        """
        Combine TF-IDF and embedding drift scores.
        
        Args:
            tfidf_drift: TF-IDF drift results
            embedding_drift: Embedding drift results
            
        Returns:
            Combined drift scores DataFrame
        """
        # Merge on term and period
        combined = pd.merge(
            tfidf_drift, 
            embedding_drift[['term', 'comparison_period', 'drift_score', 'similarity']], 
            on=['term', 'comparison_period'], 
            how='inner',
            suffixes=('_tfidf', '_embedding')
        )
        
        # Create combined score (weighted average)
        combined['combined_drift_score'] = (
            0.6 * combined['drift_score_tfidf'] + 
            0.4 * combined['drift_score_embedding']
        )
        
        return combined.sort_values('combined_drift_score', ascending=False)
    
    def create_drift_report(self, results: Dict[str, pd.DataFrame], 
                          top_n: int = 10) -> str:
        """
        Generate a comprehensive drift analysis report.
        
        Args:
            results: Results from analyze_comprehensive_drift
            top_n: Number of top terms to include in report
            
        Returns:
            Formatted report string
        """
        report = ["SEMANTIC DRIFT ANALYSIS REPORT", "=" * 40, ""]
        
        # TF-IDF Results
        if 'emerging_tfidf' in results:
            emerging = results['emerging_tfidf'].head(top_n)
            report.extend([
                f"TOP {top_n} EMERGING TERMS (TF-IDF Analysis):",
                "-" * 35
            ])
            
            for idx, row in emerging.iterrows():
                report.append(
                    f"{row['term']:25} | Drift: {row['drift_score']:6.3f} | "
                    f"Score: {row['comparison_score']:6.4f}"
                )
            report.append("")
        
        # Combined Results
        if 'combined_drift' in results:
            combined = results['combined_drift'].head(top_n)
            report.extend([
                f"TOP {top_n} TERMS (Combined Analysis):",
                "-" * 30
            ])
            
            for idx, row in combined.iterrows():
                report.append(
                    f"{row['term']:25} | Combined: {row['combined_drift_score']:6.3f} | "
                    f"TF-IDF: {row['drift_score_tfidf']:6.3f} | "
                    f"Embedding: {row['drift_score_embedding']:6.3f}"
                )
        
        return "\n".join(report)


if __name__ == "__main__":
    # Demonstration of semantic drift analysis
    print("Dog Whistle Detection - Semantic Drift Analysis Module")
    print("=" * 55)
    
    # This would typically use real preprocessed data
    print("Note: This is a demonstration with synthetic data")
    print("In practice, this module works with preprocessed social media data")
    
    # Create sample temporal segments (normally from preprocessing)
    sample_data = {
        'Q1_2024': pd.DataFrame({
            'cleaned_text': [
                'coastal elites push agenda papers',
                'traditional values under attack',
                'urban youth cultural activities'
            ]
        }),
        'Q2_2024': pd.DataFrame({
            'cleaned_text': [
                'coastal papers show bias again',
                'dual citizenship causes problems',
                'urban youth enrichment programs'
            ]
        })
    }
    
    # Initialize analyzer
    analyzer = CombinedDriftAnalyzer()
    
    print("\nRunning comprehensive drift analysis...")
    results = analyzer.analyze_comprehensive_drift(
        sample_data, 
        baseline_period='Q1_2024'
    )
    
    # Generate report
    if results:
        report = analyzer.create_drift_report(results)
        print("\n" + report)
    
    print("\nSemantic drift analysis module ready for integration!")
