"""
BERT-based semantic shift analysis for dog whistle detection.

This module implements context-aware semantic drift detection using BERT embeddings
to identify when terms are used with different meanings in domain-specific contexts.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings

class BERTSemanticAnalyzer:
    """
    BERT-based semantic shift detection for identifying dog whistles.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize BERT analyzer.
        
        Args:
            model_name: SentenceTransformer model name
        """
        self.model = SentenceTransformer(model_name)
        self.term_embeddings = {}
        self.context_embeddings = {}
        
    def extract_sentences_with_term(self, df: pd.DataFrame, term: str, 
                                   text_column: str = 'cleaned_text',
                                   context_window: int = 3) -> List[str]:
        """
        Extract sentences containing the target term with context.
        
        Args:
            df: DataFrame containing text data
            term: Target term to analyze
            text_column: Column containing text
            context_window: Number of words around term for context
            
        Returns:
            List of sentences containing the term
        """
        sentences = []
        term_lower = term.lower()
        
        for text in df[text_column].dropna():
            if isinstance(text, str) and term_lower in text.lower():
                # Split into sentences (simple approach)
                text_sentences = text.split('.')
                for sentence in text_sentences:
                    if term_lower in sentence.lower():
                        sentences.append(sentence.strip())
        
        return sentences
    
    def analyze_term_contexts(self, general_df: pd.DataFrame, domain_df: pd.DataFrame,
                            term: str, text_column: str = 'cleaned_text') -> Dict:
        """
        Analyze how a term is used in general vs domain-specific contexts.
        
        Args:
            general_df: General corpus DataFrame
            domain_df: Domain-specific corpus DataFrame  
            term: Term to analyze
            text_column: Column containing text
            
        Returns:
            Dictionary with analysis results
        """
        # Extract sentences containing the term
        general_sentences = self.extract_sentences_with_term(general_df, term, text_column)
        domain_sentences = self.extract_sentences_with_term(domain_df, term, text_column)
        
        if len(general_sentences) == 0 or len(domain_sentences) == 0:
            return {
                'term': term,
                'general_count': len(general_sentences),
                'domain_count': len(domain_sentences),
                'analysis': 'Insufficient data - term not found in one or both corpora'
            }
        
        # Generate embeddings
        print(f"Generating embeddings for '{term}' - General: {len(general_sentences)}, Domain: {len(domain_sentences)}")
        
        general_embeddings = self.model.encode(general_sentences)
        domain_embeddings = self.model.encode(domain_sentences)
        
        # Calculate centroids
        general_centroid = np.mean(general_embeddings, axis=0)
        domain_centroid = np.mean(domain_embeddings, axis=0)
        
        # Calculate similarity between centroids
        centroid_similarity = cosine_similarity([general_centroid], [domain_centroid])[0][0]
        
        # Calculate intra-cluster similarities
        general_similarities = []
        for i in range(len(general_embeddings)):
            for j in range(i+1, len(general_embeddings)):
                sim = cosine_similarity([general_embeddings[i]], [general_embeddings[j]])[0][0]
                general_similarities.append(sim)
        
        domain_similarities = []
        for i in range(len(domain_embeddings)):
            for j in range(i+1, len(domain_embeddings)):
                sim = cosine_similarity([domain_embeddings[i]], [domain_embeddings[j]])[0][0]
                domain_similarities.append(sim)
        
        # Calculate drift metrics
        avg_general_coherence = np.mean(general_similarities) if general_similarities else 0
        avg_domain_coherence = np.mean(domain_similarities) if domain_similarities else 0
        
        # Calculate cross-corpus similarities
        cross_similarities = []
        for gen_emb in general_embeddings:
            for dom_emb in domain_embeddings:
                sim = cosine_similarity([gen_emb], [dom_emb])[0][0]
                cross_similarities.append(sim)
        
        avg_cross_similarity = np.mean(cross_similarities)
        
        # Semantic shift score (lower = more drift)
        semantic_shift_score = 1 - centroid_similarity
        
        results = {
            'term': term,
            'general_count': len(general_sentences),
            'domain_count': len(domain_sentences),
            'centroid_similarity': float(centroid_similarity),
            'semantic_shift_score': float(semantic_shift_score),
            'general_coherence': float(avg_general_coherence),
            'domain_coherence': float(avg_domain_coherence),
            'cross_similarity': float(avg_cross_similarity),
            'general_sentences_sample': general_sentences[:5],
            'domain_sentences_sample': domain_sentences[:5],
            'analysis': self._interpret_results(centroid_similarity, semantic_shift_score, 
                                              avg_general_coherence, avg_domain_coherence)
        }
        
        # Store embeddings for visualization
        self.term_embeddings[term] = {
            'general': general_embeddings,
            'domain': domain_embeddings,
            'general_centroid': general_centroid,
            'domain_centroid': domain_centroid
        }
        
        return results
    
    def _interpret_results(self, centroid_sim: float, shift_score: float,
                          gen_coherence: float, dom_coherence: float) -> str:
        """Interpret the analysis results."""
        if shift_score > 0.3:
            return "HIGH semantic shift detected - potential dog whistle usage"
        elif shift_score > 0.15:
            return "MODERATE semantic shift detected - investigate further"
        elif shift_score > 0.05:
            return "LOW semantic shift detected - minor usage differences"
        else:
            return "MINIMAL semantic shift - consistent usage across corpora"
    
    def batch_analyze_terms(self, general_df: pd.DataFrame, domain_df: pd.DataFrame,
                           terms: List[str], text_column: str = 'cleaned_text') -> pd.DataFrame:
        """
        Analyze multiple terms for semantic shift.
        
        Args:
            general_df: General corpus DataFrame
            domain_df: Domain corpus DataFrame
            terms: List of terms to analyze
            text_column: Column containing text
            
        Returns:
            DataFrame with analysis results for all terms
        """
        results = []
        
        for term in terms:
            print(f"\nAnalyzing term: '{term}'")
            try:
                result = self.analyze_term_contexts(general_df, domain_df, term, text_column)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing '{term}': {e}")
                results.append({
                    'term': term,
                    'error': str(e),
                    'analysis': 'Analysis failed'
                })
        
        return pd.DataFrame(results)
    
    def visualize_term_embeddings(self, term: str, save_path: Optional[str] = None):
        """
        Visualize embeddings for a specific term using PCA.
        
        Args:
            term: Term to visualize
            save_path: Optional path to save the plot
        """
        if term not in self.term_embeddings:
            print(f"No embeddings found for term '{term}'. Run analysis first.")
            return
        
        embeddings_data = self.term_embeddings[term]
        general_embs = embeddings_data['general']
        domain_embs = embeddings_data['domain']
        
        # Combine all embeddings for PCA
        all_embeddings = np.vstack([general_embs, domain_embs])
        
        # Apply PCA
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(all_embeddings)
        
        # Split back
        general_reduced = reduced_embeddings[:len(general_embs)]
        domain_reduced = reduced_embeddings[len(general_embs):]
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        plt.scatter(general_reduced[:, 0], general_reduced[:, 1], 
                   alpha=0.6, label='General Corpus', s=50)
        plt.scatter(domain_reduced[:, 0], domain_reduced[:, 1], 
                   alpha=0.6, label='Domain Corpus', s=50)
        
        # Plot centroids
        general_centroid_reduced = pca.transform([embeddings_data['general_centroid']])
        domain_centroid_reduced = pca.transform([embeddings_data['domain_centroid']])
        
        plt.scatter(general_centroid_reduced[0, 0], general_centroid_reduced[0, 1],
                   marker='*', s=200, color='blue', label='General Centroid', 
                   edgecolors='black', linewidth=2)
        plt.scatter(domain_centroid_reduced[0, 0], domain_centroid_reduced[0, 1],
                   marker='*', s=200, color='red', label='Domain Centroid',
                   edgecolors='black', linewidth=2)
        
        # Draw line between centroids
        plt.plot([general_centroid_reduced[0, 0], domain_centroid_reduced[0, 0]],
                [general_centroid_reduced[0, 1], domain_centroid_reduced[0, 1]],
                'k--', alpha=0.5, linewidth=2, label='Centroid Distance')
        
        plt.title(f'BERT Embedding Space for Term: "{term}"')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def find_semantic_clusters(self, term: str, n_clusters: int = 3) -> Dict:
        """
        Find semantic clusters within domain usage of a term.
        
        Args:
            term: Term to analyze
            n_clusters: Number of clusters to find
            
        Returns:
            Dictionary with clustering results
        """
        if term not in self.term_embeddings:
            print(f"No embeddings found for term '{term}'. Run analysis first.")
            return {}
        
        domain_embs = self.term_embeddings[term]['domain']
        
        if len(domain_embs) < n_clusters:
            return {'error': f'Not enough samples ({len(domain_embs)}) for {n_clusters} clusters'}
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(domain_embs)
        
        # Analyze clusters
        cluster_info = {}
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            cluster_embeddings = domain_embs[cluster_indices]
            
            # Calculate cluster centroid and coherence
            cluster_centroid = np.mean(cluster_embeddings, axis=0)
            cluster_similarities = []
            for j in range(len(cluster_embeddings)):
                for k in range(j+1, len(cluster_embeddings)):
                    sim = cosine_similarity([cluster_embeddings[j]], [cluster_embeddings[k]])[0][0]
                    cluster_similarities.append(sim)
            
            cluster_info[f'cluster_{i}'] = {
                'size': len(cluster_indices),
                'indices': cluster_indices.tolist(),
                'coherence': float(np.mean(cluster_similarities)) if cluster_similarities else 0,
                'centroid': cluster_centroid
            }
        
        return {
            'term': term,
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels.tolist(),
            'clusters': cluster_info,
            'silhouette_score': self._calculate_silhouette_score(domain_embs, cluster_labels)
        }
    
    def _calculate_silhouette_score(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality."""
        try:
            from sklearn.metrics import silhouette_score
            return float(silhouette_score(embeddings, labels))
        except:
            return 0.0


def demo_bert_analysis():
    """Demonstrate BERT semantic analysis on sample data."""
    print("BERT Semantic Shift Analysis Demo")
    print("=" * 40)
    
    # This would use your actual cleaned data
    # For demo, we'll show the structure
    sample_terms = [
        "urban youth", "coastal elite", "globalist", 
        "traditional values", "real americans", "silent majority"
    ]
    
    print("Sample terms for analysis:")
    for term in sample_terms:
        print(f"  - {term}")
    
    print("\nTo run analysis:")
    print("1. Load general corpus (e.g., news articles, Wikipedia)")
    print("2. Load domain corpus (Truth Social, etc.)")
    print("3. analyzer.batch_analyze_terms(general_df, domain_df, terms)")
    print("4. analyzer.visualize_term_embeddings(term)")
    
    return sample_terms

if __name__ == "__main__":
    demo_bert_analysis()
