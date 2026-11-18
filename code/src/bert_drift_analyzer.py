"""
BERT-based semantic drift detection for dog whistle analysis.

This module implements domain-specific BERT fine-tuning and compares embeddings
of ALL domain corpus terms between original BERT and fine-tuned BERT models to
identify semantic drift as an indicator of potential dog whistle terms.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import torch
import warnings
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForMaskedLM
)
try:
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
except ImportError:
    # Fallback for different transformers versions
    try:
        from transformers.trainer import Trainer
        from transformers.training_args import TrainingArguments
        from transformers.data.data_collator import DataCollatorForLanguageModeling
    except ImportError:
        print("⚠️  Warning: Some transformers components not available. Fine-tuning may not work.")
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import re
from collections import defaultdict, Counter
import tempfile

warnings.filterwarnings('ignore')

class DomainTextDataset(Dataset):
    """Dataset for domain-specific BERT fine-tuning."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

class BERTDriftAnalyzer:
    """
    BERT-based semantic drift analyzer that compares term embeddings
    between original BERT and domain-fine-tuned BERT models.
    
    This analyzer:
    1. Fine-tunes BERT on domain-specific corpus
    2. Extracts embeddings for ALL candidate terms using both models
    3. Computes similarity between original vs fine-tuned representations
    4. Identifies terms with largest representation differences
    """
    
    def __init__(self, base_model_name: str = 'bert-base-uncased', device: Optional[str] = None):
        """
        Initialize the BERT drift analyzer.
        
        Args:
            base_model_name: Pre-trained BERT model to use
            device: Device for computation ('cuda', 'cpu', or None for auto)
        """
        self.base_model_name = base_model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize original/base model
        print(f"🚀 Loading base BERT model: {base_model_name}")
        self.original_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.original_model = AutoModel.from_pretrained(base_model_name).to(self.device)
        self.original_model.eval()
        
        # Domain model will be loaded after fine-tuning
        self.domain_model = None
        self.domain_tokenizer = None
        
        # Results storage
        self.drift_results = {}
        
        print(f"✅ Initialized BERT drift analyzer on {self.device}")
        
    def fine_tune_domain_model(self, texts: List[str], output_dir: str = "models/bert_finetuned", 
                              num_epochs: int = 3, batch_size: int = 8,
                              learning_rate: float = 5e-5, max_length: int = 512,
                              save_steps: int = 500) -> str:
        """
        Fine-tune BERT on domain-specific corpus.
        
        Args:
            texts: Domain corpus texts for fine-tuning
            output_dir: Directory to save fine-tuned model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
            max_length: Maximum sequence length
            save_steps: Steps between model saves
            
        Returns:
            Path to fine-tuned model
        """
        print(f"🔧 Fine-tuning BERT on domain corpus ({len(texts)} texts)")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model and tokenizer for fine-tuning
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        model = AutoModelForMaskedLM.from_pretrained(self.base_model_name).to(self.device)
        
        # Create dataset
        dataset = DomainTextDataset(texts, tokenizer, max_length)
        
        # Data collator for masked language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=save_steps,
            save_total_limit=2,
            prediction_loss_only=True,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=100,
            logging_dir=f"{output_dir}/logs",
            dataloader_num_workers=0  # Avoid multiprocessing issues
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        
        # Fine-tune the model
        print("⏳ Starting fine-tuning...")
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Fine-tuning completed! Model saved to {output_dir}")
        return output_dir
        
    def load_domain_model(self, model_path: str):
        """
        Load the fine-tuned domain model.
        
        Args:
            model_path: Path to fine-tuned model directory
        """
        print(f"📂 Loading fine-tuned domain model from {model_path}")
        
        self.domain_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.domain_model = AutoModel.from_pretrained(model_path).to(self.device)
        self.domain_model.eval()
        
        print("✅ Domain model loaded successfully")
        
    def extract_term_embedding(self, term: str, model, tokenizer, 
                              pooling: str = 'cls') -> np.ndarray:
        """
        Extract embedding for a single term using specified model.
        
        Args:
            term: Term to embed
            model: BERT model to use
            tokenizer: Corresponding tokenizer
            pooling: Pooling strategy ('cls', 'mean', 'max')
            
        Returns:
            Term embedding as numpy array
        """
        # Tokenize the term
        inputs = tokenizer(
            term,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
            
            if pooling == 'cls':
                # Use [CLS] token embedding
                embedding = hidden_states[:, 0, :].squeeze()
            elif pooling == 'mean':
                # Mean pooling over all tokens
                attention_mask = inputs['attention_mask']
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                embedding = (sum_embeddings / sum_mask).squeeze()
            elif pooling == 'max':
                # Max pooling over all tokens
                embedding = torch.max(hidden_states, dim=1)[0].squeeze()
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling}")
        
        return embedding.cpu().numpy()
    
    def compare_term_embeddings(self, terms: List[str], 
                               pooling: str = 'cls') -> pd.DataFrame:
        """
        Compare embeddings of terms between original and domain models.
        
        Args:
            terms: List of terms to analyze
            pooling: Pooling strategy for embeddings
            
        Returns:
            DataFrame with drift analysis results
        """
        if self.domain_model is None:
            raise ValueError("Domain model not loaded! Call load_domain_model() first.")
        
        print(f"🔍 Comparing embeddings for {len(terms)} terms...")
        
        results = []
        
        for i, term in enumerate(terms, 1):
            print(f"  [{i}/{len(terms)}] Processing: '{term}'")
            
            try:
                # Extract embeddings from both models
                original_emb = self.extract_term_embedding(
                    term, self.original_model, self.original_tokenizer, pooling
                )
                domain_emb = self.extract_term_embedding(
                    term, self.domain_model, self.domain_tokenizer, pooling
                )
                
                # Calculate similarity between original and domain embeddings
                similarity = cosine_similarity(
                    original_emb.reshape(1, -1), 
                    domain_emb.reshape(1, -1)
                )[0][0]
                
                # Drift score = 1 - similarity (higher = more drift)
                drift_score = 1 - similarity
                
                # Calculate embedding norms (for analysis)
                original_norm = np.linalg.norm(original_emb)
                domain_norm = np.linalg.norm(domain_emb)
                
                results.append({
                    'term': term,
                    'drift_score': float(drift_score),
                    'similarity': float(similarity),
                    'original_norm': float(original_norm),
                    'domain_norm': float(domain_norm),
                    'norm_ratio': float(domain_norm / original_norm),
                    'interpretation': self._interpret_drift_score(drift_score),
                    'status': 'success'
                })
                
                # Store embeddings for visualization
                self.drift_results[term] = {
                    'original_embedding': original_emb,
                    'domain_embedding': domain_emb,
                    'drift_score': drift_score
                }
                
            except Exception as e:
                print(f"    ❌ Error processing '{term}': {e}")
                results.append({
                    'term': term,
                    'drift_score': 0.0,
                    'similarity': 1.0,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Convert to DataFrame and sort by drift score
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('drift_score', ascending=False)
        
        print(f"✅ Embedding comparison completed!")
        return results_df
    
    def _interpret_drift_score(self, drift_score: float) -> str:
        """Interpret drift score magnitude."""
        if drift_score > 0.5:
            return "VERY HIGH drift - Major representation change"
        elif drift_score > 0.3:
            return "HIGH drift - Significant representation shift"
        elif drift_score > 0.2:
            return "MODERATE drift - Notable representation change"
        elif drift_score > 0.1:
            return "LOW drift - Minor representation differences"
        else:
            return "MINIMAL drift - Consistent representations"
    
    def visualize_drift_results(self, results_df: pd.DataFrame, 
                               top_n: int = 20, save_path: Optional[str] = None):
        """
        Visualize drift analysis results.
        
        Args:
            results_df: Results from compare_term_embeddings()
            top_n: Number of top terms to visualize
            save_path: Optional path to save plot
        """
        # Filter successful results and get top terms
        success_df = results_df[results_df['status'] == 'success'].head(top_n)
        
        if len(success_df) == 0:
            print("❌ No successful results to visualize")
            return
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Drift scores
        colors = ['red' if score > 0.3 else 'orange' if score > 0.2 else 'green' 
                 for score in success_df['drift_score']]
        
        bars1 = ax1.barh(range(len(success_df)), success_df['drift_score'], color=colors)
        ax1.set_yticks(range(len(success_df)))
        ax1.set_yticklabels(success_df['term'], fontsize=10)
        ax1.set_xlabel('Drift Score (1 - Cosine Similarity)')
        ax1.set_title(f'Top {len(success_df)} Terms by Semantic Drift')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add drift score labels
        for i, (bar, score) in enumerate(zip(bars1, success_df['drift_score'])):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontsize=8)
        
        # Plot 2: Similarity vs Norm Ratio
        scatter = ax2.scatter(success_df['similarity'], success_df['norm_ratio'], 
                            c=success_df['drift_score'], cmap='RdYlGn_r', 
                            s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax2.set_xlabel('Cosine Similarity (Original vs Domain)')
        ax2.set_ylabel('Norm Ratio (Domain / Original)')
        ax2.set_title('Embedding Similarity vs Magnitude Change')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Drift Score')
        
        # Add diagonal line (similarity = 1)
        ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Similarity')
        ax2.axhline(y=1.0, color='blue', linestyle='--', alpha=0.5, label='Same Magnitude')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_embedding_space(self, terms: List[str], method: str = 'pca',
                                 save_path: Optional[str] = None):
        """
        Visualize term embeddings in 2D space using dimensionality reduction.
        
        Args:
            terms: List of terms to visualize
            method: Dimensionality reduction method ('pca' or 'tsne')
            save_path: Optional path to save plot
        """
        if not terms or not all(term in self.drift_results for term in terms):
            print("❌ No embedding data available for specified terms")
            return
        
        # Collect embeddings
        original_embeddings = []
        domain_embeddings = []
        labels = []
        
        for term in terms:
            if term in self.drift_results:
                original_embeddings.append(self.drift_results[term]['original_embedding'])
                domain_embeddings.append(self.drift_results[term]['domain_embedding'])
                labels.append(term)
        
        # Combine all embeddings for dimensionality reduction
        all_embeddings = np.vstack(original_embeddings + domain_embeddings)
        
        # Apply dimensionality reduction
        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            reduced = reducer.fit_transform(all_embeddings)
            explained_var = reducer.explained_variance_ratio_
            title_suffix = f"(PC1: {explained_var[0]:.1%}, PC2: {explained_var[1]:.1%})"
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)//2))
            reduced = reducer.fit_transform(all_embeddings)
            title_suffix = ""
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Split back into original and domain
        n_terms = len(labels)
        original_reduced = reduced[:n_terms]
        domain_reduced = reduced[n_terms:]
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Plot original embeddings
        plt.scatter(original_reduced[:, 0], original_reduced[:, 1], 
                   c='blue', alpha=0.7, s=100, label='Original BERT', 
                   marker='o', edgecolors='black', linewidth=1)
        
        # Plot domain embeddings
        plt.scatter(domain_reduced[:, 0], domain_reduced[:, 1], 
                   c='red', alpha=0.7, s=100, label='Domain BERT',
                   marker='s', edgecolors='black', linewidth=1)
        
        # Draw lines connecting original to domain embeddings
        for i in range(n_terms):
            plt.plot([original_reduced[i, 0], domain_reduced[i, 0]],
                    [original_reduced[i, 1], domain_reduced[i, 1]],
                    'k--', alpha=0.5, linewidth=1)
        
        # Add term labels
        for i, term in enumerate(labels):
            # Label original embedding
            plt.annotate(f"{term}_orig", 
                        (original_reduced[i, 0], original_reduced[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
            # Label domain embedding
            plt.annotate(f"{term}_domain", 
                        (domain_reduced[i, 0], domain_reduced[i, 1]),
                        xytext=(5, -15), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        plt.title(f'BERT Embedding Space Comparison {title_suffix}')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Embedding space visualization saved to {save_path}")
        
        plt.show()
    
    def save_results(self, results_df: pd.DataFrame, output_path: str):
        """Save drift analysis results to file."""
        results_df.to_csv(output_path, index=False)
        print(f"💾 Results saved to {output_path}")
        
    def get_top_drift_terms(self, results_df: pd.DataFrame, n: int = 20) -> List[str]:
        """Get top N terms with highest drift scores."""
        success_df = results_df[results_df['status'] == 'success']
        return success_df.head(n)['term'].tolist()

def demo_bert_drift_analysis():
    """Demonstrate BERT drift analysis workflow."""
    print("🚀 BERT Semantic Drift Analysis Demo")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = BERTDriftAnalyzer()
    
    print("\n📋 Analysis workflow:")
    print("1. Fine-tune BERT on domain-specific corpus (social media)")
    print("2. Load both original and fine-tuned BERT models") 
    print("3. Extract embeddings for ALL candidate terms using both models")
    print("4. Calculate cosine similarity between original vs fine-tuned embeddings")
    print("5. Identify terms with largest representation differences")
    print("6. Visualize results and analyze potential dog whistle candidates")
    
    return analyzer

if __name__ == "__main__":
    demo_bert_drift_analysis()
