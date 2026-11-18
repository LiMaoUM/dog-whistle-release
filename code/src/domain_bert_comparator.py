"""
Domain-specific BERT fine-tuning and comparison for dog whistle detection.

This module implements multiple approaches for comparing general vs domain-specific
BERT models to detect semantic shifts and coded language usage.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForMaskedLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    pipeline
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json

warnings.filterwarnings('ignore')

class DomainBERTComparator:
    """
    Compare general BERT vs domain-fine-tuned BERT for semantic shift detection.
    """
    
    def __init__(self, base_model_name: str = 'bert-base-uncased'):
        """
        Initialize the comparator.
        
        Args:
            base_model_name: Base BERT model to use
        """
        self.base_model_name = base_model_name
        self.general_model = None
        self.domain_model = None
        self.general_tokenizer = None
        self.domain_tokenizer = None
        self.comparison_results = {}
        
    def load_general_model(self):
        """Load the general (pre-trained) BERT model."""
        print("🔄 Loading general BERT model...")
        self.general_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.general_model = AutoModel.from_pretrained(self.base_model_name)
        print("✅ General model loaded")
    
    def fine_tune_domain_model(self, domain_texts: List[str], 
                              output_dir: str = "models/domain_bert",
                              num_epochs: int = 3,
                              batch_size: int = 8) -> str:
        """
        Fine-tune BERT on domain-specific data.
        
        Args:
            domain_texts: List of domain-specific texts for training
            output_dir: Directory to save the fine-tuned model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Path to the saved model
        """
        print(f"🎯 Fine-tuning BERT on {len(domain_texts)} domain texts...")
        
        # Prepare dataset
        dataset = DomainTextDataset(domain_texts, self.general_tokenizer)
        
        # Load model for masked language modeling
        model = AutoModelForMaskedLM.from_pretrained(self.base_model_name)
        
        # Data collator for MLM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.general_tokenizer,
            mlm=True,
            mlm_probability=0.15
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            logging_steps=100,
            logging_dir=f"{output_dir}/logs",
            dataloader_drop_last=False,
            eval_strategy="no",
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        
        # Train
        print("🚀 Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        print(f"💾 Domain model saved to {output_dir}")
        
        return output_dir
    
    def load_domain_model(self, model_path: str):
        """Load the domain fine-tuned model."""
        print(f"🔄 Loading domain model from {model_path}...")
        self.domain_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.domain_model = AutoModel.from_pretrained(model_path)
        self._domain_model_path = model_path  # Store path for MLM comparisons
        print("✅ Domain model loaded")
    
    def compare_models_on_terms(self, terms: List[str], 
                               contexts: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Compare how general vs domain models represent specific terms.
        
        Args:
            terms: List of terms to analyze
            contexts: Dictionary mapping terms to their contexts
            
        Returns:
            Comparison results
        """
        if not self.general_model or not self.domain_model:
            raise ValueError("Both models must be loaded first")
        
        print(f"🔍 Comparing models on {len(terms)} terms...")
        
        results = {}
        
        for term in terms:
            print(f"  Analyzing: {term}")
            
            term_contexts = contexts.get(term, [f"The term {term} appears here."])
            
            # Get embeddings from both models
            general_embs = self._get_term_embeddings(
                term, term_contexts, self.general_model, self.general_tokenizer
            )
            domain_embs = self._get_term_embeddings(
                term, term_contexts, self.domain_model, self.domain_tokenizer
            )
            
            # Compare embeddings
            comparison = self._compare_embeddings(general_embs, domain_embs, term)
            results[term] = comparison
        
        self.comparison_results = results
        return results
    
    def _get_term_embeddings(self, term: str, contexts: List[str], 
                           model: Any, tokenizer: Any) -> np.ndarray:
        """Get embeddings for a term in various contexts."""
        embeddings = []
        
        model.eval()
        with torch.no_grad():
            for context in contexts[:10]:  # Limit to 10 contexts
                # Tokenize
                inputs = tokenizer(
                    context, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True,
                    max_length=512
                )
                
                # Get embeddings
                outputs = model(**inputs)
                
                # Use mean pooling of last hidden states
                embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        
        return np.array(embeddings)
    
    def _compare_embeddings(self, general_embs: np.ndarray, 
                          domain_embs: np.ndarray, term: str) -> Dict[str, Any]:
        """Compare embeddings between general and domain models."""
        
        # Calculate centroids
        general_centroid = np.mean(general_embs, axis=0)
        domain_centroid = np.mean(domain_embs, axis=0)
        
        # Centroid similarity
        centroid_similarity = cosine_similarity([general_centroid], [domain_centroid])[0][0]
        
        # Intra-model coherence
        general_coherence = self._calculate_coherence(general_embs)
        domain_coherence = self._calculate_coherence(domain_embs)
        
        # Cross-model similarities
        cross_sims = []
        for g_emb in general_embs:
            for d_emb in domain_embs:
                sim = cosine_similarity([g_emb], [d_emb])[0][0]
                cross_sims.append(sim)
        
        avg_cross_similarity = np.mean(cross_sims)
        
        # Semantic shift score
        shift_score = 1 - centroid_similarity
        
        # Variance difference (measure of dispersion change)
        general_variance = np.var(general_embs.flatten())
        domain_variance = np.var(domain_embs.flatten())
        variance_difference = abs(domain_variance - general_variance)
        
        return {
            'term': term,
            'centroid_similarity': float(centroid_similarity),
            'semantic_shift_score': float(shift_score),
            'general_coherence': float(general_coherence),
            'domain_coherence': float(domain_coherence),
            'cross_similarity': float(avg_cross_similarity),
            'variance_difference': float(variance_difference),
            'interpretation': self._interpret_shift_score(shift_score),
            'embeddings': {
                'general_centroid': general_centroid,
                'domain_centroid': domain_centroid,
                'general_embeddings': general_embs,
                'domain_embeddings': domain_embs
            }
        }
    
    def _calculate_coherence(self, embeddings: np.ndarray) -> float:
        """Calculate average pairwise similarity within embeddings."""
        if len(embeddings) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                similarities.append(sim)
        
        return np.mean(similarities)
    
    def _interpret_shift_score(self, shift_score: float) -> str:
        """Interpret the semantic shift score."""
        if shift_score > 0.4:
            return "VERY HIGH semantic shift - strong domain specialization"
        elif shift_score > 0.25:
            return "HIGH semantic shift - significant domain adaptation"
        elif shift_score > 0.15:
            return "MODERATE semantic shift - domain influence detected"
        elif shift_score > 0.05:
            return "LOW semantic shift - minor domain effects"
        else:
            return "MINIMAL semantic shift - consistent representations"
    
    def compare_masked_predictions(self, templates: List[str], 
                                 target_words: List[str]) -> Dict[str, Any]:
        """
        Compare masked language model predictions between general and domain models.
        
        Args:
            templates: List of template sentences with [MASK] tokens
            target_words: Words to check prediction probabilities for
            
        Returns:
            Comparison of prediction probabilities
        """
        print("🎭 Comparing masked language model predictions...")
        
        # Create pipelines
        general_mlm = pipeline(
            "fill-mask", 
            model=self.base_model_name,
            tokenizer=self.general_tokenizer
        )
        
        domain_mlm = pipeline(
            "fill-mask",
            model=self.domain_model,
            tokenizer=self.domain_tokenizer
        )
        
        results = {}
        
        for template in templates:
            print(f"  Template: {template}")
            
            # Get predictions from both models
            general_preds = general_mlm(template, top_k=20)
            domain_preds = domain_mlm(template, top_k=20)
            
            # Compare predictions for target words
            template_results = {
                'template': template,
                'general_predictions': general_preds,
                'domain_predictions': domain_preds,
                'target_word_analysis': {}
            }
            
            for word in target_words:
                general_prob = self._get_word_probability(word, general_preds)
                domain_prob = self._get_word_probability(word, domain_preds)
                
                template_results['target_word_analysis'][word] = {
                    'general_probability': general_prob,
                    'domain_probability': domain_prob,
                    'probability_ratio': domain_prob / (general_prob + 1e-8),
                    'domain_preference': domain_prob > general_prob
                }
            
            results[template] = template_results
        
        return results
    
    def _get_word_probability(self, word: str, predictions: List[Dict]) -> float:
        """Extract probability for a specific word from MLM predictions."""
        for pred in predictions:
            if pred['token_str'].lower().strip() == word.lower().strip():
                return pred['score']
        return 0.0
    
    def visualize_embedding_comparison(self, term: str, save_path: Optional[str] = None):
        """Visualize embedding differences between models for a specific term."""
        if term not in self.comparison_results:
            print(f"No comparison results found for '{term}'")
            return
        
        data = self.comparison_results[term]['embeddings']
        general_embs = data['general_embeddings']
        domain_embs = data['domain_embeddings']
        
        # Combine embeddings for PCA
        all_embeddings = np.vstack([general_embs, domain_embs])
        
        # Apply PCA
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(all_embeddings)
        
        # Split back
        general_reduced = reduced_embeddings[:len(general_embs)]
        domain_reduced = reduced_embeddings[len(general_embs):]
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        plt.scatter(general_reduced[:, 0], general_reduced[:, 1], 
                   alpha=0.7, label='General BERT', s=60, color='blue')
        plt.scatter(domain_reduced[:, 0], domain_reduced[:, 1], 
                   alpha=0.7, label='Domain BERT', s=60, color='red')
        
        # Plot centroids
        general_centroid_reduced = pca.transform([data['general_centroid']])
        domain_centroid_reduced = pca.transform([data['domain_centroid']])
        
        plt.scatter(general_centroid_reduced[0, 0], general_centroid_reduced[0, 1],
                   marker='*', s=300, color='blue', label='General Centroid', 
                   edgecolors='black', linewidth=2)
        plt.scatter(domain_centroid_reduced[0, 0], domain_centroid_reduced[0, 1],
                   marker='*', s=300, color='red', label='Domain Centroid',
                   edgecolors='black', linewidth=2)
        
        # Draw line between centroids
        plt.plot([general_centroid_reduced[0, 0], domain_centroid_reduced[0, 0]],
                [general_centroid_reduced[0, 1], domain_centroid_reduced[0, 1]],
                'k--', alpha=0.7, linewidth=3, label='Semantic Drift')
        
        # Add shift score to title
        shift_score = self.comparison_results[term]['semantic_shift_score']
        plt.title(f'General vs Domain BERT: "{term}"\nSemantic Shift Score: {shift_score:.3f}')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def compare_mlm_predictions(self, masked_sentences: List[str], 
                               top_k: int = 5) -> List[Tuple[str, List[str], List[str]]]:
        """
        Compare masked language model predictions between general and domain models.
        
        Args:
            masked_sentences: List of sentences with [MASK] tokens
            top_k: Number of top predictions to return
            
        Returns:
            List of (sentence, general_predictions, domain_predictions) tuples
        """
        if not self.general_model or not self.domain_model:
            raise ValueError("Both models must be loaded first")
        
        print(f"🎭 Comparing MLM predictions for {len(masked_sentences)} sentences...")
        
        # Load MLM models
        general_mlm = AutoModelForMaskedLM.from_pretrained(self.base_model_name)
        
        # For domain model, we need to load the MLM version
        domain_model_path = getattr(self, '_domain_model_path', None)
        if domain_model_path:
            try:
                domain_mlm = AutoModelForMaskedLM.from_pretrained(domain_model_path)
            except:
                print("⚠️  Domain MLM model not available, using general model")
                domain_mlm = general_mlm
        else:
            domain_mlm = general_mlm
        
        results = []
        
        for sentence in masked_sentences:
            # Get predictions from general model
            general_preds = self._get_mlm_predictions(
                sentence, general_mlm, self.general_tokenizer, top_k
            )
            
            # Get predictions from domain model
            domain_preds = self._get_mlm_predictions(
                sentence, domain_mlm, self.domain_tokenizer or self.general_tokenizer, top_k
            )
            
            results.append((sentence, general_preds, domain_preds))
        
        return results
    
    def _get_mlm_predictions(self, sentence: str, model: Any, 
                           tokenizer: Any, top_k: int) -> List[str]:
        """Get top-k MLM predictions for a sentence."""
        try:
            # Create MLM pipeline
            mlm_pipeline = pipeline(
                "fill-mask",
                model=model,
                tokenizer=tokenizer,
                top_k=top_k
            )
            
            # Get predictions
            predictions = mlm_pipeline(sentence)
            
            # Extract tokens
            if isinstance(predictions, list):
                return [pred['token_str'].strip() for pred in predictions]
            else:
                return [predictions['token_str'].strip()]
                
        except Exception as e:
            print(f"⚠️  MLM prediction error: {e}")
            return ["[ERROR]"] * top_k
    
    def generate_comparison_report(self, output_path: str = "results/bert_model_comparison.json"):
        """Generate a comprehensive comparison report."""
        report = {
            'comparison_summary': {
                'total_terms_analyzed': len(self.comparison_results),
                'high_shift_terms': [],
                'moderate_shift_terms': [],
                'low_shift_terms': [],
                'avg_shift_score': 0.0
            },
            'detailed_results': self.comparison_results,
            'methodology': {
                'base_model': self.base_model_name,
                'comparison_method': 'centroid_similarity',
                'shift_score_calculation': '1 - cosine_similarity(general_centroid, domain_centroid)'
            }
        }
        
        # Analyze shift scores
        shift_scores = [r['semantic_shift_score'] for r in self.comparison_results.values()]
        report['comparison_summary']['avg_shift_score'] = float(np.mean(shift_scores))
        
        for term, result in self.comparison_results.items():
            score = result['semantic_shift_score']
            if score > 0.25:
                report['comparison_summary']['high_shift_terms'].append({
                    'term': term, 'score': score
                })
            elif score > 0.15:
                report['comparison_summary']['moderate_shift_terms'].append({
                    'term': term, 'score': score
                })
            else:
                report['comparison_summary']['low_shift_terms'].append({
                    'term': term, 'score': score
                })
        
        # Remove numpy arrays for JSON serialization
        for term in report['detailed_results']:
            if 'embeddings' in report['detailed_results'][term]:
                del report['detailed_results'][term]['embeddings']
        
        # Save report
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Comparison report saved to {output_path}")
        return report


class DomainTextDataset(Dataset):
    """Dataset for domain-specific BERT fine-tuning."""
    
    def __init__(self, texts: List[str], tokenizer: Any, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
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


def demo_domain_bert_comparison():
    """Demonstrate domain BERT comparison workflow."""
    print("🚀 Domain BERT Comparison Demo")
    print("=" * 50)
    
    # This would use your actual data
    sample_domain_texts = [
        "The globalist elite are controlling our government.",
        "Urban youth are destroying our neighborhoods.",
        "Traditional values are under attack by leftists.",
        "Real Americans know what's really happening.",
        "The mainstream media won't tell you the truth.",
    ]
    
    sample_terms = ["globalist", "urban youth", "traditional values"]
    
    sample_contexts = {
        "globalist": [
            "The globalist agenda threatens our sovereignty.",
            "Globalist policies are destroying America.",
            "We must fight the globalist conspiracy."
        ],
        "urban youth": [
            "Urban youth are causing problems in cities.",
            "The urban youth issue needs addressing.",
            "Urban youth programs are ineffective."
        ]
    }
    
    print("Demo workflow:")
    print("1. Load general BERT model")
    print("2. Fine-tune on domain data")
    print("3. Compare model representations")
    print("4. Analyze semantic shifts")
    print("5. Generate comparison report")
    
    print(f"\nSample domain texts: {len(sample_domain_texts)}")
    print(f"Sample terms: {sample_terms}")
    
    return sample_domain_texts, sample_terms, sample_contexts


if __name__ == "__main__":
    demo_domain_bert_comparison()
