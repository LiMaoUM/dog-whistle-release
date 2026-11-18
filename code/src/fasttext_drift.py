"""
FastText-based semantic drift detection for emerging dog whistle identification.

This module implements semantic divergence detection by comparing domain-specific
FastText embeddings (trained on social media data) against general-purpose 
FastText embeddings to identify terms with shifted meanings.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set, Union
from collections import defaultdict, Counter
from pathlib import Path
import requests
import zipfile
import gensim
from gensim.models import FastText
from itertools import combinations
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from preprocessing import TextPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FastTextSemanticDriftDetector:
    """
    Detects semantic drift by comparing domain-specific and general FastText models.
    
    This class trains a domain-specific FastText model on social media data and
    compares it against a general-purpose model to identify terms whose meanings
    have diverged, indicating potential dog whistle usage.
    """
    
    def __init__(self, 
                 vector_size: int = 300,
                 window: int = 5,
                 min_count: int = 10,
                 epochs: int = 10,
                 min_freq_threshold: int = 20,
                 workers: int = 4,
                 analyze_ngrams: bool = True,
                 ngram_sizes: List[int] = None):
        """
        Initialize the semantic drift detector.
        
        Args:
            vector_size: Dimensionality of word vectors
            window: Context window size
            min_count: Minimum word count for inclusion in vocabulary
            epochs: Number of training epochs
            min_freq_threshold: Minimum frequency for drift analysis
            workers: Number of worker threads
            analyze_ngrams: Whether to analyze n-grams (2-grams, 3-grams)
            ngram_sizes: List of n-gram sizes to analyze (default: [2, 3])
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.min_freq_threshold = min_freq_threshold
        self.workers = workers
        self.analyze_ngrams = analyze_ngrams
        self.ngram_sizes = ngram_sizes or [2, 3]
        
        # Initialize text preprocessor for lemmatization
        self.preprocessor = TextPreprocessor()
        
        # Models - now we have three for comparison
        self.domain_model_raw = None      # Trained from scratch
        self.domain_model_finetuned = None # Fine-tuned from pre-trained
        self.general_model = None         # Pre-trained general model
        
        # Analysis results
        self.drift_results = {}
        self.vocabulary_stats = {}
        self.ngram_analysis_results = {}  # Results for n-gram analysis
        
        # Paths
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
    def preprocess_text_for_training(self, texts: List[str]) -> List[List[str]]:
        """
        Preprocess texts for FastText training using simple tokenization.
        
        FastText works best with simple tokenization that matches the pre-trained models.
        This avoids over-processing that could create inconsistencies between domain
        and general models.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of tokenized word lists
        """
        processed_sentences = []
        
        for text in texts:
            if pd.isna(text) or not isinstance(text, str):
                continue
            
            # Clean the text first
            cleaned_text = self.preprocessor.clean_text(text, 
                                                       remove_urls=True, 
                                                       remove_mentions=False, 
                                                       remove_hashtags=False)
            
            # Use FastText-compatible tokenization
            tokens = self.preprocessor.fasttext_tokenize(cleaned_text, remove_stopwords=False)
            
            if len(tokens) > 3:  # Only include sentences with meaningful content
                processed_sentences.append(tokens)
        
        logger.info(f"Preprocessed {len(texts)} texts into {len(processed_sentences)} sentences using FastText-compatible tokenization")
        return processed_sentences
    
    def generate_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """
        Generate n-grams from a list of tokens.
        
        Args:
            tokens: List of tokens
            n: Size of n-grams (e.g., 2 for bigrams, 3 for trigrams)
            
        Returns:
            List of n-gram strings
        """
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = "_".join(tokens[i:i+n])
            ngrams.append(ngram)
        return ngrams
    
    def extract_ngrams_from_sentences(self, sentences: List[List[str]], 
                                     ngram_sizes: List[int] = None) -> Dict[int, List[List[str]]]:
        """
        Extract n-grams of different sizes from sentences.
        
        Args:
            sentences: List of tokenized sentences
            ngram_sizes: List of n-gram sizes to extract
            
        Returns:
            Dictionary mapping n-gram sizes to lists of sentences with n-grams
        """
        if ngram_sizes is None:
            ngram_sizes = self.ngram_sizes
        
        ngram_sentences = {}
        
        for n in ngram_sizes:
            ngram_list = []
            ngram_count = 0
            
            for sentence in sentences:
                ngrams = self.generate_ngrams(sentence, n)
                if ngrams:
                    # Combine original tokens with n-grams
                    augmented_sentence = sentence + ngrams
                    ngram_list.append(augmented_sentence)
                    ngram_count += len(ngrams)
            
            ngram_sentences[n] = ngram_list
            logger.info(f"Generated {ngram_count} {n}-grams from {len(sentences)} sentences")
        
        return ngram_sentences
    
    def train_domain_model(self, df: pd.DataFrame, text_column: str = 'text') -> FastText:
        """
        Train domain-specific FastText model from scratch on social media data.
        Supports both unigrams and n-grams (bigrams, trigrams).
        
        Args:
            df: DataFrame containing social media posts
            text_column: Column name containing text data
            
        Returns:
            Trained FastText model
        """
        logger.info("Training domain-specific FastText model from scratch...")
        
        # Preprocess texts
        texts = df[text_column].dropna().tolist()
        sentences = self.preprocess_text_for_training(texts)
        
        if len(sentences) == 0:
            raise ValueError("No valid sentences found for training")
        
        # Add n-grams if requested
        training_sentences = sentences
        if self.analyze_ngrams:
            ngram_sentences_dict = self.extract_ngrams_from_sentences(sentences, self.ngram_sizes)
            # Use the sentences augmented with n-grams for training
            # Combine all n-gram levels into the sentences
            training_sentences = ngram_sentences_dict.get(max(self.ngram_sizes), sentences)
        
        # Train FastText model from scratch
        self.domain_model_raw = FastText(
            sentences=training_sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            workers=self.workers,
            sg=1,  # Skip-gram
            seed=42
        )
        
        # Save model
        model_path = self.models_dir / "domain_fasttext_raw.model"
        self.domain_model_raw.save(str(model_path))
        logger.info(f"Raw domain model saved to {model_path}")
        
        # Log training statistics
        vocab_size = len(self.domain_model_raw.wv.index_to_key)
        logger.info(f"Raw domain model trained with vocabulary size: {vocab_size}")
        
        # Count n-grams in vocabulary
        ngram_counts = {}
        for n in self.ngram_sizes:
            ngram_marker = "_" * (n - 1)  # n-grams have underscores: bigrams have 1, trigrams have 2
            count = sum(1 for word in self.domain_model_raw.wv.index_to_key if ngram_marker in word)
            ngram_counts[f"{n}gram"] = count
        
        self.vocabulary_stats['domain_raw'] = {
            'vocab_size': vocab_size,
            'sentences': len(sentences),
            'total_words': sum(len(sent) for sent in sentences),
            **ngram_counts  # Add n-gram statistics
        }
        
        return self.domain_model_raw
    
    def finetune_domain_model(self, df: pd.DataFrame, text_column: str = 'text', 
                             pretrained_path: Optional[str] = None) -> FastText:
        """
        Fine-tune a domain-specific FastText model starting from pre-trained vectors.
        Supports both unigrams and n-grams (bigrams, trigrams).
        
        Args:
            df: DataFrame containing social media posts
            text_column: Column name containing text data
            pretrained_path: Path to pre-trained vectors
            
        Returns:
            Fine-tuned FastText model
        """
        logger.info("Fine-tuning FastText model from pre-trained vectors...")
        
        # Preprocess texts
        texts = df[text_column].dropna().tolist()
        sentences = self.preprocess_text_for_training(texts)
        
        if len(sentences) == 0:
            raise ValueError("No valid sentences found for training")
        
        # Add n-grams if requested
        training_sentences = sentences
        if self.analyze_ngrams:
            ngram_sentences_dict = self.extract_ngrams_from_sentences(sentences, self.ngram_sizes)
            training_sentences = ngram_sentences_dict.get(max(self.ngram_sizes), sentences)
        
        # Load pre-trained vectors first if not already loaded
        if pretrained_path is None:
            pretrained_path = self.download_pretrained_model()
        
        try:
            # Try to load pre-trained FastText model
            logger.info(f"Loading pre-trained vectors from {pretrained_path}")
            
            # For fine-tuning, we need to start with a FastText model structure
            # First create a model with the same architecture
            base_model = FastText(
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                workers=self.workers,
                sg=1,
                seed=42
            )
            
            # Build vocabulary from domain data
            base_model.build_vocab(training_sentences)
            
            # Load pre-trained vectors for words that exist
            if pretrained_path.endswith('.vec'):
                pretrained_kv = KeyedVectors.load_word2vec_format(pretrained_path, binary=False)
            else:
                pretrained_model = FastText.load(pretrained_path)
                pretrained_kv = pretrained_model.wv
            
            # Initialize vectors with pre-trained where available
            logger.info("Initializing domain vocabulary with pre-trained vectors...")
            initialized_count = 0
            for word in base_model.wv.index_to_key:
                if word in pretrained_kv:
                    base_model.wv[word] = pretrained_kv[word]
                    initialized_count += 1
            
            logger.info(f"Initialized {initialized_count} words with pre-trained vectors")
            
            # Continue training on domain data
            logger.info("Fine-tuning with domain-specific data...")
            base_model.train(training_sentences, total_examples=len(training_sentences), epochs=self.epochs)
            
            self.domain_model_finetuned = base_model
            
            # Save fine-tuned model
            model_path = self.models_dir / "domain_fasttext_finetuned.model"
            self.domain_model_finetuned.save(str(model_path))
            logger.info(f"Fine-tuned domain model saved to {model_path}")
            
            # Log training statistics
            vocab_size = len(self.domain_model_finetuned.wv.index_to_key)
            logger.info(f"Fine-tuned domain model has vocabulary size: {vocab_size}")
            
            # Count n-grams in vocabulary
            ngram_counts = {}
            for n in self.ngram_sizes:
                ngram_marker = "_" * (n - 1)
                count = sum(1 for word in self.domain_model_finetuned.wv.index_to_key if ngram_marker in word)
                ngram_counts[f"{n}gram"] = count
            
            self.vocabulary_stats['domain_finetuned'] = {
                'vocab_size': vocab_size,
                'sentences': len(sentences),
                'total_words': sum(len(sent) for sent in sentences),
                'pretrained_init': initialized_count,
                **ngram_counts  # Add n-gram statistics
            }
            
            return self.domain_model_finetuned
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            logger.warning("Falling back to training from scratch")
            return self.train_domain_model(df, text_column)
    
    def load_general_model(self, model_path: Optional[str] = None) -> KeyedVectors:
        """
        Load general-purpose FastText model.
        
        Args:
            model_path: Path to pre-trained model file. If None, downloads a model.
            
        Returns:
            Loaded KeyedVectors
        """
        logger.info("Loading general-purpose FastText model...")
        
        if model_path is None:
            # Download pre-trained English FastText vectors
            model_path = self.download_pretrained_model()
        
        try:
            # Try loading as KeyedVectors first
            self.general_model = KeyedVectors.load_word2vec_format(model_path, binary=False)
            logger.info("Loaded general model as word2vec format")
        except:
            try:
                # Try loading as FastText model
                general_ft = FastText.load(model_path)
                self.general_model = general_ft.wv
                logger.info("Loaded general model as FastText format")
            except Exception as e:
                logger.error(f"Failed to load general model: {e}")
                raise
        
        vocab_size = len(self.general_model.index_to_key)
        logger.info(f"General model loaded with vocabulary size: {vocab_size}")
        
        self.vocabulary_stats['general'] = {
            'vocab_size': vocab_size
        }
        
        return self.general_model
    
    def download_pretrained_model(self) -> str:
        """
        Download pre-trained FastText vectors.
        
        Returns:
            Path to downloaded model file
        """
        model_dir = self.models_dir / "pretrained"
        model_dir.mkdir(exist_ok=True)
        
        # Try different pre-trained model sources
        model_file = model_dir / "cc.en.300.vec"
        
        if model_file.exists():
            logger.info(f"Using existing model file: {model_file}")
            return str(model_file)
        
        # For demonstration, use a smaller subset or create from Wikipedia
        logger.info("Downloading pre-trained FastText vectors...")
        
        # Option 1: Try to download smaller model first
        small_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec"
        small_file = model_dir / "wiki.en.vec"
        
        try:
            if not small_file.exists():
                logger.info("Downloading smaller Wikipedia-based FastText vectors...")
                response = requests.get(small_url, stream=True)
                response.raise_for_status()
                
                with open(small_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Downloaded general model to {small_file}")
                return str(small_file)
            else:
                return str(small_file)
                
        except Exception as e:
            logger.warning(f"Failed to download pre-trained model: {e}")
            logger.warning("Creating demo general model instead")
            return self.create_demo_general_model()
    
    def create_demo_general_model(self) -> str:
        """
        Create a demo general model using English Wikipedia data or a synthetic approach.
        This is for demonstration purposes when full pre-trained models aren't available.
        
        Returns:
            Path to demo model
        """
        logger.warning("Creating demo general model from synthetic data")
        
        demo_path = self.models_dir / "demo_general.model"
        
        # Create a simple general vocabulary model for demonstration
        # In production, this would be replaced with actual pre-trained vectors
        
        # Generate some "general" text data that would be different from social media
        general_sentences = [
            ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"],
            ["artificial", "intelligence", "machine", "learning", "natural", "language", "processing"],
            ["government", "democracy", "election", "vote", "citizen", "representative"],
            ["economy", "market", "business", "finance", "trade", "commerce"],
            ["science", "research", "technology", "innovation", "development"],
            ["education", "school", "university", "student", "teacher", "learning"],
            ["health", "medicine", "doctor", "patient", "treatment", "hospital"],
            ["environment", "climate", "nature", "conservation", "sustainability"],
            ["culture", "society", "community", "tradition", "heritage"],
            ["family", "friend", "relationship", "love", "trust", "support"]
        ]
        
        # Expand the general sentences
        expanded_general = []
        for base_sent in general_sentences:
            # Create variations
            for i in range(10):  # Create multiple variations
                variation = base_sent + [f"word_{i}", f"term_{i}"]
                expanded_general.append(variation)
        
        # Train a simple FastText model as "general" baseline
        demo_general_model = FastText(
            sentences=expanded_general,
            vector_size=self.vector_size,
            window=self.window,
            min_count=1,  # Lower threshold for demo
            epochs=5,     # Fewer epochs for demo
            workers=self.workers,
            sg=1,
            seed=42
        )
        
        # Save and set as general model
        demo_general_model.save(str(demo_path))
        self.general_model = demo_general_model.wv
        
        logger.warning("Created synthetic general model for demonstration")
        logger.warning("In production, use actual pre-trained FastText vectors")
        
        return str(demo_path)
    
    def extract_candidate_terms(self, model_type: str = 'raw', 
                              exclude_known_dogwhistles: bool = True, 
                              filter_pos: bool = False,
                              ngram_size: Optional[int] = None) -> List[str]:
        """
        Extract candidate terms for drift analysis.
        
        Args:
            model_type: Which domain model to use ('raw' or 'finetuned')
            exclude_known_dogwhistles: Whether to exclude known dog whistle terms
            filter_pos: Whether to filter to nouns and adjectives only
            ngram_size: If provided, filter by n-gram size (1 for unigrams, 2 for bigrams, etc.)
            
        Returns:
            List of candidate terms
        """
        # Select the appropriate domain model
        if model_type == 'raw' and self.domain_model_raw is not None:
            domain_model = self.domain_model_raw
        elif model_type == 'finetuned' and self.domain_model_finetuned is not None:
            domain_model = self.domain_model_finetuned
        else:
            raise ValueError(f"Domain model '{model_type}' not trained yet")
        
    # Get vocabulary from domain model
        domain_vocab = set(domain_model.wv.index_to_key)
        
        # Filter by frequency
        word_counts = {}
        for word in domain_vocab:
            # Get word count from model (approximation)
            word_counts[word] = domain_model.wv.get_vecattr(word, "count")
        
        # For n-grams, use lower frequency threshold since they're inherently less frequent
        freq_threshold = self.min_freq_threshold
        if ngram_size is not None and ngram_size > 1:
            # Use lower threshold for n-grams - typically 1/n of the unigram threshold
            freq_threshold = max(2, self.min_freq_threshold // (ngram_size + 1))
        
        # Filter by minimum frequency
        frequent_terms = [
            word for word, count in word_counts.items() 
            if count >= freq_threshold
        ]
        
        logger.info(f"Found {len(frequent_terms)} terms above frequency threshold ({freq_threshold}) in {model_type} model")
        
        # Optionally filter by POS tags
        if filter_pos:
            # POS filtering is intended for unigram tokens. When ngram_size is
            # provided, skip POS filtering because POS taggers don't handle
            # underscore-joined n-grams well.
            if ngram_size is None or ngram_size == 1:
                frequent_terms = self.filter_by_pos(frequent_terms)
        
        # Optionally exclude known dog whistles
        if exclude_known_dogwhistles:
            frequent_terms = self.filter_known_dogwhistles(frequent_terms)
        
        # If ngram_size is provided, filter tokens by underscore count
        if ngram_size is not None and ngram_size > 1:
            # n-gram tokens are stored with underscores joining tokens
            underscore_count = ngram_size - 1
            frequent_terms = [t for t in frequent_terms if t.count('_') == underscore_count]

        # Filter to keep only terms present in general model (skip for n-grams)
        if self.general_model is not None and (ngram_size is None or ngram_size == 1):
            general_vocab = set(self.general_model.index_to_key)
            shared_terms = [term for term in frequent_terms if term in general_vocab]
            logger.info(f"Found {len(shared_terms)} terms shared between {model_type} and general models (ngram_size={ngram_size})")
            return shared_terms

        return frequent_terms
    
    def filter_by_pos(self, terms: List[str]) -> List[str]:
        """
        Filter terms to keep only nouns and adjectives.
        
        Args:
            terms: List of candidate terms
            
        Returns:
            Filtered list of terms
        """
        try:
            import nltk
            from nltk.tag import pos_tag
            from nltk.tokenize import word_tokenize
            
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                nltk.download('averaged_perceptron_tagger')
            
            # POS tag the terms
            pos_tags = pos_tag(terms)
            
            # Keep only nouns (NN, NNS, NNP, NNPS) and adjectives (JJ, JJR, JJS)
            target_pos = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'}
            filtered_terms = [term for term, pos in pos_tags if pos in target_pos]
            
            logger.info(f"POS filtering: {len(terms)} -> {len(filtered_terms)} terms")
            return filtered_terms
            
        except ImportError:
            logger.warning("NLTK not available for POS filtering")
            return terms
        except Exception as e:
            logger.warning(f"POS filtering failed: {e}")
            return terms
    
    def filter_known_dogwhistles(self, terms: List[str]) -> List[str]:
        """
        Filter out known dog whistle terms.
        
        Args:
            terms: List of candidate terms
            
        Returns:
            Filtered list of terms
        """
        # Load known dog whistles from Allen AI glossary or other sources
        try:
            from .allen_ai_loader import AllenAIGlossaryLoader
            allen_loader = AllenAIGlossaryLoader()
            known_entries = allen_loader.load_entries()
            
            known_terms = set()
            for entry in known_entries:
                # Add surface forms and covert meanings
                known_terms.update([sf.lower() for sf in entry.surface_forms])
                known_terms.add(entry.covert_meaning.lower())
            
            filtered_terms = [term for term in terms if term.lower() not in known_terms]
            logger.info(f"Filtered out {len(terms) - len(filtered_terms)} known dog whistle terms")
            return filtered_terms
            
        except Exception as e:
            logger.warning(f"Could not load known dog whistles: {e}")
            return terms
    
    def compute_semantic_divergence(self, terms: List[str], model_type: str = 'raw', skip_general_check: bool = False) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """
        Compute semantic divergence scores for given terms and track unpaired words.
        
        The drift score is computed as: drift_score = 1 - cosine_similarity(domain_vector, general_vector)
        - Score of 0 = identical vectors (no drift)
        - Score of 1 = orthogonal vectors (moderate drift) 
        - Score of 2 = opposite vectors (maximum drift)
        
        For n-grams where general model doesn't have matches, skip_general_check allows
        computing drift even without general model comparison.
        
        Args:
            terms: List of terms to analyze
            model_type: Which domain model to use ('raw' or 'finetuned')
            skip_general_check: If True, compute drift for terms even if not in general model
            
        Returns:
            Tuple of (drift_scores_dict, unpaired_words_dict) where:
            - drift_scores_dict: Dictionary mapping terms to drift scores
            - unpaired_words_dict: Dictionary with 'in_domain_only' and 'in_general_only' lists
        """
        # Select the appropriate domain model
        if model_type == 'raw' and self.domain_model_raw is not None:
            domain_model = self.domain_model_raw
        elif model_type == 'finetuned' and self.domain_model_finetuned is not None:
            domain_model = self.domain_model_finetuned
        else:
            raise ValueError(f"Domain model '{model_type}' not available")
            
        if self.general_model is None and not skip_general_check:
            raise ValueError("General model must be loaded")
        
        drift_scores = {}
        unpaired_words = {
            'in_domain_only': [],
            'in_general_only': []
        }
        
        logger.info(f"Computing semantic divergence for {len(terms)} terms using {model_type} model...")
        
        for term in terms:
            try:
                # Check if term exists in both models
                domain_has_term = term in domain_model.wv
                general_has_term = self.general_model is not None and term in self.general_model
                
                if domain_has_term and general_has_term:
                    # Get normalized embeddings from both models
                    domain_vec = domain_model.wv[term]
                    general_vec = self.general_model[term]
                    
                    # Ensure vectors are normalized (unit length)
                    domain_vec_norm = domain_vec / np.linalg.norm(domain_vec)
                    general_vec_norm = general_vec / np.linalg.norm(general_vec)
                    
                    # Compute cosine similarity using dot product of normalized vectors
                    # Range: [-1, 1] where 1=identical, 0=orthogonal, -1=opposite
                    cosine_sim = np.dot(domain_vec_norm, general_vec_norm)
                    
                    # Convert to drift score: 1 - similarity
                    # Range: [0, 2] where 0=no drift, 1=moderate drift, 2=maximum drift
                    drift_score = 1.0 - cosine_sim
                    
                    drift_scores[term] = float(drift_score)
                    
                elif domain_has_term and skip_general_check:
                    # For n-grams or when general model is not available, use domain freq magnitude as proxy
                    # This gives us a basic "importance" score
                    domain_vec = domain_model.wv[term]
                    # Use vector magnitude as drift indicator (high magnitude = more established)
                    magnitude = float(np.linalg.norm(domain_vec))
                    # Normalize to 0-2 range
                    drift_score = 1.0 - (magnitude / 300.0)  # Assuming max norm ~300 for our vectors
                    drift_score = max(0.0, min(2.0, drift_score))  # Clamp to [0, 2]
                    
                    drift_scores[term] = float(drift_score)
                    
                elif domain_has_term and not general_has_term:
                    unpaired_words['in_domain_only'].append(term)
                elif not domain_has_term and general_has_term:
                    unpaired_words['in_general_only'].append(term)
                # If term is in neither model, skip silently
                
            except KeyError:
                logger.warning(f"Term '{term}' not found in one of the models")
                continue
            except Exception as e:
                logger.warning(f"Error computing drift for '{term}': {e}")
                continue
        
        logger.info(f"Computed drift scores for {len(drift_scores)} terms using {model_type} model")
        logger.info(f"Unpaired words - Domain only: {len(unpaired_words['in_domain_only'])}, General only: {len(unpaired_words['in_general_only'])}")
        
        return drift_scores, unpaired_words
    
    def get_example_sentences(self, term: str, df: pd.DataFrame, 
                            text_column: str = 'text', max_examples: int = 3) -> List[str]:
        """
        Get example sentences containing the term.
        
        Args:
            term: Term to find examples for
            df: DataFrame containing text data
            text_column: Column containing text
            max_examples: Maximum number of examples to return
            
        Returns:
            List of example sentences
        """
        # Find sentences containing the term
        mask = df[text_column].str.contains(term, case=False, na=False)
        examples = df[mask][text_column].tolist()
        
        # Return up to max_examples
        return examples[:max_examples]
    
    def analyze_semantic_drift(self, df: pd.DataFrame, 
                             text_column: str = 'text',
                             top_n: int = 50,
                             filter_pos: bool = False,
                             compare_models: bool = True) -> Dict:
        """
        Complete semantic drift analysis pipeline comparing multiple model approaches.
        
        Args:
            df: DataFrame containing social media data (FULL CORPUS)
            text_column: Column name containing text
            top_n: Number of top drift terms to return
            filter_pos: Whether to filter to nouns/adjectives only
            compare_models: Whether to compare raw vs fine-tuned models
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Starting semantic drift analysis on FULL CORPUS ({len(df)} posts)...")
        
        # Step 1: Load general model first
        if self.general_model is None:
            self.load_general_model()
        
        # Step 2: Train raw domain model from scratch
        if self.domain_model_raw is None:
            logger.info("Training raw domain model from scratch...")
            self.train_domain_model(df, text_column)
        
        # Step 3: Fine-tune domain model (if requested)
        if compare_models and self.domain_model_finetuned is None:
            logger.info("Fine-tuning domain model from pre-trained vectors...")
            try:
                self.finetune_domain_model(df, text_column)
            except Exception as e:
                logger.warning(f"Fine-tuning failed: {e}")
                compare_models = False
        
        # Step 4: Extract candidate terms from both models
        results = {}
        
        # Analyze raw model
        logger.info("Analyzing raw domain model...")
        candidate_terms_raw = self.extract_candidate_terms(model_type='raw', filter_pos=filter_pos)

        if len(candidate_terms_raw) > 0:
            drift_scores_raw, unpaired_words_raw = self.compute_semantic_divergence(candidate_terms_raw, model_type='raw')
            ranked_terms_raw = sorted(drift_scores_raw.items(), key=lambda x: x[1], reverse=True)

            # Get detailed results for top terms
            top_terms_raw = self.get_detailed_results(ranked_terms_raw[:top_n], df, text_column, 'raw')

            results['raw_model'] = {
                'total_candidates': len(candidate_terms_raw),
                'top_terms': top_terms_raw,
                'model_type': 'trained_from_scratch',
                'unpaired_words': unpaired_words_raw,
                'all_drift_scores': drift_scores_raw  # Save ALL drift scores
            }

        # If requested, analyze n-grams separately (2-grams, 3-grams)
        if self.analyze_ngrams:
            for n in self.ngram_sizes:
                logger.info(f"Analyzing raw domain model for {n}-grams...")
                candidate_terms_ng = self.extract_candidate_terms(model_type='raw', filter_pos=filter_pos, ngram_size=n)
                key_name = f"raw_model_{n}gram"

                if candidate_terms_ng:
                    # For n-grams, skip general model check since general model won't have underscored n-grams
                    drift_scores_ng, unpaired_ng = self.compute_semantic_divergence(candidate_terms_ng, model_type='raw', skip_general_check=True)
                    ranked_ng = sorted(drift_scores_ng.items(), key=lambda x: x[1], reverse=True)
                    top_terms_ng = self.get_detailed_results(ranked_ng[:top_n], df, text_column, 'raw')

                    results[key_name] = {
                        'total_candidates': len(candidate_terms_ng),
                        'top_terms': top_terms_ng,
                        'model_type': 'trained_from_scratch',
                        'unpaired_words': unpaired_ng,
                        'all_drift_scores': drift_scores_ng
                    }
        
        # Analyze fine-tuned model (if available)
        if compare_models and self.domain_model_finetuned is not None:
            logger.info("Analyzing fine-tuned domain model...")
            candidate_terms_finetuned = self.extract_candidate_terms(model_type='finetuned', filter_pos=filter_pos)
            
            if len(candidate_terms_finetuned) > 0:
                drift_scores_finetuned, unpaired_words_finetuned = self.compute_semantic_divergence(candidate_terms_finetuned, model_type='finetuned')
                ranked_terms_finetuned = sorted(drift_scores_finetuned.items(), key=lambda x: x[1], reverse=True)
                
                # Get detailed results for top terms
                top_terms_finetuned = self.get_detailed_results(ranked_terms_finetuned[:top_n], df, text_column, 'finetuned')
                
                results['finetuned_model'] = {
                    'total_candidates': len(candidate_terms_finetuned),
                    'top_terms': top_terms_finetuned,
                    'model_type': 'finetuned_from_pretrained',
                    'unpaired_words': unpaired_words_finetuned,
                    'all_drift_scores': drift_scores_finetuned  # Save ALL drift scores
                }

            # N-gram analysis on fine-tuned model
            if self.analyze_ngrams:
                for n in self.ngram_sizes:
                    logger.info(f"Analyzing fine-tuned domain model for {n}-grams...")
                    candidate_terms_ng = self.extract_candidate_terms(model_type='finetuned', filter_pos=filter_pos, ngram_size=n)
                    key_name = f"finetuned_model_{n}gram"

                    if candidate_terms_ng:
                        # For n-grams, skip general model check since general model won't have underscored n-grams
                        drift_scores_ng, unpaired_ng = self.compute_semantic_divergence(candidate_terms_ng, model_type='finetuned', skip_general_check=True)
                        ranked_ng = sorted(drift_scores_ng.items(), key=lambda x: x[1], reverse=True)
                        top_terms_ng = self.get_detailed_results(ranked_ng[:top_n], df, text_column, 'finetuned')

                        results[key_name] = {
                            'total_candidates': len(candidate_terms_ng),
                            'top_terms': top_terms_ng,
                            'model_type': 'finetuned_from_pretrained',
                            'unpaired_words': unpaired_ng,
                            'all_drift_scores': drift_scores_ng
                        }
        
        # Choose primary results for top-level access (domain model preferred)
        primary_model = 'domain_model' if 'domain_model' in results else list(results.keys())[0]
        primary_results = results[primary_model]
        
        # Store comprehensive results
        self.drift_results = {
            'analysis_date': datetime.now().isoformat(),
            'corpus_size': len(df),
            'model_comparison': compare_models,
            'results': results,
            'model_params': {
                'vector_size': self.vector_size,
                'window': self.window,
                'min_count': self.min_count,
                'epochs': self.epochs,
                'min_freq_threshold': self.min_freq_threshold,
                'filter_pos': filter_pos
            },
            'vocabulary_stats': self.vocabulary_stats,
            # Add top-level keys expected by generate_report()
            'total_candidates': primary_results['total_candidates'],
            'top_terms': primary_results['top_terms']
        }
        
        logger.info(f"Analysis complete on full corpus of {len(df)} posts!")
        return self.drift_results
    
    def get_detailed_results(self, ranked_terms: List[Tuple[str, float]], 
                           df: pd.DataFrame, text_column: str, model_type: str) -> List[Dict]:
        """
        Get detailed results including examples for top terms.
        
        Args:
            ranked_terms: List of (term, score) tuples
            df: DataFrame containing text data
            text_column: Column containing text
            model_type: Type of model used
            
        Returns:
            List of detailed term results
        """
        results = []
        
        # Select the appropriate domain model
        if model_type == 'raw':
            domain_model = self.domain_model_raw
        elif model_type == 'finetuned':
            domain_model = self.domain_model_finetuned
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        for term, drift_score in ranked_terms:
            # Get domain frequency
            domain_freq = domain_model.wv.get_vecattr(term, "count")
            
            # Get example sentences
            examples = self.get_example_sentences(term, df, text_column)
            
            result = {
                'term': term,
                'drift_score': drift_score,
                'domain_freq': int(domain_freq),
                'examples': examples,
                'model_type': model_type
            }
            results.append(result)
        
        return results
    
    def save_results(self, output_path: str = "semantic_drift_candidates.jsonl"):
        """
        Save analysis results to JSONL file.
        
        Args:
            output_path: Path to output file
        """
        if not self.drift_results:
            logger.warning("No results to save")
            return
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            'analysis_date': self.drift_results['analysis_date'],
            'corpus_size': self.drift_results['corpus_size'],
            'model_comparison': self.drift_results['model_comparison'],
            'model_params': self.drift_results['model_params'],
            'vocabulary_stats': self.drift_results['vocabulary_stats']
        }
        
        with open(output_file, 'w') as f:
            # Write metadata as first line
            f.write(json.dumps({'metadata': metadata}) + '\n')
            
            # Write results for each model type
            for model_name, model_results in self.drift_results['results'].items():
                # Write model header with enhanced information
                model_header = {
                    'model_type': model_name,
                    'total_candidates': model_results['total_candidates'],
                    'training_method': model_results['model_type'],
                    'unpaired_words_stats': {
                        'in_domain_only': len(model_results.get('unpaired_words', {}).get('in_domain_only', [])),
                        'in_general_only': len(model_results.get('unpaired_words', {}).get('in_general_only', []))
                    },
                    'total_drift_scores': len(model_results.get('all_drift_scores', {}))
                }
                f.write(json.dumps(model_header) + '\n')
                
                # Write ALL drift scores (not just top terms)
                all_scores = model_results.get('all_drift_scores', {})
                for term, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                    f.write(json.dumps({
                        'term': term,
                        'drift_score': score,
                        'data_type': 'drift_score'
                    }) + '\n')
                
                # Write unpaired words
                unpaired = model_results.get('unpaired_words', {})
                if unpaired.get('in_domain_only'):
                    f.write(json.dumps({
                        'unpaired_type': 'in_domain_only',
                        'terms': unpaired['in_domain_only'],
                        'count': len(unpaired['in_domain_only']),
                        'data_type': 'unpaired_words'
                    }) + '\n')
                
                if unpaired.get('in_general_only'):
                    f.write(json.dumps({
                        'unpaired_type': 'in_general_only', 
                        'terms': unpaired['in_general_only'],
                        'count': len(unpaired['in_general_only']),
                        'data_type': 'unpaired_words'
                    }) + '\n')
                
                # Write detailed top terms for reference
                f.write(json.dumps({
                    'data_type': 'top_terms_with_examples',
                    'top_terms': model_results['top_terms']
                }) + '\n')

            # If we performed explicit n-gram analysis, include a separate section
            if any(k.endswith('2gram') or k.endswith('3gram') for k in self.drift_results['results'].keys()):
                f.write(json.dumps({'data_type': 'ngram_analysis_summary'}) + '\n')
                for model_name, model_results in self.drift_results['results'].items():
                    if model_name.endswith('2gram') or model_name.endswith('3gram'):
                        ng_header = {
                            'model_type': model_name,
                            'total_candidates': model_results['total_candidates'],
                            'training_method': model_results['model_type'],
                            'total_drift_scores': len(model_results.get('all_drift_scores', {}))
                        }
                        f.write(json.dumps(ng_header) + '\n')

                        all_scores = model_results.get('all_drift_scores', {})
                        for term, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                            f.write(json.dumps({
                                'term': term,
                                'drift_score': score,
                                'data_type': 'ngram_drift_score'
                            }) + '\n')
        
        logger.info(f"Results saved to {output_file}")
        
        # Also save a comprehensive summary
        summary_file = output_file.parent / "comprehensive_drift_summary.json"
        comparison_summary = {
            'analysis_date': self.drift_results['analysis_date'],
            'corpus_size': self.drift_results['corpus_size'],
            'models_compared': list(self.drift_results['results'].keys()),
            'vocabulary_comparison': self.vocabulary_stats,
            'comprehensive_statistics': {},
            'top_terms_comparison': {},
            'unpaired_words_summary': {}
        }
        
        # Extract comprehensive statistics for each model
        for model_name, model_results in self.drift_results['results'].items():
            # Top terms comparison
            comparison_summary['top_terms_comparison'][model_name] = [
                {
                    'term': term['term'],
                    'drift_score': term['drift_score'],
                    'domain_freq': term['domain_freq']
                }
                for term in model_results['top_terms'][:10]
            ]
            
            # Comprehensive statistics
            all_scores = model_results.get('all_drift_scores', {})
            unpaired = model_results.get('unpaired_words', {})
            
            comparison_summary['comprehensive_statistics'][model_name] = {
                'total_drift_scores': len(all_scores),
                'avg_drift_score': np.mean(list(all_scores.values())) if all_scores else 0,
                'max_drift_score': max(all_scores.values()) if all_scores else 0,
                'min_drift_score': min(all_scores.values()) if all_scores else 0,
                'std_drift_score': np.std(list(all_scores.values())) if all_scores else 0
            }
            
            # Unpaired words summary
            comparison_summary['unpaired_words_summary'][model_name] = {
                'in_domain_only_count': len(unpaired.get('in_domain_only', [])),
                'in_general_only_count': len(unpaired.get('in_general_only', [])),
                'sample_domain_only': unpaired.get('in_domain_only', [])[:10],  # First 10 for review
                'sample_general_only': unpaired.get('in_general_only', [])[:10]   # First 10 for review
            }
        
        with open(summary_file, 'w') as f:
            json.dump(comparison_summary, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        logger.info(f"Comprehensive summary saved to {summary_file}")
        logger.info(f"Analysis includes ALL drift scores and unpaired words tracking")
    
    def visualize_drift_distribution(self, save_path: Optional[str] = None):
        """
        Create visualizations of drift score distribution.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.drift_results or not self.drift_results.get('top_terms'):
            logger.warning("No results to visualize")
            return
        
        # Extract data for plotting
        terms = [item['term'] for item in self.drift_results['top_terms']]
        scores = [item['drift_score'] for item in self.drift_results['top_terms']]
        frequencies = [item['domain_freq'] for item in self.drift_results['top_terms']]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Top terms by drift score
        top_20_terms = terms[:20]
        top_20_scores = scores[:20]
        
        ax1.barh(range(len(top_20_terms)), top_20_scores)
        ax1.set_yticks(range(len(top_20_terms)))
        ax1.set_yticklabels(top_20_terms)
        ax1.set_xlabel('Semantic Drift Score')
        ax1.set_title('Top 20 Terms by Semantic Drift')
        ax1.invert_yaxis()
        
        # Drift score vs frequency scatter
        ax2.scatter(frequencies, scores, alpha=0.6)
        ax2.set_xlabel('Domain Frequency (log scale)')
        ax2.set_ylabel('Semantic Drift Score')
        ax2.set_title('Semantic Drift vs Term Frequency')
        ax2.set_xscale('log')
        
        # Add trend line
        z = np.polyfit(np.log(frequencies), scores, 1)
        p = np.poly1d(z)
        x_trend = np.logspace(np.log10(min(frequencies)), np.log10(max(frequencies)), 100)
        ax2.plot(x_trend, p(np.log(x_trend)), "r--", alpha=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def generate_report(self) -> str:
        """
        Generate a text report of the analysis results.
        
        Returns:
            Formatted report string
        """
        if not self.drift_results:
            return "No analysis results available"
        
        report = []
        report.append("SEMANTIC DRIFT ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Analysis Date: {self.drift_results['analysis_date']}")
        report.append(f"Total Candidates Evaluated: {self.drift_results['total_candidates']}")
        
        # Model parameters
        report.append("\nModel Parameters:")
        for param, value in self.drift_results['model_params'].items():
            report.append(f"  {param}: {value}")
        
        # Vocabulary statistics
        report.append("\nVocabulary Statistics:")
        for model_type, stats in self.drift_results['vocabulary_stats'].items():
            report.append(f"  {model_type.title()} Model:")
            for stat, value in stats.items():
                report.append(f"    {stat}: {value:,}")
        
        # Top terms
        report.append(f"\nTop {len(self.drift_results['top_terms'])} Terms by Semantic Drift:")
        for i, term_data in enumerate(self.drift_results['top_terms'][:10], 1):
            report.append(f"\n{i:2d}. {term_data['term']}")
            report.append(f"    Drift Score: {term_data['drift_score']:.4f}")
            report.append(f"    Domain Frequency: {term_data['domain_freq']:,}")
            if term_data['examples']:
                report.append("    Example Usage:")
                for example in term_data['examples'][:2]:
                    # Truncate long examples
                    ex_text = example[:100] + "..." if len(example) > 100 else example
                    report.append(f"      \"{ex_text}\"")
        
        return "\n".join(report)
    
    def generate_comprehensive_report(self) -> str:
        """
        Generate a comprehensive text report comparing both model approaches.
        
        Returns:
            Formatted report string
        """
        if not self.drift_results:
            return "No analysis results available"
        
        report = []
        report.append("COMPREHENSIVE SEMANTIC DRIFT ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Analysis Date: {self.drift_results['analysis_date']}")
        report.append(f"Full Corpus Size: {self.drift_results['corpus_size']:,} posts")
        report.append(f"Model Comparison: {self.drift_results['model_comparison']}")
        
        # Model parameters
        report.append("\nModel Parameters:")
        for param, value in self.drift_results['model_params'].items():
            report.append(f"  {param}: {value}")
        
        # Vocabulary statistics comparison
        report.append("\nVocabulary Statistics Comparison:")
        for model_type, stats in self.drift_results['vocabulary_stats'].items():
            report.append(f"  {model_type.replace('_', ' ').title()} Model:")
            for stat, value in stats.items():
                if isinstance(value, (int, float)):
                    report.append(f"    {stat}: {value:,}")
                else:
                    report.append(f"    {stat}: {value}")
        
        # Results comparison
        report.append("\nMODEL COMPARISON RESULTS:")
        report.append("-" * 40)
        
        for model_name, model_results in self.drift_results['results'].items():
            report.append(f"\n{model_name.replace('_', ' ').upper()} MODEL:")
            report.append(f"  Training Method: {model_results['model_type']}")
            report.append(f"  Total Candidates: {model_results['total_candidates']:,}")
            
            if model_results['top_terms']:
                report.append(f"  Top 10 High-Drift Terms:")
                for i, term_data in enumerate(model_results['top_terms'][:10], 1):
                    report.append(f"\n    {i:2d}. {term_data['term']}")
                    report.append(f"        Drift Score: {term_data['drift_score']:.4f}")
                    report.append(f"        Domain Frequency: {term_data['domain_freq']:,}")
                    if term_data['examples']:
                        report.append("        Example Usage:")
                        for example in term_data['examples'][:1]:  # Just one example for brevity
                            ex_text = example[:80] + "..." if len(example) > 80 else example
                            report.append(f"          \"{ex_text}\"")
        
        # Model comparison insights
        report.append("\nMODEL COMPARISON INSIGHTS:")
        report.append("-" * 30)
        
        if len(self.drift_results['results']) >= 2:
            models = list(self.drift_results['results'].keys())
            model1_terms = set(term['term'] for term in self.drift_results['results'][models[0]]['top_terms'][:20])
            model2_terms = set(term['term'] for term in self.drift_results['results'][models[1]]['top_terms'][:20])
            
            overlap = model1_terms.intersection(model2_terms)
            unique_to_model1 = model1_terms - model2_terms
            unique_to_model2 = model2_terms - model1_terms
            
            report.append(f"  Terms in both top-20 lists: {len(overlap)}")
            report.append(f"  Terms unique to {models[0]}: {len(unique_to_model1)}")
            report.append(f"  Terms unique to {models[1]}: {len(unique_to_model2)}")
            
            if overlap:
                report.append(f"\n  Overlapping high-drift terms:")
                for term in list(overlap)[:5]:
                    report.append(f"    - {term}")
        
        return "\n".join(report)
    
    def visualize_model_comparison(self, save_path: Optional[str] = None):
        """
        Create visualizations comparing both model approaches.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.drift_results or not self.drift_results.get('results'):
            logger.warning("No results to visualize")
            return
        
        # Create subplots for comparison
        n_models = len(self.drift_results['results'])
        fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        model_names = list(self.drift_results['results'].keys())
        
        for i, (model_name, model_results) in enumerate(self.drift_results['results'].items()):
            if not model_results['top_terms']:
                continue
                
            # Extract data
            terms = [item['term'] for item in model_results['top_terms']]
            scores = [item['drift_score'] for item in model_results['top_terms']]
            frequencies = [item['domain_freq'] for item in model_results['top_terms']]
            
            # Top terms by drift score
            top_15_terms = terms[:15]
            top_15_scores = scores[:15]
            
            axes[0, i].barh(range(len(top_15_terms)), top_15_scores, alpha=0.7)
            axes[0, i].set_yticks(range(len(top_15_terms)))
            axes[0, i].set_yticklabels(top_15_terms, fontsize=8)
            axes[0, i].set_xlabel('Semantic Drift Score')
            axes[0, i].set_title(f'Top Terms - {model_name.replace("_", " ").title()}')
            axes[0, i].invert_yaxis()
            
            # Drift score vs frequency scatter
            axes[1, i].scatter(frequencies, scores, alpha=0.6)
            axes[1, i].set_xlabel('Domain Frequency (log scale)')
            axes[1, i].set_ylabel('Semantic Drift Score')
            axes[1, i].set_title(f'Drift vs Frequency - {model_name.replace("_", " ").title()}')
            axes[1, i].set_xscale('log')
            
            # Add trend line
            if len(frequencies) > 1:
                z = np.polyfit(np.log(frequencies), scores, 1)
                p = np.poly1d(z)
                x_trend = np.logspace(np.log10(min(frequencies)), np.log10(max(frequencies)), 100)
                axes[1, i].plot(x_trend, p(np.log(x_trend)), "r--", alpha=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison visualization saved to {save_path}")
        
        plt.show()
    

def main():
    """
    Main function for running FastText semantic drift analysis on FULL CORPUS.
    """
    import sys
    from pathlib import Path
    
    # Add src to path for imports
    sys.path.append(str(Path(__file__).parent))
    
    # Initialize detector with optimized parameters for full corpus
    detector = FastTextSemanticDriftDetector(
        vector_size=300,
        window=5,
        min_count=5,  # Lower threshold for full corpus
        epochs=10,
        min_freq_threshold=10,  # Lower threshold for full corpus
        workers=4
    )
    
    try:
        # Load FULL data - no sampling!
        data_path = Path("../data/standardized/all_platforms_combined.csv")
        if data_path.exists():
            logger.info(f"Loading FULL CORPUS from {data_path}")
            df = pd.read_csv(data_path)
            
            logger.info(f"Loaded {len(df)} posts from all platforms")
            logger.info(f"Platform distribution: {df['platform'].value_counts().to_dict()}")
            
            # Run comprehensive analysis comparing both model types
            results = detector.analyze_semantic_drift(
                df, 
                text_column='text', 
                top_n=100,  # More terms for full analysis
                filter_pos=False,  # Can enable later
                compare_models=True
            )
            
            # Save comprehensive results
            detector.save_results("../results/semantic_drift_full_corpus.jsonl")
            
            # Generate and save comprehensive report
            report = detector.generate_comprehensive_report()
            print(report)
            
            with open("../results/semantic_drift_full_corpus_report.txt", 'w') as f:
                f.write(report)
            
            # Generate visualizations for both models
            detector.visualize_model_comparison("../results/semantic_drift_comparison.png")
            
            logger.info("FULL CORPUS analysis complete!")
            
        else:
            logger.error(f"Data file not found: {data_path}")
            logger.info("Please ensure the standardized data exists")
    
    except Exception as e:
        logger.error(f"FULL CORPUS analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
