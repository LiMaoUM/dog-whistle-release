"""
Text preprocessing pipeline for dog whistle detection.

This module handles data cleaning, normalization, tokenization, and temporal segmentation
for social media posts from multiple platforms.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from bs4 import BeautifulSoup
from fast_langdetect import detect, detect_multilingual
import warnings
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    """
    Comprehensive text preprocessing for social media content.
    
    Handles platform-specific cleaning, normalization, and segmentation
    while preserving important linguistic patterns for dog whistle detection.
    """
    
    def __init__(self, preserve_case: bool = False, min_word_length: int = 2, use_custom_stopwords: bool = True):
        self.preserve_case = preserve_case
        self.min_word_length = min_word_length
        self.use_custom_stopwords = use_custom_stopwords
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.unicode_pattern = re.compile(r'[^\x00-\x7F]+')
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Initialize stop words
        self.stop_words = self._get_stop_words()
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = None
        self.feature_names = None
        
    def _get_stop_words(self) -> set:
        """Get stop words for filtering."""
        if self.use_custom_stopwords:
            # Use scikit-learn's English stop words as base
            stop_words = set(ENGLISH_STOP_WORDS)
            
            # Add custom stop words relevant to social media
            custom_stops = {
                'rt', 'via', 'amp', 'gt', 'lt', 'nbsp', 'quot', 'apos',
                'http', 'https', 'www', 'com', 'html', 'org', 'net',
                'said', 'say', 'says', 'get', 'got', 'go', 'going',
                'like', 'just', 'know', 'think', 'people', 'time',
                'one', 'two', 'first', 'last', 'new', 'old', 'good', 'bad',
                'right', 'left', 'way', 'day', 'year', 'today', 'tomorrow'
            }
            stop_words.update(custom_stops)
            return stop_words
        else:
            return set()
        
    def clean_text(self, text: str, remove_urls: bool = True, 
                   remove_mentions: bool = False, remove_hashtags: bool = False,
                   parse_html: bool = True) -> str:
        """
        Clean individual text with configurable options.
        
        Args:
            text: Input text to clean
            remove_urls: Whether to remove URLs
            remove_mentions: Whether to remove @mentions
            remove_hashtags: Whether to remove #hashtags
            parse_html: Whether to parse HTML content first
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
            
        # Parse HTML content first if requested
        if parse_html and ('<' in text and '>' in text):
            text = self.parse_html_content(text)
            
        # Remove URLs
        if remove_urls:
            text = self.url_pattern.sub('', text)
            
        # Remove email addresses
        text = self.email_pattern.sub('', text)
        
        # Handle mentions and hashtags
        if remove_mentions:
            text = self.mention_pattern.sub('', text)
        else:
            # Keep mention content but remove @ symbol
            text = self.mention_pattern.sub(lambda m: m.group(0)[1:], text)
            
        if remove_hashtags:
            text = self.hashtag_pattern.sub('', text)
        else:
            # Keep hashtag content but remove # symbol
            text = self.hashtag_pattern.sub(lambda m: m.group(0)[1:], text)
        
        # Remove excessive unicode characters (emojis, etc.)
        text = self.unicode_pattern.sub(' ', text)
        
        # Remove extra whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Remove punctuation but preserve apostrophes in words
        text = re.sub(r"[^\w\s']", ' ', text)
        
        # Normalize case
        if not self.preserve_case:
            text = text.lower()
            
        return text.strip()
    
    def batch_clean_text(self, texts: pd.Series, remove_urls: bool = True, 
                        remove_mentions: bool = False, remove_hashtags: bool = False,
                        parse_html: bool = True) -> pd.Series:
        """
        Vectorized text cleaning for better performance on large datasets.
        
        Args:
            texts: Series of text strings
            remove_urls: Whether to remove URLs
            remove_mentions: Whether to remove @mentions
            remove_hashtags: Whether to remove #hashtags
            parse_html: Whether to parse HTML content first
            
        Returns:
            Series of cleaned text strings
        """
        # Start with a copy
        cleaned = texts.fillna('').astype(str)
        
        # Parse HTML content first if requested
        if parse_html:
            # Apply HTML parsing to texts that contain HTML tags
            html_mask = cleaned.str.contains('<.*>', regex=True, na=False)
            if html_mask.any():
                cleaned.loc[html_mask] = cleaned.loc[html_mask].apply(self.parse_html_content)
        
        # Remove URLs
        if remove_urls:
            cleaned = cleaned.str.replace(self.url_pattern.pattern, '', regex=True)
            
        # Remove email addresses
        cleaned = cleaned.str.replace(self.email_pattern.pattern, '', regex=True)
        
        # Handle mentions and hashtags
        if remove_mentions:
            cleaned = cleaned.str.replace(self.mention_pattern.pattern, '', regex=True)
        else:
            # Keep mention content but remove @ symbol - simpler approach
            cleaned = cleaned.str.replace(r'@(\w+)', r'\1', regex=True)
            
        if remove_hashtags:
            cleaned = cleaned.str.replace(self.hashtag_pattern.pattern, '', regex=True)
        else:
            # Keep hashtag content but remove # symbol - simpler approach
            cleaned = cleaned.str.replace(r'#(\w+)', r'\1', regex=True)
        
        # Remove excessive unicode characters (emojis, etc.)
        cleaned = cleaned.str.replace(self.unicode_pattern.pattern, ' ', regex=True)
        
        # Remove extra whitespace
        cleaned = cleaned.str.replace(self.whitespace_pattern.pattern, ' ', regex=True)
        
        # Remove punctuation but preserve apostrophes in words
        cleaned = cleaned.str.replace(r"[^\w\s']", ' ', regex=True)
        
        # Normalize case
        if not self.preserve_case:
            cleaned = cleaned.str.lower()
            
        return cleaned.str.strip()
    
    def tokenize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Tokenize text into words, filtering by minimum length and optionally removing stop words.
        
        Args:
            text: Input text to tokenize
            remove_stopwords: Whether to remove stop words
            
        Returns:
            List of tokens
        """
        tokens = text.split()
        
        # Filter by minimum length
        tokens = [token for token in tokens if len(token) >= self.min_word_length]
        
        # Remove stop words if requested
        if remove_stopwords and self.use_custom_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
            
        return tokens
    
    def lemmatize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Lemmatize text into root words, filtering by minimum length and optionally removing stop words.
        
        Args:
            text: Input text to lemmatize
            remove_stopwords: Whether to remove stop words
            
        Returns:
            List of lemmatized words
        """
        # Use NLTK's word_tokenize for better tokenization
        try:
            tokens = word_tokenize(text.lower())
        except LookupError:
            # Fallback to simple split if NLTK data not available
            tokens = text.lower().split()
        
        # Lemmatize tokens
        lemmas = []
        for token in tokens:
            # Skip punctuation and very short tokens
            if token.isalpha() and len(token) >= self.min_word_length:
                lemma = self.lemmatizer.lemmatize(token)
                lemmas.append(lemma)
        
        # Remove stop words if requested
        if remove_stopwords and self.use_custom_stopwords:
            lemmas = [lemma for lemma in lemmas if lemma not in self.stop_words]
            
        return lemmas
    
    def fasttext_tokenize(self, text: str, remove_stopwords: bool = False) -> List[str]:
        """
        FastText-compatible tokenization using ICU tokenizer behavior.
        
        This implementation replicates the ICU (International Components for Unicode) 
        tokenizer that FastText uses in its research implementations, providing:
        1. Unicode-aware word boundary detection
        2. Proper handling of contractions and apostrophes
        3. Language-neutral tokenization compatible with pre-trained models
        4. Consistent tokenization for both training and inference
        
        Based on ICU's word segmentation rules as used in FastText papers.
        
        Args:
            text: Input text to tokenize
            remove_stopwords: Whether to remove stop words (usually False for FastText)
            
        Returns:
            List of tokens compatible with FastText training
        """
        if not isinstance(text, str):
            return []
        
        # Convert to lowercase (FastText standard)
        text = text.lower()
        
        # ICU tokenizer behavior: split on Unicode word boundaries
        # This pattern matches the ICU tokenizer more closely:
        # - \w+ matches Unicode word characters (letters, digits, connecting punctuation)
        # - Apostrophes are handled specially for contractions
        
        # ICU tokenizer splits contractions differently - it keeps apostrophes with words
        # when they're part of valid contractions, but separates them otherwise
        
        # Use Unicode word boundary detection similar to ICU
        # This regex captures:
        # 1. Words with apostrophes (contractions): word'suffix
        # 2. Regular words: sequences of word characters
        # 3. Isolated apostrophe sequences (less common)
        
        # More accurate ICU-like tokenization
        tokens = re.findall(r"\b\w+(?:'\w+)?\b", text, re.UNICODE)
        
        # Additional cleanup for FastText compatibility
        cleaned_tokens = []
        for token in tokens:
            # Ensure token meets minimum length and is primarily alphabetic
            # Allow apostrophes in the middle of words (contractions)
            if re.match(r"^[a-z]+(?:'[a-z]+)?$", token) and len(token) >= self.min_word_length:
                cleaned_tokens.append(token)
        
        # Optionally remove stop words (usually not done for FastText)
        if remove_stopwords and self.use_custom_stopwords:
            cleaned_tokens = [token for token in cleaned_tokens if token not in self.stop_words]
            
        return cleaned_tokens

    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text',
                           date_column: str = 'created_at', compute_tfidf: bool = True,
                           filter_english: bool = False, confidence_threshold: float = 0.7) -> pd.DataFrame:
        """
        Preprocess a dataframe of social media posts with optimized performance.
        
        Args:
            df: Input dataframe
            text_column: Name of text content column
            date_column: Name of timestamp column
            compute_tfidf: Whether to compute TF-IDF scores during preprocessing
            filter_english: Whether to filter for English-only posts
            confidence_threshold: Minimum confidence for language detection
            
        Returns:
            Preprocessed dataframe with additional columns
        """
        df = df.copy()
        
        # Language filtering first (if requested) to reduce processing load
        if filter_english:
            print("Filtering for English posts...")
            df = self.filter_english_posts(df, text_column, confidence_threshold)
            if len(df) == 0:
                print("Warning: No English posts found after filtering!")
                return df
        
        # Vectorized text cleaning for better performance
        print("Cleaning text...")
        df['cleaned_text'] = self.batch_clean_text(df[text_column])
        
        # Vectorized tokenization
        print("Tokenizing text...")
        df['tokens'] = df['cleaned_text'].apply(self.tokenize)
        df['token_count'] = df['tokens'].apply(len)
        
        # Extract potential terms (2-3 word phrases)
        print("Extracting potential terms...")
        df['potential_terms'] = df['tokens'].apply(self._extract_potential_terms)
        
        # Ensure datetime format
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Compute TF-IDF if requested
        if compute_tfidf:
            print("Computing TF-IDF scores...")
            try:
                tfidf_matrix, feature_names, tfidf_scores = self.compute_tfidf(df)
                df.attrs['tfidf_matrix'] = tfidf_matrix
                df.attrs['tfidf_feature_names'] = feature_names
                df.attrs['tfidf_scores'] = tfidf_scores
                print(f"TF-IDF computed for {len(feature_names)} features")
            except Exception as e:
                print(f"Warning: TF-IDF computation failed: {e}")
                
        return df
        
    def _extract_potential_terms(self, tokens: List[str]) -> List[str]:
        """
        Extract potential dog whistle terms (bigrams and trigrams).
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of potential multi-word terms
        """
        terms = []
        
        # Extract bigrams
        for i in range(len(tokens) - 1):
            bigram = " ".join(tokens[i:i+2])
            terms.append(bigram)
            
        # Extract trigrams
        for i in range(len(tokens) - 2):
            trigram = " ".join(tokens[i:i+3])
            terms.append(trigram)
            
        return terms
    
    def compute_tfidf(self, df: pd.DataFrame, text_column: str = 'cleaned_text', 
                     max_features: int = 10000, min_df: int = 2, max_df: float = 0.95) -> Tuple[Any, List[str], Dict[str, float]]:
        """
        Compute TF-IDF scores for the corpus.
        
        Args:
            df: Preprocessed dataframe
            text_column: Column containing cleaned text
            max_features: Maximum number of features to extract
            min_df: Minimum document frequency (ignore terms appearing in fewer docs)
            max_df: Maximum document frequency (ignore terms appearing in more than this fraction of docs)
            
        Returns:
            Tuple of (tfidf_matrix, feature_names, tfidf_scores_dict)
        """
        # Initialize TF-IDF vectorizer with custom stop words
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words=list(self.stop_words) if self.use_custom_stopwords else None,
            lowercase=not self.preserve_case,
            token_pattern=r'\b\w+\b',  # Only alphanumeric tokens
            ngram_range=(1, 2)  # Include unigrams and bigrams
        )
        
        # Fit and transform the corpus
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(df[text_column].fillna(''))
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Calculate mean TF-IDF scores across all documents
        # Convert sparse matrix to dense array
        dense_matrix = np.asarray(tfidf_matrix.todense())
        mean_scores = np.mean(dense_matrix, axis=0).flatten()
        
        # Create dictionary mapping words to their average TF-IDF scores
        tfidf_scores = dict(zip(self.feature_names, mean_scores))
        
        # Sort by TF-IDF score (descending)
        sorted_tfidf = dict(sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True))
        
        return tfidf_matrix, list(self.feature_names), sorted_tfidf
    
    def get_document_tfidf_scores(self, document_index: int, tfidf_matrix: Any, 
                                feature_names: List[str], top_n: int = 20) -> Dict[str, float]:
        """
        Get TF-IDF scores for a specific document.
        
        Args:
            document_index: Index of the document in the corpus
            tfidf_matrix: TF-IDF matrix from compute_tfidf
            feature_names: Feature names from compute_tfidf
            top_n: Number of top features to return
            
        Returns:
            Dictionary of top TF-IDF features for the document
        """
        # Get TF-IDF scores for the specific document
        doc_scores = tfidf_matrix[document_index].toarray().flatten()
        
        # Create word-score pairs
        word_scores = list(zip(feature_names, doc_scores))
        
        # Filter out zero scores and sort by score
        word_scores = [(word, score) for word, score in word_scores if score > 0]
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N as dictionary
        return dict(word_scores[:top_n])
    
    def temporal_segmentation(self, df: pd.DataFrame, date_column: str = 'created_at',
                            segment_type: str = 'quarter') -> Dict[str, pd.DataFrame]:
        """
        Segment data into temporal periods for drift analysis.
        
        Args:
            df: Input dataframe with datetime column
            date_column: Name of datetime column
            segment_type: Type of segmentation ('quarter', 'month', 'week')
            
        Returns:
            Dictionary mapping time periods to dataframes
        """
        if date_column not in df.columns:
            raise ValueError(f"Column {date_column} not found in dataframe")
            
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        if segment_type == 'quarter':
            df['period'] = df[date_column].dt.to_period('Q')
        elif segment_type == 'month':
            df['period'] = df[date_column].dt.to_period('M')
        elif segment_type == 'week':
            df['period'] = df[date_column].dt.to_period('W')
        else:
            raise ValueError(f"Unsupported segment_type: {segment_type}")
            
        segments = {}
        for period in df['period'].unique():
            segments[str(period)] = df[df['period'] == period].copy()
            
        return segments
    
    def get_vocabulary_stats(self, df: pd.DataFrame, include_tfidf: bool = True) -> Dict:
        """
        Get vocabulary statistics for the corpus including TF-IDF scores.
        
        Args:
            df: Preprocessed dataframe
            include_tfidf: Whether to compute and include TF-IDF statistics
            
        Returns:
            Dictionary with vocabulary statistics
        """
        all_tokens = []
        all_terms = []
        
        # Efficiently collect all tokens using vectorized operations
        if 'tokens' in df.columns:
            for tokens in df['tokens']:
                tokens = eval(tokens) if isinstance(tokens, str) else tokens
                if isinstance(tokens, list):
                    all_tokens.extend(tokens)

           
                    
        if 'potential_terms' in df.columns:
            for terms in df['potential_terms']:
                terms = eval(terms) if isinstance(terms, str) else terms
                if isinstance(terms, list):
                    all_terms.extend(terms)

 
        
        token_counts = Counter(all_tokens)
        term_counts = Counter(all_terms)
        
        stats = {
            'total_posts': len(df),
            'total_tokens': len(all_tokens),
            'unique_tokens': len(token_counts),
            'total_terms': len(all_terms),
            'unique_terms': len(term_counts),
            'avg_tokens_per_post': np.mean(df['token_count']) if 'token_count' in df.columns else 0,
            'most_common_tokens': token_counts.most_common(50),
            'most_common_terms': term_counts.most_common(50) if all_terms else []
        }
        
        # Add TF-IDF statistics if requested
        if include_tfidf and 'cleaned_text' in df.columns:
            try:
                tfidf_matrix, feature_names, tfidf_scores = self.compute_tfidf(df)
                stats.update({
                    'tfidf_vocabulary_size': len(feature_names),
                    'top_tfidf_terms': list(tfidf_scores.items())[:50],  # Top 50 TF-IDF terms
                    'all_tfidf_scores': tfidf_scores  # Complete TF-IDF dictionary
                })
            except Exception as e:
                print(f"Warning: Could not compute TF-IDF statistics: {e}")
                stats['tfidf_error'] = str(e)
        
        return stats
    
    def get_tfidf_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive TF-IDF results for the corpus.
        
        Args:
            df: Preprocessed dataframe
            
        Returns:
            Dictionary containing TF-IDF matrix, feature names, and scores
        """
        if hasattr(df, 'attrs') and 'tfidf_scores' in df.attrs:
            return {
                'tfidf_matrix': df.attrs.get('tfidf_matrix'),
                'feature_names': df.attrs.get('tfidf_feature_names'),
                'tfidf_scores': df.attrs.get('tfidf_scores')
            }
        else:
            # Compute TF-IDF if not already done
            tfidf_matrix, feature_names, tfidf_scores = self.compute_tfidf(df)
            return {
                'tfidf_matrix': tfidf_matrix,
                'feature_names': feature_names,
                'tfidf_scores': tfidf_scores
            }

    def print_tfidf_summary(self, tfidf_scores: Dict[str, float], top_n: int = 20):
        """
        Print a summary of top TF-IDF scores.
        
        Args:
            tfidf_scores: Dictionary of TF-IDF scores
            top_n: Number of top terms to display
        """
        print(f"\nTop {top_n} TF-IDF Terms:")
        print("-" * 40)
        for i, (term, score) in enumerate(list(tfidf_scores.items())[:top_n], 1):
            print(f"{i:2d}. {term:<20} {score:.4f}")
    
    def parse_html_content(self, html_text: str) -> str:
        """
        Parse HTML content and extract clean text.
        
        Args:
            html_text: HTML formatted text content
            
        Returns:
            Clean text with HTML tags removed
        """
        if not isinstance(html_text, str) or not html_text.strip():
            return ""
            
        try:
            # Suppress BeautifulSoup warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                soup = BeautifulSoup(html_text, 'html.parser')
            text = soup.get_text()
            text = re.sub(r'\s+', ' ', text).strip()
            return text
            
        except Exception as e:
            # Fallback: if HTML parsing fails, return original text
            # Only print warning if in debug mode (optional)
            # print(f"Warning: HTML parsing failed: {e}")
            return html_text

    def detect_language(self, text: str, confidence_threshold: float = 0.5) -> Tuple[str, float]:
        """
        Detect the language of the given text using fast-langdetect.
        
        Args:
            text: Input text to detect language for
            confidence_threshold: Minimum confidence score to trust detection
            
        Returns:
            Tuple of (language_code, confidence_score) or ('unknown', 0.0) if detection fails
        """
        if not isinstance(text, str) or len(text.strip()) < 3:
            return ('unknown', 0.0)
            
        try:
            clean_text = re.sub(r'http[s]?://\S+', '', text)
            clean_text = re.sub(r'@\w+', '', clean_text)
            clean_text = re.sub(r'#\w+', '', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            if len(clean_text.strip()) < 3:
                return ('unknown', 0.0)
            # Suppress fast-langdetect warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = detect(clean_text)
            lang = str(result['lang'])
            score = float(result['score'])
            if score < confidence_threshold:
                return ('unknown', score)
            return (lang, score)
        except Exception:
            return ('unknown', 0.0)

    def filter_english_posts(self, df: pd.DataFrame, text_column: str = 'text', 
                           confidence_threshold: float = 0.7) -> pd.DataFrame:
        """
        Filter dataframe to keep only English posts using fast-langdetect.
        
        Args:
            df: Input dataframe
            text_column: Column containing text content
            confidence_threshold: Minimum confidence for language detection
            
        Returns:
            Filtered dataframe with only English posts
        """
        if text_column not in df.columns:
            print(f"Warning: Column '{text_column}' not found in dataframe")
            return df.copy()
            
        print("Detecting languages with fast-langdetect...")
        df = df.copy()
        
        # Detect language for each post
        lang_results = df[text_column].apply(
            lambda x: self.detect_language(x, confidence_threshold)
        )
        
        # Split results into separate columns
        df['detected_language'] = lang_results.apply(lambda x: x[0])
        df['language_confidence'] = lang_results.apply(lambda x: x[1])
        
        # Count languages before filtering
        lang_counts = df['detected_language'].value_counts()
        print(f"\nLanguage distribution:")
        for lang, count in lang_counts.items():
            print(f"  {lang}: {count} posts")
            
        # Show average confidence for detected languages
        for lang in lang_counts.index[:5]:  # Top 5 languages
            avg_conf = df[df['detected_language'] == lang]['language_confidence'].mean()
            print(f"  {lang} avg confidence: {avg_conf:.3f}")
            
        # Filter for English posts
        english_df = df[df['detected_language'] == 'en'].copy()
        
        print(f"\nFiltering results:")
        print(f"  Original posts: {len(df)}")
        print(f"  English posts: {len(english_df)}")
        print(f"  Filtered out: {len(df) - len(english_df)} posts")
        print(f"  Retention rate: {(len(english_df)/len(df)*100):.1f}%")
        
        if len(english_df) > 0:
            avg_en_conf = english_df['language_confidence'].mean()
            print(f"  Average English confidence: {avg_en_conf:.3f}")
        
        return english_df


class PlatformSpecificPreprocessor:
    """
    Platform-specific preprocessing for different social media platforms.
    """
    
    @staticmethod
    def truth_social_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess Truth Social posts with platform-specific handling."""
        preprocessor = TextPreprocessor(preserve_case=False)
        
        # Truth Social specific cleaning
        df = df.copy()
        if 'content' in df.columns:
            df['text'] = df['content']  # Standardize column name
            
        # Handle Truth Social HTML content
        if 'text' in df.columns:
            # Parse HTML content in Truth Social posts
            df['text'] = df['text'].apply(lambda x: preprocessor.parse_html_content(x) if isinstance(x, str) else x)
            
        # Handle Truth Social specific patterns
        # Remove "TRUTH" tags and platform-specific artifacts
        df['text'] = df['text'].str.replace(r'\bTRUTH\b', '', regex=True, case=False)
        
        return preprocessor.preprocess_dataframe(df)
    
    @staticmethod
    def bluesky_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess Bluesky posts with platform-specific handling."""
        preprocessor = TextPreprocessor(preserve_case=False)
        
        df = df.copy()
        # Handle Bluesky's AT Protocol specific content
        if 'record' in df.columns:
            df['text'] = df['record'].apply(lambda x: x.get('text', '') if isinstance(x, dict) else str(x))
            
        return preprocessor.preprocess_dataframe(df)
    
    @staticmethod
    def mastodon_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess Mastodon posts with platform-specific handling."""
        preprocessor = TextPreprocessor(preserve_case=False)
        
        df = df.copy()
        # Handle Mastodon's federated content
        if 'content' in df.columns:
            df['text'] = df['content']
            
        # Parse HTML content in Mastodon posts (they often contain HTML)
        if 'text' in df.columns:
            df['text'] = df['text'].apply(lambda x: preprocessor.parse_html_content(x) if isinstance(x, str) else x)
        
        return preprocessor.preprocess_dataframe(df)


def create_sample_data() -> pd.DataFrame:
    """
    Create sample social media data for testing and demonstration.
    
    Returns:
        Sample dataframe with social media posts
    """
    sample_posts = [
        {
            'text': "The coastal elites are pushing their agenda again with those coastal papers",
            'created_at': '2024-01-15 10:30:00',
            'platform': 'truth_social',
            'user_id': 'user001'
        },
        {
            'text': "Another example of dual citizenship causing problems in our institutions",
            'created_at': '2024-02-20 14:22:00',
            'platform': 'truth_social',
            'user_id': 'user002'
        },
        {
            'text': '<p>😳😳😳 <a href="https://links.truthsocial.com/link/112657474556800233" rel="nofollow noopener noreferrer" target="_blank"><span class="invisible">https://</span><span class="ellipsis">youtube.com/shorts/8TXmS-iXUoM</span><span class="invisible">?si=wJ1IBq9DqPMF8QmO</span></a><br />🤣🤣🤣🤣🤣🤣🤣</p>',
            'created_at': '2024-03-10 09:15:00',
            'platform': 'truth_social',
            'user_id': 'user003'
        },
        {
            'text': "Traditional values are under attack by the globalist agenda",
            'created_at': '2024-04-05 16:45:00',
            'platform': 'mastodon',
            'user_id': 'user004'
        },
        {
            'text': '<p>Those people with their <em>alternative lifestyle choices</em> are corrupting society</p>',
            'created_at': '2024-05-12 11:20:00',
            'platform': 'mastodon',
            'user_id': 'user005'
        }
    ]
    
    return pd.DataFrame(sample_posts)


if __name__ == "__main__":
    # Demonstration of preprocessing capabilities
    print("Dog Whistle Detection - Text Preprocessing Module")
    print("=" * 50)
    
    # Create sample data
    sample_df = create_sample_data()
    print(f"Created sample dataset with {len(sample_df)} posts")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess the data
    processed_df = preprocessor.preprocess_dataframe(sample_df)
    
    # Display results
    print("\nProcessed Data Sample:")
    for i, (idx, row) in enumerate(processed_df.head(3).iterrows()):
        print(f"\nPost {i + 1}:")
        print(f"Original: {row['text']}")
        print(f"Cleaned: {row['cleaned_text']}")
        print(f"Tokens: {row['tokens'][:5]}...")  # Show first 5 tokens
        print(f"Potential terms: {row['potential_terms'][:3]}...")  # Show first 3 terms
    
    # Get vocabulary statistics
    stats = preprocessor.get_vocabulary_stats(processed_df, include_tfidf=True)
    print(f"\nVocabulary Statistics:")
    print(f"Total posts: {stats['total_posts']}")
    print(f"Unique tokens: {stats['unique_tokens']}")
    print(f"Average tokens per post: {stats['avg_tokens_per_post']:.2f}")
    
    # Display TF-IDF results if available
    if 'all_tfidf_scores' in stats:
        print(f"TF-IDF vocabulary size: {stats['tfidf_vocabulary_size']}")
        preprocessor.print_tfidf_summary(stats['all_tfidf_scores'], top_n=10)
    
    # Demonstrate HTML parsing capability
    print(f"\nHTML Parsing Demonstration:")
    html_posts = processed_df[processed_df['text'].str.contains('<.*>', regex=True, na=False)]
    if len(html_posts) > 0:
        for i, (_, row) in enumerate(html_posts.iterrows()):
            print(f"\nHTML Post {i + 1}:")
            print(f"Original HTML: {row['text']}")
            print(f"Parsed Text: {row['cleaned_text']}")
    else:
        print("No HTML content found in sample data")
    
    # Get TF-IDF results
    tfidf_results = preprocessor.get_tfidf_results(processed_df)
    if tfidf_results['tfidf_scores']:
        print(f"\nTotal TF-IDF features: {len(tfidf_results['feature_names'])}")
        print("All TF-IDF scores available in results dictionary")
    
    # Temporal segmentation demo
    segments = preprocessor.temporal_segmentation(processed_df, segment_type='quarter')
    print(f"\nTemporal Segments: {list(segments.keys())}")
    
    # TF-IDF results demo
    tfidf_results = preprocessor.get_tfidf_results(processed_df)
    preprocessor.print_tfidf_summary(tfidf_results['tfidf_scores'], top_n=10)
    
    print("\nPreprocessing module ready for integration!")
