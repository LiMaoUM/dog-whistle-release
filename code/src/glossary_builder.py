"""
Glossary building and term management for dog whistle detection.

This module manages the construction and maintenance of a dynamic glossary
of confirmed dog whistle terms with their metadata, targets, and examples.
Now includes integration with Allen AI Dog Whistle Glossary data.
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict, Counter
import re
from pathlib import Path

from .llm_vetting import TermAnalysis, ClassificationResult
from .allen_ai_loader import AllenAIGlossaryLoader, DogWhistleEntry, load_allen_ai_glossary


@dataclass
class GlossaryEntry:
    """Data class for individual glossary entries."""
    term: str
    category: str  # e.g., 'racial', 'religious', 'political', 'gender'
    target_group: str
    definition: str
    confidence_score: float
    platforms: List[str]
    first_detected: str
    last_updated: str
    example_posts: List[str]
    related_terms: List[str]
    usage_frequency: int
    geographic_distribution: Dict[str, int]
    temporal_pattern: Dict[str, float]  # time period -> usage intensity
    validation_status: str  # 'confirmed', 'suspected', 'disputed'
    sources: List[str]  # research sources, reports, etc.


class DogWhistleGlossary:
    """
    Dynamic glossary system for dog whistle terms.
    
    Manages the lifecycle of term detection, validation, and categorization
    with support for updates, version control, and export formats.
    """
    
    def __init__(self, glossary_path: Optional[str] = None):
        self.glossary_path = glossary_path or "results/emergent_dogwhistles.json"
        self.entries: Dict[str, GlossaryEntry] = {}
        self.categories = self._initialize_categories()
        self.load_glossary()
    
    def _initialize_categories(self) -> Dict[str, str]:
        """Initialize standard categories for dog whistle terms."""
        return {
            'racial': 'Terms targeting racial or ethnic groups',
            'religious': 'Terms targeting religious groups',
            'political': 'Terms targeting political groups or ideologies',
            'gender': 'Terms targeting gender or sexual identity',
            'economic': 'Terms targeting economic classes',
            'geographic': 'Terms targeting geographic regions or populations',
            'immigration': 'Terms targeting immigrants or immigration',
            'cultural': 'Terms targeting cultural practices or values',
            'antisemitic': 'Terms with antisemitic implications',
            'islamophobic': 'Terms with anti-Muslim implications',
            'other': 'Terms not fitting other categories'
        }
    
    def load_glossary(self) -> None:
        """Load existing glossary from file if it exists."""
        try:
            if Path(self.glossary_path).exists():
                with open(self.glossary_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for entry_data in data:
                    entry = GlossaryEntry(**entry_data)
                    self.entries[entry.term] = entry
                    
                print(f"Loaded {len(self.entries)} entries from existing glossary")
        except Exception as e:
            print(f"Could not load existing glossary: {e}")
            self.entries = {}
    
    def load_allen_ai_data(self, force_refresh: bool = False) -> Tuple[List[DogWhistleEntry], pd.DataFrame]:
        """
        Load Allen AI Dog Whistle Glossary data.
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            Tuple of (allen_ai_entries, dataframe)
        """
        print("Loading Allen AI Dog Whistle Glossary data...")
        try:
            allen_ai_entries, df = load_allen_ai_glossary(force_refresh=force_refresh)
            print(f"✓ Loaded {len(allen_ai_entries)} Allen AI glossary entries")
            return allen_ai_entries, df
        except Exception as e:
            print(f"❌ Failed to load Allen AI data: {e}")
            return [], pd.DataFrame()
    
    def integrate_allen_ai_entries(self, allen_ai_entries: List[DogWhistleEntry], 
                                  overwrite_existing: bool = False) -> int:
        """
        Integrate Allen AI entries into the main glossary.
        
        Args:
            allen_ai_entries: List of Allen AI DogWhistleEntry objects
            overwrite_existing: If True, overwrite existing entries
            
        Returns:
            Number of entries added/updated
        """
        added_count = 0
        
        for allen_entry in allen_ai_entries:
            # Convert Allen AI entry to GlossaryEntry format
            glossary_entry = self._convert_allen_ai_entry(allen_entry)
            
            # Check if entry already exists
            if glossary_entry.term in self.entries:
                if overwrite_existing:
                    print(f"Updating existing entry: {glossary_entry.term}")
                    self.entries[glossary_entry.term] = glossary_entry
                    added_count += 1
                else:
                    print(f"Skipping existing entry: {glossary_entry.term}")
            else:
                self.entries[glossary_entry.term] = glossary_entry
                added_count += 1
        
        print(f"✓ Integrated {added_count} Allen AI entries into glossary")
        return added_count
    
    def _convert_allen_ai_entry(self, allen_entry: DogWhistleEntry) -> GlossaryEntry:
        """
        Convert an Allen AI DogWhistleEntry to a GlossaryEntry.
        
        Args:
            allen_entry: Allen AI DogWhistleEntry object
            
        Returns:
            GlossaryEntry object
        """
        # Map Allen AI persona to our category system
        persona_category_map = {
            'racist': 'racial',
            'antisemitic': 'antisemitic',
            'transphobic': 'gender',
            'homophobic': 'gender',
            'islamophobic': 'islamophobic',
            'nationalist': 'political',
            'white supremacist': 'racial'
        }
        
        category = persona_category_map.get(allen_entry.persona.lower(), 'other')
        
        # Create target group string from targets list
        target_group = '; '.join(allen_entry.targets) if allen_entry.targets else allen_entry.persona
        
        # Set high confidence for Allen AI entries since they're curated
        confidence_score = 0.9
        
        # Current timestamp
        current_time = datetime.now().isoformat()
        
        return GlossaryEntry(
            term=allen_entry.surface_form,
            category=category,
            target_group=target_group,
            definition=f"{allen_entry.description}. Covert meaning: {allen_entry.covert_meaning}",
            confidence_score=confidence_score,
            platforms=['multiple'],  # Allen AI data spans multiple platforms
            first_detected=current_time,
            last_updated=current_time,
            example_posts=allen_entry.examples,
            related_terms=[],  # Will be populated later through analysis
            usage_frequency=len(allen_entry.examples),  # Proxy based on examples
            geographic_distribution={},  # Not available in Allen AI data
            temporal_pattern={},  # Not available in Allen AI data
            validation_status='confirmed',  # Allen AI entries are pre-validated
            sources=['Allen AI Dog Whistle Glossary', 'https://dogwhistles.allen.ai/glossary']
        )
    
    def analyze_allen_ai_coverage(self, allen_ai_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the coverage and characteristics of Allen AI data.
        
        Args:
            allen_ai_df: DataFrame of Allen AI entries
            
        Returns:
            Dictionary with analysis results
        """
        if allen_ai_df.empty:
            return {"error": "No Allen AI data to analyze"}
        
        analysis = {
            'total_entries': len(allen_ai_df),
            'persona_distribution': allen_ai_df['persona'].value_counts().to_dict(),
            'type_distribution': allen_ai_df['type'].value_counts().to_dict(),
            'register_distribution': allen_ai_df['register'].value_counts().to_dict(),
            'entries_with_examples': (allen_ai_df['num_examples'] > 0).sum(),
            'avg_examples_per_entry': allen_ai_df['num_examples'].mean(),
            'entries_with_targets': (allen_ai_df['num_targets'] > 0).sum(),
            'avg_targets_per_entry': allen_ai_df['num_targets'].mean(),
            'most_common_terms': allen_ai_df['surface_form'].head(10).tolist(),
            'coverage_by_category': self._map_allen_ai_to_categories(allen_ai_df)
        }
        
        return analysis
    
    def _map_allen_ai_to_categories(self, allen_ai_df: pd.DataFrame) -> Dict[str, int]:
        """Map Allen AI personas to our category system for coverage analysis."""
        persona_category_map = {
            'racist': 'racial',
            'antisemitic': 'antisemitic',
            'transphobic': 'gender',
            'homophobic': 'gender',
            'islamophobic': 'islamophobic',
            'nationalist': 'political',
            'white supremacist': 'racial'
        }
        
        category_counts = defaultdict(int)
        for persona in allen_ai_df['persona']:
            category = persona_category_map.get(persona.lower(), 'other')
            category_counts[category] += 1
        
        return dict(category_counts)
    
    def compare_with_allen_ai(self, allen_ai_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare current glossary with Allen AI data.
        
        Args:
            allen_ai_df: DataFrame of Allen AI entries
            
        Returns:
            Comparison analysis
        """
        if allen_ai_df.empty:
            return {"error": "No Allen AI data for comparison"}
        
        current_terms = set(self.entries.keys())
        allen_ai_terms = set(allen_ai_df['surface_form'].str.lower())
        current_terms_lower = set(term.lower() for term in current_terms)
        
        overlap = current_terms_lower.intersection(allen_ai_terms)
        only_in_current = current_terms_lower - allen_ai_terms
        only_in_allen_ai = allen_ai_terms - current_terms_lower
        
        comparison = {
            'current_glossary_size': len(current_terms),
            'allen_ai_size': len(allen_ai_terms),
            'overlap_size': len(overlap),
            'overlap_percentage': len(overlap) / len(allen_ai_terms) * 100 if allen_ai_terms else 0,
            'terms_only_in_current': list(only_in_current)[:20],  # First 20
            'terms_only_in_allen_ai': list(only_in_allen_ai)[:20],  # First 20
            'potential_additions': len(only_in_allen_ai),
            'overlapping_terms': list(overlap)[:20]  # First 20
        }
        
        return comparison
    
    def add_term_analysis(self, analysis: TermAnalysis, 
                         platform: str = "unknown",
                         additional_contexts: List[str] = None) -> bool:
        """
        Add a term analysis to the glossary.
        
        Args:
            analysis: TermAnalysis object from LLM vetting
            platform: Source platform
            additional_contexts: Additional example contexts
            
        Returns:
            True if term was added/updated, False otherwise
        """
        # Only add confirmed or likely dog whistles
        if analysis.classification not in [
            ClassificationResult.CONFIRMED_DOG_WHISTLE,
            ClassificationResult.LIKELY_DOG_WHISTLE
        ]:
            return False
        
        term = analysis.term.lower().strip()
        
        # Determine category
        category = self._categorize_term(analysis)
        
        # Determine validation status
        validation_status = (
            "confirmed" if analysis.classification == ClassificationResult.CONFIRMED_DOG_WHISTLE
            else "suspected"
        )
        
        # Prepare contexts
        example_posts = analysis.example_contexts[:]
        if additional_contexts:
            example_posts.extend(additional_contexts)
        
        current_time = datetime.now().isoformat()
        
        if term in self.entries:
            # Update existing entry
            entry = self.entries[term]
            entry.confidence_score = max(entry.confidence_score, analysis.confidence)
            entry.last_updated = current_time
            
            # Add platform if not already present
            if platform not in entry.platforms:
                entry.platforms.append(platform)
            
            # Add new examples (avoid duplicates)
            for post in example_posts:
                if post not in entry.example_posts:
                    entry.example_posts.append(post)
            
            # Update validation status if more confident
            if (validation_status == "confirmed" and 
                entry.validation_status == "suspected"):
                entry.validation_status = validation_status
            
            entry.usage_frequency += 1
            
        else:
            # Create new entry
            entry = GlossaryEntry(
                term=term,
                category=category,
                target_group=analysis.target_group or "Unknown",
                definition=self._generate_definition(analysis),
                confidence_score=analysis.confidence,
                platforms=[platform],
                first_detected=current_time,
                last_updated=current_time,
                example_posts=example_posts[:10],  # Limit examples
                related_terms=[],
                usage_frequency=1,
                geographic_distribution={},
                temporal_pattern={},
                validation_status=validation_status,
                sources=[f"LLM Analysis ({analysis.analysis_timestamp})"]
            )
            
            self.entries[term] = entry
        
        # Find and update related terms
        self._update_related_terms(term)
        
        return True
    
    def _categorize_term(self, analysis: TermAnalysis) -> str:
        """
        Automatically categorize a term based on its analysis.
        
        Args:
            analysis: TermAnalysis object
            
        Returns:
            Category string
        """
        term_lower = analysis.term.lower()
        reasoning_lower = analysis.reasoning.lower()
        target_lower = (analysis.target_group or "").lower()
        
        # Keyword-based categorization
        category_keywords = {
            'racial': ['black', 'white', 'hispanic', 'latino', 'asian', 'racial', 'ethnic', 'urban'],
            'religious': ['muslim', 'jewish', 'christian', 'religious', 'faith'],
            'antisemitic': ['jewish', 'jew', 'israel', 'zionist', 'globalist'],
            'islamophobic': ['muslim', 'islam', 'sharia', 'jihad'],
            'political': ['liberal', 'conservative', 'democrat', 'republican', 'socialist', 'communist'],
            'gender': ['lgbtq', 'gay', 'lesbian', 'trans', 'gender', 'women', 'feminist'],
            'immigration': ['immigrant', 'border', 'illegal', 'alien', 'migrant'],
            'economic': ['elite', 'wealthy', 'poor', 'class', 'economic'],
            'geographic': ['coastal', 'urban', 'rural', 'southern', 'northern'],
            'cultural': ['traditional', 'values', 'culture', 'heritage', 'family']
        }
        
        combined_text = f"{term_lower} {reasoning_lower} {target_lower}"
        
        # Score each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                category_scores[category] = score
        
        # Return highest scoring category or 'other'
        if category_scores:
            return max(category_scores, key=category_scores.get)
        else:
            return 'other'
    
    def _generate_definition(self, analysis: TermAnalysis) -> str:
        """
        Generate a definition based on the LLM analysis.
        
        Args:
            analysis: TermAnalysis object
            
        Returns:
            Generated definition string
        """
        # Extract key points from reasoning
        reasoning = analysis.reasoning
        
        # Simple definition generation
        if analysis.target_group:
            definition = f"A coded term that appears to target {analysis.target_group}. "
        else:
            definition = "A potential dog whistle term. "
        
        # Add context from reasoning (first sentence)
        first_sentence = reasoning.split('.')[0] + '.'
        if len(first_sentence) < 200:
            definition += first_sentence
        
        return definition.strip()
    
    def _update_related_terms(self, term: str) -> None:
        """
        Find and update related terms in the glossary.
        
        Args:
            term: Term to find relations for
        """
        if term not in self.entries:
            return
        
        current_entry = self.entries[term]
        
        # Find related terms based on shared words and categories
        term_words = set(term.split())
        
        for other_term, other_entry in self.entries.items():
            if other_term == term:
                continue
            
            # Check for shared words
            other_words = set(other_term.split())
            shared_words = term_words.intersection(other_words)
            
            # Check for same category and target
            same_category = current_entry.category == other_entry.category
            same_target = current_entry.target_group == other_entry.target_group
            
            # Add as related if criteria met
            if (len(shared_words) > 0 or (same_category and same_target)):
                if other_term not in current_entry.related_terms:
                    current_entry.related_terms.append(other_term)
                if term not in other_entry.related_terms:
                    other_entry.related_terms.append(term)
    
    def update_temporal_patterns(self, temporal_segments: Dict[str, pd.DataFrame]) -> None:
        """
        Update temporal usage patterns for glossary terms.
        
        Args:
            temporal_segments: Dictionary mapping time periods to dataframes
        """
        # Count term usage across time periods
        for period, df in temporal_segments.items():
            period_term_counts = Counter()
            
            # Count occurrences of each glossary term
            for _, row in df.iterrows():
                text = str(row.get('cleaned_text', '')).lower()
                for term in self.entries.keys():
                    if term in text:
                        period_term_counts[term] += 1
            
            # Update temporal patterns
            total_posts = len(df)
            for term, count in period_term_counts.items():
                if term in self.entries:
                    # Calculate normalized frequency
                    frequency = count / total_posts if total_posts > 0 else 0
                    self.entries[term].temporal_pattern[period] = frequency
    
    def get_terms_by_category(self, category: str) -> List[GlossaryEntry]:
        """Get all terms in a specific category."""
        return [entry for entry in self.entries.values() if entry.category == category]
    
    def get_terms_by_target(self, target_group: str) -> List[GlossaryEntry]:
        """Get all terms targeting a specific group."""
        return [
            entry for entry in self.entries.values()
            if target_group.lower() in entry.target_group.lower()
        ]
    
    def get_trending_terms(self, recent_periods: int = 3) -> List[Tuple[str, float]]:
        """
        Get terms showing upward trends in recent periods.
        
        Args:
            recent_periods: Number of recent periods to consider
            
        Returns:
            List of (term, trend_score) tuples
        """
        trending = []
        
        for term, entry in self.entries.items():
            if len(entry.temporal_pattern) < 2:
                continue
            
            # Get recent periods (sorted by period name)
            periods = sorted(entry.temporal_pattern.keys())
            if len(periods) < recent_periods:
                continue
            
            recent = periods[-recent_periods:]
            values = [entry.temporal_pattern[p] for p in recent]
            
            # Calculate trend (simple slope)
            if len(values) >= 2:
                trend = (values[-1] - values[0]) / len(values)
                if trend > 0:
                    trending.append((term, trend))
        
        return sorted(trending, key=lambda x: x[1], reverse=True)
    
    def export_glossary(self, format_type: str = 'json', filepath: Optional[str] = None) -> str:
        """
        Export glossary in specified format.
        
        Args:
            format_type: Export format ('json', 'csv', 'markdown')
            filepath: Output file path
            
        Returns:
            Path to exported file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"results/glossary_export_{timestamp}.{format_type}"
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == 'json':
            self._export_json(filepath)
        elif format_type == 'csv':
            self._export_csv(filepath)
        elif format_type == 'markdown':
            self._export_markdown(filepath)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return filepath
    
    def _export_json(self, filepath: str) -> None:
        """Export as JSON file."""
        export_data = []
        for entry in self.entries.values():
            export_data.append(asdict(entry))
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def _export_csv(self, filepath: str) -> None:
        """Export as CSV file."""
        data = []
        for entry in self.entries.values():
            data.append({
                'term': entry.term,
                'category': entry.category,
                'target_group': entry.target_group,
                'definition': entry.definition,
                'confidence_score': entry.confidence_score,
                'platforms': ';'.join(entry.platforms),
                'validation_status': entry.validation_status,
                'usage_frequency': entry.usage_frequency,
                'first_detected': entry.first_detected,
                'last_updated': entry.last_updated
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8')
    
    def _export_markdown(self, filepath: str) -> None:
        """Export as Markdown documentation."""
        lines = [
            "# Dog Whistle Terms Glossary",
            "",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total terms: {len(self.entries)}",
            ""
        ]
        
        # Group by category
        for category in sorted(self.categories.keys()):
            category_entries = self.get_terms_by_category(category)
            if not category_entries:
                continue
            
            lines.extend([
                f"## {category.title()} Terms",
                "",
                f"*{self.categories[category]}*",
                ""
            ])
            
            for entry in sorted(category_entries, key=lambda x: x.confidence_score, reverse=True):
                lines.extend([
                    f"### {entry.term}",
                    "",
                    f"**Target Group:** {entry.target_group}",
                    f"**Confidence:** {entry.confidence_score:.1f}%",
                    f"**Status:** {entry.validation_status}",
                    f"**Platforms:** {', '.join(entry.platforms)}",
                    "",
                    f"**Definition:** {entry.definition}",
                    "",
                    "**Examples:**",
                ])
                
                for example in entry.example_posts[:3]:  # Show top 3 examples
                    lines.append(f"- \"{example}\"")
                
                lines.extend(["", "---", ""])
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def save_glossary(self) -> None:
        """Save current glossary to the specified path."""
        self.export_glossary('json', self.glossary_path)
        print(f"Glossary saved with {len(self.entries)} terms")
    
    def generate_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics for the glossary."""
        if not self.entries:
            return {}
        
        # Basic counts
        total_terms = len(self.entries)
        confirmed_terms = len([e for e in self.entries.values() if e.validation_status == "confirmed"])
        
        # Category distribution
        category_counts = Counter(entry.category for entry in self.entries.values())
        
        # Target group analysis
        target_counts = Counter(entry.target_group for entry in self.entries.values())
        
        # Platform distribution
        platform_counts = Counter()
        for entry in self.entries.values():
            for platform in entry.platforms:
                platform_counts[platform] += 1
        
        # Confidence distribution
        confidences = [entry.confidence_score for entry in self.entries.values()]
        
        return {
            'total_terms': total_terms,
            'confirmed_terms': confirmed_terms,
            'suspected_terms': total_terms - confirmed_terms,
            'category_distribution': dict(category_counts),
            'target_distribution': dict(target_counts.most_common(10)),
            'platform_distribution': dict(platform_counts),
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'confidence_std': np.std(confidences) if confidences else 0,
            'last_updated': max(entry.last_updated for entry in self.entries.values()) if self.entries else None
        }


def create_sample_glossary() -> DogWhistleGlossary:
    """Create a sample glossary for testing purposes."""
    glossary = DogWhistleGlossary(glossary_path=None)  # Don't load from file
    
    # Add some sample entries
    sample_entries = [
        {
            'term': 'urban youth',
            'category': 'racial',
            'target_group': 'Black and minority youth',
            'definition': 'A coded term that uses "urban" as a proxy for Black or minority communities.',
            'confidence_score': 85.0,
            'platforms': ['truth_social', 'twitter'],
            'validation_status': 'confirmed'
        },
        {
            'term': 'coastal elites',
            'category': 'political',
            'target_group': 'Liberal/progressive Americans',
            'definition': 'Geographic dog whistle targeting liberal Americans in coastal urban areas.',
            'confidence_score': 90.0,
            'platforms': ['truth_social'],
            'validation_status': 'confirmed'
        },
        {
            'term': 'traditional values',
            'category': 'cultural',
            'target_group': 'LGBTQ+ individuals, progressives',
            'definition': 'Coded language opposing LGBTQ+ rights and progressive social changes.',
            'confidence_score': 70.0,
            'platforms': ['truth_social', 'bluesky'],
            'validation_status': 'suspected'
        }
    ]
    
    current_time = datetime.now().isoformat()
    
    for entry_data in sample_entries:
        entry = GlossaryEntry(
            term=entry_data['term'],
            category=entry_data['category'],
            target_group=entry_data['target_group'],
            definition=entry_data['definition'],
            confidence_score=entry_data['confidence_score'],
            platforms=entry_data['platforms'],
            first_detected=current_time,
            last_updated=current_time,
            example_posts=[f"Sample context for {entry_data['term']}"],
            related_terms=[],
            usage_frequency=1,
            geographic_distribution={},
            temporal_pattern={},
            validation_status=entry_data['validation_status'],
            sources=['Sample data']
        )
        
        glossary.entries[entry_data['term']] = entry
    
    return glossary


if __name__ == "__main__":
    # Demonstration of glossary building
    print("Dog Whistle Detection - Glossary Builder Module")
    print("=" * 50)
    
    # Create sample glossary
    glossary = create_sample_glossary()
    
    print(f"Created sample glossary with {len(glossary.entries)} terms")
    
    # Show summary statistics
    stats = glossary.generate_summary_stats()
    print(f"\nSummary Statistics:")
    print(f"Total terms: {stats['total_terms']}")
    print(f"Confirmed: {stats['confirmed_terms']}")
    print(f"Average confidence: {stats['avg_confidence']:.1f}%")
    
    # Show category distribution
    print(f"\nCategory Distribution:")
    for category, count in stats['category_distribution'].items():
        print(f"  {category}: {count}")
    
    # Export examples
    try:
        # Export as markdown
        md_path = glossary.export_glossary('markdown', 'results/sample_glossary.md')
        print(f"\nExported sample glossary to: {md_path}")
        
        # Export as JSON
        json_path = glossary.export_glossary('json', 'results/sample_glossary.json')
        print(f"Exported sample glossary to: {json_path}")
        
    except Exception as e:
        print(f"Export error: {e}")
    
    print("\nGlossary builder module ready for integration!")
