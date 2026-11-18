"""
LLM-based vetting and classification of potential dog whistle terms.

This module uses GPT-4 and other language models to classify terms as potential
dog whistles and identify their targets through few-shot prompting and structured analysis.
"""

import os
import json
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import openai
from dotenv import load_dotenv
import re
from collections import Counter, defaultdict

# Load environment variables
load_dotenv()


class ClassificationResult(Enum):
    """Enum for dog whistle classification results."""
    CONFIRMED_DOG_WHISTLE = "confirmed_dog_whistle"
    LIKELY_DOG_WHISTLE = "likely_dog_whistle"
    UNCLEAR = "unclear"
    NOT_DOG_WHISTLE = "not_dog_whistle"
    ERROR = "error"


@dataclass
class TermAnalysis:
    """Data class for storing term analysis results."""
    term: str
    classification: ClassificationResult
    confidence: float
    target_group: Optional[str]
    reasoning: str
    example_contexts: List[str]
    platform: Optional[str]
    analysis_timestamp: str


class LLMVetter:
    """
    LLM-based vetting system for dog whistle term classification.
    
    Uses structured prompting with GPT-4 to identify coded language and
    determine likely targets of dog whistle terms.
    """
    
    def __init__(self, model: str = "gpt-4", max_retries: int = 3, 
                 delay_between_calls: float = 1.0):
        self.model = model
        self.max_retries = max_retries
        self.delay_between_calls = delay_between_calls
        
        # Initialize OpenAI client
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = openai.OpenAI()
        
        # Load prompts and examples
        self.base_prompt = self._get_base_prompt()
        self.few_shot_examples = self._get_few_shot_examples()
        
    def _get_base_prompt(self) -> str:
        """Get the base prompt template for dog whistle classification."""
        return """You are an expert in sociolinguistics and hate speech detection with deep knowledge of coded language, dog whistles, and implicit bias in social media communication.

Your task is to analyze terms used in social media posts to determine if they function as "dog whistles" - coded language that appears neutral to general audiences but conveys specific meaning to in-group members, often targeting marginalized groups.

ANALYSIS FRAMEWORK:
1. Context Analysis: Examine how the term is used across multiple examples
2. Historical Context: Consider known dog whistles and coded language patterns
3. Targeting Assessment: Identify if/which groups are being targeted
4. Confidence Rating: Provide numerical confidence (0-100)

CLASSIFICATION CATEGORIES:
- CONFIRMED_DOG_WHISTLE: Clear coded language with established targeting pattern
- LIKELY_DOG_WHISTLE: Strong indicators but some ambiguity
- UNCLEAR: Insufficient evidence or highly ambiguous
- NOT_DOG_WHISTLE: Likely legitimate usage without coded meaning

RESPONSE FORMAT:
{
    "classification": "[CATEGORY]",
    "confidence": [0-100],
    "target_group": "[specific group or null]",
    "reasoning": "[detailed explanation]"
}

Here are some examples of previous analyses:"""

    def _get_few_shot_examples(self) -> List[Dict[str, Any]]:
        """Get few-shot examples for prompt engineering."""
        return [
            {
                "term": "urban youth",
                "contexts": [
                    "The urban youth are causing problems in our neighborhoods again",
                    "More programs for urban youth won't solve the crime issue",
                    "Urban youth culture is corrupting traditional values"
                ],
                "analysis": {
                    "classification": "CONFIRMED_DOG_WHISTLE",
                    "confidence": 85,
                    "target_group": "Black and minority youth",
                    "reasoning": "Term 'urban youth' is a well-documented dog whistle that uses 'urban' as a proxy for 'Black' or minority communities. The contexts show usage patterns typical of coded racial messaging, implying criminality and cultural threat without explicit racial language."
                }
            },
            {
                "term": "traditional values",
                "contexts": [
                    "We need to return to traditional values in education",
                    "Traditional values are under attack by modern society",
                    "Our traditional values built this great nation"
                ],
                "analysis": {
                    "classification": "LIKELY_DOG_WHISTLE",
                    "confidence": 70,
                    "target_group": "LGBTQ+ individuals, religious minorities, progressives",
                    "reasoning": "While 'traditional values' can have legitimate uses, in these contexts it appears to function as coded language opposing LGBTQ+ rights, religious diversity, and progressive social changes. The framing suggests an us-vs-them dynamic typical of dog whistle usage."
                }
            },
            {
                "term": "coastal elites",
                "contexts": [
                    "The coastal elites don't understand real America",
                    "Coastal elites are pushing their agenda on us",
                    "Stop listening to coastal elites and their media"
                ],
                "analysis": {
                    "classification": "CONFIRMED_DOG_WHISTLE",
                    "confidence": 90,
                    "target_group": "Liberal/progressive Americans, particularly those in urban areas",
                    "reasoning": "Term functions as geographic dog whistle targeting liberal Americans, particularly those in coastal urban areas. Creates artificial division between 'real Americans' and others, a classic dog whistle pattern that delegitimizes political opposition through coded geographic/cultural othering."
                }
            }
        ]
    
    def _format_prompt(self, term: str, contexts: List[str]) -> str:
        """
        Format the complete prompt for GPT-4 analysis.
        
        Args:
            term: Term to analyze
            contexts: List of context sentences containing the term
            
        Returns:
            Formatted prompt string
        """
        prompt = self.base_prompt + "\n\n"
        
        # Add few-shot examples
        for example in self.few_shot_examples:
            prompt += f"TERM: \"{example['term']}\"\n"
            prompt += "CONTEXTS:\n"
            for i, context in enumerate(example['contexts'], 1):
                prompt += f"{i}. \"{context}\"\n"
            prompt += f"ANALYSIS: {json.dumps(example['analysis'], indent=2)}\n\n"
        
        # Add current term for analysis
        prompt += f"Now analyze this term:\n\nTERM: \"{term}\"\n"
        prompt += "CONTEXTS:\n"
        for i, context in enumerate(contexts, 1):
            prompt += f"{i}. \"{context}\"\n"
        
        prompt += "\nANALYSIS:"
        
        return prompt
    
    def analyze_term(self, term: str, contexts: List[str], 
                    platform: Optional[str] = None) -> TermAnalysis:
        """
        Analyze a single term for dog whistle characteristics.
        
        Args:
            term: Term to analyze
            contexts: List of contexts where term appears
            platform: Social media platform (for additional context)
            
        Returns:
            TermAnalysis object with classification results
        """
        if not contexts:
            return TermAnalysis(
                term=term,
                classification=ClassificationResult.ERROR,
                confidence=0.0,
                target_group=None,
                reasoning="No contexts provided for analysis",
                example_contexts=[],
                platform=platform,
                analysis_timestamp=pd.Timestamp.now().isoformat()
            )
        
        # Limit contexts to prevent token overflow
        limited_contexts = contexts[:8]  # Use up to 8 contexts
        
        prompt = self._format_prompt(term, limited_contexts)
        
        # Make API call with retries
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,  # Low temperature for consistent analysis
                    max_tokens=500
                )
                
                # Parse response
                response_text = response.choices[0].message.content.strip()
                analysis_data = self._parse_response(response_text)
                
                return TermAnalysis(
                    term=term,
                    classification=ClassificationResult(analysis_data["classification"].lower()),
                    confidence=float(analysis_data["confidence"]),
                    target_group=analysis_data.get("target_group"),
                    reasoning=analysis_data["reasoning"],
                    example_contexts=limited_contexts,
                    platform=platform,
                    analysis_timestamp=pd.Timestamp.now().isoformat()
                )
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for term '{term}': {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay_between_calls * (attempt + 1))
                else:
                    return TermAnalysis(
                        term=term,
                        classification=ClassificationResult.ERROR,
                        confidence=0.0,
                        target_group=None,
                        reasoning=f"API error after {self.max_retries} attempts: {str(e)}",
                        example_contexts=limited_contexts,
                        platform=platform,
                        analysis_timestamp=pd.Timestamp.now().isoformat()
                    )
        
        # Add delay between API calls
        time.sleep(self.delay_between_calls)
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the LLM response into structured data.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Parsed analysis dictionary
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback parsing for non-JSON responses
                return self._fallback_parse(response_text)
        except json.JSONDecodeError:
            return self._fallback_parse(response_text)
    
    def _fallback_parse(self, response_text: str) -> Dict[str, Any]:
        """
        Fallback parsing for non-JSON responses.
        
        Args:
            response_text: Raw response text
            
        Returns:
            Parsed analysis dictionary
        """
        # Basic keyword extraction
        classification = "unclear"
        confidence = 50
        target_group = None
        reasoning = response_text
        
        text_lower = response_text.lower()
        
        if "confirmed_dog_whistle" in text_lower:
            classification = "confirmed_dog_whistle"
            confidence = 80
        elif "likely_dog_whistle" in text_lower:
            classification = "likely_dog_whistle"
            confidence = 70
        elif "not_dog_whistle" in text_lower:
            classification = "not_dog_whistle"
            confidence = 75
        
        # Extract confidence if mentioned
        conf_match = re.search(r'confidence[:\s]*(\d+)', text_lower)
        if conf_match:
            confidence = int(conf_match.group(1))
        
        return {
            "classification": classification,
            "confidence": confidence,
            "target_group": target_group,
            "reasoning": reasoning
        }
    
    def batch_analyze(self, terms_with_contexts: Dict[str, List[str]],
                     platform: Optional[str] = None) -> List[TermAnalysis]:
        """
        Analyze multiple terms in batch.
        
        Args:
            terms_with_contexts: Dictionary mapping terms to their contexts
            platform: Social media platform
            
        Returns:
            List of TermAnalysis objects
        """
        results = []
        total_terms = len(terms_with_contexts)
        
        print(f"Analyzing {total_terms} terms...")
        
        for i, (term, contexts) in enumerate(terms_with_contexts.items(), 1):
            print(f"Analyzing term {i}/{total_terms}: '{term}'")
            analysis = self.analyze_term(term, contexts, platform)
            results.append(analysis)
            
            # Progress update
            if i % 5 == 0 or i == total_terms:
                print(f"Completed {i}/{total_terms} terms")
        
        return results


class ResultsProcessor:
    """
    Process and analyze LLM vetting results.
    """
    
    @staticmethod
    def to_dataframe(analyses: List[TermAnalysis]) -> pd.DataFrame:
        """Convert TermAnalysis objects to DataFrame."""
        data = []
        for analysis in analyses:
            data.append({
                'term': analysis.term,
                'classification': analysis.classification.value,
                'confidence': analysis.confidence,
                'target_group': analysis.target_group,
                'reasoning': analysis.reasoning,
                'platform': analysis.platform,
                'num_contexts': len(analysis.example_contexts),
                'analysis_timestamp': analysis.analysis_timestamp
            })
        return pd.DataFrame(data)
    
    @staticmethod
    def filter_confirmed_dog_whistles(analyses: List[TermAnalysis],
                                    min_confidence: float = 70.0) -> List[TermAnalysis]:
        """Filter for confirmed dog whistles above confidence threshold."""
        return [
            analysis for analysis in analyses
            if analysis.classification in [
                ClassificationResult.CONFIRMED_DOG_WHISTLE,
                ClassificationResult.LIKELY_DOG_WHISTLE
            ] and analysis.confidence >= min_confidence
        ]
    
    @staticmethod
    def generate_summary_report(analyses: List[TermAnalysis]) -> str:
        """Generate a summary report of the analysis results."""
        df = ResultsProcessor.to_dataframe(analyses)
        
        report = ["LLM VETTING ANALYSIS SUMMARY", "=" * 30, ""]
        
        # Overall statistics
        total_terms = len(df)
        confirmed = len(df[df['classification'] == 'confirmed_dog_whistle'])
        likely = len(df[df['classification'] == 'likely_dog_whistle'])
        unclear = len(df[df['classification'] == 'unclear'])
        not_dog_whistle = len(df[df['classification'] == 'not_dog_whistle'])
        
        report.extend([
            f"Total terms analyzed: {total_terms}",
            f"Confirmed dog whistles: {confirmed} ({confirmed/total_terms*100:.1f}%)",
            f"Likely dog whistles: {likely} ({likely/total_terms*100:.1f}%)",
            f"Unclear: {unclear} ({unclear/total_terms*100:.1f}%)",
            f"Not dog whistles: {not_dog_whistle} ({not_dog_whistle/total_terms*100:.1f}%)",
            ""
        ])
        
        # Top confirmed dog whistles
        confirmed_df = df[df['classification'] == 'confirmed_dog_whistle'].sort_values('confidence', ascending=False)
        if not confirmed_df.empty:
            report.extend([
                "TOP CONFIRMED DOG WHISTLES:",
                "-" * 25
            ])
            for _, row in confirmed_df.head(10).iterrows():
                report.append(f"{row['term']:25} | Confidence: {row['confidence']:3.0f}% | Target: {row['target_group'] or 'Unknown'}")
            report.append("")
        
        # Target group analysis
        target_groups = df[df['target_group'].notna()]['target_group'].value_counts()
        if not target_groups.empty:
            report.extend([
                "TARGETED GROUPS:",
                "-" * 15
            ])
            for group, count in target_groups.items():
                report.append(f"{group}: {count} terms")
        
        return "\n".join(report)
    
    @staticmethod
    def export_results(analyses: List[TermAnalysis], filepath: str) -> None:
        """Export results to JSON file."""
        export_data = []
        for analysis in analyses:
            export_data.append({
                'term': analysis.term,
                'classification': analysis.classification.value,
                'confidence': analysis.confidence,
                'target_group': analysis.target_group,
                'reasoning': analysis.reasoning,
                'example_contexts': analysis.example_contexts,
                'platform': analysis.platform,
                'analysis_timestamp': analysis.analysis_timestamp
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results exported to {filepath}")


def create_sample_terms_for_testing() -> Dict[str, List[str]]:
    """Create sample terms with contexts for testing the vetting system."""
    return {
        "dual citizenship": [
            "The dual citizenship issue is causing problems in our institutions",
            "People with dual citizenship can't be trusted in government",
            "Dual citizenship creates divided loyalties"
        ],
        "cultural enrichment": [
            "Another example of cultural enrichment in our cities",
            "The cultural enrichment is really working out well",
            "More cultural enrichment programs announced"
        ],
        "family values": [
            "We need to return to family values",
            "Family values are under attack",
            "Traditional family values built this nation"
        ],
        "medical freedom": [
            "Fighting for medical freedom and choice",
            "Medical freedom is a fundamental right",
            "Protecting medical freedom from tyranny"
        ]
    }


if __name__ == "__main__":
    # Demonstration of LLM vetting system
    print("Dog Whistle Detection - LLM Vetting Module")
    print("=" * 45)
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set in environment")
        print("Set your API key in .env file to test LLM functionality")
        print("\nModule structure ready for integration!")
    else:
        print("API key found - running sample analysis...")
        
        try:
            # Initialize vetter
            vetter = LLMVetter()
            
            # Create sample data
            sample_terms = create_sample_terms_for_testing()
            
            # Analyze first term as demo
            first_term = list(sample_terms.keys())[0]
            first_contexts = sample_terms[first_term]
            
            print(f"\nAnalyzing sample term: '{first_term}'")
            analysis = vetter.analyze_term(first_term, first_contexts)
            
            print(f"\nResults:")
            print(f"Classification: {analysis.classification.value}")
            print(f"Confidence: {analysis.confidence}%")
            print(f"Target Group: {analysis.target_group}")
            print(f"Reasoning: {analysis.reasoning[:200]}...")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
    
    print("\nLLM vetting module ready for integration!")
