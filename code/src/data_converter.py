"""
Data converter for transforming platform-specific JSON data into standardized format.

This module converts JSON data from Truth Social, Mastodon, and Bluesky
into the study's standardized format with fields: text, created_at, platform, user_id.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EchoDataConverter:
    """
    Converter for echo-data JSON files to standardized format.
    
    Handles platform-specific field mappings:
    - Truth Social/Mastodon: content -> text, created_at, _id -> user_id
    - Bluesky: record.text -> text, indexedAt -> created_at, uri -> user_id
    """
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the data converter.
        
        Args:
            input_dir: Path to echo-data directory containing JSON files
            output_dir: Path to output directory for converted data
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Platform detection patterns
        self.platform_patterns = {
            'truth_social': ['truth', 'truthsocial'],
            'mastodon': ['mastodon'],
            'bluesky': ['bluesky', 'bsky']
        }
    
    def detect_platform(self, filename: str) -> str:
        """
        Detect platform from filename.
        
        Args:
            filename: Name of the JSON file
            
        Returns:
            Detected platform name
        """
        filename_lower = filename.lower()
        
        for platform, patterns in self.platform_patterns.items():
            if any(pattern in filename_lower for pattern in patterns):
                return platform
                
        # Default fallback - try to infer from file structure when loading
        return 'unknown'
    
    def convert_truth_social_mastodon(self, data: List[Dict]) -> List[Dict]:
        """
        Convert Truth Social or Mastodon data to standardized format.
        
        Expected fields: content, created_at, _id
        
        Args:
            data: List of JSON records
            
        Returns:
            List of standardized records
        """
        converted = []
        
        for record in data:
            try:
                standardized = {
                    'text': record.get('content', ''),
                    'created_at': record.get('created_at', ''),
                    'platform': 'truth_social',  # Will be updated by caller
                    'user_id': record.get('_id', record.get('id', ''))
                }
                
                # Skip empty text
                if standardized['text'].strip():
                    converted.append(standardized)
                    
            except Exception as e:
                logger.warning(f"Error converting record: {e}")
                continue
                
        return converted
    
    def convert_bluesky(self, data: List[Dict]) -> List[Dict]:
        """
        Convert Bluesky data to standardized format.
        
        Expected fields: record.text, indexedAt, uri
        
        Args:
            data: List of JSON records
            
        Returns:
            List of standardized records
        """
        converted = []
        
        for record in data:
            try:
                # Extract text from record field
                text = ''
                if 'record' in record and isinstance(record['record'], dict):
                    text = record['record'].get('text', '')
                elif 'text' in record:
                    text = record['text']
                
                standardized = {
                    'text': text,
                    'created_at': record.get('indexedAt', record.get('createdAt', '')),
                    'platform': 'bluesky',
                    'user_id': record.get('uri', record.get('did', ''))
                }
                
                # Skip empty text
                if standardized['text'].strip():
                    converted.append(standardized)
                    
            except Exception as e:
                logger.warning(f"Error converting Bluesky record: {e}")
                continue
                
        return converted
    
    def convert_file(self, input_path: Path, platform: str = None) -> Optional[pd.DataFrame]:
        """
        Convert a single JSON file to standardized format.
        
        Args:
            input_path: Path to input JSON file
            platform: Platform name (auto-detected if None)
            
        Returns:
            DataFrame with standardized data or None if error
        """
        try:
            logger.info(f"Converting file: {input_path}")
            
            # Load JSON data
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Ensure data is a list
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                logger.error(f"Unexpected data format in {input_path}")
                return None
            
            # Auto-detect platform if not provided
            if platform is None:
                platform = self.detect_platform(input_path.name)
            
            # Convert based on platform
            if platform in ['truth_social', 'mastodon']:
                converted_data = self.convert_truth_social_mastodon(data)
                # Update platform field
                for record in converted_data:
                    record['platform'] = platform
            elif platform == 'bluesky':
                converted_data = self.convert_bluesky(data)
            else:
                # Try to auto-detect from data structure
                if data and isinstance(data[0], dict):
                    sample_record = data[0]
                    if 'record' in sample_record and 'indexedAt' in sample_record:
                        converted_data = self.convert_bluesky(data)
                    elif 'content' in sample_record:
                        converted_data = self.convert_truth_social_mastodon(data)
                        # Use filename to determine platform
                        detected_platform = self.detect_platform(input_path.name)
                        for record in converted_data:
                            record['platform'] = detected_platform if detected_platform != 'unknown' else 'mastodon'
                    else:
                        logger.error(f"Could not auto-detect platform for {input_path}")
                        return None
                else:
                    logger.error(f"Empty or invalid data in {input_path}")
                    return None
            
            if not converted_data:
                logger.warning(f"No valid records found in {input_path}")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(converted_data)
            
            # Validate required columns
            required_columns = ['text', 'created_at', 'platform', 'user_id']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns in converted data from {input_path}")
                return None
            
            logger.info(f"Successfully converted {len(df)} records from {input_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error converting file {input_path}: {e}")
            return None
    
    def convert_directory(self, subdirectory: str = None) -> Dict[str, pd.DataFrame]:
        """
        Convert all JSON files in a directory or subdirectory.
        
        Args:
            subdirectory: Optional subdirectory within input_dir
            
        Returns:
            Dictionary mapping filenames to DataFrames
        """
        search_dir = self.input_dir / subdirectory if subdirectory else self.input_dir
        
        if not search_dir.exists():
            logger.error(f"Directory does not exist: {search_dir}")
            return {}
        
        converted_files = {}
        json_files = list(search_dir.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in {search_dir}")
            return {}
        
        logger.info(f"Found {len(json_files)} JSON files in {search_dir}")
        
        for json_file in json_files:
            df = self.convert_file(json_file)
            if df is not None:
                converted_files[json_file.name] = df
        
        return converted_files
    
    def save_converted_data(self, dataframes: Dict[str, pd.DataFrame], 
                          format: str = 'csv') -> None:
        """
        Save converted data to output directory.
        
        Args:
            dataframes: Dictionary mapping filenames to DataFrames
            format: Output format ('csv', 'json', or 'parquet')
        """
        if not dataframes:
            logger.warning("No data to save")
            return
        
        for filename, df in dataframes.items():
            # Create output filename
            base_name = Path(filename).stem
            
            if format == 'csv':
                output_path = self.output_dir / f"{base_name}_standardized.csv"
                df.to_csv(output_path, index=False, escapechar='\\', quoting=1)
            elif format == 'json':
                output_path = self.output_dir / f"{base_name}_standardized.json"
                df.to_json(output_path, orient='records', indent=2)
            elif format == 'parquet':
                output_path = self.output_dir / f"{base_name}_standardized.parquet"
                df.to_parquet(output_path, index=False)
            else:
                logger.error(f"Unsupported format: {format}")
                continue
            
            logger.info(f"Saved {len(df)} records to {output_path}")
    
    def convert_all_and_save(self, output_format: str = 'csv') -> pd.DataFrame:
        """
        Convert all echo-data files and save to output directory.
        
        Args:
            output_format: Output format ('csv', 'json', or 'parquet')
            
        Returns:
            Combined DataFrame of all converted data
        """
        logger.info("Starting conversion of all echo-data files")
        
        # Convert all files
        all_dataframes = self.convert_directory()
        
        if not all_dataframes:
            logger.error("No files were successfully converted")
            return pd.DataFrame()
        
        # Save individual files
        self.save_converted_data(all_dataframes, output_format)
        
        # Combine all data
        combined_df = pd.concat(all_dataframes.values(), ignore_index=True)
        
        # Save combined file
        combined_filename = f"all_platforms_combined.{output_format}"
        combined_path = self.output_dir / combined_filename
        
        if output_format == 'csv':
            combined_df.to_csv(combined_path, index=False, escapechar='\\', quoting=1)
        elif output_format == 'json':
            combined_df.to_json(combined_path, orient='records', indent=2)
        elif output_format == 'parquet':
            combined_df.to_parquet(combined_path, index=False)
        
        logger.info(f"Saved combined dataset with {len(combined_df)} records to {combined_path}")
        
        # Print summary statistics
        self.print_conversion_summary(combined_df)
        
        return combined_df
    
    def print_conversion_summary(self, df: pd.DataFrame) -> None:
        """
        Print summary statistics of converted data.
        
        Args:
            df: Combined DataFrame of converted data
        """
        print("\n" + "="*50)
        print("DATA CONVERSION SUMMARY")
        print("="*50)
        
        print(f"Total records converted: {len(df):,}")
        
        if 'platform' in df.columns:
            print("\nRecords by platform:")
            platform_counts = df['platform'].value_counts()
            for platform, count in platform_counts.items():
                print(f"  {platform}: {count:,}")
        
        if 'created_at' in df.columns:
            try:
                df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
                date_range = df['created_at'].dropna()
                if not date_range.empty:
                    print(f"\nDate range: {date_range.min()} to {date_range.max()}")
            except:
                pass
        
        if 'text' in df.columns:
            text_lengths = df['text'].str.len()
            print(f"\nText statistics:")
            print(f"  Average length: {text_lengths.mean():.1f} characters")
            print(f"  Median length: {text_lengths.median():.1f} characters")
            print(f"  Max length: {text_lengths.max()} characters")
        
        print("\nOutput directory:", self.output_dir)
        print("="*50)


def main():
    """Main function for testing the data converter."""
    # Example usage
    input_dir = "./data/echo-data"
    output_dir = "./data/standardized"
    
    converter = EchoDataConverter(input_dir, output_dir)
    
    # Convert all data and save
    combined_df = converter.convert_all_and_save(output_format='csv')
    
    return combined_df


if __name__ == "__main__":
    main()
