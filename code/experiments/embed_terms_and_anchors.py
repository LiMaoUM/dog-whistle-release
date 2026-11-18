"""
Script to embed text from data_all and extract term/anchor embeddings.
Optimized for speed: batch encoding, efficient caching, checkpoint support.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import json
import os

# Load data with proper dtypes to save memory
print("Loading data...")
data = pd.read_csv(
    "./data/english_posts_processed.csv",
    dtype={'text': 'object', 'platform': 'category'},
    low_memory=False
)

# Try to load Turkish data with error handling
try:
    tr_data = pd.read_csv(
        "./data/tr_english_posts_processed.csv",
        dtype={'text': 'object', 'platform': 'category'},
        on_bad_lines='skip',  # Skip malformed lines
    )

except Exception as e:
    print(f"Warning: Could not load Turkish data: {e}")
    print("Proceeding with English data only...")
    tr_data = pd.DataFrame(columns=['text', 'platform'])

data_all = pd.concat([data[['text','platform']], tr_data[['text','platform']]], ignore_index=True)

# Add unique IDs for each post
data_all['post_id'] = range(len(data_all))

print(f"Total posts: {len(data_all)}")

# Initialize model
print("Loading model...")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda:0")

# Define terms and anchors
terms = ['adolph', 'banker', 'fairy', 'caliphate', 'frog', 'maga']

anchors = {
    "pos": [
        "good", "benefit", "help", "protect", "justice",
        "support", "safe", "improve", "positive"
    ],
    "neg": [
        "bad", "harm", "hate", "threat", "danger",
        "disgust", "evil", "corrupt", "destroy", "negative"
    ],
    "left": [
        "democrat", "democrats", "democratic", "dnc",
        "biden", "harris",
        "liberal", "progressive", "left-wing", "left wing",
        "aoc", "bernie", "sanders",
        "union", "labor"
    ],
    "right": [
        "republican", "republicans", "gop",
        "trump",
        "conservative", "right-wing", "right wing",
        "libertarian",
        "fox", "heritage foundation",
        "second amendment", "2a"
    ],
}

# 1. Embed all texts - use larger batch size for GPU efficiency
print("Embedding all texts with larger batch size...")
output_dir = Path("./results")
output_dir.mkdir(exist_ok=True)
text_embeddings_path = output_dir / "text_embeddings.npy"

# Use memory mapping for large arrays
text_embeddings = model.encode(data_all['text'].tolist(), show_progress_bar=True, batch_size=256)

# 2. Extract term embeddings for texts containing those terms
print("\nExtracting term contexts and embeddings...")
term_results = []

for term in terms:
    mask = data_all['text'].str.contains(term, case=False, na=False)
    matching_indices = np.where(mask.values)[0]  # Use numpy for speed
    matching_texts = data_all.loc[mask, 'text'].tolist()
    
    if len(matching_texts) == 0:
        print(f"  {term}: no matches")
        continue
    
    print(f"  {term}: {len(matching_texts)} matches")
    
    # Batch encode marked texts with larger batch size
    marked_texts = [text.replace(term, f"[{term.upper()}]") for text in matching_texts]
    term_embeddings = model.encode(marked_texts, show_progress_bar=False, batch_size=256)
    
    term_results.append({
        'term': term,
        'post_ids': matching_indices.tolist(),
        'embeddings': term_embeddings
    })

print(f"Total term occurrences: {sum(len(r['post_ids']) for r in term_results)}")


# 4. Save results
print("\nSaving results...")

# Save metadata mapping (post_id -> text, platform)
print("Saving metadata mapping...")
metadata_map = {
    int(row['post_id']): {
        'text': row['text'],
        'platform': str(row['platform'])
    }
    for _, row in data_all.iterrows()
}
with open(output_dir / "post_metadata.json", "w") as f:
    json.dump(metadata_map, f)
print(f"Saved: {output_dir}/post_metadata.json ({len(metadata_map)} posts)")

# Save all text embeddings as NumPy
print("Saving text embeddings...")
np.save(output_dir / "text_embeddings.npy", text_embeddings)
print(f"Saved: {output_dir}/text_embeddings.npy (shape: {text_embeddings.shape})")

# Save term embeddings with IDs
print("Saving term embeddings...")
term_embeddings_data = {}
for term_info in term_results:
    term = term_info['term']
    term_embeddings_data[term] = {
        'post_ids': term_info['post_ids'],
        'embeddings_file': f"term_embeddings_{term}.npy"
    }
    np.save(output_dir / f"term_embeddings_{term}.npy", term_info['embeddings'])
    print(f"  Saved term '{term}': {len(term_info['post_ids'])} embeddings")

with open(output_dir / "term_embeddings_index.json", "w") as f:
    json.dump(term_embeddings_data, f)
print(f"Saved: {output_dir}/term_embeddings_index.json")

# Save anchor embeddings - extract from posts containing anchor words
print("\nExtracting anchor contexts and embeddings...")
anchor_results = {}

for category, words in anchors.items():
    print(f"\n  Processing anchor category: {category}")
    category_post_ids = []
    category_embeddings_list = []
    category_words_found = []
    
    for anchor_word in words:
        # Find posts containing this anchor word
        mask = data_all['text'].str.contains(anchor_word, case=False, na=False)
        matching_indices = np.where(mask.values)[0]
        matching_texts = data_all.loc[mask, 'text'].tolist()
        
        if len(matching_texts) == 0:
            print(f"    {anchor_word}: no matches")
            continue
        
        print(f"    {anchor_word}: {len(matching_texts)} matches")
        
        # Batch encode marked texts with larger batch size
        marked_texts = [text.replace(anchor_word, f"[{anchor_word.upper()}]") for text in matching_texts]
        anchor_embeddings = model.encode(marked_texts, show_progress_bar=False, batch_size=256)
        
        category_post_ids.extend(matching_indices.tolist())
        category_embeddings_list.append(anchor_embeddings)
        category_words_found.extend([anchor_word] * len(matching_indices))
    
    if len(category_embeddings_list) > 0:
        all_category_embeddings = np.vstack(category_embeddings_list)
        
        anchor_results[category] = {
            'words': list(set(category_words_found)),
            'post_ids': category_post_ids,
            'anchor_words_per_post': category_words_found,
            'embeddings_file': f"anchor_embeddings_{category}.npy",
            'total_contexts': len(category_post_ids)
        }
        
        np.save(output_dir / f"anchor_embeddings_{category}.npy", all_category_embeddings)
        print(f"  Saved anchor '{category}': {len(category_post_ids)} total context embeddings")
    else:
        print(f"  Warning: No matches found for category '{category}'")

with open(output_dir / "anchor_embeddings_index.json", "w") as f:
    json.dump(anchor_results, f)
print(f"Saved: {output_dir}/anchor_embeddings_index.json")

print("\n✓ Done! Storage layout:")
print(f"  - post_metadata.json: ID -> (text, platform)")
print(f"  - text_embeddings.npy: All embeddings (index by post_id)")
print(f"  - term_embeddings_index.json: Term info + file references")
print(f"  - term_embeddings_*.npy: Individual term contextual embeddings")
print(f"  - anchor_embeddings_index.json: Anchor info + file references")
print(f"  - anchor_embeddings_*.npy: Individual anchor contextual embeddings")
