"""
Prepare the Shakespeare dataset for word-level language modeling.
The processed data will be saved as train.bin and val.bin containing the word indices,
and meta.pkl containing the word encoder, decoder, and other related information.
"""
import os
import pickle
import requests
import numpy as np
from collections import Counter
import re

def download_shakespeare_dataset(file_path: str) -> None:
    """Download the tiny Shakespeare dataset if it doesn't exist."""
    if not os.path.exists(file_path):
        dataset_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(file_path, 'w') as file:
            file.write(requests.get(dataset_url).text)

def preprocess_text(text: str) -> list[str]:
    """Preprocess text into words and punctuation with basic cleaning."""
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    
    # Handle punctuation and special characters
    words = []
    current_word = ''
    
    for char in text:
        if char.isalnum() or char == "'":  # Part of a word or contraction
            current_word += char
        else:
            # Save the current word if it exists
            if current_word:
                words.append(current_word)
                current_word = ''
            
            # Add punctuation/whitespace as separate token if not just a space
            if not char.isspace():
                words.append(char)
    
    # Add the last word if it exists
    if current_word:
        words.append(current_word)
    
    return words

def create_vocabulary(words: list[str], min_freq: int = 2) -> tuple[int, dict, dict]:
    """Create word-to-index and index-to-word mappings."""
    # Count word frequencies
    word_counts = Counter(words)
    
    # Filter rare words and create vocabulary
    vocab_words = ['<UNK>'] + [word for word, count in word_counts.items() 
                              if count >= min_freq]
    vocabulary_size = len(vocab_words)
    
    # Create mappings
    word_to_index = {word: idx for idx, word in enumerate(vocab_words)}
    index_to_word = {idx: word for idx, word in enumerate(vocab_words)}
    
    print(f"Total unique tokens: {len(word_counts):,}")
    print(f"Tokens in vocabulary: {vocabulary_size:,}")
    
    return vocabulary_size, word_to_index, index_to_word

def encode_words(words: list[str], word_to_index: dict) -> list[int]:
    """Encode words to integer indices."""
    return [word_to_index.get(word, 0) for word in words]  # 0 is <UNK> token

def main() -> None:
    # Set up file paths
    input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
    download_shakespeare_dataset(input_file_path)

    # Read and preprocess data
    with open(input_file_path, 'r') as file:
        text_data = file.read()
    words = preprocess_text(text_data)
    print(f"Total tokens in dataset: {len(words):,}")

    # Create vocabulary
    vocabulary_size, word_to_index, index_to_word = create_vocabulary(words)

    # Encode all words
    word_indices = encode_words(words, word_to_index)

    # Split into training and validation sets (90-10 split)
    split_idx = int(len(word_indices) * 0.9)
    training_indices = word_indices[:split_idx]
    validation_indices = word_indices[split_idx:]

    print(f"Training set has {len(training_indices):,} tokens")
    print(f"Validation set has {len(validation_indices):,} tokens")

    # Save encoded data
    training_indices = np.array(training_indices, dtype=np.uint16)
    validation_indices = np.array(validation_indices, dtype=np.uint16)
    training_indices.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    validation_indices.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

    # Save metadata
    metadata = {
        'vocabulary_size': vocabulary_size,
        'index_to_word': index_to_word,
        'word_to_index': word_to_index,
    }
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as file:
        pickle.dump(metadata, file)

if __name__ == '__main__':
    main() 