"""
Prepare the Shakespeare dataset for character-level language modeling.
Instead of encoding with GPT-2 BPE tokens, we map characters to integers.
The processed data will be saved as train.bin and val.bin containing the ids,
and meta.pkl containing the encoder, decoder, and other related information.
"""
import os
import pickle
import requests
import numpy as np

def download_shakespeare_dataset(file_path: str) -> None:
    """Download the tiny Shakespeare dataset if it doesn't exist."""
    if not os.path.exists(file_path):
        dataset_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(file_path, 'w') as file:
            file.write(requests.get(dataset_url).text)

def create_character_mappings(text_data: str) -> tuple[int, dict, dict]:
    """Create mappings between characters and integers."""
    unique_characters: list[str] = sorted(list(set(text_data)))
    vocabulary_size: int = len(unique_characters)
    print("All the unique characters:", ''.join(unique_characters))
    print(f"Vocabulary size: {vocabulary_size:,}")
    
    char_to_index: dict[str, int] = { char: index for index, char in enumerate(unique_characters) }
    index_to_char: dict[int, str] = { index: char for index, char in enumerate(unique_characters) }
    return vocabulary_size, char_to_index, index_to_char

def encode_text(text: str, char_to_index: dict) -> list[int]:
    """Encode string to integer list."""
    return [char_to_index[char] for char in text]

def decode_ids(ids: list[int], index_to_char: dict[int, str]) -> str:
    """Decode integer list to string."""
    return ''.join([index_to_char[i] for i in ids])

def main() -> None:
    # Set up file paths
    input_file_path: str = os.path.join(os.path.dirname(__file__), 'input.txt')
    download_shakespeare_dataset(input_file_path)

    # Read data
    with open(input_file_path, 'r') as file:
        text_data: str = file.read()
    print(f"Length of dataset in characters: {len(text_data):,}")

    # Create character mappings
    vocabulary_size: int
    char_to_index: dict
    index_to_char: dict
    vocabulary_size, char_to_index, index_to_char = create_character_mappings(text_data)

    # Split into training and validation sets
    total_length: int = len(text_data)
    training_data: str = text_data[:int(total_length * 0.9)]
    validation_data: str = text_data[int(total_length * 0.9):]

    # Encode data
    training_ids: list[int] = encode_text(training_data, char_to_index)
    validation_ids: list[int] = encode_text(validation_data, char_to_index)
    print(f"Training set has {len(training_ids):,} tokens")
    print(f"Validation set has {len(validation_ids):,} tokens")

    # Save encoded data
    training_ids: np.ndarray[np.uint16] = np.array(training_ids, dtype=np.uint16)
    validation_ids: np.ndarray[np.uint16] = np.array(validation_ids, dtype=np.uint16)
    training_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    validation_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

    # Save metadata
    metadata: dict = {
        'vocabulary_size': vocabulary_size,
        'index_to_char': index_to_char,
        'char_to_index': char_to_index,
    }
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as file:
        pickle.dump(metadata, file)

if __name__ == '__main__':
    main()

# Example output:
# Length of dataset in characters:  1115394
# All the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# Vocabulary size: 65
# Training set has 1003854 tokens
# Validation set has 111540 tokens
