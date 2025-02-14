"""
Data loader module for character-level, word-level, and BPE tokenization of Shakespeare datasets.
"""
import os
import pickle
import numpy as np
import torch
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Dict, List, Any, Optional
import gc
import tiktoken

def clear_memory():
    """Clear memory cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        # MPS doesn't have explicit cache clearing like CUDA
        # but we can force synchronization
        torch.mps.synchronize()
    gc.collect()

class TokenType(Enum):
    CHAR = 'char'
    WORD = 'word'
    BPE = 'bpe'  # New BPE token type

@dataclass
class DataConfig:
    token_type: TokenType
    context_window: int
    batch_size: int
    bpe_encoding: str = "gpt2"  # Default to GPT-2's BPE encoding
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary with enum handling."""
        return {
            'token_type': self.token_type.value,
            'context_window': self.context_window,
            'batch_size': self.batch_size,
            'bpe_encoding': self.bpe_encoding
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataConfig':
        """Create config from dictionary with enum handling."""
        config_dict['token_type'] = TokenType(config_dict['token_type'])
        return cls(**config_dict)

@dataclass
class DataInfo:
    vocab_size: int
    train_size: int
    val_size: int
    index_to_token: Dict[int, str]
    token_to_index: Dict[str, int]

class ShakespeareDataLoader:
    def __init__(self, config: DataConfig, device: torch.device = None):
        self.config = config
        # Use provided device or default to CPU
        self.device = device if device is not None else torch.device('cpu')
        print(f"DataLoader using device: {self.device}")
        
        # Initialize BPE tokenizer if using BPE
        self.bpe_tokenizer = None
        if config.token_type == TokenType.BPE:
            try:
                self.bpe_tokenizer = tiktoken.get_encoding(config.bpe_encoding)
                print(f"Using BPE tokenizer: {config.bpe_encoding}")
            except Exception as e:
                print(f"Error initializing BPE tokenizer: {e}")
                raise
        
        # Load raw text for BPE or processed data for other token types
        if config.token_type == TokenType.BPE:
            self.train_data, self.val_data, self.meta = self._load_and_tokenize_text()
        else:
            # Load data based on token type
            self.data_dir = os.path.join(
                os.path.dirname(__file__), 
                'data', 
                f'shakespeare_{config.token_type.value}'
            )
            self.train_data, self.val_data, self.meta = self._load_data()
        
        # Create sequences
        self.train_sequences, self.train_targets = self._create_sequences(self.train_data)
        self.val_sequences, self.val_targets = self._create_sequences(self.val_data)
        
        # Create data info
        if config.token_type == TokenType.BPE:
            self.info = DataInfo(
                vocab_size=self.bpe_tokenizer.n_vocab,
                train_size=len(self.train_sequences),
                val_size=len(self.val_sequences),
                index_to_token={i: str(i) for i in range(self.bpe_tokenizer.n_vocab)},
                token_to_index={str(i): i for i in range(self.bpe_tokenizer.n_vocab)}
            )
        else:
            self.info = DataInfo(
                vocab_size=self.meta['vocabulary_size'],
                train_size=len(self.train_sequences),
                val_size=len(self.val_sequences),
                index_to_token=self.meta[f'index_to_{config.token_type.value}'],
                token_to_index=self.meta[f'{config.token_type.value}_to_index']
            )
        
        # Clear memory after initialization
        del self.train_data
        del self.val_data
        clear_memory()
    
    def _load_and_tokenize_text(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Load raw text and tokenize using BPE."""
        # Load raw text file
        text_path = os.path.join(os.path.dirname(__file__), 'data/shakespeare_char/input.txt')
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into train and validation (90-10 split)
        split_idx = int(len(text) * 0.9)
        train_text = text[:split_idx]
        val_text = text[split_idx:]
        
        # Tokenize using BPE
        train_tokens = np.array(self.bpe_tokenizer.encode(train_text), dtype=np.int32)
        val_tokens = np.array(self.bpe_tokenizer.encode(val_text), dtype=np.int32)
        
        # Create minimal metadata for BPE
        meta = {
            'vocabulary_size': self.bpe_tokenizer.n_vocab,
            'encoding_name': self.config.bpe_encoding
        }
        
        return train_tokens, val_tokens, meta
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Load binary data and metadata for char/word tokenization."""
        # Load metadata
        with open(os.path.join(self.data_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        
        # Load training and validation data
        train_data = np.fromfile(os.path.join(self.data_dir, 'train.bin'), dtype=np.uint16)
        val_data = np.fromfile(os.path.join(self.data_dir, 'val.bin'), dtype=np.uint16)
        
        return train_data, val_data, meta
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-target sequences for training with MPS optimization."""
        sequences = []
        targets = []
        
        # Process in smaller chunks for MPS memory constraints
        chunk_size = 500  # Smaller chunk size for MPS
        for i in range(0, len(data) - self.config.context_window, chunk_size * self.config.context_window):
            end_idx = min(i + chunk_size * self.config.context_window, len(data) - self.config.context_window)
            
            # Create sequences for this chunk
            for j in range(i, end_idx, self.config.context_window):
                seq = data[j:j + self.config.context_window]
                target = data[j + 1:j + self.config.context_window + 1]
                sequences.append(seq)
                targets.append(target)
            
            # Clear memory more frequently for MPS
            if len(sequences) % (chunk_size * 5) == 0:
                clear_memory()
        
        return np.array(sequences), np.array(targets)
    
    def get_batch(self, sequences: np.ndarray, targets: np.ndarray, batch_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Create batches with memory optimization."""
        try:
            # Get batch data
            batch_sequences = sequences[batch_idx:batch_idx + self.config.batch_size]
            batch_targets = targets[batch_idx:batch_idx + self.config.batch_size]
            
            # Skip if batch is incomplete
            if len(batch_sequences) < self.config.batch_size:
                return None, None
            
            if self.config.token_type == TokenType.BPE:
                # For BPE, directly use token indices without one-hot encoding
                # Create tensors on CPU first
                x = torch.tensor(batch_sequences, dtype=torch.long)
                y = torch.tensor(batch_targets.reshape(-1), dtype=torch.long)
                
                # Move to device
                x = x.to(self.device)
                y = y.to(self.device)
                
                return x, y
            else:
                # For character/word tokens, use one-hot encoding
                # Create tensors on CPU first
                x = torch.zeros(
                    batch_sequences.shape[0], 
                    batch_sequences.shape[1], 
                    self.info.vocab_size, 
                    dtype=torch.float32
                )
                
                # Fill one-hot tensor efficiently
                for i in range(batch_sequences.shape[0]):
                    x[i, range(batch_sequences.shape[1]), batch_sequences[i]] = 1
                
                # Create target tensor
                y = torch.tensor(batch_targets.reshape(-1), dtype=torch.long)
                
                # Move both tensors to device
                x = x.to(self.device)
                y = y.to(self.device)
                
                return x, y
            
        except Exception as e:
            print(f"Error in get_batch: {e}")
            clear_memory()
            return None, None
    
    def decode_tokens(self, indices: List[int]) -> str:
        """Convert token indices back to text."""
        try:
            if self.config.token_type == TokenType.BPE:
                return self.bpe_tokenizer.decode(indices)
            else:
                tokens = [self.info.index_to_token[idx] for idx in indices]
                if self.config.token_type == TokenType.CHAR:
                    return ''.join(tokens)
                else:
                    return ' '.join(tokens).replace(' <EOL> ', '\n')
        finally:
            clear_memory()
    
    def encode_tokens(self, text: str) -> List[int]:
        """Convert text to token indices."""
        try:
            if self.config.token_type == TokenType.BPE:
                return self.bpe_tokenizer.encode(text)
            elif self.config.token_type == TokenType.CHAR:
                tokens = list(text)
            else:
                tokens = text.split()
            
            return [self.info.token_to_index.get(token, 0) for token in tokens]  # 0 is UNK token
        finally:
            clear_memory()
    
    def get_train_batches(self):
        """Get training data batch indices."""
        return range(0, len(self.train_sequences), self.config.batch_size)
    
    def get_val_batches(self):
        """Get validation data batch indices."""
        return range(0, len(self.val_sequences), self.config.batch_size)
    
    @property
    def train_size(self) -> int:
        return self.info.train_size
    
    @property
    def val_size(self) -> int:
        return self.info.val_size
    
    @property
    def vocab_size(self) -> int:
        return self.info.vocab_size
    
    def __del__(self):
        """Clean up when the data loader is deleted."""
        clear_memory() 