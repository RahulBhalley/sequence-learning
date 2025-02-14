import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
import json
from tqdm import tqdm
from data_loader import ShakespeareDataLoader, DataConfig, TokenType
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
import time
import math
from torch.utils.tensorboard import SummaryWriter
import gc
import argparse
from accelerate import Accelerator
from accelerate.utils import set_seed

# Import TPU-specific modules conditionally
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    HAS_TPU = True
except ImportError:
    HAS_TPU = False

# Initialize accelerator with device selection
def init_accelerator(device_str: str = None, mixed_precision: str = 'no'):
    """Initialize accelerator with optional device override and mixed precision"""
    if device_str and device_str != 'auto':
        # Manual device override
        if device_str == 'cpu':
            device_placement_policy = 'cpu'
            # Force no mixed precision for CPU
            mixed_precision = 'no'
        elif device_str.startswith('cuda'):
            device_placement_policy = 'cuda'
            # Enable multi-GPU if available
            multi_gpu = torch.cuda.device_count() > 1
        elif device_str == 'mps':
            device_placement_policy = 'mps'
            # Force no mixed precision for MPS
            mixed_precision = 'no'
            multi_gpu = False
        elif device_str == 'tpu':
            device_placement_policy = 'tpu'
            # TPU supports bfloat16 by default
            if mixed_precision == 'no':
                mixed_precision = 'bf16'
            multi_gpu = False
        else:
            raise ValueError(f"Unsupported device: {device_str}")
        
        accelerator = Accelerator(
            device_placement_policy=device_placement_policy,
            mixed_precision=mixed_precision,
            # Enable TPU-specific optimizations
            dispatch_batches=device_str == 'tpu',
            # Enable multi-GPU if available
            split_batches=multi_gpu,
            gradient_accumulation_steps=model_config.gradient_accumulation_steps if 'model_config' in globals() else 8
        )
    else:
        # Let Accelerate automatically choose the best device
        # Check for multi-GPU
        multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
        accelerator = Accelerator(
            mixed_precision=mixed_precision,
            split_batches=multi_gpu,
            gradient_accumulation_steps=model_config.gradient_accumulation_steps if 'model_config' in globals() else 8
        )
    
    return accelerator, accelerator.device

# Initialize globals
accelerator = None
device = None

def initialize_globals(args):
    """Initialize global accelerator and device"""
    global accelerator, device
    accelerator, device = init_accelerator(args.device, args.mixed_precision)
    
    # Print distributed training info
    if accelerator.num_processes > 1:
        print(f"\nDistributed Training Configuration:")
        print(f"Number of GPUs: {accelerator.num_processes}")
        print(f"Process rank: {accelerator.process_index}")
        print(f"Local process rank: {accelerator.local_process_index}")
        print(f"Device placement: {accelerator.device}")
        if accelerator.distributed_type == "MULTI_GPU":
            print("Using DistributedDataParallel (DDP)")
    
    print(f"\nUsing device: {device}")
    print(f"Mixed precision mode: {args.mixed_precision}")
    return accelerator, device

def get_available_devices():
    """Get list of available devices for argparse choices"""
    devices = ['auto', 'cpu']
    
    # Check CUDA availability
    if torch.cuda.is_available():
        devices.extend([f'cuda:{i}' for i in range(torch.cuda.device_count())])
    
    # Check MPS availability
    if torch.backends.mps.is_available():
        devices.append('mps')
    
    # Check TPU availability
    try:
        import torch_xla.core.xla_model as xm
        if xm.xrt_world_size() > 0:
            devices.append('tpu')
    except ImportError:
        pass
    
    return devices

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Transformer model on Shakespeare text')
    
    # Device and distributed training settings
    parser.add_argument('--device', type=str, default='auto', 
                       choices=get_available_devices(),
                       help='Device to use (auto, cpu, cuda:N, or mps)')
    parser.add_argument('--multi_gpu', action='store_true',
                       help='Enable multi-GPU training using DistributedDataParallel')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training (automatically set by torch.distributed.launch)')
    
    # Mixed precision settings
    parser.add_argument('--mixed_precision', type=str, default='no',
                       choices=['no', 'fp16', 'bf16'],
                       help='Mixed precision mode (no, fp16, or bf16). Note: bf16 requires recent NVIDIA GPU')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Model architecture
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='Number of warmup steps')
    parser.add_argument('--clip_grad_norm', type=float, default=0.5, help='Gradient clipping norm')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Number of steps to accumulate gradients')
    parser.add_argument('--max_context_window', type=int, default=2048, help='Maximum context window size')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of training epochs')
    
    # Data configuration
    parser.add_argument('--context_window', type=int, default=128, help='Context window size for training')
    parser.add_argument('--token_type', type=str, default='bpe', choices=['char', 'word', 'bpe'], help='Tokenization type')
    parser.add_argument('--bpe_encoding', type=str, default='gpt2', help='BPE encoding type')
    
    # Generation parameters
    parser.add_argument('--gen_length', type=int, default=500, help='Length of text to generate during inference')
    parser.add_argument('--gen_every', type=int, default=10, help='Generate sample text every N epochs (0 to disable)')
    
    # Other parameters
    parser.add_argument('--save_dir', type=str, default='model_comparison', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Adjust batch size based on number of GPUs if multi-GPU is enabled
    if args.multi_gpu and torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"\nUsing {n_gpus} GPUs")
            print(f"Batch size per GPU: {args.batch_size}")
            print(f"Total batch size: {args.batch_size * n_gpus}")
            print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
            print(f"Effective total batch size: {args.batch_size * n_gpus * args.gradient_accumulation_steps}")
    
    return args

def clear_memory():
    """Clear memory cache and run garbage collection."""
    if torch.backends.mps.is_available():
        # MPS doesn't have explicit cache clearing like CUDA
        # but we can force synchronization
        torch.mps.synchronize()
    elif HAS_TPU and device.type == 'xla':
        # Synchronize TPU operations
        xm.mark_step()
    gc.collect()

batch_size = 4

@dataclass
class ModelConfig:
    # Transformer parameters
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    
    # Positional encoding
    max_context_window: int = 2048
    
    # Training parameters
    learning_rate: float = 0.0001
    batch_size: int = batch_size
    warmup_steps: int = 4000
    clip_grad_norm: float = 0.5
    gradient_accumulation_steps: int = 8  # Number of steps to accumulate gradients
    effective_batch_size: int = batch_size * gradient_accumulation_steps  # For reference
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Move to device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]

class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, config.d_model).to(device)
        self.pos_encoder = PositionalEncoding(config.d_model, config.max_context_window).to(device)
        
        # Transformer encoder with KV caching
        encoder_layers = []
        for _ in range(config.num_layers):
            layer = nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(
                    embed_dim=config.d_model,
                    num_heads=config.nhead,
                    dropout=config.dropout,
                    batch_first=True
                ).to(device),
                'norm1': nn.LayerNorm(config.d_model).to(device),
                'ff': nn.Sequential(
                    nn.Linear(config.d_model, config.dim_feedforward),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.dim_feedforward, config.d_model)
                ).to(device),
                'norm2': nn.LayerNorm(config.d_model).to(device),
                'dropout': nn.Dropout(config.dropout).to(device)
            })
            encoder_layers.append(layer)
        self.encoder_layers = nn.ModuleList(encoder_layers).to(device)
        
        # Output layer
        self.decoder = nn.Linear(config.d_model, vocab_size).to(device)
        
        # Initialize parameters
        self.init_weights()
        
        # KV cache
        self.kv_cache = None
        
        # Move entire model to device
        self.to(device)
    
    def init_weights(self) -> None:
        """Initialize weights following the Transformer paper recommendations.
        - Linear layers: Xavier/Glorot initialization for weights, zeros for biases
        - Embeddings: Normal distribution with mean=0, std=d_model^(-0.5)
        - Layer Norm: weight=1, bias=0
        """
        for p in self.parameters():
            if p.dim() > 1:
                # Linear weights
                nn.init.xavier_uniform_(p)
        
        # Special handling for embedding
        std = (self.config.d_model ** -0.5)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=std)
        
        # Initialize decoder bias to zeros
        if hasattr(self.decoder, 'bias') and self.decoder.bias is not None:
            nn.init.constant_(self.decoder.bias, 0.0)
        
        # Set LayerNorm weight to 1 and bias to 0
        for layer in self.encoder_layers:
            nn.init.constant_(layer['norm1'].weight, 1.0)
            nn.init.constant_(layer['norm1'].bias, 0.0)
            nn.init.constant_(layer['norm2'].weight, 1.0)
            nn.init.constant_(layer['norm2'].bias, 0.0)
            
            # Initialize attention output projection with smaller weights
            if hasattr(layer['self_attn'], 'out_proj'):
                nn.init.xavier_uniform_(layer['self_attn'].out_proj.weight, gain=0.1)
                if layer['self_attn'].out_proj.bias is not None:
                    nn.init.constant_(layer['self_attn'].out_proj.bias, 0.0)
    
    def init_kv_cache(self, batch_size: int):
        """Initialize KV cache for faster generation"""
        self.kv_cache = []
        for _ in range(len(self.encoder_layers)):
            layer_cache = {
                'key': torch.zeros(batch_size, 0, self.config.d_model, device=device),
                'value': torch.zeros(batch_size, 0, self.config.d_model, device=device)
            }
            self.kv_cache.append(layer_cache)
    
    def clear_kv_cache(self):
        """Clear the KV cache"""
        self.kv_cache = None
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, use_cache: bool = False) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len] for BPE tokens
                 or [batch_size, seq_len, vocab_size] for one-hot encoded tokens
            src_mask: Optional mask for padding
            use_cache: Whether to use KV cache during generation
        """
        # Initialize KV cache if needed
        if use_cache and self.kv_cache is None:
            self.init_kv_cache(src.size(0))
        
        # Handle both one-hot and index-based inputs
        if src.dim() == 3:  # One-hot encoded input
            # Get indices from one-hot
            src_indices = src.argmax(dim=-1)
            x = self.embedding(src_indices) * math.sqrt(self.config.d_model)
        else:  # Index-based input (BPE tokens)
            x = self.embedding(src) * math.sqrt(self.config.d_model)
        
        x = self.pos_encoder(x)
        
        # Process through encoder layers with KV caching
        for i, layer in enumerate(self.encoder_layers):
            # Self attention with optional KV cache
            if use_cache:
                # Compute attention only for new tokens
                q = layer['norm1'](x[:, -1:])  # Only last token for query
                k = v = layer['norm1'](x)  # All tokens for key/value
                
                # Concatenate with cached KV if available
                if self.kv_cache[i]['key'].size(1) > 0:
                    k = torch.cat([self.kv_cache[i]['key'], k], dim=1)
                    v = torch.cat([self.kv_cache[i]['value'], v], dim=1)
                
                # Update KV cache
                self.kv_cache[i]['key'] = k
                self.kv_cache[i]['value'] = v
                
                # Compute attention (no need for mask with KV cache)
                attn_output, _ = layer['self_attn'](q, k, v)
                x = x + layer['dropout'](attn_output)
            else:
                # Standard attention for training
                x = layer['norm1'](x)
                attn_output, _ = layer['self_attn'](x, x, x, attn_mask=src_mask)
                x = x + layer['dropout'](attn_output)
            
            # Feed forward
            x = x + layer['dropout'](layer['ff'](layer['norm2'](x)))
        
        # Decode
        output = self.decoder(x)
        
        return output
    
    def get_model_size(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

def generate_square_subsequent_mask(sz: int, batch_size: int = 1, num_heads: int = 8) -> torch.Tensor:
    """Generate causal mask for transformer.
    Args:
        sz: sequence length
        batch_size: batch size for broadcasting
        num_heads: number of attention heads
    """
    if sz == 1:  # Special case for single token
        return torch.zeros(batch_size * num_heads, 1, 1).to(device)
    
    # Create base mask
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    
    # Expand for batch size and number of heads
    mask = mask.unsqueeze(0).expand(batch_size * num_heads, -1, -1)
    
    # Move to device
    mask = mask.to(device)
    
    return mask

def generate_text(model: TransformerModel, 
                 data_loader: ShakespeareDataLoader,
                 start_text: Optional[str] = None,
                 length: int = 500,
                 temperature: float = 0.8) -> None:
    """Generate text using the trained model with single GPU optimization"""
    # Only generate text on the main process
    if not accelerator.is_local_main_process:
        return
    
    # Get the underlying model if using DDP
    unwrapped_model = accelerator.unwrap_model(model)
    
    # Move model to CPU or first GPU for generation
    if torch.cuda.is_available():
        generation_device = torch.device('cuda:0')
    else:
        generation_device = torch.device('cpu')
    
    # Store original device for model restoration
    original_device = next(unwrapped_model.parameters()).device
    
    try:
        # Move model to generation device
        unwrapped_model = unwrapped_model.to(generation_device)
        unwrapped_model.eval()
        
        # Clear any existing KV cache
        unwrapped_model.clear_kv_cache()
        
        # Initialize with start text or random token
        if start_text is None:
            current_idx = torch.randint(0, data_loader.vocab_size, (1,)).item()
            first_token = data_loader.decode_tokens([current_idx])
            print(f"\nStarting token: '{first_token}'", end='', flush=True)
            current_indices = [current_idx]
        else:
            current_indices = data_loader.encode_tokens(start_text)
            print(f"\n{start_text}", end='', flush=True)
        
        # Convert to tensor and move to generation device
        if data_loader.config.token_type == TokenType.BPE:
            current_sequence = torch.tensor(current_indices, dtype=torch.long).unsqueeze(0).to(generation_device)
        else:
            # Create one-hot encoded tensor for char/word tokens
            x = torch.zeros(1, len(current_indices), data_loader.vocab_size, device=generation_device)
            x[0, range(len(current_indices)), current_indices] = 1
            current_sequence = x
        
        with torch.no_grad():
            # Process the initial sequence to build up the KV cache
            initial_mask = generate_square_subsequent_mask(current_sequence.size(1)).to(generation_device)
            output = unwrapped_model(current_sequence, initial_mask, use_cache=True)
            
            # Generate new tokens one at a time
            for i in range(length):
                if data_loader.config.token_type == TokenType.BPE:
                    # For BPE, use the last token index directly
                    next_input = current_sequence[:, -1:]
                else:
                    # For char/word, use one-hot of last token
                    next_input = current_sequence[:, -1:, :]
                
                output = unwrapped_model(next_input, None, use_cache=True)
                next_token_logits = output[0, -1, :] / temperature
                next_token_probs = torch.softmax(next_token_logits, dim=0)
                
                # Sample from the distribution
                next_token = torch.multinomial(next_token_probs, 1).item()
                
                # Ensure synchronization based on device type
                if generation_device.type == 'cuda':
                    torch.cuda.synchronize()
                elif generation_device.type == 'mps':
                    torch.mps.synchronize()
                
                # Decode and print the new token immediately
                new_token = data_loader.decode_tokens([next_token])
                print(new_token, end='', flush=True)
                
                # Append to sequence
                if data_loader.config.token_type == TokenType.BPE:
                    next_token_tensor = torch.tensor([[next_token]], device=generation_device)
                    current_sequence = torch.cat([current_sequence, next_token_tensor], dim=1)
                else:
                    next_token_tensor = torch.zeros(1, 1, data_loader.vocab_size, device=generation_device)
                    next_token_tensor[0, 0, next_token] = 1
                    current_sequence = torch.cat([current_sequence, next_token_tensor], dim=1)
                
                # Periodically clear memory (but not KV cache)
                if i % 100 == 0:
                    clear_memory()
        
        print()  # New line at the end
        
    finally:
        # Clean up
        unwrapped_model.clear_kv_cache()
        # Move model back to original device
        unwrapped_model = unwrapped_model.to(original_device)
        clear_memory()

class NoamLRScheduler:
    """
    Learning rate scheduler with warmup and decay as described in
    'Attention is All You Need' paper. Learning rate increases linearly
    during warmup and then decays proportionally to the inverse square
    root of the step number.
    """
    def __init__(self, optimizer: optim.Optimizer, d_model: int, warmup_steps: int, factor: float = 1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0
        self._rate = 0
    
    def step(self):
        """Update parameters and learning rate"""
        self._step += 1
        rate = self._get_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        
    def _get_lr(self):
        """Calculate learning rate according to formula"""
        step = self._step
        return self.factor * (self.d_model ** (-0.5) *
                min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))
    
    def get_last_lr(self):
        """Return last computed learning rate"""
        return [self._rate]
    
    def state_dict(self):
        """Return scheduler state for checkpointing"""
        return {
            'step': self._step,
            'rate': self._rate
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint"""
        self._step = state_dict['step']
        self._rate = state_dict['rate']

def train_epoch(model: TransformerModel,
                data_loader: ShakespeareDataLoader,
                optimizer: optim.Optimizer,
                scheduler: Any,
                criterion: nn.Module,
                clip_grad_norm: float) -> float:
    """Train for one epoch with gradient accumulation and memory optimization"""
    model.train()
    total_loss = 0
    
    # Get config from model (handle DDP case)
    model_config = model.module.config if hasattr(model, 'module') else model.config
    
    # Calculate total steps for this epoch
    total_batches = len(data_loader.get_train_batches())
    accumulation_steps = model_config.gradient_accumulation_steps
    
    # Move criterion to device (will be handled by accelerator)
    criterion = accelerator.prepare(criterion)
    
    # Create progress bar (only show on main process)
    pbar = tqdm(data_loader.get_train_batches(), 
                desc='Training', 
                disable=not accelerator.is_local_main_process)
    
    # Wrap data loader with TPU parallel loader if using TPU
    if HAS_TPU and device.type == 'xla':
        pbar = pl.ParallelLoader(pbar, [device]).per_device_loader(device)
    
    for batch_idx, batch_num in enumerate(pbar):
        try:
            # Get batch
            x, y = data_loader.get_batch(data_loader.train_sequences, 
                                       data_loader.train_targets, 
                                       batch_num)
            if x is None:  # Skip incomplete batches
                continue
            
            # Move tensors to appropriate device (handled by accelerator)
            x, y = accelerator.prepare(x, y)
            
            # Create mask for the batch with correct number of heads
            mask = generate_square_subsequent_mask(x.size(1), x.size(0), model_config.nhead)
            mask = accelerator.prepare(mask)
            
            # Forward pass
            with accelerator.accumulate(model):
                if data_loader.config.token_type == TokenType.BPE:
                    output = model(x, mask)  # x is already indices for BPE
                else:
                    x_indices = x.argmax(dim=-1)
                    output = model(x_indices, mask)  # Convert one-hot to indices
                
                output = output.view(-1, data_loader.vocab_size)
                y = y.view(-1)
                loss = criterion(output, y)
                
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), clip_grad_norm)
                
                optimizer.step()
                scheduler.step()  # Step after each effective batch
                optimizer.zero_grad()
            
            # Get current learning rate for display
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
            
            # Update stats (scale loss back up for reporting)
            total_loss += loss.item() * accumulation_steps
            
            # Only update progress bar on main process
            if accelerator.is_local_main_process:
                pbar.set_postfix({
                    'loss': f'{loss.item() * accumulation_steps:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
            
            # TPU-specific: mark step for XLA compilation
            if HAS_TPU and device.type == 'xla':
                xm.mark_step()
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
            
        finally:
            # Clean up tensors
            if 'output' in locals():
                del output
            if 'loss' in locals():
                del loss
            if 'mask' in locals():
                del mask
            if (batch_idx + 1) % 10 == 0:  # Clear memory less frequently
                clear_memory()
    
    # Gather and average loss across all processes
    total_loss = accelerator.gather(torch.tensor(total_loss, device=device)).mean().item()
    return total_loss / len(data_loader.get_train_batches())

def validate(model: TransformerModel,
            data_loader: ShakespeareDataLoader,
            criterion: nn.Module) -> float:
    """Validate the model"""
    model.eval()
    total_loss = 0
    batch_size = data_loader.config.batch_size * 2  # Use larger batches for validation
    
    # Get the underlying model if using DDP
    unwrapped_model = accelerator.unwrap_model(model)
    
    # Move criterion to device (will be handled by accelerator)
    criterion = accelerator.prepare(criterion)
    
    with torch.no_grad():
        for batch_idx in data_loader.get_val_batches():
            try:
                # Get batch
                x, y = data_loader.get_batch(data_loader.val_sequences, 
                                           data_loader.val_targets, 
                                           batch_idx)
                if x is None:  # Skip incomplete batches
                    continue
                
                # Move tensors to appropriate device
                x, y = accelerator.prepare(x, y)
                
                # Create mask for the batch with correct number of heads
                mask = generate_square_subsequent_mask(x.size(1), x.size(0), unwrapped_model.config.nhead)
                mask = accelerator.prepare(mask)
                
                # Forward pass
                if data_loader.config.token_type == TokenType.BPE:
                    output = model(x, mask)  # x is already indices for BPE
                else:
                    output = model(x.argmax(dim=-1), mask)  # Convert one-hot to indices
                
                output = output.view(-1, data_loader.vocab_size)
                y = y.view(-1)
                loss = criterion(output, y)
                
                # Gather loss from all processes
                loss = accelerator.gather(loss).mean()
                
                total_loss += loss.item() * (x.size(0) / batch_size)
                
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue
            
            finally:
                # Clean up tensors
                if 'output' in locals():
                    del output
                if 'loss' in locals():
                    del loss
                if 'mask' in locals():
                    del mask
                if batch_idx % (batch_size * 5) == 0:  # Clear memory less frequently
                    clear_memory()
    
    return total_loss / len(data_loader.get_val_batches())

# Reuse the checkpoint management functions from rnn-text-gen.py
def find_latest_checkpoint(save_dir: str) -> Optional[str]:
    """Find the latest checkpoint in the save directory."""
    if not os.path.exists(save_dir):
        return None
        
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    if not checkpoints:
        return None
        
    # Extract epoch numbers and find the latest
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(save_dir, latest_checkpoint)

def save_checkpoint(model: TransformerModel, 
                   optimizer: optim.Optimizer,
                   scheduler: Any,
                   epoch: int,
                   train_loss: float,
                   val_loss: float,
                   save_path: str,
                   data_config: DataConfig,
                   history: Dict[str, List] = None):
    """Save checkpoint with enhanced information."""
    # Get the underlying model if using DDP
    unwrapped_model = accelerator.unwrap_model(model)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scheduler_type': type(scheduler).__name__ if scheduler is not None else None,
        'scheduler_config': {
            'warmup_steps': scheduler.warmup_steps if isinstance(scheduler, NoamLRScheduler) else None,
            'factor': scheduler.factor if isinstance(scheduler, NoamLRScheduler) else None,
            'd_model': scheduler.d_model if isinstance(scheduler, NoamLRScheduler) else None,
            'last_lr': scheduler.get_last_lr()[0] if scheduler is not None else None,
            'step': scheduler._step if isinstance(scheduler, NoamLRScheduler) else None
        } if scheduler is not None else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_config': unwrapped_model.config.to_dict(),
        'data_config': data_config.to_dict(),
        'model_size': unwrapped_model.get_model_size(),
        'timestamp': datetime.now().isoformat(),
        'history': history,
        'learning_rate': optimizer.param_groups[0]['lr']  # Current learning rate
    }
    
    torch.save(checkpoint, save_path)
    
    # Save a human-readable summary
    summary_path = save_path.replace('.pt', '_summary.json')
    summary = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_config': unwrapped_model.config.to_dict(),
        'data_config': data_config.to_dict(),
        'model_size': unwrapped_model.get_model_size(),
        'timestamp': checkpoint['timestamp'],
        'learning_rate': checkpoint['learning_rate'],
        'scheduler_type': checkpoint['scheduler_type'],
        'scheduler_config': checkpoint['scheduler_config']
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

def load_checkpoint(path: str,
                   model: TransformerModel,
                   optimizer: Optional[optim.Optimizer] = None,
                   scheduler: Optional[Any] = None) -> dict:
    """Load checkpoint with enhanced information."""
    # Load checkpoint to CPU first to avoid device mismatch
    checkpoint = torch.load(path, map_location='cpu')
    
    # Move model state dict to the correct device before loading
    model_state = {k: v.to(device) for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(model_state)
    
    if optimizer is not None:
        # Move optimizer state to correct device
        optimizer_state = checkpoint['optimizer_state_dict']
        for state in optimizer_state['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        optimizer.load_state_dict(optimizer_state)
    
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load scheduler state: {e}")
            # Reconstruct scheduler state if possible
            if isinstance(scheduler, NoamLRScheduler):
                scheduler_config = checkpoint.get('scheduler_config', {})
                if scheduler_config:
                    scheduler._step = scheduler_config.get('step', 0)
                    scheduler._rate = scheduler_config.get('last_lr', scheduler._rate)
                    scheduler.warmup_steps = scheduler_config.get('warmup_steps', scheduler.warmup_steps)
                    scheduler.factor = scheduler_config.get('factor', scheduler.factor)
                    scheduler.d_model = scheduler_config.get('d_model', scheduler.d_model)
    
    return checkpoint

def resume_or_start_training(model: TransformerModel,
                           data_loader: ShakespeareDataLoader,
                           optimizer: optim.Optimizer,
                           scheduler: Any,
                           criterion: nn.Module,
                           n_epochs: int,
                           save_dir: str = 'checkpoints',
                           gen_every: int = 1) -> Dict[str, Any]:
    """Resume training from latest checkpoint or start fresh."""
    latest_checkpoint = find_latest_checkpoint(save_dir)
    start_epoch = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    if latest_checkpoint:
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        checkpoint = load_checkpoint(latest_checkpoint, model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint.get('history', history)
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("Starting fresh training...")
    
    return train_model(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        n_epochs=n_epochs,
        save_dir=save_dir,
        start_epoch=start_epoch,
        initial_history=history,
        gen_every=gen_every
    )

def train_model(model: TransformerModel,
                data_loader: ShakespeareDataLoader,
                optimizer: optim.Optimizer,
                scheduler: Any,
                criterion: nn.Module,
                n_epochs: int,
                save_dir: str = 'checkpoints',
                start_epoch: int = 0,
                initial_history: Optional[Dict] = None,
                gen_every: int = 1):
    """Main training loop with memory optimization"""
    print("Starting training...")
    os.makedirs(save_dir, exist_ok=True)
    
    # Get config from model (handle DDP case)
    model_config = model.module.config if hasattr(model, 'module') else model.config
    
    try:
        # Initialize TensorBoard writer
        tensorboard_dir = os.path.join(save_dir, 'tensorboard')
        writer = SummaryWriter(tensorboard_dir)
        
        best_val_loss = float('inf')
        no_improve_count = 0
        start_time = time.time()
        final_epoch = start_epoch
        
        # Training history
        history = initial_history or {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        for epoch in range(start_epoch, n_epochs):
            final_epoch = epoch
            
            # Train and validate
            train_loss = train_epoch(model, data_loader, optimizer, scheduler, criterion, model_config.clip_grad_norm)
            clear_memory()  # Clear memory after training
            
            val_loss = validate(model, data_loader, criterion)
            clear_memory()  # Clear memory after validation
            
            # Update history and logging
            current_lr = optimizer.param_groups[0]['lr']
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rates'].append(current_lr)
            
            # Log metrics to TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Save checkpoints and handle early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
                
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    save_path=checkpoint_path,
                    data_config=data_loader.config,
                    history=history
                )
                clear_memory()  # Clear memory after saving checkpoint
            
            # Generate sample text periodically if enabled
            if gen_every > 0 and epoch % gen_every == 0:
                print("\nGenerated text sample:")
                print("-" * 50)
                start_text = 'T' if data_loader.config.token_type == TokenType.CHAR else 'The'
                generate_text(model, data_loader, start_text=start_text)
                print("-" * 50)
                clear_memory()  # Clear memory after text generation
            
            # Print progress
            print(f'Epoch {epoch+1}/{n_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f} (best: {best_val_loss:.4f})')
            print(f'  Learning Rate: {current_lr:.6f}')
            
    finally:
        # Clean up
        writer.close()
        clear_memory()
    
    return {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'training_time': time.time() - start_time,
        'history': history,
        'epochs_without_improvement': no_improve_count
    }

def compare_schedulers(model: TransformerModel,
                    data_loader: ShakespeareDataLoader,
                    base_save_dir: str = 'scheduler_comparison',
                    n_epochs: int = 50) -> Dict[str, Any]:
    """Compare different learning rate schedulers for Transformer."""
    results = {}
    
    # Create TensorBoard writer for comparison
    comparison_writer = SummaryWriter(os.path.join(base_save_dir, 'tensorboard_comparison'))
    
    # Define scheduler configurations to compare
    scheduler_configs = {
        'no_scheduler': {
            'type': 'none',
            'params': {}
        },
        'noam': {  # Original Transformer scheduler
            'type': 'noam',
            'params': {
                'model_size': model.config.d_model,
                'warmup_steps': 4000,
                'factor': 2.0
            }
        },
        'cosine_warm_restarts': {
            'type': 'cosine_warm_restarts',
            'params': {
                'T_0': len(data_loader.get_train_batches()) * 5,
                'T_mult': 2,
                'eta_min': model.config.learning_rate * 0.01
            }
        },
        'one_cycle': {
            'type': 'one_cycle',
            'params': {
                'max_lr': model.config.learning_rate * 10,
                'epochs': n_epochs,
                'steps_per_epoch': len(data_loader.get_train_batches()),
                'pct_start': 0.3,
                'anneal_strategy': 'cos'
            }
        },
        'reduce_on_plateau': {
            'type': 'reduce_on_plateau',
            'params': {
                'mode': 'min',
                'factor': 0.5,
                'patience': 3,
                'min_lr': model.config.learning_rate * 0.001
            }
        }
    }
    
    for scheduler_name, config in scheduler_configs.items():
        print(f"\nTraining with {scheduler_name}...")
        save_dir = os.path.join(base_save_dir, scheduler_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Create fresh model instance
        model_copy = TransformerModel(
            vocab_size=data_loader.vocab_size,
            config=model.config
        ).to(device)
        model_copy.load_state_dict(model.state_dict())  # Copy weights
        
        # Setup optimizer
        optimizer = optim.Adam(
            model_copy.parameters(), 
            lr=model.config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Setup scheduler based on configuration
        if config['type'] == 'none':
            scheduler = None
        elif config['type'] == 'noam':
            scheduler = NoamLRScheduler(optimizer, **config['params'])
        elif config['type'] == 'cosine_warm_restarts':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **config['params'])
        elif config['type'] == 'one_cycle':
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, **config['params'])
        elif config['type'] == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config['params'])
        
        criterion = nn.CrossEntropyLoss()
        
        # Train model
        model_results = train_model(
            model=model_copy,
            data_loader=data_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            n_epochs=n_epochs,
            save_dir=save_dir
        )
        
        # Generate sample text
        print(f"\nGenerating sample text for {scheduler_name}:")
        print("-" * 50)
        generate_text(
            model=model_copy,
            data_loader=data_loader,
            start_text='The' if data_loader.config.token_type == TokenType.WORD else 'T',
            length=500
        )
        print("-" * 50)
        
        # Log comparison metrics to TensorBoard
        comparison_writer.add_scalar(f'Best_Val_Loss/{scheduler_name}', model_results['best_val_loss'], 0)
        comparison_writer.add_scalar(f'Training_Time/{scheduler_name}', model_results['training_time'], 0)
        
        # Plot learning rate curve
        if model_results['history']['learning_rates']:
            comparison_writer.add_figure(
                f'Learning_Rate_Curve/{scheduler_name}',
                plot_learning_rate_curve(model_results['history']['learning_rates']),
                0
            )
        
        # Save results
        results[scheduler_name] = {
            'scheduler_config': config,
            'final_train_loss': model_results['train_loss'],
            'final_val_loss': model_results['val_loss'],
            'best_val_loss': model_results['best_val_loss'],
            'training_time': model_results['training_time'],
            'history': model_results['history']
        }
    
    # Save comparison results
    with open(os.path.join(base_save_dir, 'scheduler_comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Close comparison writer
    comparison_writer.close()
    
    return results

def plot_learning_rate_curve(learning_rates):
    """Create a matplotlib figure of the learning rate curve."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(learning_rates)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True)
    
    return fig

def compare_models(data_config: DataConfig, 
                data_loader: ShakespeareDataLoader,
                model_configs: List[ModelConfig], 
                n_epochs: int = 10,
                base_save_dir: str = 'model_comparison') -> Dict[str, Any]:
    """Compare different Transformer architectures with TensorBoard support."""
    results = {}
    
    # Create TensorBoard writer for comparison
    comparison_writer = SummaryWriter(os.path.join(base_save_dir, 'tensorboard_comparison'))
    
    for config in model_configs:
        # Create save directory for this model
        model_name = f"d{config.d_model}_h{config.nhead}_l{config.num_layers}"
        if config.dim_feedforward != config.d_model * 4:  # Non-standard feedforward dim
            model_name += f"_ff{config.dim_feedforward}"
        
        print(f"\nTraining {model_name}...")
        print(f"Batch size: {config.batch_size}, Gradient accumulation steps: {config.gradient_accumulation_steps}")
        print(f"Effective batch size: {config.effective_batch_size}")
        
        save_dir = os.path.join(base_save_dir, model_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump({
                'model_config': config.to_dict(),
                'data_config': data_config.to_dict()
            }, f, indent=2)
        
        # Create model
        model = TransformerModel(
            vocab_size=data_loader.vocab_size,
            config=config
        ).to(device)
        
        # Setup optimizer with Transformer-specific parameters
        optimizer = optim.Adam(
            model.parameters(),
            lr=1e-7,  # Start with very small learning rate for Noam scheduler
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Use Noam scheduler (original Transformer scheduler)
        scheduler = NoamLRScheduler(
            optimizer,
            d_model=config.d_model,
            warmup_steps=config.warmup_steps,
            factor=2.0
        )
        
        # Force an initial scheduler step to set the starting learning rate
        scheduler.step()
        
        criterion = nn.CrossEntropyLoss()
        
        # Train model
        print(f"\nTraining {model_name}...")
        model_results = resume_or_start_training(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            n_epochs=n_epochs,
            save_dir=save_dir
        )
        
        # Generate sample text
        print(f"\nGenerating sample text for {model_name}:")
        print("-" * 50)
        generate_text(
            model=model,
            data_loader=data_loader,
            start_text='The' if data_config.token_type == TokenType.WORD else 'T',
            length=500
        )
        print("-" * 50)
        
        # Log comparison metrics to TensorBoard
        comparison_writer.add_scalar(f'Best_Val_Loss/{model_name}', model_results['best_val_loss'], 0)
        comparison_writer.add_scalar(f'Training_Time/{model_name}', model_results['training_time'], 0)
        comparison_writer.add_scalar(f'Model_Size/{model_name}', model.get_model_size(), 0)
        
        # Save results
        results[model_name] = {
            'config': config.to_dict(),
            'data_config': data_config.to_dict(),
            'final_train_loss': model_results['train_loss'],
            'final_val_loss': model_results['val_loss'],
            'best_val_loss': model_results['best_val_loss'],
            'training_time': model_results['training_time'],
            'model_size': model.get_model_size()
        }
        
        # Save comparison results
        with open(os.path.join(base_save_dir, 'comparison_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    # Close comparison writer
    comparison_writer.close()
    
    return results

def main():
    """Main function to run the training"""
    args = parse_args()
    
    # Initialize accelerator and device first
    accelerator, device = initialize_globals(args)
    
    # Set random seed
    set_seed(args.seed)
    
    # Data configuration
    data_config = DataConfig(
        token_type=TokenType[args.token_type.upper()],
        context_window=args.context_window,
        batch_size=args.batch_size,
        bpe_encoding=args.bpe_encoding
    )
    
    # Load data with explicit device
    data_loader = ShakespeareDataLoader(data_config, device=device)
    if accelerator.is_local_main_process:
        print(f"Vocabulary size: {data_loader.vocab_size}")
        print(f"Training sequences: {data_loader.train_size}")
        print(f"Validation sequences: {data_loader.val_size}")
    
    # Create model configuration
    model_config = ModelConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_context_window=args.max_context_window,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        clip_grad_norm=args.clip_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Create model
    model = TransformerModel(
        vocab_size=data_loader.vocab_size,
        config=model_config
    )
    
    # Get model size before DDP wrapping
    model_size = model.get_model_size()
    
    # Setup optimizer with Transformer-specific parameters
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-7,  # Start with very small learning rate for Noam scheduler
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # Use Noam scheduler (original Transformer scheduler)
    scheduler = NoamLRScheduler(
        optimizer,
        d_model=model_config.d_model,
        warmup_steps=model_config.warmup_steps,
        factor=2.0
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Prepare for distributed training with accelerate and mixed precision
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    
    # Force an initial scheduler step to set the starting learning rate
    scheduler.step()
    
    # Print training configuration only on main process
    if accelerator.is_local_main_process:
        print("\nTraining Configuration:")
        print(f"Device: {device}")
        print(f"Mixed Precision: {args.mixed_precision}")
        print(f"Model Parameters: {model_size:,}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
        print(f"Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps}")
        print(f"Generate text every {args.gen_every} epochs" if args.gen_every > 0 else "Text generation during training disabled")
    
    # Train model
    if accelerator.is_local_main_process:
        print("\nStarting training...")
    model_results = resume_or_start_training(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        n_epochs=args.n_epochs,
        save_dir=args.save_dir,
        gen_every=args.gen_every
    )
    
    # Print final results only on main process
    if accelerator.is_local_main_process:
        print("\nTraining completed!")
        print(f"Best validation loss: {model_results['best_val_loss']:.4f}")
        print(f"Training time: {model_results['training_time']:.2f} seconds")
        
        # Generate final sample
        print("\nGenerating final sample:")
        print("-" * 50)
        # Get the unwrapped model for generation
        unwrapped_model = accelerator.unwrap_model(model)
        generate_text(
            model=unwrapped_model,
            data_loader=data_loader,
            start_text='The' if data_config.token_type == TokenType.WORD else 'T',
            length=args.gen_length
        )
        print("-" * 50)

if __name__ == '__main__':
    main()
