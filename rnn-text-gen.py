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
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import math

# Set device
# device = torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class RNNType(Enum):
    RNN = 'rnn'
    LSTM = 'lstm'
    GRU = 'gru'

@dataclass
class ModelConfig:
    # Basic parameters
    rnn_type: RNNType
    hidden_size: int
    num_layers: int
    dropout: float
    
    # Architecture-specific parameters
    bidirectional: bool = False
    
    # LSTM-specific parameters
    lstm_proj_size: Optional[int] = None
    
    # RNN-specific parameters
    nonlinearity: str = 'tanh'  # 'tanh' or 'relu'
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    clip_grad_norm: float = 0.5
    gradient_accumulation_steps: int = 4  # Number of steps to accumulate gradients
    effective_batch_size: int = batch_size * gradient_accumulation_steps  # For reference
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary with enum handling."""
        config_dict = asdict(self)
        config_dict['rnn_type'] = self.rnn_type.value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary with enum handling."""
        config_dict['rnn_type'] = RNNType(config_dict['rnn_type'])
        return cls(**config_dict)

@dataclass
class DataConfig:
    token_type: TokenType
    seq_length: int
    batch_size: int
    bpe_encoding: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary with enum handling."""
        return {
            'token_type': self.token_type.value,
            'seq_length': self.seq_length,
            'batch_size': self.batch_size,
            'bpe_encoding': self.bpe_encoding
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataConfig':
        """Create config from dictionary with enum handling."""
        config_dict['token_type'] = TokenType(config_dict['token_type'])
        return cls(**config_dict)

def load_shakespeare_data():
    """Load and return the Shakespeare dataset and metadata."""
    data_dir = os.path.join(os.path.dirname(__file__), 'data/shakespeare_char')
    
    # Load the metadata
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    # Load training and validation data
    train_data = np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint16)
    val_data = np.fromfile(os.path.join(data_dir, 'val.bin'), dtype=np.uint16)
    
    return train_data, val_data, meta

def create_sequences(data, seq_length):
    """Create input-target sequences for training"""
    sequences = []
    targets = []
    for i in range(0, len(data) - seq_length, seq_length):
        seq = data[i:i + seq_length]
        target = data[i + 1:i + seq_length + 1]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def generate_text(model: RNNModel, 
                 data_loader: ShakespeareDataLoader,
                 start_text: Optional[str] = None,
                 length: int = 250,
                 temperature: float = 0.8) -> None:
    """Generate text using the trained model"""
    model.eval()
    
    # Initialize with start text or random token
    if start_text is None:
        current_idx = torch.randint(0, data_loader.vocab_size, (1,)).item()
        first_token = data_loader.decode_tokens([current_idx])
        print(f"\nStarting token: '{first_token}'", end='', flush=True)
    else:
        indices = data_loader.encode_tokens(start_text)
        current_idx = indices[0]
        print(f"\n{start_text}", end='', flush=True)
    
    # Initialize hidden state
    hidden = model.init_hidden(batch_size=1)
    
    # Generate tokens
    with torch.no_grad():
        for i in range(length):
            # Prepare input
            if data_loader.config.token_type == TokenType.BPE:
                # For BPE, use token index directly
                x = torch.tensor([[current_idx]], dtype=torch.long, device=device)
            else:
                # For char/word tokens, use one-hot encoding
                x = torch.zeros(1, 1, data_loader.vocab_size, device=device)
                x[0, 0, current_idx] = 1
            
            # Forward pass
            output, hidden = model(x, hidden)
            
            # Apply temperature
            output = output.squeeze() / temperature
            probs = torch.softmax(output, dim=0)
            
            # Sample from the distribution
            current_idx = torch.multinomial(probs, 1).item()
            
            # Ensure synchronization before decoding
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
            
            # Decode and print the new token immediately
            new_token = data_loader.decode_tokens([current_idx])
            print(new_token, end='', flush=True)
    
    print()  # New line at the end

class RNNModel(nn.Module):
    def __init__(self, input_size: int, config: ModelConfig, output_size: int):
        super(RNNModel, self).__init__()
        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        
        # Add embedding layer for BPE tokens
        self.embedding = nn.Embedding(input_size, config.hidden_size)
        
        # Select RNN cell type
        rnn_cells = {
            RNNType.RNN: nn.RNN,
            RNNType.LSTM: nn.LSTM,
            RNNType.GRU: nn.GRU
        }
        rnn_cell = rnn_cells[config.rnn_type]
        
        # Prepare RNN kwargs based on cell type
        rnn_kwargs = {
            'input_size': config.hidden_size,  # Use hidden_size as input size since we're using embeddings
            'hidden_size': config.hidden_size,
            'num_layers': config.num_layers,
            'dropout': config.dropout if config.num_layers > 1 else 0,
            'batch_first': True,
            'bidirectional': config.bidirectional
        }
        
        # Add cell-specific parameters
        if config.rnn_type == RNNType.RNN:
            rnn_kwargs['nonlinearity'] = config.nonlinearity
        elif config.rnn_type == RNNType.LSTM and config.lstm_proj_size:
            rnn_kwargs['proj_size'] = config.lstm_proj_size
        
        # Main RNN layer
        self.rnn = rnn_cell(**rnn_kwargs)
        
        # Adjust output size for bidirectional
        output_multiplier = 2 if config.bidirectional else 1
        effective_hidden_size = config.lstm_proj_size or config.hidden_size
        
        # Output layer
        self.decoder = nn.Linear(effective_hidden_size * output_multiplier, output_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input: torch.Tensor, hidden: Any) -> Tuple[torch.Tensor, Any]:
        """
        Args:
            input: Tensor, shape [batch_size, seq_len] for BPE tokens
                  or [batch_size, seq_len, vocab_size] for one-hot encoded tokens
            hidden: Initial hidden state
        """
        # Handle both one-hot and index-based inputs
        if input.dim() == 3:  # One-hot encoded input
            # Project one-hot input to hidden size
            input_indices = input.argmax(dim=-1)
            x = self.embedding(input_indices)
        else:  # Index-based input (BPE tokens)
            x = self.embedding(input)
        
        # Apply dropout to embeddings
        x = self.dropout(x)
        
        # RNN forward pass
        if isinstance(hidden, tuple):  # LSTM has cell state
            output, (hidden, cell) = self.rnn(x, hidden)
            hidden = (hidden, cell)
        else:  # RNN/GRU
            output, hidden = self.rnn(x, hidden)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Reshape output for decoder if necessary
        if len(output.shape) == 3:
            batch_size, seq_len, hidden_size = output.shape
            output = output.reshape(-1, hidden_size)
        
        # Decode
        output = self.decoder(output)
        
        return output, hidden

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        num_directions = 2 if self.config.bidirectional else 1
        effective_hidden_size = self.config.lstm_proj_size or self.config.hidden_size
        hidden_size = (self.config.num_layers * num_directions, batch_size, effective_hidden_size)
        
        if self.config.rnn_type == RNNType.LSTM:
            return (torch.zeros(*hidden_size, device=device),
                   torch.zeros(*hidden_size, device=device))
        else:
            return torch.zeros(*hidden_size, device=device)
    
    def get_model_size(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

def get_batch(sequences, targets, batch_idx, batch_size, vocab_size):
    """Create one-hot encoded batches"""
    # Get batch data
    batch_sequences = sequences[batch_idx:batch_idx + batch_size]
    batch_targets = targets[batch_idx:batch_idx + batch_size]
    
    # Convert to one-hot encoded tensors
    x = torch.zeros(batch_sequences.shape[0], batch_sequences.shape[1], vocab_size, device=device)
    for i in range(batch_sequences.shape[0]):
        x[i, range(batch_sequences.shape[1]), batch_sequences[i]] = 1
    
    y = torch.tensor(batch_targets.reshape(-1), dtype=torch.long, device=device)
    
    return x, y

class CyclicCosineScheduler:
    """
    Cyclical Learning Rate Scheduler with Cosine Annealing.
    Features:
    - Cosine annealing within each cycle
    - Optional cycle length multiplier for progressive cycle length increase
    - Learning rate decay across cycles
    - Warm restarts at the beginning of each cycle
    """
    def __init__(self, 
                 optimizer: optim.Optimizer,
                 init_lr: float,
                 min_lr: float,
                 cycle_length: int,
                 cycle_mult: float = 1.0,
                 decay: float = 1.0):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.cycle_length = cycle_length
        self.cycle_mult = cycle_mult
        self.decay = decay
        self._step = 0
        self._cycle = 0
        self._rate = init_lr
        
        # Initialize optimizer with initial learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = init_lr
    
    def step(self):
        """Update parameters and learning rate"""
        # Calculate current cycle and position within cycle
        cycle_length = int(self.cycle_length * (self.cycle_mult ** self._cycle))
        cycle_position = self._step % cycle_length
        
        # Check if we're starting a new cycle
        if cycle_position == 0 and self._step > 0:
            self._cycle += 1
        
        # Calculate cosine-annealed learning rate
        cosine_position = cycle_position / cycle_length
        cosine_value = 0.5 * (1 + math.cos(math.pi * cosine_position))
        
        # Apply cycle decay
        cycle_decay = self.decay ** self._cycle
        
        # Calculate new learning rate
        lr_range = (self.init_lr * cycle_decay - self.min_lr)
        self._rate = self.min_lr + lr_range * cosine_value
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._rate
        
        self._step += 1
    
    def get_last_lr(self):
        """Return last computed learning rate"""
        return [self._rate]
    
    def state_dict(self):
        """Return scheduler state for checkpointing"""
        return {
            'step': self._step,
            'cycle': self._cycle,
            'rate': self._rate,
            'init_lr': self.init_lr,
            'min_lr': self.min_lr,
            'cycle_length': self.cycle_length,
            'cycle_mult': self.cycle_mult,
            'decay': self.decay
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint with backward compatibility."""
        # Handle legacy checkpoints that might have different formats
        if isinstance(state_dict, dict):
            # New format with full state
            if 'step' in state_dict:
                self._step = state_dict['step']
                self._cycle = state_dict.get('cycle', 0)
                self._rate = state_dict.get('rate', self.init_lr)
                
                # Optional parameters with defaults
                self.init_lr = state_dict.get('init_lr', self.init_lr)
                self.min_lr = state_dict.get('min_lr', self.min_lr)
                self.cycle_length = state_dict.get('cycle_length', self.cycle_length)
                self.cycle_mult = state_dict.get('cycle_mult', self.cycle_mult)
                self.decay = state_dict.get('decay', self.decay)
            # Legacy format with just the learning rate
            elif 'base_lrs' in state_dict:
                self._rate = state_dict['base_lrs'][0]
            else:
                print("Warning: Unrecognized scheduler state format. Starting fresh.")
                self._step = 0
                self._cycle = 0
                self._rate = self.init_lr
        else:
            print("Warning: Invalid scheduler state. Starting fresh.")
            self._step = 0
            self._cycle = 0
            self._rate = self.init_lr
        
        # Update optimizer with current rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._rate

def train_epoch(model, data_loader, optimizer, scheduler, criterion):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    optimizer.zero_grad(set_to_none=True)  # Initial gradient clear
    
    pbar = tqdm(data_loader.get_train_batches(), desc='Training')
    for batch_idx in pbar:
        try:
            # Get batch
            x, y = data_loader.get_batch(data_loader.train_sequences, 
                                       data_loader.train_targets, 
                                       batch_idx)
            if x is None:  # Skip incomplete batches
                continue
            
            # Initialize hidden state
            hidden = model.init_hidden(data_loader.config.batch_size)
            
            # Forward pass
            output, _ = model(x, hidden)
            loss = criterion(output, y) / model.config.gradient_accumulation_steps  # Scale loss for accumulation
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation and optimization step
            if (batch_idx + 1) % model.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), model.config.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                # Update learning rate based on scheduler type
                if scheduler is not None:
                    if isinstance(scheduler, (CyclicCosineScheduler, optim.lr_scheduler.OneCycleLR)):
                        scheduler.step()  # Step every accumulated batch
                    elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        pass  # Will step after epoch with validation loss
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update stats (scale loss back up for reporting)
            total_loss += loss.item() * model.config.gradient_accumulation_steps
            pbar.set_postfix({
                'loss': f'{loss.item() * model.config.gradient_accumulation_steps:.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    # Handle any remaining gradients
    if (batch_idx + 1) % model.config.gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.config.clip_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    return total_loss / len(data_loader.get_train_batches())

def validate(model, data_loader, criterion):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx in data_loader.get_val_batches():
            # Get batch
            x, y = data_loader.get_batch(data_loader.val_sequences, 
                                       data_loader.val_targets, 
                                       batch_idx)
            if x is None:  # Skip incomplete batches
                continue
            
            # Forward pass
            hidden = model.init_hidden(data_loader.config.batch_size)
            output, _ = model(x, hidden)
            loss = criterion(output, y)
            total_loss += loss.item()
    
    return total_loss / len(data_loader.get_val_batches())

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

def resume_or_start_training(model: RNNModel,
                           data_loader: ShakespeareDataLoader,
                           optimizer: optim.Optimizer,
                           scheduler: Any,
                           criterion: nn.Module,
                           n_epochs: int,
                           save_dir: str = 'checkpoints') -> Dict[str, Any]:
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
        initial_history=history
    )

def train_model(model, data_loader, optimizer, scheduler, criterion, n_epochs, save_dir='checkpoints', start_epoch=0, initial_history=None):
    """Main training loop with enhanced tracking and TensorBoard support"""
    print("Starting training...")
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if we've already completed training
    if start_epoch >= n_epochs:
        print(f"Training already completed ({start_epoch} >= {n_epochs} epochs)")
        return {
            'train_loss': initial_history['train_loss'][-1] if initial_history and initial_history['train_loss'] else 0.0,
            'val_loss': initial_history['val_loss'][-1] if initial_history and initial_history['val_loss'] else 0.0,
            'best_val_loss': min(initial_history['val_loss']) if initial_history and initial_history['val_loss'] else 0.0,
            'training_time': 0.0,
            'history': initial_history or {'train_loss': [], 'val_loss': [], 'learning_rates': []},
            'epochs_without_improvement': 0
        }
    
    # Initialize TensorBoard writer
    tensorboard_dir = os.path.join(save_dir, 'tensorboard')
    writer = SummaryWriter(tensorboard_dir)
    
    # Log model architecture
    dummy_input = torch.zeros(1, 1, data_loader.vocab_size, device=device)
    writer.add_graph(model, (dummy_input, model.init_hidden(1)))
    
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
    
    # Find best validation loss from history
    if history['val_loss']:
        best_val_loss = min(history['val_loss'])
    
    for epoch in range(start_epoch, n_epochs):
        final_epoch = epoch
        # Train
        train_loss = train_epoch(model, data_loader, optimizer, scheduler, criterion)
        
        # Validate
        val_loss = validate(model, data_loader, criterion)
        
        # Step schedulers that update per epoch
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                scheduler.step(epoch)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(current_lr)
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save checkpoint only if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            
            # Save both epoch checkpoint and best model
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            best_model_path = os.path.join(save_dir, 'best_model.pt')
            
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
            
            # Update symlink to best model
            try:
                # Remove existing symlink if it exists
                if os.path.islink(best_model_path):
                    os.unlink(best_model_path)
                elif os.path.exists(best_model_path):
                    os.remove(best_model_path)
                # Create new symlink
                os.symlink(os.path.basename(checkpoint_path), best_model_path)
            except OSError as e:
                print(f"Warning: Could not create symlink - {e}")
                # Fallback to copying the file
                import shutil
                shutil.copy2(checkpoint_path, best_model_path)
            
            print(f"New best model saved! Val Loss: {val_loss:.4f}")
        else:
            no_improve_count += 1
            print(f"No improvement for {no_improve_count} epochs")
            
            # Save periodic checkpoint
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
        
        print(f'Epoch {epoch+1}/{n_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f} (best: {best_val_loss:.4f})')
        print(f'  Learning Rate: {current_lr:.6f}')
        
        # Generate and log sample text
        print("\nGenerated text sample:")
        print("-" * 50)
        start_text = 'T' if data_loader.config.token_type == TokenType.CHAR else 'The'
        generate_text(model, data_loader, start_text=start_text, length=500)
        print("-" * 50)
        
        # Log generated text to TensorBoard
        writer.add_text('Generated Text', '', epoch)
        
        # Log parameter histograms
        for name, param in model.named_parameters():
            writer.add_histogram(f'Parameters/{name}', param.data, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
    
    training_time = time.time() - start_time
    
    # Save final training history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'history': history,
            'training_time': training_time,
            'best_val_loss': best_val_loss,
            'final_epoch': final_epoch,
            'epochs_without_improvement': no_improve_count
        }, f, indent=2)
    
    # Close TensorBoard writer
    writer.close()
    
    return {
        'train_loss': train_loss if 'train_loss' in locals() else history['train_loss'][-1] if history['train_loss'] else 0.0,
        'val_loss': val_loss if 'val_loss' in locals() else history['val_loss'][-1] if history['val_loss'] else 0.0,
        'best_val_loss': best_val_loss,
        'training_time': training_time,
        'history': history,
        'epochs_without_improvement': no_improve_count
    }

def save_checkpoint(model: RNNModel, 
                   optimizer: optim.Optimizer,
                   scheduler: Any,
                   epoch: int,
                   train_loss: float,
                   val_loss: float,
                   save_path: str,
                   data_config: DataConfig,
                   history: Dict[str, List] = None):
    """Save checkpoint with enhanced information."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_config': model.config.to_dict(),
        'data_config': data_config.to_dict(),
        'model_size': model.get_model_size(),
        'timestamp': datetime.now().isoformat(),
        'history': history
    }
    
    torch.save(checkpoint, save_path)
    
    # Save a human-readable summary
    summary_path = save_path.replace('.pt', '_summary.json')
    summary = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_config': model.config.to_dict(),
        'data_config': data_config.to_dict(),
        'model_size': model.get_model_size(),
        'timestamp': checkpoint['timestamp']
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

def load_checkpoint(path: str, model: RNNModel, 
                   optimizer: Optional[optim.Optimizer] = None,
                   scheduler: Optional[Any] = None) -> dict:
    """Load checkpoint with enhanced information."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint

def compare_models(data_config: DataConfig, 
                data_loader: ShakespeareDataLoader,
                model_configs: List[ModelConfig], 
                n_epochs: int = 10,
                base_save_dir: str = 'model_comparison') -> Dict[str, Any]:
    """Compare different RNN architectures with TensorBoard support."""
    results = {}
    
    # Create TensorBoard writer for comparison
    comparison_writer = SummaryWriter(os.path.join(base_save_dir, 'tensorboard_comparison'))
    
    for config in model_configs:
        # Create save directory for this model
        model_name = f"{config.rnn_type.value}_l{config.num_layers}_h{config.hidden_size}"
        if config.bidirectional:
            model_name += "_bidir"
        if config.lstm_proj_size:
            model_name += f"_proj{config.lstm_proj_size}"
        
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
        model = RNNModel(
            input_size=data_loader.vocab_size,
            config=config,
            output_size=data_loader.vocab_size
        ).to(device)
        
        # Setup training with cyclical learning rate
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Use cyclical scheduler with cosine annealing
        scheduler = CyclicCosineScheduler(
            optimizer,
            init_lr=config.learning_rate,
            min_lr=config.learning_rate * 0.01,  # Minimum LR is 1% of initial
            cycle_length=len(data_loader.get_train_batches()) * 5,  # 5 epochs per cycle
            cycle_mult=1.2,  # Increase cycle length by 20% each time
            decay=0.95  # Decay maximum learning rate by 5% each cycle
        )
        
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

def compare_schedulers(model: RNNModel,
                    data_loader: ShakespeareDataLoader,
                    base_save_dir: str = 'scheduler_comparison',
                    n_epochs: int = 50) -> Dict[str, Any]:
    """Compare different learning rate schedulers."""
    results = {}
    
    # Create TensorBoard writer for comparison
    comparison_writer = SummaryWriter(os.path.join(base_save_dir, 'tensorboard_comparison'))
    
    # Define scheduler configurations to compare
    scheduler_configs = {
        'no_scheduler': {
            'type': 'none',
            'params': {}
        },
        'cyclic_cosine': {
            'type': 'cyclic_cosine',
            'params': {
                'init_lr': model.config.learning_rate,
                'min_lr': model.config.learning_rate * 0.01,
                'cycle_length': len(data_loader.get_train_batches()) * 5,
                'cycle_mult': 1.2,
                'decay': 0.95
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
        'cosine_annealing_warm_restarts': {
            'type': 'cosine_warm_restarts',
            'params': {
                'T_0': len(data_loader.get_train_batches()) * 5,
                'T_mult': 2,
                'eta_min': model.config.learning_rate * 0.01
            }
        }
    }
    
    for scheduler_name, config in scheduler_configs.items():
        print(f"\nTraining with {scheduler_name}...")
        save_dir = os.path.join(base_save_dir, scheduler_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Create fresh model instance
        model_copy = RNNModel(
            input_size=data_loader.vocab_size,
            config=model.config,
            output_size=data_loader.vocab_size
        ).to(device)
        model_copy.load_state_dict(model.state_dict())  # Copy weights
        
        # Setup optimizer
        optimizer = optim.Adam(model_copy.parameters(), lr=model.config.learning_rate)
        
        # Setup scheduler based on configuration
        if config['type'] == 'none':
            scheduler = None
        elif config['type'] == 'cyclic_cosine':
            scheduler = CyclicCosineScheduler(optimizer, **config['params'])
        elif config['type'] == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config['params'])
        elif config['type'] == 'one_cycle':
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, **config['params'])
        elif config['type'] == 'cosine_warm_restarts':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **config['params'])
        
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

def main():
    """Main function to run the training"""
    print(f"Using device: {device}")
    
    # Data configuration
    data_config = DataConfig(
        token_type=TokenType.BPE,  # Use BPE tokenization
        seq_length=128,
        batch_size=32,
        bpe_encoding="gpt2"  # Using GPT-2's BPE encoding
    )
    
    # Load data
    data_loader = ShakespeareDataLoader(data_config)
    print(f"Vocabulary size: {data_loader.vocab_size}")
    print(f"Training sequences: {data_loader.train_size}")
    print(f"Validation sequences: {data_loader.val_size}")
    
    # First compare different model architectures
    print("\nComparing different model architectures...")
    model_configs = [
        # Basic RNN
        ModelConfig(
            rnn_type=RNNType.RNN,
            hidden_size=512,
            num_layers=3,
            dropout=0.2,
            nonlinearity='tanh',
            learning_rate=0.001,  # Initial learning rate
            clip_grad_norm=0.5
        ),
        # LSTM
        ModelConfig(
            rnn_type=RNNType.LSTM,
            hidden_size=512,
            num_layers=3,
            dropout=0.2,
            bidirectional=False,
            learning_rate=0.001,  # Initial learning rate
            clip_grad_norm=0.5
        ),
        # Bidirectional LSTM
        ModelConfig(
            rnn_type=RNNType.LSTM,
            hidden_size=512,
            num_layers=3,
            dropout=0.2,
            bidirectional=True,
            learning_rate=0.001,  # Initial learning rate
            clip_grad_norm=0.5
        ),
        # GRU
        ModelConfig(
            rnn_type=RNNType.GRU,
            hidden_size=512,
            num_layers=3,
            dropout=0.2,
            learning_rate=0.001,  # Initial learning rate
            clip_grad_norm=0.5
        ),
        # LSTM with projection
        ModelConfig(
            rnn_type=RNNType.LSTM,
            hidden_size=1024,
            num_layers=3,
            dropout=0.2,
            lstm_proj_size=512,
            learning_rate=0.001,  # Initial learning rate
            clip_grad_norm=0.5
        )
    ]
    
    architecture_results = compare_models(
        data_config=data_config,
        data_loader=data_loader,
        model_configs=model_configs,
        n_epochs=100,
        base_save_dir='model_comparison'
    )
    
    # Find the best model configuration
    best_model_name = min(architecture_results.items(), key=lambda x: x[1]['best_val_loss'])[0]
    best_model_config = next(config for config in model_configs 
                           if f"{config.rnn_type.value}_l{config.num_layers}_h{config.hidden_size}" in best_model_name)
    
    print(f"\nBest model architecture: {best_model_name}")
    
    # Create model with best architecture for scheduler comparison
    best_model = RNNModel(
        input_size=data_loader.vocab_size,
        config=best_model_config,
        output_size=data_loader.vocab_size
    ).to(device)
    
    # Compare different schedulers using the best model
    print("\nComparing different learning rate schedulers...")
    scheduler_results = compare_schedulers(
        model=best_model,
        data_loader=data_loader,
        base_save_dir='scheduler_comparison',
        n_epochs=50
    )
    
    # Print final comparison summary
    print("\nScheduler Comparison Results:")
    print("-" * 50)
    for scheduler_name, results in scheduler_results.items():
        print(f"\n{scheduler_name}:")
        print(f"  Best Validation Loss: {results['best_val_loss']:.4f}")
        print(f"  Training Time: {results['training_time']:.2f} seconds")
        print(f"  Final Learning Rate: {results['history']['learning_rates'][-1]:.2e}")

if __name__ == '__main__':
    main()
    