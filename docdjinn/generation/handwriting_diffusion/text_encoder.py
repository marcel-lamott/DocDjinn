"""
Transformer-based text encoder for conditioning diffusion model.
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder block."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            attention_mask: [batch_size, seq_len] - 1 for valid, 0 for padding
        """
        # Self-attention with residual
        attn_output, _ = self.self_attn(
            x, x, x,
            key_padding_mask=(1 - attention_mask).bool() if attention_mask is not None else None
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class TextEncoder(nn.Module):
    """
    Transformer-based text encoder for character-level conditioning.
    """
    
    def __init__(
        self,
        vocab_size: int,
        char_embed_dim: int = 256,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_length: int = 128,
        dropout: float = 0.1,
        output_dim: int = 512
    ):
        """
        Args:
            vocab_size: Size of character vocabulary
            char_embed_dim: Dimension of character embeddings
            d_model: Hidden dimension of transformer
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            d_ff: Dimension of feed-forward layer
            max_length: Maximum sequence length
            dropout: Dropout probability
            output_dim: Output dimension for conditioning
        """
        super().__init__()
        
        self.d_model = d_model
        self.output_dim = output_dim
        
        # Character embedding
        self.char_embedding = nn.Embedding(vocab_size, char_embed_dim, padding_idx=0)
        
        # Project char embeddings to model dimension
        self.input_projection = nn.Linear(char_embed_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_length)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: [batch_size, seq_len] - Token indices
            attention_mask: [batch_size, seq_len] - 1 for valid, 0 for padding
        
        Returns:
            Encoded text features [batch_size, seq_len, output_dim]
        """
        # Character embedding
        x = self.char_embedding(input_ids)  # [B, seq_len, char_embed_dim]
        
        # Project to model dimension
        x = self.input_projection(x)  # [B, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Normalize
        x = self.norm(x)
        
        # Project to output dimension
        x = self.output_projection(x)  # [B, seq_len, output_dim]
        
        return x
    
    def get_sequence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Get single embedding for entire sequence (mean pooling over valid tokens).
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            Pooled embedding [batch_size, output_dim]
        """
        # Get token-level embeddings
        token_embeddings = self.forward(input_ids, attention_mask)  # [B, seq_len, output_dim]
        
        # Mean pooling over valid tokens
        if attention_mask is not None:
            # Expand mask to match embedding dimension
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = token_embeddings.mean(dim=1)
        
        return pooled


if __name__ == "__main__":
    # Test the text encoder
    vocab_size = 100
    batch_size = 4
    seq_len = 32
    
    model = TextEncoder(
        vocab_size=vocab_size,
        char_embed_dim=256,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        max_length=128,
        output_dim=512
    )
    
    # Random input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, seq_len//2:] = 0  # Simulate padding
    
    # Forward pass
    output = model(input_ids, attention_mask)
    pooled = model.get_sequence_embedding(input_ids, attention_mask)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Pooled shape: {pooled.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
