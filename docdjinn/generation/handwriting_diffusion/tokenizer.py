"""
Character-level tokenizer for handwriting generation.
Supports special tokens and can be saved/loaded for inference.
"""
import json
import os
from typing import List, Dict, Optional
import numpy as np


class CharTokenizer:
    """Character-level tokenizer with special tokens."""
    
    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        max_length: int = 128
    ):
        """
        Initialize tokenizer.
        
        Args:
            vocab: Character to index mapping. If None, will be built from data.
            max_length: Maximum sequence length for padding/truncation.
        """
        self.max_length = max_length
        
        if vocab is None:
            # Initialize with special tokens only
            self.char_to_idx = {
                self.PAD_TOKEN: 0,
                self.UNK_TOKEN: 1,
                self.SOS_TOKEN: 2,
                self.EOS_TOKEN: 3,
            }
        else:
            self.char_to_idx = vocab
        
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from list of texts.
        
        Args:
            texts: List of text strings to build vocabulary from.
        """
        # Collect all unique characters
        unique_chars = set()
        for text in texts:
            unique_chars.update(text)
        
        # Sort for deterministic ordering
        unique_chars = sorted(list(unique_chars))
        
        # Add to vocabulary (starting after special tokens)
        for char in unique_chars:
            if char not in self.char_to_idx:
                self.char_to_idx[char] = len(self.char_to_idx)
        
        # Update reverse mapping
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        print(f"Built vocabulary with {self.vocab_size} characters")
        print(f"Sample characters: {list(unique_chars)[:20]}")
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True,
        return_attention_mask: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Encode text to token indices.
        
        Args:
            text: Input text string.
            add_special_tokens: Whether to add SOS/EOS tokens.
            padding: Whether to pad to max_length.
            truncation: Whether to truncate to max_length.
            return_attention_mask: Whether to return attention mask.
        
        Returns:
            Dictionary with 'input_ids' and optionally 'attention_mask'.
        """
        # Convert characters to indices
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.char_to_idx[self.SOS_TOKEN])
        
        for char in text:
            token_ids.append(
                self.char_to_idx.get(char, self.char_to_idx[self.UNK_TOKEN])
            )
        
        if add_special_tokens:
            token_ids.append(self.char_to_idx[self.EOS_TOKEN])
        
        # Truncation
        if truncation and len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            if add_special_tokens:
                token_ids[-1] = self.char_to_idx[self.EOS_TOKEN]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(token_ids)
        
        # Padding
        if padding and len(token_ids) < self.max_length:
            padding_length = self.max_length - len(token_ids)
            token_ids.extend([self.char_to_idx[self.PAD_TOKEN]] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        result = {
            'input_ids': np.array(token_ids, dtype=np.int64)
        }
        
        if return_attention_mask:
            result['attention_mask'] = np.array(attention_mask, dtype=np.float32)
        
        return result
    
    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True,
        return_attention_mask: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Encode batch of texts.
        
        Args:
            texts: List of text strings.
            add_special_tokens: Whether to add SOS/EOS tokens.
            padding: Whether to pad to max_length.
            truncation: Whether to truncate to max_length.
            return_attention_mask: Whether to return attention mask.
        
        Returns:
            Dictionary with batched 'input_ids' and optionally 'attention_mask'.
        """
        batch_encoding = [
            self.encode(
                text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                return_attention_mask=return_attention_mask
            )
            for text in texts
        ]
        
        result = {
            'input_ids': np.stack([enc['input_ids'] for enc in batch_encoding])
        }
        
        if return_attention_mask:
            result['attention_mask'] = np.stack([enc['attention_mask'] for enc in batch_encoding])
        
        return result
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token indices to text.
        
        Args:
            token_ids: List of token indices.
            skip_special_tokens: Whether to skip special tokens in output.
        
        Returns:
            Decoded text string.
        """
        chars = []
        special_tokens = {
            self.char_to_idx[self.PAD_TOKEN],
            self.char_to_idx[self.UNK_TOKEN],
            self.char_to_idx[self.SOS_TOKEN],
            self.char_to_idx[self.EOS_TOKEN]
        }
        
        for idx in token_ids:
            if skip_special_tokens and idx in special_tokens:
                continue
            chars.append(self.idx_to_char.get(idx, self.UNK_TOKEN))
        
        return ''.join(chars)
    
    def save(self, save_path: str) -> None:
        """
        Save tokenizer to file.
        
        Args:
            save_path: Path to save tokenizer (JSON file).
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        config = {
            'char_to_idx': self.char_to_idx,
            'max_length': self.max_length,
            'vocab_size': self.vocab_size
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"Tokenizer saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: str) -> "CharTokenizer":
        """
        Load tokenizer from file.
        
        Args:
            load_path: Path to load tokenizer from (JSON file).
        
        Returns:
            Loaded tokenizer instance.
        """
        with open(load_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        tokenizer = cls(
            vocab=config['char_to_idx'],
            max_length=config['max_length']
        )
        
        print(f"Tokenizer loaded from {load_path}")
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        
        return tokenizer
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
    
    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size}, max_length={self.max_length})"


def build_tokenizer_from_csv(csv_path: str, max_length: int = 128) -> CharTokenizer:
    """
    Build tokenizer from IAM dataset CSV file.
    
    Args:
        csv_path: Path to dataset_metadata.csv
        max_length: Maximum sequence length
    
    Returns:
        Built tokenizer
    """
    import pandas as pd
    
    print(f"Loading texts from {csv_path}...")
    df = pd.read_csv(csv_path)
    texts = df['text'].astype(str).tolist()
    
    print(f"Building vocabulary from {len(texts)} samples...")
    tokenizer = CharTokenizer(max_length=max_length)
    tokenizer.build_vocab(texts)
    
    return tokenizer


if __name__ == "__main__":
    # Example: Build tokenizer from IAM dataset
    tokenizer = build_tokenizer_from_csv(
        "../iam_dataset_processed/dataset_metadata.csv",
        max_length=128
    )
    
    # Save tokenizer
    tokenizer.save("../training/tokenizer.json")
    
    # Test encoding
    test_text = "Hello, World!"
    encoded = tokenizer.encode(test_text)
    print(f"\nTest encoding for: '{test_text}'")
    print(f"Input IDs: {encoded['input_ids'][:20]}")
    print(f"Attention mask: {encoded['attention_mask'][:20]}")
    
    # Test decoding
    decoded = tokenizer.decode(encoded['input_ids'])
    print(f"Decoded: '{decoded}'")
