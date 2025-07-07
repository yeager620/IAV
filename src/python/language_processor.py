"""
Language processing for converting text commands to embeddings.
"""

import numpy as np
import time
import logging
from typing import Optional, Dict, List
import re

logger = logging.getLogger(__name__)

class LanguageProcessor:
    """Simple language processor for drone commands"""
    
    def __init__(self, config: dict):
        self.config = config
        self.embedding_dim = config.get('embedding_dim', 512)
        
        # Command vocabulary and embeddings
        self.command_vocab = self._build_command_vocabulary()
        self.word_embeddings = self._initialize_word_embeddings()
        
        # Performance tracking
        self.total_encodings = 0
        self.encoding_times = []
    
    def initialize(self) -> bool:
        """Initialize language processor"""
        try:
            logger.info("Initializing language processor")
            
            # Test encoding
            test_embedding = self.encode_command("test command")
            if test_embedding.shape[0] != self.embedding_dim:
                raise ValueError(f"Unexpected embedding dimension: {test_embedding.shape[0]}")
            
            logger.info(f"Language processor initialized with {self.embedding_dim}D embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Language processor initialization failed: {e}")
            return False
    
    def encode_command(self, command: str) -> np.ndarray:
        """
        Encode text command to embedding vector
        
        Args:
            command: Natural language command string
            
        Returns:
            embedding: Command embedding vector [embedding_dim]
        """
        start_time = time.perf_counter()
        
        try:
            # Preprocess command
            processed_command = self._preprocess_command(command)
            
            # Tokenize
            tokens = self._tokenize(processed_command)
            
            # Convert to embeddings
            embedding = self._tokens_to_embedding(tokens)
            
            # Track performance
            end_time = time.perf_counter()
            encoding_time = (end_time - start_time) * 1000
            self.encoding_times.append(encoding_time)
            if len(self.encoding_times) > 100:
                self.encoding_times.pop(0)
            self.total_encodings += 1
            
            return embedding
            
        except Exception as e:
            logger.error(f"Command encoding failed: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def _preprocess_command(self, command: str) -> str:
        """Preprocess command text"""
        # Convert to lowercase
        command = command.lower().strip()
        
        # Remove extra whitespace
        command = re.sub(r'\s+', ' ', command)
        
        # Handle common abbreviations and variations
        replacements = {
            'take off': 'takeoff',
            'land': 'landing',
            'go up': 'ascend',
            'go down': 'descend',
            'turn left': 'rotate left',
            'turn right': 'rotate right',
            'move forward': 'forward',
            'move backward': 'backward',
            'move back': 'backward',
            'stop': 'hover',
            'stay': 'hover',
            'hold position': 'hover'
        }
        
        for old, new in replacements.items():
            command = command.replace(old, new)
        
        return command
    
    def _tokenize(self, command: str) -> List[str]:
        """Simple tokenization"""
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', command)
        return tokens
    
    def _tokens_to_embedding(self, tokens: List[str]) -> np.ndarray:
        """Convert tokens to embedding vector"""
        if not tokens:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Get embeddings for each token
        token_embeddings = []
        for token in tokens:
            if token in self.word_embeddings:
                token_embeddings.append(self.word_embeddings[token])
            else:
                # Unknown token - use random embedding
                token_embeddings.append(np.random.normal(0, 0.1, self.embedding_dim))
        
        if not token_embeddings:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Simple averaging (could be improved with attention)
        embedding = np.mean(token_embeddings, axis=0)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def _build_command_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary for drone commands"""
        vocab = {
            # Movement commands
            'takeoff': 0, 'landing': 1, 'hover': 2,
            'forward': 3, 'backward': 4, 'left': 5, 'right': 6,
            'up': 7, 'down': 8, 'ascend': 9, 'descend': 10,
            
            # Rotation commands  
            'rotate': 11, 'turn': 12, 'spin': 13,
            'clockwise': 14, 'counterclockwise': 15,
            
            # Speed modifiers
            'fast': 16, 'slow': 17, 'quickly': 18, 'slowly': 19,
            'gently': 20, 'carefully': 21,
            
            # Distance/amount modifiers
            'little': 22, 'bit': 23, 'small': 24, 'large': 25,
            'meters': 26, 'feet': 27, 'degrees': 28,
            
            # Action words
            'move': 29, 'go': 30, 'fly': 31, 'navigate': 32,
            'approach': 33, 'avoid': 34, 'follow': 35,
            
            # Objects and targets
            'obstacle': 36, 'target': 37, 'object': 38,
            'wall': 39, 'ground': 40, 'ceiling': 41,
            
            # Prepositions and articles
            'to': 42, 'towards': 43, 'away': 44, 'from': 45,
            'the': 46, 'a': 47, 'an': 48,
            
            # Numbers
            'one': 49, 'two': 50, 'three': 51, 'four': 52, 'five': 53,
            '1': 49, '2': 50, '3': 51, '4': 52, '5': 53
        }
        
        return vocab
    
    def _initialize_word_embeddings(self) -> Dict[str, np.ndarray]:
        """Initialize word embeddings for vocabulary"""
        embeddings = {}
        
        # Create embeddings for each word in vocabulary
        for word, idx in self.command_vocab.items():
            # Use a simple deterministic embedding based on word index
            # In practice, this would be pre-trained embeddings
            embedding = np.random.RandomState(idx).normal(0, 0.1, self.embedding_dim)
            
            # Add some semantic structure for related words
            if word in ['takeoff', 'up', 'ascend']:
                embedding[0] = 1.0  # Positive vertical motion
            elif word in ['landing', 'down', 'descend']:
                embedding[0] = -1.0  # Negative vertical motion
            elif word in ['forward']:
                embedding[1] = 1.0  # Forward motion
            elif word in ['backward']:
                embedding[1] = -1.0  # Backward motion
            elif word in ['left']:
                embedding[2] = -1.0  # Left motion
            elif word in ['right']:
                embedding[2] = 1.0  # Right motion
            elif word in ['rotate', 'turn', 'spin']:
                embedding[3] = 1.0  # Rotation
            elif word in ['hover', 'stop']:
                embedding[4] = 1.0  # Stop/hover
            elif word in ['fast', 'quickly']:
                embedding[5] = 1.0  # High speed
            elif word in ['slow', 'slowly', 'gently', 'carefully']:
                embedding[5] = -1.0  # Low speed
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            embeddings[word] = embedding.astype(np.float32)
        
        return embeddings
    
    def get_performance_stats(self) -> dict:
        """Get language processing performance statistics"""
        return {
            'total_encodings': self.total_encodings,
            'avg_encoding_time_ms': np.mean(self.encoding_times) if self.encoding_times else 0,
            'vocabulary_size': len(self.command_vocab),
            'embedding_dimension': self.embedding_dim
        }
    
    def get_supported_commands(self) -> List[str]:
        """Get list of supported command patterns"""
        return [
            "take off", "takeoff", "land", "landing",
            "hover", "stop", "hold position",
            "move forward", "go forward", "forward",
            "move backward", "go backward", "backward", 
            "move left", "go left", "left",
            "move right", "go right", "right",
            "go up", "ascend", "up",
            "go down", "descend", "down",
            "turn left", "rotate left",
            "turn right", "rotate right",
            "move slowly", "move carefully",
            "move quickly", "move fast"
        ]