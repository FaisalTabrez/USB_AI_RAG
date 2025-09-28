# embeddings/embedder.py
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
import open_clip
import os

class MultimodalEmbedder:
    def __init__(self, text_model_name="all-MiniLM-L6-v2", clip_model_name="ViT-B-32", pretrained="laion2b_s13b_b90k"):
        """Initialize embedder with text and image models"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Text embedder
        print(f"Loading text model: {text_model_name}")
        self.text_model = SentenceTransformer(text_model_name)
        self.text_dim = self.text_model.get_sentence_embedding_dimension() or 384  # fallback to common dim

        # Image embedder (CLIP)
        print(f"Loading CLIP model: {clip_model_name}")
        try:
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                clip_model_name, pretrained=pretrained
            )
            self.clip_model.to(self.device).eval()
            self.clip_dim = self.clip_model.visual.output_dim
        except ImportError:
            print("Warning: open_clip_torch not available, using fallback image processing")
            self.clip_model = None
            self.preprocess = None
            self.clip_dim = 512  # Standard CLIP dimension

        # Projection layer to align dimensions
        self.projection_dim = min(self.text_dim, self.clip_dim)
        self.projection = torch.nn.Linear(self.clip_dim, self.projection_dim).to(self.device)

        print(f"Text embedding dimension: {self.text_dim}")
        print(f"CLIP embedding dimension: {self.clip_dim}")
        print(f"Projection dimension: {self.projection_dim}")

    def text_to_embedding(self, text):
        """Convert text to embedding vector"""
        if isinstance(text, str):
            text = [text]

        with torch.no_grad():
            embeddings = self.text_model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
            return embeddings.astype("float32")

    def image_to_embedding(self, image_path):
        """Convert image to embedding vector"""
        try:
            if self.clip_model is None or self.preprocess is None:
                # Fallback: use a simple hash-based embedding or text description
                print(f"Warning: CLIP not available, using fallback for {image_path}")
                return self._fallback_image_embedding(image_path)

            img = Image.open(image_path).convert('RGB')
            img_processed = self.preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Get CLIP embedding
                clip_embedding = self.clip_model.encode_image(img_processed)
                clip_embedding = clip_embedding / clip_embedding.norm(dim=-1, keepdim=True)  # Normalize

                # Project to common dimension
                projected = self.projection(clip_embedding.float())
                projected = projected / projected.norm(dim=-1, keepdim=True)  # Normalize again

                return projected.cpu().numpy().astype("float32")

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Return zero vector as fallback
            return np.zeros((1, self.projection_dim), dtype="float32")

    def _fallback_image_embedding(self, image_path):
        """Fallback image embedding when CLIP is not available"""
        # Simple hash-based embedding as fallback
        import hashlib
        hash_obj = hashlib.md5(image_path.encode())
        hash_bytes = hash_obj.digest()

        # Convert hash to embedding vector
        embedding = np.frombuffer(hash_bytes, dtype=np.float32)
        # Pad or truncate to match projection_dim
        if len(embedding) < self.projection_dim:
            embedding = np.pad(embedding, (0, self.projection_dim - len(embedding)))
        else:
            embedding = embedding[:self.projection_dim]

        return embedding.reshape(1, -1).astype("float32")

    def embed_text_and_image(self, text, image_path):
        """Get embeddings for both text and image in the same vector space"""
        text_emb = self.text_to_embedding(text)
        image_emb = self.image_to_embedding(image_path)

        # Ensure same batch size
        if text_emb.shape[0] == 1 and image_emb.shape[0] == 1:
            # Combine for joint retrieval
            combined = np.concatenate([text_emb, image_emb], axis=1)
            return combined.astype("float32")
        else:
            return text_emb.astype("float32")

    def get_embedding_dim(self):
        """Get the dimension of the embedding vectors"""
        return self.projection_dim

# Global embedder instance
embedder = None

def get_embedder():
    """Get or create the global embedder instance"""
    global embedder
    if embedder is None:
        embedder = MultimodalEmbedder()
    return embedder

def text_to_embedding(text):
    """Convenience function for text embedding"""
    return get_embedder().text_to_embedding(text)

def image_to_embedding(image_path):
    """Convenience function for image embedding"""
    return get_embedder().image_to_embedding(image_path)

def embed_text_and_image(text, image_path):
    """Convenience function for combined embedding"""
    return get_embedder().embed_text_and_image(text, image_path)

def get_embedding_dim():
    """Get the embedding dimension"""
    return get_embedder().get_embedding_dim()
