# rag/llm_client.py
import os
import sys
from utils.install_check import prompt_yes_no

class LLMClient:
    """Client for local LLM inference"""

    def __init__(self, model_path="models/llama-2-7b-chat.ggmlv3.q4_0.bin"):
        """Initialize LLM client"""
        self.model_path = model_path
        self.llm = None
        self.model_loaded = False

        # Try to load the model
        self._load_model()

    def _load_model(self):
        """Load the LLM model"""
        try:
            # Try llama-cpp-python first
            from llama_cpp import Llama
            print(f"Loading model: {self.model_path}")

            # Check if model file exists
            if not os.path.exists(self.model_path):
                print(f"Model file not found: {self.model_path}")
                print("Please download a GGML model file to continue.")
                print("You can download models from: https://huggingface.co/TheBloke")
                return False

            # Initialize LLM
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=2048,  # Context window
                n_threads=os.cpu_count() or 4,
                verbose=False
            )

            self.model_loaded = True
            print("Model loaded successfully!")
            return True

        except ImportError:
            print("llama-cpp-python not installed.")
            if self._prompt_install_llama_cpp():
                return self._load_model()
            return False

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def _prompt_install_llama_cpp(self):
        """Prompt user to install llama-cpp-python"""
        print("\nllama-cpp-python is required for local LLM inference.")
        print("Install with: pip install llama-cpp-python")

        if prompt_yes_no("Install llama-cpp-python now?"):
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python"])
                print("Installation complete!")
                return True
            except Exception as e:
                print(f"Installation failed: {e}")
                return False

        return False

    def generate(self, prompt, max_tokens=512, temperature=0.1):
        """Generate text from prompt"""
        if not self.model_loaded or not self.llm:
            return "Error: Model not loaded. Please check your model file and try again."

        try:
            # Generate response
            response = self.llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["User:", "Question:", "Evidence:"],  # Stop at common delimiters
                echo=False
            )

            return response["choices"][0]["text"].strip()

        except Exception as e:
            return f"Error during generation: {e}"

    def generate_stream(self, prompt, max_tokens=512, temperature=0.1):
        """Generate text with streaming (if supported)"""
        if not self.model_loaded or not self.llm:
            yield "Error: Model not loaded."
            return

        try:
            # For now, just return the full response
            # In a full implementation, you'd use streaming API if available
            response = self.generate(prompt, max_tokens, temperature)
            yield response

        except Exception as e:
            yield f"Error: {e}"

    def is_model_loaded(self):
        """Check if model is loaded"""
        return self.model_loaded

    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.model_loaded:
            return {"error": "Model not loaded"}

        return {
            "model_path": self.model_path,
            "loaded": True,
            "context_window": getattr(self.llm, 'n_ctx', 'Unknown'),
            "threads": getattr(self.llm, 'n_threads', 'Unknown')
        }

# Global LLM client instance
_llm_client = None

def get_llm_client(model_path="models/llama-2-7b-chat.ggmlv3.q4_0.bin"):
    """Get or create global LLM client"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient(model_path)
    return _llm_client

def generate_response(prompt, max_tokens=512, temperature=0.1):
    """Convenience function to generate response"""
    client = get_llm_client()
    return client.generate(prompt, max_tokens, temperature)

def is_model_available():
    """Check if a model is available and loaded"""
    client = get_llm_client()
    return client.is_model_loaded()

def setup_model_interactive():
    """Interactive model setup"""
    print("LLM Model Setup")
    print("===============")

    # Ask for model path
    model_path = input("Enter path to GGML model file (or press Enter for default): ").strip()
    if not model_path:
        model_path = "models/llama-2-7b-chat.ggmlv3.q4_0.bin"

    # Create client with specified model
    client = LLMClient(model_path)

    if client.is_model_loaded():
        print("Model setup complete!")
        return client
    else:
        print("Model setup failed. Please check the model file and try again.")
        return None

if __name__ == "__main__":
    # Test the LLM client
    client = get_llm_client()

    if client.is_model_loaded():
        test_prompt = "Hello! Can you help me with a question about documents?"
        response = client.generate(test_prompt, max_tokens=100)
        print(f"Test response: {response}")
    else:
        print("Model not loaded. Please set up a model first.")
        setup_model_interactive()
