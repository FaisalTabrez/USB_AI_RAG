#!/usr/bin/env python3
"""
Complete setup and run script for Multimodal RAG System
This script handles the entire installation and setup process.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(cmd, shell=False):
    """Run a command and return success status"""
    try:
        print(f"Running: {cmd}")
        result = subprocess.run(cmd if shell else cmd.split(),
                              shell=shell, check=True, capture_output=True, text=True)
        print(f"✅ Success: {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {cmd}")
        print(f"Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Error: {cmd} - {e}")
        return False

def install_python_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")

    # Upgrade pip first
    if not run_command("python -m pip install --upgrade pip"):
        return False

    # Install from requirements.txt
    if not run_command("python -m pip install -r requirements.txt"):
        print("⚠️  Some packages may have failed to install")
        print("You can try installing them manually or run the system with reduced functionality")

    return True

def install_system_dependencies():
    """Install system dependencies based on OS"""
    print("\n🔧 Installing system dependencies...")

    os_name = platform.system().lower()

    if os_name == "linux":
        # Ubuntu/Debian
        commands = [
            "sudo apt update",
            "sudo apt install -y tesseract-ocr",
            "sudo apt install -y ffmpeg",
            "sudo apt install -y build-essential",
            "sudo apt install -y cmake",
            "sudo apt install -y git"
        ]
    elif os_name == "darwin":  # macOS
        commands = [
            "brew install tesseract",
            "brew install ffmpeg",
            "brew install cmake",
            "brew install git"
        ]
    elif os_name == "windows":
        print("For Windows, please install manually:")
        print("1. Download and install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Download and install FFmpeg from: https://ffmpeg.org/download.html")
        print("3. Install Visual Studio Build Tools")
        print("4. Install Git from: https://git-scm.com/download/win")
        return True
    else:
        print(f"Unsupported OS: {os_name}")
        print("Please install manually: tesseract-ocr, ffmpeg, cmake, git")
        return True

    success = True
    for cmd in commands:
        if not run_command(cmd, shell=True):
            success = False

    return success

def setup_project():
    """Setup the project structure"""
    print("\n🏗️  Setting up project structure...")

    # Create necessary directories
    dirs = ["data", "db", "models", "uploads"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}")

    # Create sample data
    print("📝 Creating sample data...")
    try:
        from setup_demo import create_sample_data
        create_sample_data()
        print("✅ Sample data created")
    except ImportError:
        print("⚠️  Could not create sample data (setup_demo.py not found)")

    return True

def check_dependencies():
    """Check if all dependencies are installed"""
    print("\n🔍 Checking dependencies...")

    try:
        # Test Python imports
        import faiss
        import torch
        import transformers
        import sentence_transformers
        print("✅ Python packages OK")

        # Test system binaries
        import shutil
        if shutil.which("tesseract") and shutil.which("ffmpeg"):
            print("✅ System binaries OK")
        else:
            print("⚠️  Some system binaries missing")

        return True

    except ImportError as e:
        print(f"❌ Missing Python package: {e}")
        return False

def download_sample_model():
    """Download a small sample model"""
    print("\n🤖 Setting up sample model...")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    print("For demonstration, we recommend downloading TinyLlama-1.1B-Chat")
    print("Visit: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-GGUF")
    print("Download: TinyLlama-1.1B-Chat.Q4_K_M.gguf")
    print("Place the file in the models/ directory")

    # For now, create a placeholder
    placeholder_path = models_dir / "model_placeholder.txt"
    with open(placeholder_path, 'w') as f:
        f.write("Place your GGML model file here\n")
        f.write("Recommended: TinyLlama-1.1B-Chat.Q4_K_M.gguf\n")
        f.write("Download from: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-GGUF\n")

    print("✅ Model placeholder created")

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")

    try:
        # Test basic imports
        from utils.install_check import run_system_checks
        from embeddings.embedder import get_embedding_dim
        from indexer.faiss_index import get_index

        print("✅ Basic imports successful")

        # Test embedder
        embedder_dim = get_embedding_dim()
        print(f"✅ Embedder initialized (dimension: {embedder_dim})")

        # Test index
        index = get_index()
        print(f"✅ Index initialized ({index.get_stats()})")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main installation function"""
    print("🚀 Multimodal RAG System - Complete Installation")
    print("=" * 60)

    # Step 1: Install system dependencies
    if not install_system_dependencies():
        print("❌ System dependencies installation failed")
        print("Please install manually and try again")
        return False

    # Step 2: Install Python dependencies
    if not install_python_dependencies():
        print("❌ Python dependencies installation failed")
        print("Please check your internet connection and try again")
        return False

    # Step 3: Setup project structure
    if not setup_project():
        print("❌ Project setup failed")
        return False

    # Step 4: Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed")
        print("Some components may not work correctly")
        print("You can still try running the system")

    # Step 5: Download sample model (instructions)
    download_sample_model()

    # Step 6: Test installation
    if test_installation():
        print("✅ Installation test passed!")
    else:
        print("⚠️  Installation test failed, but system may still work")

    # Final instructions
    print("\n" + "=" * 60)
    print("🎉 Installation Complete!")
    print("=" * 60)

    print("\n📋 Next Steps:")
    print("1. Download a GGML model file to the models/ directory")
    print("2. Ingest your data: python main.py ingest data/")
    print("3. Start web interface: python webapp/app.py")
    print("4. Or use CLI: python main.py query 'your question'")

    print("\n🔗 Useful Links:")
    print("• Models: https://huggingface.co/TheBloke")
    print("• Documentation: See README.md")
    print("• Demo: python setup_demo.py")

    print("\n💡 Quick Test:")
    print("python main.py setup")
    print("python setup_demo.py")
    print("python main.py ingest data/")
    print("python main.py query 'What are the key findings?'")

    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Setup completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Setup failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)
