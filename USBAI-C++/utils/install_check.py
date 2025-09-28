# utils/install_check.py
import sys, platform, shutil, subprocess, importlib

OS = platform.system().lower()

def _pip_install(pkgs):
    """Attempt pip install (user opted in)"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + pkgs)
        print(f"Successfully installed: {' '.join(pkgs)}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages: {e}")
        return False
    return True

def prompt_yes_no(question, default="y"):
    """Get yes/no input from user"""
    ans = input(f"{question} [{'Y/n' if default=='y' else 'y/N'}]: ").strip().lower()
    if ans == "" and default == "y":
        return True
    return ans.startswith("y")

def check_binary(bin_name, install_cmds):
    """Check if binary is available, prompt for installation if missing"""
    path = shutil.which(bin_name)
    if path:
        print(f"[✓] {bin_name} found at {path}")
        return True

    print(f"[✗] {bin_name} not found.")
    print("Install instructions:")
    for osname, cmd in install_cmds.items():
        print(f"  {osname}: {cmd}")

    if prompt_yes_no("Show these instructions and continue (do not auto-install)?"):
        return False
    return False

def check_python_packages(pkgs):
    """Check if Python packages are installed"""
    missing = []
    for p in pkgs:
        try:
            importlib.import_module(p)
        except (ImportError, ModuleNotFoundError):
            missing.append(p)

    if missing:
        print(f"Missing Python packages: {', '.join(missing)}")
        if prompt_yes_no("Attempt to pip-install missing python packages now?"):
            return _pip_install(missing)
        else:
            print("Please install them manually:")
            print(f"  {sys.executable} -m pip install {' '.join(missing)}")
            return False

    print("[✓] All Python packages present")
    return True

def run_system_checks():
    """Run all system dependency checks"""
    print("Checking system dependencies...")

    # Check system binaries
    binaries_to_check = {
        "tesseract": {
            "linux": "sudo apt install tesseract-ocr",
            "darwin": "brew install tesseract",
            "windows": "choco install tesseract"
        },
        "ffmpeg": {
            "linux": "sudo apt install ffmpeg",
            "darwin": "brew install ffmpeg",
            "windows": "choco install ffmpeg"
        }
    }

    all_good = True
    for bin_name, install_cmds in binaries_to_check.items():
        if not check_binary(bin_name, {OS: install_cmds.get(OS, "Package manager not configured")}):
            all_good = False

    # Check Python packages
    python_packages = [
        "faiss", "sentence_transformers", "torch", "transformers",
        "llama_cpp", "pytesseract", "docx", "fitz", "open_clip_torch",
        "numpy", "flask", "pydub", "PIL", "librosa", "soundfile"
    ]

    if not check_python_packages(python_packages):
        all_good = False

    if all_good:
        print("All dependencies satisfied!")
    else:
        print("Some dependencies are missing. Please install them to continue.")

    return all_good

if __name__ == "__main__":
    run_system_checks()
