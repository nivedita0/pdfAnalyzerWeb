import subprocess
import sys
import os

def setup_environment():
    print("Setting up the environment...")
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists("venv"):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])
    
    # Determine the pip path based on the OS
    if sys.platform == "win32":
        pip_path = os.path.join("venv", "Scripts", "pip")
        python_path = os.path.join("venv", "Scripts", "python")
    else:
        pip_path = os.path.join("venv", "bin", "pip")
        python_path = os.path.join("venv", "bin", "python")
    
    # Upgrade pip
    print("Upgrading pip...")
    subprocess.run([pip_path, "install", "--upgrade", "pip"])
    
    # Install requirements
    print("Installing requirements...")
    subprocess.run([pip_path, "install", "-r", "requirements.txt"])
    
    print("\nSetup complete! To activate the environment:")
    if sys.platform == "win32":
        print("venv\\Scripts\\activate")
    else:
        print("source venv/bin/activate")
    print("\nThen run the application with:")
    print("python app.py")

if __name__ == "__main__":
    setup_environment() 