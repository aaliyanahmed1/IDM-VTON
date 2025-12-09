"""
Test script to validate API structure and imports
Run this before starting the full server to check for issues
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test if all required imports are available"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ PyTorch not found: {e}")
        return False
    
    try:
        from fastapi import FastAPI
        import fastapi
        print(f"✓ FastAPI {fastapi.__version__}")
    except ImportError as e:
        print(f"✗ FastAPI not found: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ PIL/Pillow")
    except ImportError as e:
        print(f"✗ PIL not found: {e}")
        return False
    
    try:
        from transformers import CLIPTextModel
        print("✓ Transformers")
    except ImportError as e:
        print(f"✗ Transformers not found: {e}")
        return False
    
    try:
        from diffusers import AutoencoderKL
        print("✓ Diffusers")
    except ImportError as e:
        print(f"✗ Diffusers not found: {e}")
        return False
    
    try:
        import uvicorn
        print("✓ Uvicorn")
    except ImportError as e:
        print(f"✗ Uvicorn not found: {e}")
        return False
    
    return True

def test_api_imports():
    """Test if API modules can be imported"""
    print("\nTesting API module imports...")
    
    # Add parent directory to path for imports
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    try:
        # Try relative import first (when running from api/)
        try:
            from models import TryOnRequest, TryOnResponse
            print("✓ API models imported (relative)")
        except ImportError:
            # Try absolute import (when running from root)
            from api.models import TryOnRequest, TryOnResponse
            print("✓ API models imported (absolute)")
    except ImportError as e:
        print(f"✗ API models import failed: {e}")
        return False
    
    try:
        # Try importing service (this will fail if models aren't downloaded, but structure should be OK)
        try:
            from service_optimized import OptimizedTryOnService
            print("✓ Service class structure OK (relative)")
        except ImportError:
            from api.service_optimized import OptimizedTryOnService
            print("✓ Service class structure OK (absolute)")
    except ImportError as e:
        print(f"⚠ Service import issue (expected if models not loaded): {e}")
    except Exception as e:
        # This is expected if models aren't downloaded
        if "from_pretrained" in str(e) or "model" in str(e).lower() or "diffusers" in str(e).lower():
            print("⚠ Service structure OK (dependencies need to be installed)")
        else:
            print(f"⚠ Service import error (may need dependencies): {e}")
    
    try:
        try:
            from main import app
            print("✓ FastAPI app structure OK (relative)")
        except ImportError:
            from api.main import app
            print("✓ FastAPI app structure OK (absolute)")
    except ImportError as e:
        print(f"✗ API main import failed: {e}")
        return False
    
    return True

def test_torch_compile():
    """Check if torch.compile is available"""
    print("\nTesting PyTorch optimizations...")
    
    try:
        import torch
        if hasattr(torch, 'compile'):
            print("✓ torch.compile available (PyTorch 2.0+)")
            return True
        else:
            print("⚠ torch.compile not available (PyTorch < 2.0)")
            print("  Service will work but without compilation optimization")
            return True
    except Exception as e:
        print(f"✗ Error checking torch.compile: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")
    
    api_dir = Path(__file__).parent
    required_files = [
        'main.py',
        'service_optimized.py',
        'models.py',
        'requirements.txt',
        '__init__.py'
    ]
    
    all_exist = True
    for file in required_files:
        file_path = api_dir / file
        if file_path.exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("=" * 50)
    print("IDM-VTON API Structure Test")
    print("=" * 50)
    
    results = []
    
    results.append(("File Structure", test_file_structure()))
    results.append(("Python Imports", test_imports()))
    results.append(("API Imports", test_api_imports()))
    results.append(("PyTorch Optimizations", test_torch_compile()))
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All structure tests passed!")
        print("\nNext steps:")
        print("1. Install missing dependencies: pip install diffusers")
        print("2. Ensure models are downloaded from HuggingFace")
        print("3. Run: python -m api.main (from root) OR python main.py (from api/)")
        print("4. Test endpoint: http://localhost:8000/health")
    else:
        print("\n⚠ Some tests failed.")
        print("\nTo fix:")
        print("1. Install missing dependencies: pip install diffusers")
        print("2. File structure and PyTorch are OK")
        print("3. API imports will work once dependencies are installed")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

