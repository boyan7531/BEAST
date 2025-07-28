#!/usr/bin/env python3
"""
Setup Validation Script for Enhanced Training System

This script validates that your environment is properly set up for running
the enhanced training pipeline.

Usage:
    python validate_setup.py
"""

import os
import sys
import importlib
import subprocess
from pathlib import Path


def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_check(description, status, details=""):
    """Print a check result"""
    status_symbol = "✅" if status else "❌"
    print(f"{status_symbol} {description}")
    if details:
        print(f"   {details}")


def check_python_version():
    """Check Python version"""
    print_header("Python Environment")
    
    version = sys.version_info
    python_ok = version.major == 3 and version.minor >= 8
    
    print_check(
        f"Python Version: {version.major}.{version.minor}.{version.micro}",
        python_ok,
        "Requires Python 3.8+" if not python_ok else "Good!"
    )
    
    return python_ok


def check_required_packages():
    """Check if required packages are installed"""
    print_header("Required Packages")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'tqdm': 'TQDM',
        'psutil': 'PSUtil (for system monitoring)',
        'yaml': 'PyYAML (for config files)'
    }
    
    all_packages_ok = True
    
    for package, description in required_packages.items():
        try:
            if package == 'sklearn':
                importlib.import_module('sklearn')
            elif package == 'yaml':
                importlib.import_module('yaml')
            else:
                importlib.import_module(package)
            
            print_check(f"{description}", True)
        except ImportError:
            print_check(f"{description}", False, f"Install with: pip install {package}")
            all_packages_ok = False
    
    return all_packages_ok


def check_pytorch_setup():
    """Check PyTorch setup and CUDA availability"""
    print_header("PyTorch Setup")
    
    try:
        import torch
        
        # Check PyTorch version
        torch_version = torch.__version__
        print_check(f"PyTorch Version: {torch_version}", True)
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_check(
                f"CUDA Available: {device_name}",
                True,
                f"GPU Memory: {memory_gb:.1f} GB"
            )
        else:
            print_check(
                "CUDA Available",
                False,
                "Will use CPU (training will be slower)"
            )
        
        # Test basic tensor operations
        try:
            x = torch.randn(2, 3)
            y = torch.randn(3, 2)
            z = torch.mm(x, y)
            print_check("Basic tensor operations", True)
        except Exception as e:
            print_check("Basic tensor operations", False, f"Error: {e}")
            return False
        
        return True
        
    except ImportError:
        print_check("PyTorch Installation", False, "Install with: pip install torch torchvision")
        return False


def check_dataset():
    """Check if MVFouls dataset is available"""
    print_header("Dataset Availability")
    
    # Check main dataset directory
    dataset_dir = Path("mvfouls")
    dataset_exists = dataset_dir.exists() and dataset_dir.is_dir()
    
    print_check(f"Dataset directory 'mvfouls'", dataset_exists)
    
    if not dataset_exists:
        print("   Please ensure the MVFouls dataset is available in the current directory")
        return False
    
    # Check required splits
    required_splits = ["train", "test"]
    splits_ok = True
    
    for split in required_splits:
        split_path = dataset_dir / split
        split_exists = split_path.exists() and split_path.is_dir()
        print_check(f"Split '{split}'", split_exists)
        
        if split_exists:
            # Count subdirectories (action classes)
            subdirs = [d for d in split_path.iterdir() if d.is_dir()]
            print(f"     Found {len(subdirs)} action classes")
        else:
            splits_ok = False
    
    return dataset_exists and splits_ok


def check_enhanced_components():
    """Check if enhanced training components are available"""
    print_header("Enhanced Training Components")
    
    required_files = [
        'complete_training_integration.py',
        'enhanced_training_config.py',
        'enhanced_training_error_handling.py',
        'enhanced_data_pipeline.py',
        'adaptive_loss_system.py',
        'advanced_training_strategies.py',
        'curriculum_learning_system.py',
        'performance_monitor.py',
        'model_ensemble_system.py',
        'hyperparameter_optimizer.py'
    ]
    
    all_files_ok = True
    
    for filename in required_files:
        file_exists = Path(filename).exists()
        print_check(f"Component: {filename}", file_exists)
        if not file_exists:
            all_files_ok = False
    
    return all_files_ok


def check_integration_modules():
    """Check if integration modules are available"""
    print_header("Integration Modules")
    
    integration_files = [
        'enhanced_pipeline_integration.py',
        'adaptive_loss_integration.py',
        'advanced_training_integration.py',
        'curriculum_learning_integration.py',
        'performance_monitor_integration.py',
        'ensemble_integration.py',
        'hyperparameter_optimization_integration.py'
    ]
    
    all_integration_ok = True
    
    for filename in integration_files:
        file_exists = Path(filename).exists()
        print_check(f"Integration: {filename}", file_exists)
        if not file_exists:
            all_integration_ok = False
    
    return all_integration_ok


def check_system_resources():
    """Check system resources"""
    print_header("System Resources")
    
    try:
        import psutil
        
        # Check memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / 1e9
        memory_ok = memory_gb >= 8  # Recommend at least 8GB
        
        print_check(
            f"System Memory: {memory_gb:.1f} GB",
            memory_ok,
            "Recommend at least 8GB for training" if not memory_ok else "Sufficient"
        )
        
        # Check disk space
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / 1e9
        disk_ok = disk_free_gb >= 5  # Recommend at least 5GB free
        
        print_check(
            f"Free Disk Space: {disk_free_gb:.1f} GB",
            disk_ok,
            "Recommend at least 5GB free" if not disk_ok else "Sufficient"
        )
        
        # Check CPU
        cpu_count = psutil.cpu_count()
        print_check(f"CPU Cores: {cpu_count}", True)
        
        return memory_ok and disk_ok
        
    except ImportError:
        print_check("System resource check", False, "Install psutil for detailed checks")
        return True  # Don't fail validation for this


def test_basic_import():
    """Test basic import of the enhanced training system"""
    print_header("Import Test")
    
    try:
        from complete_training_integration import CompleteEnhancedTrainingSystem
        print_check("Import CompleteEnhancedTrainingSystem", True)
        
        from enhanced_training_config import create_default_config
        print_check("Import configuration system", True)
        
        from enhanced_training_error_handling import ErrorHandler
        print_check("Import error handling system", True)
        
        # Test basic functionality
        config = create_default_config()
        print_check("Create default configuration", True)
        
        return True
        
    except Exception as e:
        print_check("Import enhanced training system", False, f"Error: {e}")
        return False


def run_mini_test():
    """Run a mini functionality test"""
    print_header("Mini Functionality Test")
    
    try:
        import torch
        from adaptive_loss_system import DynamicLossSystem
        
        # Test adaptive loss system
        loss_system = DynamicLossSystem(
            num_action_classes=8,
            num_severity_classes=4
        )
        
        # Test forward pass
        batch_size = 2
        action_logits = torch.randn(batch_size, 8, requires_grad=True)
        severity_logits = torch.randn(batch_size, 4, requires_grad=True)
        action_targets = torch.randint(0, 8, (batch_size,))
        severity_targets = torch.randint(0, 4, (batch_size,))
        
        total_loss, loss_components = loss_system(
            action_logits, severity_logits, action_targets, severity_targets
        )
        
        print_check("Adaptive loss system test", True)
        
        # Test backward pass
        total_loss.backward()
        print_check("Gradient computation test", True)
        
        return True
        
    except Exception as e:
        print_check("Mini functionality test", False, f"Error: {e}")
        return False


def main():
    """Run complete setup validation"""
    print("🔍 Enhanced Training System Setup Validation")
    print("This script will check if your environment is ready for training.")
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("PyTorch Setup", check_pytorch_setup),
        ("Dataset Availability", check_dataset),
        ("Enhanced Components", check_enhanced_components),
        ("Integration Modules", check_integration_modules),
        ("System Resources", check_system_resources),
        ("Import Test", test_basic_import),
        ("Mini Functionality Test", run_mini_test)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} failed with exception: {e}")
            results.append((check_name, False))
    
    # Summary
    print_header("Validation Summary")
    
    passed_checks = sum(1 for _, result in results if result)
    total_checks = len(results)
    
    print(f"Checks passed: {passed_checks}/{total_checks}")
    
    if passed_checks == total_checks:
        print("\n🎉 ALL CHECKS PASSED!")
        print("✅ Your environment is ready for enhanced training!")
        print("\nNext steps:")
        print("1. Run quick test: python run_enhanced_training.py --mode quick_test")
        print("2. Run production: python run_enhanced_training.py --mode production")
    else:
        print(f"\n⚠️  {total_checks - passed_checks} checks failed!")
        print("❌ Please address the issues above before running training.")
        print("\nFailed checks:")
        for check_name, result in results:
            if not result:
                print(f"  - {check_name}")
        
        print("\nRecommended actions:")
        print("1. Install missing packages: pip install torch torchvision numpy scikit-learn matplotlib tqdm psutil pyyaml")
        print("2. Ensure MVFouls dataset is in the 'mvfouls/' directory")
        print("3. Verify all enhanced training files are present")
    
    return passed_checks == total_checks


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)