#!/usr/bin/env python3
"""
Dependency Checker for ECG RLHF GRPO Training
==============================================

This script validates all dependencies, configurations, and requirements
before running GRPO training to catch issues early.

Usage:
    python check_dependencies.py [--fix]

Options:
    --fix    Attempt to automatically fix common issues
"""

import sys
import os
import subprocess
import importlib.metadata
from pathlib import Path
from typing import List, Tuple, Dict
import argparse


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_section(title: str):
    """Print a section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")


def print_check(name: str, passed: bool, message: str = ""):
    """Print check result"""
    status = f"{Colors.GREEN}✓ PASS{Colors.END}" if passed else f"{Colors.RED}✗ FAIL{Colors.END}"
    print(f"{status} {name}")
    if message:
        print(f"       {message}")


def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ WARNING: {message}{Colors.END}")


def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}✗ ERROR: {message}{Colors.END}")


def print_info(message: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ INFO: {message}{Colors.END}")


def get_package_version(package_name: str) -> str:
    """Get installed package version"""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_python_version() -> bool:
    """Check Python version is >= 3.10"""
    version = sys.version_info
    required = (3, 10)
    passed = version >= required

    print_check(
        "Python Version",
        passed,
        f"Current: {version.major}.{version.minor}.{version.micro}, Required: >= {required[0]}.{required[1]}"
    )
    return passed


def check_cuda_available() -> bool:
    """Check if CUDA is available"""
    try:
        import torch
        available = torch.cuda.is_available()
        if available:
            n_gpus = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
            print_check(
                "CUDA Availability",
                True,
                f"Found {n_gpus} GPU(s): {', '.join(gpu_names)}"
            )
        else:
            print_check("CUDA Availability", False, "No CUDA devices found")
        return available
    except ImportError:
        print_check("CUDA Availability", False, "PyTorch not installed")
        return False


def check_gpu_memory() -> bool:
    """Check GPU memory availability"""
    try:
        import torch
        if not torch.cuda.is_available():
            return False

        n_gpus = torch.cuda.device_count()
        all_sufficient = True

        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / 1024**3

            # Recommend at least 16GB for GRPO training
            sufficient = total_gb >= 16
            all_sufficient = all_sufficient and sufficient

            print_check(
                f"GPU {i} Memory",
                sufficient,
                f"{total_gb:.1f} GB (Recommended: >= 16 GB)"
            )

        return all_sufficient
    except Exception as e:
        print_error(f"Failed to check GPU memory: {e}")
        return False


def check_core_packages() -> Dict[str, bool]:
    """Check core ML framework versions"""
    packages = {
        'torch': '2.6.0',
        'transformers': '4.56.0',
        'accelerate': '0.34.2',
        'peft': '>=0.13.0',
        'datasets': '>=2.14.0',
    }

    results = {}
    for package, expected in packages.items():
        version = get_package_version(package)
        if version is None:
            print_check(package, False, f"Not installed (Required: {expected})")
            results[package] = False
        else:
            # Simple version check (exact match or >= for flexible requirements)
            if expected.startswith('>='):
                passed = True  # Just check it's installed for >= requirements
            else:
                passed = version == expected

            status = "✓" if passed else "⚠"
            print_check(
                package,
                passed,
                f"v{version} (Expected: {expected})"
            )
            results[package] = passed

    return results


def check_ray_opentelemetry_compatibility() -> Tuple[bool, str]:
    """Check Ray and OpenTelemetry version compatibility"""
    ray_version = get_package_version('ray')
    otel_version = get_package_version('opentelemetry-api')

    if ray_version is None:
        print_check("Ray", False, "Not installed")
        return False, "Ray not installed"

    if otel_version is None:
        print_check("OpenTelemetry", False, "Not installed")
        return False, "OpenTelemetry not installed"

    # Known compatibility issue: Ray 2.50.1 has issues with OpenTelemetry 1.26.0+
    ray_major_minor = '.'.join(ray_version.split('.')[:2])
    otel_major_minor = '.'.join(otel_version.split('.')[:2])

    # Check for known incompatibility
    incompatible = (ray_major_minor == '2.50' and otel_major_minor >= '1.26')

    if incompatible:
        print_check(
            "Ray ↔ OpenTelemetry Compatibility",
            False,
            f"Ray {ray_version} has issues with OpenTelemetry {otel_version}"
        )
        return False, f"Incompatible versions: Ray {ray_version}, OpenTelemetry {otel_version}"
    else:
        print_check(
            "Ray ↔ OpenTelemetry Compatibility",
            True,
            f"Ray {ray_version}, OpenTelemetry {otel_version}"
        )
        return True, ""


def check_verl_installation() -> bool:
    """Check if VERL is properly installed"""
    try:
        import verl
        if verl.__file__ is not None:
            verl_path = Path(verl.__file__).parent.parent
            print_check(
                "VERL Installation",
                True,
                f"Installed at: {verl_path}"
            )
        else:
            print_check(
                "VERL Installation",
                True,
                "Installed (builtin module)"
            )
        return True
    except ImportError:
        print_check("VERL Installation", False, "VERL not found")
        return False
    except Exception as e:
        print_check("VERL Installation", False, f"Error checking VERL: {e}")
        return False


def check_vllm_installation() -> bool:
    """Check vLLM installation (required for GRPO rollout)"""
    version = get_package_version('vllm')
    if version is None:
        print_check("vLLM", False, "Not installed (Required for GRPO rollout)")
        return False
    else:
        print_check("vLLM", True, f"v{version}")
        return True


def check_data_files() -> bool:
    """Check if required data files exist"""
    data_dir = Path(__file__).parent / "data" / "processed"

    required_files = [
        "ECG_Knowledge_Basic_Q_A_grpo.parquet",
        "ECG_Knowledge_Basic_Q_A_val.parquet",
    ]

    all_exist = True
    for filename in required_files:
        filepath = data_dir / filename
        exists = filepath.exists()
        all_exist = all_exist and exists

        if exists:
            size_mb = filepath.stat().st_size / 1024**2
            print_check(filename, True, f"Found ({size_mb:.2f} MB)")
        else:
            print_check(filename, False, f"Not found at {filepath}")

    return all_exist


def check_model_access() -> bool:
    """Check if model is accessible"""
    model_path = "meta-llama/Llama-3.2-3B-Instruct"

    try:
        from transformers import AutoTokenizer
        print_info(f"Checking access to {model_path}...")

        # Try to load tokenizer (faster than loading full model)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print_check(
            "Model Access",
            True,
            f"{model_path} is accessible"
        )
        return True
    except Exception as e:
        print_check(
            "Model Access",
            False,
            f"Cannot access {model_path}: {str(e)[:100]}"
        )
        return False


def check_environment_variables() -> bool:
    """Check recommended environment variables"""
    all_good = True

    # Check CUDA_VISIBLE_DEVICES
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_devices:
        print_check(
            "CUDA_VISIBLE_DEVICES",
            True,
            f"Set to: {cuda_devices}"
        )
    else:
        print_warning("CUDA_VISIBLE_DEVICES not set (will use all GPUs)")

    # Check Ray workarounds for OpenTelemetry issue
    ray_workarounds = [
        'RAY_USAGE_STATS_ENABLED',
        'RAY_OTEL_ENABLED',
    ]

    missing_workarounds = []
    for var in ray_workarounds:
        if os.environ.get(var) != '0':
            missing_workarounds.append(var)

    if missing_workarounds:
        print_warning(
            f"Ray workarounds not set: {', '.join(missing_workarounds)}\n"
            f"       Consider setting these to '0' to avoid telemetry issues"
        )
        all_good = False
    else:
        print_check("Ray Workarounds", True, "Telemetry disabled")

    return all_good


def suggest_fixes(issues: List[str]):
    """Suggest fixes for common issues"""
    if not issues:
        return

    print_section("Suggested Fixes")

    for issue in issues:
        print(f"\n{Colors.YELLOW}Issue:{Colors.END} {issue}")

        if "Ray" in issue and "OpenTelemetry" in issue:
            print(f"{Colors.GREEN}Fix:{Colors.END}")
            print("  Add these lines to your training script or shell script:")
            print("    export RAY_USAGE_STATS_ENABLED=0")
            print("    export RAY_OTEL_ENABLED=0")
            print("    export RAY_DASHBOARD_METRICS_ENABLED=0")

        elif "data" in issue.lower() or "file" in issue.lower():
            print(f"{Colors.GREEN}Fix:{Colors.END}")
            print("  Run data preparation:")
            print("    python prepare_data.py")

        elif "model" in issue.lower() or "access" in issue.lower():
            print(f"{Colors.GREEN}Fix:{Colors.END}")
            print("  1. Login to Hugging Face:")
            print("     huggingface-cli login")
            print("  2. Accept model license at:")
            print("     https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")

        elif "vllm" in issue.lower():
            print(f"{Colors.GREEN}Fix:{Colors.END}")
            print("  Install vLLM:")
            print("    pip install 'vllm>=0.5.0'")

        elif "verl" in issue.lower():
            print(f"{Colors.GREEN}Fix:{Colors.END}")
            print("  Install VERL:")
            print("    cd verl && pip install -e .")


def apply_automatic_fixes():
    """Apply automatic fixes where possible"""
    print_section("Applying Automatic Fixes")

    # Set Ray environment variables
    os.environ['RAY_USAGE_STATS_ENABLED'] = '0'
    os.environ['RAY_OTEL_ENABLED'] = '0'
    os.environ['RAY_DASHBOARD_METRICS_ENABLED'] = '0'
    print_info("Set Ray environment variables to disable telemetry")

    print_warning("Note: These fixes only apply to the current session")
    print_info("Add them to your training script for permanent effect")


def main():
    parser = argparse.ArgumentParser(description='Check GRPO training dependencies')
    parser.add_argument('--fix', action='store_true', help='Apply automatic fixes')
    args = parser.parse_args()

    print(f"{Colors.BOLD}ECG RLHF GRPO - Dependency Checker{Colors.END}")
    print("=" * 70)

    # Early check: Python environment
    version = sys.version_info
    if version < (3, 10):
        print_error(f"Python {version.major}.{version.minor} detected!")
        print_warning("You may be running this with system Python instead of conda environment")
        print_info("Try: conda activate rlhf && python check_dependencies.py")
        print()

    all_passed = True
    issues = []

    # Python version
    print_section("Python Environment")
    if not check_python_version():
        all_passed = False
        issues.append("Python version < 3.10")

    # CUDA and GPU
    print_section("GPU & CUDA")
    if not check_cuda_available():
        all_passed = False
        issues.append("CUDA not available")
    else:
        if not check_gpu_memory():
            print_warning("Some GPUs have less than 16GB memory")

    # Core packages
    print_section("Core Packages")
    package_results = check_core_packages()
    if not all(package_results.values()):
        all_passed = False
        issues.append("Some core packages missing or wrong version")

    # VERL and vLLM
    print_section("RLHF Framework")
    if not check_verl_installation():
        all_passed = False
        issues.append("VERL not installed")

    if not check_vllm_installation():
        all_passed = False
        issues.append("vLLM not installed")

    # Ray compatibility
    print_section("Ray Configuration")
    compatible, msg = check_ray_opentelemetry_compatibility()
    if not compatible:
        issues.append(msg)
        # This is a known issue with workaround, don't fail completely
        print_warning("Known compatibility issue - workarounds available")

    check_environment_variables()

    # Data files
    print_section("Data Files")
    if not check_data_files():
        issues.append("Required data files missing")
        # Don't fail for missing data, might be intentional
        print_warning("Some data files missing - run prepare_data.py if needed")

    # Model access (optional check, might require HF login)
    print_section("Model Access")
    print_info("Checking model access (may take a moment)...")
    if not check_model_access():
        print_warning("Model access check failed - you may need to login")
        print_info("This check can fail if model isn't cached yet")

    # Summary
    print_section("Summary")

    if all_passed and not issues:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All checks passed!{Colors.END}")
        print(f"{Colors.GREEN}Your environment is ready for GRPO training.{Colors.END}\n")
        return 0
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠ Some issues detected{Colors.END}\n")

        if issues:
            suggest_fixes(issues)

        if args.fix:
            apply_automatic_fixes()
        else:
            print(f"\n{Colors.BLUE}Tip: Run with --fix to apply automatic fixes{Colors.END}")

        print(f"\n{Colors.YELLOW}You can still try to run training, but issues may occur.{Colors.END}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
