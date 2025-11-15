"""
Fix R-Python integration for sits package

This script helps diagnose and fix issues when trying to use R sits package
from Python via rpy2/reticulate.

Usage:
    python fix_r_python_integration.py
"""

import os
import sys
import subprocess

def check_r_installation():
    """Check if R is installed and accessible"""
    print("=" * 70)
    print("STEP 1: Checking R Installation")
    print("=" * 70)

    try:
        result = subprocess.run(['R', '--version'],
                              capture_output=True, text=True, check=True)
        print("✓ R is installed:")
        print(result.stdout.split('\n')[0])
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ R is not found in PATH")
        print("  Install R: conda install -c conda-forge r-base")
        return False

    return True

def check_sits_in_r():
    """Check if sits package is installed in R"""
    print("\n" + "=" * 70)
    print("STEP 2: Checking sits Package in R")
    print("=" * 70)

    r_code = """
    if (!require("sits", quietly = TRUE)) {
        cat("✗ sits package NOT installed in R\\n")
        cat("  Install with: install.packages('sits')\\n")
        quit(status = 1)
    } else {
        cat("✓ sits package installed:\\n")
        cat(paste("  Version:", packageVersion("sits"), "\\n"))
        cat(paste("  Location:", find.package("sits"), "\\n"))
    }
    """

    try:
        result = subprocess.run(['R', '--vanilla', '--slave', '-e', r_code],
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print("\nTo install sits in R:")
        print("  1. Open R or RStudio")
        print("  2. Run: install.packages('sits')")
        print("  3. Or via conda: conda install -c conda-forge r-sits")
        return False

def check_rpy2():
    """Check if rpy2 is installed"""
    print("\n" + "=" * 70)
    print("STEP 3: Checking rpy2 (R-Python Interface)")
    print("=" * 70)

    try:
        import rpy2
        print(f"✓ rpy2 installed: version {rpy2.__version__}")
        return True
    except ImportError:
        print("✗ rpy2 not installed")
        print("  Install with: pip install rpy2")
        return False

def fix_r_home():
    """Set R_HOME environment variable"""
    print("\n" + "=" * 70)
    print("STEP 4: Setting R_HOME Environment Variable")
    print("=" * 70)

    # Get R home directory
    r_code = 'cat(R.home())'
    try:
        result = subprocess.run(['R', '--vanilla', '--slave', '-e', r_code],
                              capture_output=True, text=True, check=True)
        r_home = result.stdout.strip()

        print(f"R_HOME detected: {r_home}")

        # Set environment variable
        os.environ['R_HOME'] = r_home

        print("✓ R_HOME environment variable set")
        print(f"\nAdd this to your shell profile (~/.bashrc or ~/.zshrc):")
        print(f'export R_HOME="{r_home}"')

        return r_home
    except subprocess.CalledProcessError:
        print("✗ Could not detect R_HOME")
        return None

def test_rpy2_sits():
    """Test if sits can be imported via rpy2"""
    print("\n" + "=" * 70)
    print("STEP 5: Testing sits Import via rpy2")
    print("=" * 70)

    try:
        from rpy2.robjects.packages import importr

        print("Attempting to import sits...")
        sits = importr('sits')

        print("✓ SUCCESS! sits imported via rpy2")
        print(f"  sits functions available: {len(dir(sits))}")
        return True

    except Exception as e:
        print(f"✗ FAILED to import sits via rpy2")
        print(f"  Error: {e}")
        print("\nThis is usually caused by:")
        print("  1. sits not installed in R")
        print("  2. R_HOME not set correctly")
        print("  3. Conda R vs system R conflict")
        return False

def provide_solutions():
    """Provide alternative solutions"""
    print("\n" + "=" * 70)
    print("SOLUTIONS")
    print("=" * 70)

    print("\n⭐ RECOMMENDED: Use Separate R and Python Sessions")
    print("-" * 70)
    print("You DON'T need to mix R and Python in the same session!")
    print()
    print("Workflow:")
    print("  1. Run sits code in R (or RStudio)")
    print("     library(sits)")
    print("     # ... extract data ...")
    print("     sits_to_fusets_csv(data, 'output.csv')")
    print()
    print("  2. Run FuseTS code in Python (Jupyter)")
    print("     from fusets.io.sits_bridge import load_sits_csv")
    print("     data = load_sits_csv('output.csv')")
    print()

    print("\n🔧 ALTERNATIVE: Fix rpy2 Integration")
    print("-" * 70)
    print("If you really need R-Python integration:")
    print()
    print("Option A: Use conda-managed R")
    print("  conda create -n fusets_r python=3.9 r-base r-sits")
    print("  conda activate fusets_r")
    print("  pip install rpy2 fusets")
    print()
    print("Option B: Set R_HOME manually in Python")
    print("  import os")
    print("  os.environ['R_HOME'] = '/path/to/R'  # from 'R.home()' in R")
    print("  import rpy2.robjects as ro")
    print()
    print("Option C: Use separate conda environments")
    print("  # R environment")
    print("  conda create -n sits_r r-base r-sits")
    print()
    print("  # Python environment")
    print("  conda create -n fusets_py python=3.9")
    print("  conda activate fusets_py")
    print("  pip install fusets")
    print()
    print("  # Use CSV/GeoTIFF files to bridge the two")

def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  R-Python Integration Diagnostics for sits + FuseTS".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    # Run diagnostics
    r_ok = check_r_installation()
    if not r_ok:
        provide_solutions()
        return

    sits_ok = check_sits_in_r()
    rpy2_ok = check_rpy2()

    if not rpy2_ok:
        provide_solutions()
        return

    r_home = fix_r_home()

    if sits_ok and rpy2_ok and r_home:
        test_rpy2_sits()

    # Always show solutions
    provide_solutions()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"R installed: {'✓' if r_ok else '✗'}")
    print(f"sits installed: {'✓' if sits_ok else '✗'}")
    print(f"rpy2 installed: {'✓' if rpy2_ok else '✗'}")
    print(f"R_HOME set: {'✓' if r_home else '✗'}")
    print()
    print("For the notebooks in /home/unika_sianturi/work/FuseTS/notebooks/:")
    print("  → You DON'T need rpy2")
    print("  → Just run R code in R, Python code in Python")
    print("  → Use CSV/GeoTIFF files to transfer data between them")
    print()

if __name__ == "__main__":
    main()
