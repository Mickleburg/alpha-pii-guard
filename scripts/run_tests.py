"""Script to run test suite."""

import argparse
import subprocess
import sys
from pathlib import Path

from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def run_pytest(
    test_dir: str = "tests",
    verbose: bool = True,
    coverage: bool = False,
    markers: str = None,
    failfast: bool = False
) -> int:
    """
    Run pytest programmatically.
    
    Args:
        test_dir: Directory containing tests
        verbose: Verbose output
        coverage: Generate coverage report
        markers: Pytest markers to filter tests
        failfast: Stop on first failure
        
    Returns:
        Exit code
    """
    cmd = ["pytest", test_dir]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    if markers:
        cmd.extend(["-m", markers])
    
    if failfast:
        cmd.append("-x")
    
    cmd.append("--tb=short")
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=Path.cwd())
    
    return result.returncode


def run_unit_tests() -> int:
    """Run unit tests only."""
    logger.info("Running unit tests...")
    return run_pytest(test_dir="tests", markers="not integration")


def run_integration_tests() -> int:
    """Run integration tests."""
    logger.info("Running integration tests...")
    return run_pytest(test_dir="tests", markers="integration")


def run_all_tests(coverage: bool = False) -> int:
    """Run all tests."""
    logger.info("Running all tests...")
    return run_pytest(test_dir="tests", coverage=coverage)


def run_specific_test(test_file: str) -> int:
    """Run specific test file."""
    logger.info(f"Running specific test: {test_file}")
    return run_pytest(test_dir=test_file, verbose=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run test suite")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests"
    )
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests only"
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests only"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Run specific test file"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--failfast",
        action="store_true",
        help="Stop on first failure"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("="*60)
    logger.info("RUNNING TEST SUITE")
    logger.info("="*60)
    
    # Run tests
    if args.file:
        exit_code = run_specific_test(args.file)
    elif args.unit:
        exit_code = run_unit_tests()
    elif args.integration:
        exit_code = run_integration_tests()
    else:
        # Default: run all tests
        exit_code = run_all_tests(coverage=args.coverage)
    
    logger.info("="*60)
    if exit_code == 0:
        logger.info("ALL TESTS PASSED")
    else:
        logger.error(f"TESTS FAILED (exit code: {exit_code})")
    logger.info("="*60)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
