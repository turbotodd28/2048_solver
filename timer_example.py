#!/usr/bin/env python3
"""
Example script demonstrating the Timer utility for measuring runtime.
"""

import time
from src.timer_utils import Timer, format_runtime


def some_work():
    """Simulate some work"""
    time.sleep(2.5)  # Simulate 2.5 seconds of work


def main():
    # Method 1: Using the Timer context manager (recommended)
    print("=== Method 1: Timer Context Manager ===")
    with Timer("Total runtime"):
        some_work()
        print("Work completed!")
    
    print()
    
    # Method 2: Manual timing with format_runtime function
    print("=== Method 2: Manual Timing ===")
    start_time = time.time()
    some_work()
    runtime_seconds = time.time() - start_time
    print(f"Total runtime: {format_runtime(runtime_seconds)}")
    
    print()
    
    # Method 3: Nested timers for different sections
    print("=== Method 3: Nested Timers ===")
    with Timer("Total runtime"):
        with Timer("Section 1"):
            time.sleep(1.0)
        
        with Timer("Section 2"):
            time.sleep(1.5)
        
        print("All sections completed!")


if __name__ == "__main__":
    main() 