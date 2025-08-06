#!/usr/bin/env python3
"""
Utility functions for measuring and formatting runtime.
"""

import time


def format_runtime(seconds):
    """Format runtime in hours, minutes, and seconds with decimal precision"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes:02d}m {seconds_remainder:05.2f}s"
    elif minutes > 0:
        return f"{minutes}m {seconds_remainder:05.2f}s"
    else:
        return f"{seconds_remainder:.2f}s"


class Timer:
    """Context manager for measuring and formatting runtime"""
    
    def __init__(self, description="Total runtime"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        runtime_seconds = self.end_time - self.start_time
        print(f"{self.description}: {format_runtime(runtime_seconds)}")
    
    def elapsed(self):
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0
        return time.time() - self.start_time 