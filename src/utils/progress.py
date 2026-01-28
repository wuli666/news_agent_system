"""
Progress Bar Utilities

Simple dynamic progress display without external dependencies.
"""

import sys
import time
from typing import Optional


class ProgressBar:
    """
    Simple progress bar that updates in place.

    Example:
        with ProgressBar(total=10, desc="Processing") as pbar:
            for i in range(10):
                pbar.update(current_url="https://example.com/page" + str(i))
                time.sleep(0.1)
    """

    def __init__(self, total: int = 100, desc: str = "", width: int = 40):
        """
        Initialize progress bar.

        Args:
            total: Total number of items
            desc: Description prefix
            width: Width of progress bar in characters
        """
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
        self.start_time = time.time()
        self.current_info = ""

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit - print newline."""
        sys.stdout.write('\n')
        sys.stdout.flush()

    def update(self, n: int = 1, current_url: Optional[str] = None, status: str = ""):
        """
        Update progress bar.

        Args:
            n: Increment amount
            current_url: Current URL being processed
            status: Status message
        """
        self.current += n
        self.current = min(self.current, self.total)

        # Build info text
        info_parts = []
        if current_url:
            # Truncate long URLs
            if len(current_url) > 60:
                display_url = current_url[:57] + "..."
            else:
                display_url = current_url
            info_parts.append(display_url)
        if status:
            info_parts.append(status)

        self.current_info = " | ".join(info_parts)

        self._render()

    def _render(self):
        """Render the progress bar."""
        # Calculate percentage
        percent = self.current / self.total if self.total > 0 else 0
        filled_width = int(self.width * percent)

        # Build bar
        bar = '█' * filled_width + '░' * (self.width - filled_width)

        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        if self.current > 0 and elapsed > 0:
            rate = self.current / elapsed
            eta = (self.total - self.current) / rate if rate > 0 else 0
            time_info = f"{elapsed:.1f}s | ETA: {eta:.1f}s"
        else:
            time_info = "0.0s"

        # Build output line
        output = f"\r{self.desc}: [{bar}] {self.current}/{self.total} ({percent*100:.0f}%) | {time_info}"

        # Add current info if available
        if self.current_info:
            # Truncate if too long
            max_info_len = 80
            if len(self.current_info) > max_info_len:
                self.current_info = self.current_info[:max_info_len-3] + "..."
            output += f"\n  ▶ {self.current_info}"
            # Clear previous info line
            output = "\r\033[K" + output.replace("\n", "\n\033[K")

        # Write to stdout
        sys.stdout.write('\r\033[K' + output)
        sys.stdout.flush()

    def set_description(self, desc: str):
        """Update description."""
        self.desc = desc
        self._render()

    def close(self):
        """Close the progress bar."""
        sys.stdout.write('\n')
        sys.stdout.flush()


class SimpleSpinner:
    """
    Simple spinner for indefinite progress.

    Example:
        with SimpleSpinner("Loading") as spinner:
            for item in items:
                spinner.update(f"Processing {item}")
                process(item)
    """

    FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    def __init__(self, desc: str = ""):
        self.desc = desc
        self.frame_index = 0
        self.current_info = ""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        sys.stdout.write('\n')
        sys.stdout.flush()

    def update(self, info: str = ""):
        """Update spinner with new info."""
        self.current_info = info
        self._render()

    def _render(self):
        """Render the spinner."""
        frame = self.FRAMES[self.frame_index % len(self.FRAMES)]
        self.frame_index += 1

        output = f"\r{frame} {self.desc}"
        if self.current_info:
            # Truncate long info
            if len(self.current_info) > 80:
                display_info = self.current_info[:77] + "..."
            else:
                display_info = self.current_info
            output += f": {display_info}"

        sys.stdout.write('\r\033[K' + output)
        sys.stdout.flush()

    def close(self):
        """Close the spinner."""
        sys.stdout.write('\n')
        sys.stdout.flush()
