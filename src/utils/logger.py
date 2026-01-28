"""
Enhanced Logger with Colorful Output

Features:
- Colorful stage indicators
- Clean phase separators
- Filtered verbose logs
"""

import logging
import colorlog
from typing import Optional


# ANSI color codes
COLORS = {
    'reset': '\033[0m',
    'cyan': '\033[96m',
    'blue': '\033[94m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'red': '\033[91m',
    'magenta': '\033[95m',
    'white': '\033[97m',
    'purple': '\033[35m',
}


# Stage emoji and colors
STAGE_STYLES = {
    "coordinator": {"emoji": "🎬", "color": "cyan"},
    "supervisor": {"emoji": "🎯", "color": "magenta"},
    "collector": {"emoji": "📰", "color": "blue"},
    "collection": {"emoji": "📦", "color": "blue"},
    "analysis": {"emoji": "🔍", "color": "yellow"},
    "summary": {"emoji": "📝", "color": "green"},
    "graphrag": {"emoji": "🕸️", "color": "purple"},
    "extractor": {"emoji": "🔬", "color": "cyan"},
    "indexer": {"emoji": "💾", "color": "blue"},
    "retriever": {"emoji": "🔎", "color": "yellow"},
    "reasoner": {"emoji": "🧠", "color": "magenta"},
    "gatekeeper": {"emoji": "🚪", "color": "cyan"},
    "planner": {"emoji": "📋", "color": "blue"},
    "auditor": {"emoji": "✅", "color": "green"},
}


def setup_colorful_logger(log_level: str = "INFO", verbose: bool = False):
    """
    Setup colorful logger with clean output.

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR)
        verbose: If False, suppress verbose logs from httpx, arxiv, etc.
    """

    # Color formatter
    log_format = "%(log_color)s%(asctime)s %(levelname)-8s%(reset)s %(message)s"

    color_formatter = colorlog.ColoredFormatter(
        log_format,
        datefmt='%H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(color_formatter)
    root_logger.addHandler(console_handler)

    # Filter verbose loggers if not in verbose mode
    if not verbose:
        # Suppress httpx detailed logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

        # Suppress arxiv info logs
        logging.getLogger("arxiv").setLevel(logging.WARNING)

        # Suppress crawler debug logs
        logging.getLogger("src.services.crawler").setLevel(logging.WARNING)

        # Suppress langsmith trace logs
        logging.getLogger("langsmith").setLevel(logging.WARNING)

        # Suppress other verbose libraries
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)


def print_stage_header(stage: str, message: str = ""):
    """
    Print a colorful stage header.

    Args:
        stage: Stage name (coordinator, collector, analysis, etc.)
        message: Optional additional message
    """
    stage_lower = stage.lower()
    style = STAGE_STYLES.get(stage_lower, {"emoji": "📌", "color": "white"})

    emoji = style["emoji"]
    separator = "═" * 60

    # Print separator
    print(f"\n{COLORS['cyan']}{separator}{COLORS['reset']}")

    # Print stage name
    color_code = COLORS.get(style["color"], COLORS['white'])
    reset_code = COLORS['reset']

    stage_text = f"{emoji} {stage.upper()}"
    if message:
        stage_text += f": {message}"

    print(f"{color_code}{stage_text}{reset_code}")
    print(f"{COLORS['cyan']}{separator}{COLORS['reset']}\n")


def print_substage(name: str, status: str = ""):
    """
    Print a substage indicator.

    Args:
        name: Substage name
        status: Optional status (running, done, failed)
    """
    status_symbols = {
        "running": "⏳",
        "done": "✅",
        "failed": "❌",
        "": "▶️"
    }

    symbol = status_symbols.get(status, "▶️")

    color = "green" if status == "done" else "yellow" if status == "running" else "red" if status == "failed" else "white"
    color_code = COLORS.get(color, COLORS['white'])
    reset_code = COLORS['reset']

    print(f"  {color_code}{symbol} {name}{reset_code}")


def print_metric(name: str, value: any, unit: str = ""):
    """
    Print a metric.

    Args:
        name: Metric name
        value: Metric value
        unit: Optional unit
    """
    value_str = f"{value} {unit}" if unit else str(value)
    print(f"  📊 {name}: {COLORS['cyan']}{value_str}{COLORS['reset']}")


def print_success(message: str):
    """Print success message."""
    print(f"{COLORS['green']}✅ {message}{COLORS['reset']}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{COLORS['yellow']}⚠️  {message}{COLORS['reset']}")


def print_error(message: str):
    """Print error message."""
    print(f"{COLORS['red']}❌ {message}{COLORS['reset']}")


def print_info(message: str, emoji: str = "ℹ️"):
    """Print info message."""
    print(f"{COLORS['cyan']}{emoji} {message}{COLORS['reset']}")


# Create a stage-aware logger
class StageLogger:
    """Logger with stage awareness."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.current_stage = None

    def set_stage(self, stage: str, message: str = ""):
        """Set current stage and print header."""
        self.current_stage = stage
        print_stage_header(stage, message)

    def substage(self, name: str, status: str = ""):
        """Print substage."""
        print_substage(name, status)

    def metric(self, name: str, value: any, unit: str = ""):
        """Print metric."""
        print_metric(name, value, unit)

    def success(self, message: str):
        """Log success."""
        print_success(message)
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning."""
        print_warning(message)
        self.logger.warning(message)

    def error(self, message: str):
        """Log error."""
        print_error(message)
        self.logger.error(message)

    def info(self, message: str, emoji: str = "ℹ️"):
        """Log info."""
        print_info(message, emoji)
        self.logger.info(message)

    def debug(self, message: str):
        """Log debug (standard logger)."""
        self.logger.debug(message)


def get_stage_logger(name: str) -> StageLogger:
    """Get a stage-aware logger."""
    return StageLogger(name)
