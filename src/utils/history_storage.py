"""
Historical news data storage.

Similar to TrendRadar, stores news data in daily folders for historical analysis.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class HistoryStorage:
    """Store and retrieve historical news data."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize history storage.

        Args:
            output_dir: Directory to store historical data
        """
        if output_dir is None:
            # Default: project_root/output
            project_root = Path(__file__).resolve().parents[2]
            output_dir = project_root / "output"

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"History storage initialized at: {self.output_dir}")

    def _get_date_folder(self, date: datetime) -> Path:
        """
        Get folder path for a specific date.

        Args:
            date: Target date

        Returns:
            Path to date folder (e.g., output/2025年11月19日/)
        """
        folder_name = date.strftime("%Y年%m月%d日")
        return self.output_dir / folder_name

    def _get_json_file_path(self, date: datetime) -> Path:
        """
        Get JSON file path for a specific date and time.

        Args:
            date: Target datetime

        Returns:
            Path to JSON file (e.g., output/2025年11月19日/json/15时30分.json)
        """
        date_folder = self._get_date_folder(date)
        json_folder = date_folder / "json"
        json_folder.mkdir(parents=True, exist_ok=True)

        file_name = date.strftime("%H时%M分.json")
        return json_folder / file_name

    def save_news_data(
        self,
        news_data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> Path:
        """
        Save news data to historical storage.

        Args:
            news_data: News data dictionary
            timestamp: Timestamp (defaults to now)

        Returns:
            Path to saved file
        """
        if timestamp is None:
            timestamp = datetime.now()

        file_path = self._get_json_file_path(timestamp)

        # Add metadata
        data_to_save = {
            "timestamp": timestamp.isoformat(),
            "date": timestamp.strftime("%Y-%m-%d"),
            "time": timestamp.strftime("%H:%M:%S"),
            "data": news_data
        }

        # Save to JSON
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved news data to: {file_path}")
        return file_path

    def load_news_data(self, date: datetime, time_str: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load news data for a specific date/time.

        Args:
            date: Target date
            time_str: Time string (e.g., "15时30分"), None for latest

        Returns:
            News data dictionary or None if not found
        """
        date_folder = self._get_date_folder(date)
        json_folder = date_folder / "json"

        if not json_folder.exists():
            logger.warning(f"No data found for date: {date.strftime('%Y-%m-%d')}")
            return None

        # If time_str not specified, get the latest file
        if time_str is None:
            json_files = sorted(json_folder.glob("*.json"), reverse=True)
            if not json_files:
                return None
            file_path = json_files[0]
        else:
            file_path = json_folder / f"{time_str}.json"

        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None

        # Load JSON
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded news data from: {file_path}")
        return data

    def get_available_dates(self) -> List[datetime]:
        """
        Get list of all available dates with data.

        Returns:
            List of datetime objects
        """
        dates = []

        for folder in self.output_dir.iterdir():
            if not folder.is_dir() or folder.name.startswith("."):
                continue

            # Parse folder name (格式: YYYY年MM月DD日)
            try:
                import re
                match = re.match(r"(\d{4})年(\d{2})月(\d{2})日", folder.name)
                if match:
                    year, month, day = match.groups()
                    date = datetime(int(year), int(month), int(day))
                    dates.append(date)
            except Exception as e:
                logger.warning(f"Failed to parse folder name: {folder.name}, error: {e}")

        return sorted(dates)

    def get_date_range(self) -> Optional[tuple[datetime, datetime]]:
        """
        Get available date range.

        Returns:
            (earliest_date, latest_date) or None if no data
        """
        dates = self.get_available_dates()
        if not dates:
            return None
        return (dates[0], dates[-1])

    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Statistics dictionary
        """
        dates = self.get_available_dates()
        total_files = 0
        total_size = 0

        for date_folder in self.output_dir.iterdir():
            if date_folder.is_dir():
                for file_path in date_folder.rglob("*.json"):
                    total_files += 1
                    total_size += file_path.stat().st_size

        date_range = self.get_date_range()

        return {
            "total_days": len(dates),
            "total_files": total_files,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "earliest_date": date_range[0].strftime("%Y-%m-%d") if date_range else None,
            "latest_date": date_range[1].strftime("%Y-%m-%d") if date_range else None,
        }


# Global storage instance
_storage = HistoryStorage()


def get_storage() -> HistoryStorage:
    """Get global history storage instance."""
    return _storage
