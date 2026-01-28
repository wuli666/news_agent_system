"""
Task Manager for GraphRAG

Provides async task tracking for long-running operations like:
- Graph building
- Batch ingestion
- Report generation
"""

import threading
import uuid
import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Represents an async task."""
    task_id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0  # 0-100
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class TaskManager:
    """
    Singleton task manager for tracking async operations.

    Thread-safe implementation for concurrent task updates.
    """

    _instance: Optional["TaskManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "TaskManager":
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._tasks: Dict[str, Task] = {}
                    cls._instance._task_lock = threading.Lock()
        return cls._instance

    def create_task(self, name: str) -> str:
        """
        Create a new task.

        Args:
            name: Task description

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())[:8]

        with self._task_lock:
            self._tasks[task_id] = Task(
                task_id=task_id,
                name=name,
                status=TaskStatus.PENDING,
                message="Task created"
            )

        logger.info(f"Created task {task_id}: {name}")
        return task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get task by ID.

        Args:
            task_id: Task identifier

        Returns:
            Task object or None
        """
        with self._task_lock:
            return self._tasks.get(task_id)

    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> bool:
        """
        Update task status and progress.

        Args:
            task_id: Task identifier
            status: New status
            progress: Progress percentage (0-100)
            message: Status message
            result: Task result data
            error: Error message if failed

        Returns:
            Success status
        """
        with self._task_lock:
            task = self._tasks.get(task_id)
            if not task:
                logger.warning(f"Task {task_id} not found")
                return False

            if status is not None:
                task.status = status
            if progress is not None:
                task.progress = max(0, min(100, progress))
            if message is not None:
                task.message = message
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error

            task.updated_at = datetime.now()

            if status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                task.completed_at = datetime.now()

        return True

    def complete_task(
        self,
        task_id: str,
        result: Optional[Dict[str, Any]] = None,
        message: str = "Task completed"
    ) -> bool:
        """
        Mark task as completed.

        Args:
            task_id: Task identifier
            result: Task result data
            message: Completion message

        Returns:
            Success status
        """
        return self.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=100,
            message=message,
            result=result
        )

    def fail_task(
        self,
        task_id: str,
        error: str,
        message: str = "Task failed"
    ) -> bool:
        """
        Mark task as failed.

        Args:
            task_id: Task identifier
            error: Error details
            message: Failure message

        Returns:
            Success status
        """
        logger.error(f"Task {task_id} failed: {error}")
        return self.update_task(
            task_id,
            status=TaskStatus.FAILED,
            message=message,
            error=error
        )

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 50
    ) -> List[Task]:
        """
        List tasks with optional status filter.

        Args:
            status: Filter by status
            limit: Maximum number of tasks

        Returns:
            List of Task objects
        """
        with self._task_lock:
            tasks = list(self._tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]

        # Sort by created_at descending
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        return tasks[:limit]

    def delete_task(self, task_id: str) -> bool:
        """
        Delete a task.

        Args:
            task_id: Task identifier

        Returns:
            Success status
        """
        with self._task_lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                return True
        return False

    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """
        Remove completed/failed tasks older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of tasks removed
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        removed = 0

        with self._task_lock:
            to_remove = [
                task_id
                for task_id, task in self._tasks.items()
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
                and task.created_at < cutoff
            ]

            for task_id in to_remove:
                del self._tasks[task_id]
                removed += 1

        if removed:
            logger.info(f"Cleaned up {removed} old tasks")

        return removed


# Global task manager instance
task_manager = TaskManager()
