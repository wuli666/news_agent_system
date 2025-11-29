"""
Python REPL tool for on-demand data处理/计算。
默认通过环境变量 ENABLE_PYTHON_REPL 控制开关。
"""
import logging
import os
from typing import Annotated

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL


logger = logging.getLogger(__name__)


def _is_python_repl_enabled() -> bool:
    env_enabled = os.getenv("ENABLE_PYTHON_REPL", "false").lower()
    return env_enabled in ("true", "1", "yes", "on")


# 实例化 REPL（只有启用时才创建）
repl = PythonREPL() if _is_python_repl_enabled() else None


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to do further analysis or calculation."],
):
    """Execute ad-hoc Python code for分析/计算。需要 ENABLE_PYTHON_REPL=true 才会运行。"""
    if not _is_python_repl_enabled():
        msg = "Python REPL tool is disabled. Set ENABLE_PYTHON_REPL=true to enable."
        logger.warning(msg)
        return f"Tool disabled: {msg}"

    if not isinstance(code, str):
        msg = f"Invalid input: code must be a string, got {type(code)}"
        logger.error(msg)
        return f"Error executing code:\n```python\n{code}\n```\nError: {msg}"

    logger.info("Executing Python REPL code")
    try:
        result = repl.run(code)
        if isinstance(result, str) and ("Error" in result or "Exception" in result):
            logger.error(result)
            return f"Error executing code:\n```python\n{code}\n```\nError: {result}"
    except BaseException as e:
        msg = repr(e)
        logger.error(msg)
        return f"Error executing code:\n```python\n{code}\n```\nError: {msg}"

    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
