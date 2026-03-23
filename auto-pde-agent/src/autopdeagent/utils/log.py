import logging
from pathlib import Path

from rich.logging import RichHandler


def _setup_root_logger() -> None:
    # === 🔄 修改 1：替换 Logger 名称 ===
    logger = logging.getLogger("autopdeagent")
    logger.setLevel(logging.DEBUG)
    _handler = RichHandler(
        show_path=False,
        show_time=False,
        show_level=False,
        markup=True,
    )
    _formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)


def add_file_handler(path: Path | str, level: int = logging.DEBUG, *, print_path: bool = True) -> None:
    # === 🔄 修改 2：替换 Logger 名称 ===
    logger = logging.getLogger("autopdeagent")
    handler = logging.FileHandler(path)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if print_path:
        print(f"Logging to '{path}'")


_setup_root_logger()
# === 🔄 修改 3：替换 Logger 名称 ===
logger = logging.getLogger("autopdeagent")


__all__ = ["logger"]