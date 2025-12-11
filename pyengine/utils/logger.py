import os
import sys
from typing import Optional, TextIO


class Logger:
    """
    Write everything to both stdout (optional) and a log file.

    Typical use
    -----------
    >>> sys.stdout = Logger("logs/run.txt")   # mirror everything
    >>> print("hello")                        # prints and logs
    >>> sys.stdout.close()                    # tidy-up at the end
    """

    def __init__(
        self,
        path: str,
        mode: str = "w",
        print_to_stdout: bool = True,
        encoding: str = "utf-8",
        buffering: int = 1,
    ):
        if mode not in {"w", "a"}:
            raise ValueError(f"Unknown mode {mode!r}; use 'w' or 'a'.")
        self.terminal: Optional[TextIO] = sys.stdout if print_to_stdout else None

        # ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # open the backing log file (line-buffered by default)
        self.log: TextIO = open(path, mode, encoding=encoding, buffering=buffering)

        # expose common file-object attributes
        self.encoding = encoding
        self.newlines = self.log.newlines
        self.closed = False

    # ------------------------------------------------------------------ #
    #                  Minimum API PyTorch/Dynamo expects
    # ------------------------------------------------------------------ #
    def isatty(self) -> bool:
        """Return True if attached to an interactive terminal."""
        return bool(self.terminal) and self.terminal.isatty()

    def fileno(self) -> int:  # sometimes required
        return self.log.fileno()

    # ------------------------------------------------------------------ #
    #                            Core I/O
    # ------------------------------------------------------------------ #
    def write(self, message: str) -> None:
        if self.closed:
            raise ValueError("I/O operation on closed Logger")
        if self.terminal:
            self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        if self.closed:
            return
        if self.terminal:
            self.terminal.flush()
        self.log.flush()

    # ------------------------------------------------------------------ #
    #                      Convenience / safety
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        if not self.closed:
            self.flush()
            self.log.close()
            self.closed = True

    def __del__(self):
        # close gracefully when the object is garbage-collected
        try:
            self.close()
        except Exception:  # pragma: no cover
            pass

    # Context-manager support so you can use `with Logger(...) as log:`
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # Forward any attribute lookups we haven’t defined to the underlying file
    def __getattr__(self, item):
        return getattr(self.log, item)
