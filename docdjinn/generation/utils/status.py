import itertools
import sys
import time
import threading


class StatusLine:
    _BAR = "|/-\\"
    _DOTS = "▖▘▝▗"
    _ARROW = "←↖↑↗→↘↓↙"
    _BRAILLE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    _CIRCLE = "◐◓◑◒"
    _DASH = "━╾╼━"

    def __init__(self, delay: float = 0.1):
        self._stop = False
        self._spinner = itertools.cycle(self._BRAILLE)  # rotating chars
        self._delay = delay
        self.message = ""
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.start_time = time.time()
        self.thread.start()

    def stop(self):
        self._stop = True
        self.thread.join()
        # clear line on exit
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

    def update_message(self, msg: str):
        with self.lock:
            self.message = msg

    def log(self, msg: str):
        # Print a normal log above the status line
        sys.stdout.write("\r" + " " * 80 + "\r")  # clear status line
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()

    def _run(self):
        while not self._stop:
            with self.lock:
                spinner_char = next(self._spinner)
                elapsed = int(time.time() - self.start_time)
                line = f"[{spinner_char}] {self.message} | waiting {elapsed}s"
            sys.stdout.write("\r" + line[:79])  # overwrite line
            sys.stdout.flush()
            time.sleep(self._delay)


# Example usage
if __name__ == "__main__":
    status = StatusLine()
    status.start()

    for i in range(5):
        status.update_message(f"Awaiting {5 - i} batches")
        time.sleep(3)
        status.log(f"Batch {i + 1} ended!")

    status.stop()
    print("Done.")


def get_progress_bar():
    from rich.progress import (
        Progress,
        TimeElapsedColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
    )

    return Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        "[yellow]({task.completed}/{task.total})",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
