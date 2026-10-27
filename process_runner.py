"""Stream owned child output and release workers that inherit its output file."""
import codecs
import contextlib
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time


def _writers(capture):
    """Find only processes still holding this invocation's unique output file."""
    owned = os.fstat(capture.fileno())
    result = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        for fd in ('1', '2'):
            try:
                current = (entry / 'fd' / fd).stat()
            except OSError:
                continue
            if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                result.append(int(entry.name))
                break
    return result


def _signal_writers(capture, sig):
    for pid in _writers(capture):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def execute(command, log=None, own_group=True):
    # A TFF worker can outlive its Python parent and inherit stdout/stderr.
    # A pipe iterator waits for that worker's EOF even after the parent exits.
    # Follow a regular file instead, and use the direct child's exit status.
    with tempfile.TemporaryFile(mode='w+b') as capture:
        process = subprocess.Popen(command, stdout=capture, stderr=subprocess.STDOUT,
                                   start_new_session=own_group)
        stream = open(log, 'a', encoding='utf-8') if log is not None else contextlib.nullcontext()
        position = 0
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

        def forward(output, final=False):
            nonlocal position
            # Read a bounded snapshot, including an unterminated final line.
            end = os.fstat(capture.fileno()).st_size
            while position < end:
                data = os.pread(capture.fileno(), min(65536, end-position), position)
                if not data:
                    break
                position += len(data)
                message = decoder.decode(data)
                print(message, end='', flush=True)
                if output is not None:
                    output.write(message)
                    output.flush()
            if final:
                message = decoder.decode(b'', final=True)
                print(message, end='', flush=True)
                if output is not None:
                    output.write(message)
                    output.flush()

        try:
            with stream as output:
                while process.poll() is None:
                    forward(output)
                    time.sleep(0.2)
                forward(output, final=True)
            if process.returncode:
                raise RuntimeError(f'Workflow process exited {process.returncode}; inspect log; completed fits kept.')
        finally:
            if process.poll() is None:
                try:
                    if own_group:
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if own_group:
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
            # Match the unique inherited file, never all workers or GPU users.
            _signal_writers(capture, signal.SIGTERM)
            deadline = time.monotonic() + 3
            while _writers(capture) and time.monotonic() < deadline:
                time.sleep(0.1)
            _signal_writers(capture, signal.SIGKILL)
