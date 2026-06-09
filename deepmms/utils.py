"""
Shared utilities for Deep-MMS: timestamped printing, stdout/stderr suppression,
and the JAX x64 configuration that must run once on package import.
"""

import jax
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from datetime import datetime

jax.config.update("jax_enable_x64", True)

printf = lambda x: print(
    f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}//{x}", flush=True
)


@contextmanager
def suppress_stdout_stderr():
    """
    Context manager that redirects both stdout and stderr to /dev/null.

    Useful for silencing verbose third-party library output during data loading.
    """
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)
