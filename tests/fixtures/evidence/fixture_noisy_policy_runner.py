from __future__ import annotations

import os

chunk = b"x" * (1024 * 1024)
for _index in range(16):
    os.write(1, chunk)
