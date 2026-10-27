#!/usr/bin/env python3
"""Start the benchmark UI on your own machine.

  python scripts/ui.py [--port 8770]

Then open http://127.0.0.1:8770 in a browser. Stop it with Ctrl+C.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genbench.ui.server import serve


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args(argv)
    server = serve(args.port)
    print(f"NSE GenBench UI: http://127.0.0.1:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
