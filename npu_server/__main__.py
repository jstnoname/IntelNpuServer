from __future__ import annotations

import sys

from npu_server.cli.server_cli import CommandLineInterface


def main() -> int:
    return CommandLineInterface().startup(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
