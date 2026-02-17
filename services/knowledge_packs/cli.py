#!/usr/bin/env python3
"""
Knowledge Pack CLI for EDISON
Usage:
    python -m services.knowledge_packs.cli install <pack>
    python -m services.knowledge_packs.cli update <pack>
    python -m services.knowledge_packs.cli status [pack]
    python -m services.knowledge_packs.cli uninstall <pack>
"""

import argparse
import json
import sys

from .manager import KnowledgePackManager


def main():
    parser = argparse.ArgumentParser(description="EDISON Knowledge Pack Manager")
    sub = parser.add_subparsers(dest="command")

    install_p = sub.add_parser("install", help="Install a knowledge pack")
    install_p.add_argument("pack", help="Pack ID to install")

    update_p = sub.add_parser("update", help="Update a knowledge pack")
    update_p.add_argument("pack", help="Pack ID to update")

    status_p = sub.add_parser("status", help="Show pack status")
    status_p.add_argument("pack", nargs="?", help="Optional pack ID")

    uninstall_p = sub.add_parser("uninstall", help="Uninstall a knowledge pack")
    uninstall_p.add_argument("pack", help="Pack ID to uninstall")

    args = parser.parse_args()
    mgr = KnowledgePackManager()

    if args.command == "install":
        result = mgr.install(args.pack)
        print(json.dumps(result, indent=2))
    elif args.command == "update":
        result = mgr.update(args.pack)
        print(json.dumps(result, indent=2))
    elif args.command == "status":
        result = mgr.status(args.pack if hasattr(args, "pack") else None)
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "uninstall":
        result = mgr.uninstall(args.pack)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
