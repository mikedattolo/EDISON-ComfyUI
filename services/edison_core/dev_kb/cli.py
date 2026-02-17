#!/usr/bin/env python3
"""
Developer Knowledge Base CLI for EDISON
Usage:
    python -m services.edison_core.dev_kb.cli install <pack>
    python -m services.edison_core.dev_kb.cli add-repo <name> <path_or_git_url>
    python -m services.edison_core.dev_kb.cli update <name>
    python -m services.edison_core.dev_kb.cli status [name]
    python -m services.edison_core.dev_kb.cli uninstall <name>
"""

import argparse
import json
import sys

from .manager import DevKBManager


def main():
    parser = argparse.ArgumentParser(description="EDISON Developer Knowledge Base")
    sub = parser.add_subparsers(dest="command")

    add_repo = sub.add_parser("add-repo", help="Add and index a local or git repo")
    add_repo.add_argument("name", help="Name for the repo")
    add_repo.add_argument("path", help="Local path or git URL")
    add_repo.add_argument("--collection", default="code_examples", help="Target collection")

    update_p = sub.add_parser("update", help="Update a repo or pack")
    update_p.add_argument("name", help="Name to update")

    status_p = sub.add_parser("status", help="Show status")
    status_p.add_argument("name", nargs="?", help="Optional name")

    uninstall_p = sub.add_parser("uninstall", help="Uninstall a repo or pack")
    uninstall_p.add_argument("name", help="Name to uninstall")

    args = parser.parse_args()
    mgr = DevKBManager()

    if args.command == "add-repo":
        result = mgr.add_repo(args.name, args.path, args.collection)
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "update":
        result = mgr.update_repo(args.name)
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "status":
        result = mgr.status(getattr(args, "name", None))
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "uninstall":
        result = mgr.uninstall(args.name)
        print(json.dumps(result, indent=2, default=str))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
