import argparse
import re
import subprocess
from pathlib import Path


def parse_issues(markdown_text: str):
    pattern = re.compile(r"^## Issue\s+\d+:\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(markdown_text))

    issues = []

    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        body = markdown_text[start:end].strip()

        if title and body:
            issues.append((title, body))

    return issues


def create_issue(title: str, body: str, dry_run: bool):
    if dry_run:
        print("=" * 80)
        print(f"TITLE: {title}")
        print("-" * 80)
        print(body)
        print()
        return

    subprocess.run(
        ["gh", "issue", "create", "--title", title, "--body", body],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create GitHub Issues from docs/issues.md"
    )
    parser.add_argument(
        "--file",
        default="docs/issues.md",
        help="Path to issues markdown file",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually create GitHub Issues. Without this, dry-run only.",
    )
    args = parser.parse_args()

    markdown_path = Path(args.file)

    if not markdown_path.exists():
        raise SystemExit(f"Error: file not found: {markdown_path}")

    markdown_text = markdown_path.read_text(encoding="utf-8")
    issues = parse_issues(markdown_text)

    if not issues:
        raise SystemExit("Error: no issues found. Expected headings like '## Issue 1: title'.")

    print(f"Found {len(issues)} issues.")

    for title, body in issues:
        create_issue(title, body, dry_run=not args.apply)

    if not args.apply:
        print("Dry-run only. To create issues, run with --apply.")


if __name__ == "__main__":
    main()
