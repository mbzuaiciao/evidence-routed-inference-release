"""Reject unsupported math delimiters in canonical Markdown."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

UNSUPPORTED_DELIMITERS: tuple[str, ...] = (r"\(", r"\)", r"\[", r"\]")


@dataclass(frozen=True, slots=True)
class MathDelimiterIssue:
    """One unsupported delimiter at a canonical Markdown location."""

    path: Path
    line: int
    column: int
    delimiter: str


def find_unsupported_delimiters(text: str) -> tuple[tuple[int, int, str], ...]:
    """Return one-based line and column locations for unsupported tokens."""
    issues: list[tuple[int, int, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for delimiter in UNSUPPORTED_DELIMITERS:
            offset = 0
            while True:
                index = line.find(delimiter, offset)
                if index < 0:
                    break
                issues.append((line_number, index + 1, delimiter))
                offset = index + len(delimiter)
    return tuple(issues)


def canonical_markdown_paths(repo_root: Path) -> tuple[Path, ...]:
    """Return only repository-owned canonical and top-level policy Markdown."""
    paths = [
        path
        for path in (repo_root / "AGENTS.md", repo_root / "README.md")
        if path.is_file()
    ]
    docs_root = repo_root / "docs"
    if docs_root.is_dir():
        paths.extend(path for path in docs_root.rglob("*.md") if path.is_file())
    return tuple(sorted(paths, key=lambda path: path.relative_to(repo_root).as_posix()))


def check_canonical_markdown(repo_root: Path) -> tuple[MathDelimiterIssue, ...]:
    """Check canonical Markdown without traversing any Obsidian directory."""
    issues: list[MathDelimiterIssue] = []
    for path in canonical_markdown_paths(repo_root):
        text = path.read_text(encoding="utf-8")
        issues.extend(
            MathDelimiterIssue(path, line, column, delimiter)
            for line, column, delimiter in find_unsupported_delimiters(text)
        )
    return tuple(issues)


def main() -> int:
    """Run the canonical Markdown delimiter audit for this repository."""
    repo_root = Path(__file__).resolve().parents[1]
    issues = check_canonical_markdown(repo_root)
    for issue in issues:
        relative = issue.path.relative_to(repo_root)
        print(
            f"{relative}:{issue.line}:{issue.column}: "
            f"unsupported math delimiter {issue.delimiter!r}"
        )
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
