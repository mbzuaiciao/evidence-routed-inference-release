"""Tests for the canonical Markdown math-delimiter policy."""

from pathlib import Path

from scripts.check_markdown_math import (
    check_canonical_markdown,
    find_unsupported_delimiters,
)


def test_inline_dollar_math_is_accepted() -> None:
    assert find_unsupported_delimiters("Use $S = \\mu / \\sigma$ inline.\n") == ()


def test_display_dollar_math_is_accepted() -> None:
    assert find_unsupported_delimiters("Before\n\n$$\nS = 1\n$$\n\nAfter\n") == ()


def test_parenthesized_latex_delimiters_are_rejected() -> None:
    issues = find_unsupported_delimiters(r"Unsupported \(S = 1\) inline.")
    assert tuple(issue[2] for issue in issues) == (r"\(", r"\)")


def test_bracketed_latex_delimiters_are_rejected() -> None:
    issues = find_unsupported_delimiters("Unsupported\n\\[\nS = 1\n\\]\n")
    assert tuple(issue[2] for issue in issues) == (r"\[", r"\]")


def test_user_owned_obsidian_notes_are_not_scanned(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "canonical.md").write_text("Accepted $S$.\n", encoding="utf-8")
    personal = tmp_path / "inference-obsidian/mynotes/personal.md"
    personal.parent.mkdir(parents=True)
    personal.write_text(r"User-owned \(syntax\).", encoding="utf-8")

    assert check_canonical_markdown(tmp_path) == ()


def test_generated_mirror_front_matter_does_not_cause_false_positive(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "canonical.md").write_text("Accepted $$S = 1$$.\n", encoding="utf-8")
    mirror = tmp_path / "inference-obsidian/implementation/Canonical.md"
    mirror.parent.mkdir(parents=True)
    mirror.write_text(
        '---\ncanonical_source: "docs/\\(legacy\\).md"\n---\n',
        encoding="utf-8",
    )

    assert check_canonical_markdown(tmp_path) == ()
