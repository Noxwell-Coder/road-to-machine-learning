#!/usr/bin/env python3
"""
Scan all Markdown files under the repo for broken relative links and
missing in-file / cross-file heading anchors (GitHub-style slug rules).

Usage (from repo root):  python3 tools/check_md_links.py
Exit code 1 if any issue is found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"!?\[([^\]]*)\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def text_outside_code_fences(text: str) -> str:
    """Replace fenced code with newlines so regex scanners do not see false links."""
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            lines.append("\n")
            continue
        if not in_fence:
            lines.append(line)
        else:
            lines.append("\n")
    return "".join(lines)


def base_slug(title: str) -> str:
    """GitHub-like heading slug: lowercase, non-alnum runs -> single hyphen."""
    t = title.strip()
    t = re.sub(r"[*_`]+", "", t)
    t = t.lower().strip()
    t = re.sub(r"[^a-z0-9]+", "-", t)
    t = re.sub(r"-+", "-", t)
    return t.strip("-")


def heading_slugs(headings: list[str]) -> list[str]:
    """Ordered slug ids with -1, -2 suffixes for duplicates (GitHub behavior)."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw in headings:
        b = base_slug(raw)
        if not b:
            continue
        n = seen.get(b, 0)
        if n == 0:
            seen[b] = 1
            out.append(b)
        else:
            seen[b] = n + 1
            out.append(f"{b}-{n}")
    return out


def extract_headings(text: str) -> list[str]:
    """Heading texts in order, skipping fenced code blocks."""
    lines = text.splitlines()
    in_fence = False
    headings: list[str] = []
    for line in lines:
        st = line.lstrip()
        if st.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADING_RE.match(line)
        if m:
            headings.append(m.group(2))
    return headings


def strip_angle(url: str) -> str:
    u = url.strip()
    if u.startswith("<") and u.endswith(">"):
        return u[1:-1].strip()
    return u


def parse_link_target(raw: str) -> tuple[str, str]:
    """Return (path_without_fragment, fragment) fragment may be ''."""
    raw = strip_angle(raw.strip())
    if raw.startswith("#"):
        return "", unquote(raw[1:])
    if "#" in raw:
        path, _, frag = raw.partition("#")
        return path.strip(), unquote(frag) if frag else ""
    return raw.strip(), ""


def is_external(url: str) -> bool:
    u = url.lower()
    return bool(
        re.match(r"^[a-z][a-z0-9+.-]*:", u)
        or u.startswith("//")
        or u.startswith("mailto:")
    )


def resolve_path(source: Path, path_part: str) -> Path | None:
    """Resolve path_part relative to source's directory; return absolute or None."""
    if not path_part:
        return source.resolve()
    p = Path(path_part)
    if p.is_absolute():
        cand = p
    else:
        cand = (source.parent / p).resolve()
    try:
        cand.relative_to(ROOT.resolve())
    except ValueError:
        return None
    return cand


def main() -> int:
    issues: list[str] = []
    md_files = sorted(ROOT.rglob("*.md"))
    # Skip vendored or huge if any
    skip_dirs = {".git", ".linkcheck-venv", "node_modules"}
    md_files = [p for p in md_files if not any(part in skip_dirs for part in p.parts)]

    slug_cache: dict[Path, list[str]] = {}

    def slugs_for(path: Path) -> list[str]:
        path = path.resolve()
        if path not in slug_cache:
            try:
                txt = path.read_text(encoding="utf-8")
            except OSError:
                slug_cache[path] = []
            else:
                slug_cache[path] = heading_slugs(extract_headings(txt))
        return slug_cache[path]

    for src in md_files:
        try:
            raw = src.read_text(encoding="utf-8")
        except OSError as e:
            issues.append(f"{src.relative_to(ROOT)}: cannot read ({e})")
            continue

        content = text_outside_code_fences(raw)

        for m in LINK_RE.finditer(content):
            raw = m.group(2).strip()
            if not raw or is_external(raw):
                continue
            path_part, frag = parse_link_target(raw)

            target = resolve_path(src, path_part)
            rel = src.relative_to(ROOT)

            if target is None:
                issues.append(f"{rel}: link target leaves repo: {raw!r}")
                continue

            if path_part == "":
                # same file anchor only
                if frag:
                    ids = slugs_for(src)
                    if frag not in ids:
                        issues.append(
                            f"{rel}: missing anchor #{frag} (same file); "
                            f"known slugs sample: {ids[:8]}{'...' if len(ids) > 8 else ''}"
                        )
                continue

            if not target.exists():
                issues.append(f"{rel}: missing file {path_part!r} -> {target.relative_to(ROOT)}")
                continue

            if target.is_dir():
                # allow links to dirs if they contain README.md (optional)
                readme = target / "README.md"
                if not readme.is_file():
                    issues.append(
                        f"{rel}: link points to directory without README.md: {path_part!r}"
                    )
                if frag:
                    issues.append(f"{rel}: anchor on directory link not checked: #{frag}")
                continue

            if target.suffix.lower() != ".md":
                if frag:
                    issues.append(f"{rel}: anchor on non-md file not checked: {path_part!r}#{frag}")
                continue

            if frag:
                ids = slugs_for(target)
                if frag not in ids:
                    issues.append(
                        f"{rel}: missing anchor #{frag} in "
                        f"{target.relative_to(ROOT)}; "
                        f"first headings slugs: {ids[:12]}{'...' if len(ids) > 12 else ''}"
                    )

    if issues:
        print(f"Found {len(issues)} issue(s):\n", file=sys.stderr)
        for line in issues:
            print(line, file=sys.stderr)
        return 1

    print(f"OK: scanned {len(md_files)} markdown files; no broken relative links or missing anchors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
