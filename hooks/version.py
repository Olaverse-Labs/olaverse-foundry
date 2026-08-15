"""
MkDocs hook: substitute the package version into the docs at build time.

Write ``{{ foundry_version }}`` anywhere in a Markdown page and it renders as
the version in ``foundry/__init__.py``. The version badge on the landing page
used to be hand-written, so it kept advertising an old release after each bump.

This is a native MkDocs hook (``hooks:`` in mkdocs.yml), not a plugin, so it
adds no dependency. The version is read by parsing the source rather than
importing the package — the docs build shouldn't depend on the package's
imports resolving.

It lives outside ``docs/`` on purpose: anything under the docs directory is
copied into the built site, and this file has no business being published.
"""
import re
from pathlib import Path

PLACEHOLDER = "{{ foundry_version }}"

_VERSION_RE = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']', re.M)

# hooks/ sits at the repo root, alongside foundry/.
_INIT = Path(__file__).resolve().parent.parent / "foundry" / "__init__.py"


def _read_version() -> str:
    match = _VERSION_RE.search(_INIT.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(
            f"Could not find __version__ in {_INIT}. The version badge in the "
            "docs is generated from it, so the build is stopped rather than "
            "publishing a page with an unsubstituted placeholder."
        )
    return match.group(1)


def on_page_markdown(markdown: str, page=None, config=None, files=None) -> str:
    if PLACEHOLDER not in markdown:
        return markdown
    return markdown.replace(PLACEHOLDER, _read_version())
