# Sphinx configuration for CREAM documentation.
#
# Renders the Markdown files under docs/ via MyST-Parser.  Designed to build
# on Read the Docs without any system dependencies beyond what is listed in
# docs/requirements.txt.

from __future__ import annotations

project = "CREAM"
author = "ss0832"
copyright = "2026, ss0832"

# The short X.Y version.  Keep in sync with Cargo.toml [package] version.
version = "0.1"
release = "0.1.0"

# ── General ──────────────────────────────────────────────────────────────────

extensions = [
    "myst_parser",
    "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

# Allow implicit cross-references to every heading across the documentation.
autosectionlabel_prefix_document = True

master_doc = "index"
exclude_patterns = ["_build", ".venv", "requirements.txt"]

# MyST extensions used by the documentation.
myst_enable_extensions = [
    "colon_fence",   # ::: fenced directives
    "deflist",       # definition lists
    "tasklist",      # - [ ] task syntax in Markdown
    "linkify",       # auto-link bare URLs
    "substitution",  # {{ name }} replacements
]

myst_heading_anchors = 3

# ── HTML output ──────────────────────────────────────────────────────────────

html_theme = "furo"
html_title = "CREAM documentation"
html_static_path: list[str] = []
html_show_sphinx = False
