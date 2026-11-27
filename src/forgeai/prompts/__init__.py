"""Prompt templates live next to this file as Markdown, versioned in Git.

Templates use `$name` placeholders (string.Template) so JSON braces in prompts need no
escaping. A missing variable raises immediately rather than shipping a broken prompt.
"""

from __future__ import annotations

from functools import cache
from importlib import resources
from string import Template


@cache
def _template(name: str) -> Template:
    text = resources.files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8")
    return Template(text)


def render(name: str, **values: object) -> str:
    return _template(name).substitute({k: str(v) for k, v in values.items()}).strip()
