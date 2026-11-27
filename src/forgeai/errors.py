"""Domain errors. Every failure ForgeAI expects has a type, so callers can decide what to do."""

from __future__ import annotations


class ForgeError(Exception):
    """Base class for all expected ForgeAI failures."""


class ModelOutputError(ForgeError):
    """The model returned output that failed validation, even after a repair attempt."""


class BudgetExceeded(ForgeError):
    """The run hit its token or call budget. Not a crash: the loop stops and reports why."""


class GroundingError(ForgeError):
    """A claim about the documents could not be verified against the documents."""


class ToolError(ForgeError):
    """A tool was called with invalid arguments or failed to run."""


class ModelUnavailable(ForgeError):
    """The model endpoint could not be reached."""
