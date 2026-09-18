"""Errors for same-model agent C2C."""


class AgentC2CError(Exception):
    """Base error for the agent communication runtime."""


class ModelMismatchError(AgentC2CError):
    """Capsule was produced by a different model geometry."""


class TimelineError(AgentC2CError):
    """KV slice is not a valid prefix of a single shared timeline."""


class CapsuleError(AgentC2CError):
    """Capsule is missing, corrupt, or over budget."""


class AgentStateError(AgentC2CError):
    """Unknown agent, empty state, or illegal transition."""
