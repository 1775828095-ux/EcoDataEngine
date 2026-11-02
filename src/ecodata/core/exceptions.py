class EcoDataError(Exception):
    """Base error for expected data or configuration failures."""


class SchemaError(EcoDataError):
    """Input data does not satisfy the required schema."""


class SemanticError(EcoDataError):
    """A requested operation conflicts with the variable semantics."""


class ValidationError(EcoDataError):
    """Independent validation did not pass."""

    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report
