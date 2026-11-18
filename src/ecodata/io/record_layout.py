"""Separate text annotations after a validated, declared record layout."""
from decimal import Decimal, InvalidOperation


class RecordLayoutError(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.code=code


def split_declared_record(tokens, width, *, numeric_columns, integer_columns=(), optional_tail=0):
    """Return declared cells and trailing annotations without changing field offsets."""
    extra=tokens[width:]
    cells=tokens[:width]
    if len(cells)!=width and len(cells)!=width-optional_tail:
        raise RecordLayoutError("record_column_count_mismatch")
    if extra:
        try:
            Decimal(extra[0].upper().replace("D","E"))
        except InvalidOperation:
            if not any(char.isalpha() for char in extra[0]):
                raise RecordLayoutError("unrecognized_trailing_columns")
        else:
            raise RecordLayoutError("unrecognized_trailing_columns")
    try:
        for offset in integer_columns:
            int(cells[offset])
        for offset in numeric_columns:
            value=Decimal(cells[offset].upper().replace("D","E"))
            if not value.is_finite():
                raise RecordLayoutError("invalid_numeric_value")
    except (ValueError,InvalidOperation,IndexError) as error:
        if isinstance(error,RecordLayoutError):
            raise
        raise RecordLayoutError("invalid_identity_or_numeric_prefix") from error
    return cells,extra
