class DataError(Exception):
    """Base class for data-layer failures."""


class InvalidTickerError(DataError):
    """Provider returned nothing for a ticker."""


class InsufficientDataError(DataError):
    """Overlapping history is too short (or empty) to stabilise the model."""


InsuffientDataError = InsufficientDataError  # spelling used in the spec; kept as an alias
