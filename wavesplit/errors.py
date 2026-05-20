class WaveSplitError(Exception):
    """Base exception for expected WaveSplit failures."""


class InputValidationError(WaveSplitError):
    """Raised when uploaded inputs cannot be processed."""


class AlignmentError(WaveSplitError):
    """Raised when transcript/audio alignment fails."""


class PipelineError(WaveSplitError):
    """Raised when the processing pipeline cannot complete."""
