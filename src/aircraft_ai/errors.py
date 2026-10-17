"""Domain exceptions for aircraft damage classification and captioning."""


class AircraftAIError(Exception):
    """Base exception for all domain errors in aircraft_ai."""
    pass


class DatasetError(AircraftAIError):
    """Raised when dataset files or directory structures are missing or malformed."""
    pass


class DatasetNotFoundError(DatasetError):
    """Raised when the expected dataset folder cannot be located."""
    pass


class CorruptImageError(DatasetError):
    """Raised when an image fails PIL verification or cannot be opened."""
    pass


class UnsafeArchiveError(DatasetError):
    """Raised when a tar member contains path traversal or unsafe file types."""
    pass


class ModelError(AircraftAIError):
    """Base exception for model building, loading, or inference errors."""
    pass


class ModelNotTrainedError(ModelError):
    """Raised when requesting inference but model weights artifact does not exist."""
    pass


class PreprocessingMismatchError(ModelError):
    """Raised when inference preprocessing does not match training preprocessing."""
    pass


class BlipError(AircraftAIError):
    """Base exception for BLIP model and processor errors."""
    pass


class BlipLoadError(BlipError):
    """Raised when pretrained BLIP weights cannot be downloaded or loaded from cache."""
    pass


class InvalidTaskError(BlipError):
    """Raised when an unsupported task string is provided (must be 'caption' or 'summary')."""
    pass
