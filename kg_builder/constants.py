"""Constants and enumerations."""

from enum import Enum


class MutationStrategy(str, Enum):
    """Graph mutation strategies."""

    SIMPLE = "simple"
    INCREMENTAL = "incremental"
