"""Server-facing adapters and protocol checks for external Aladdin baselines.

These adapters deliberately stop at a validated execution plan locally. The
server implementation supplies the official repository, model runner, and
dataset-specific operations.
"""

from .adapters import (
    DataEnvGymAdapter,
    DataEvolverAdapter,
    DittoAdapter,
    ExternalBaselineAdapter,
    GepaAdapter,
    build_adapter,
)
from .protocol import (
    ExternalBaselineConfig,
    ExternalBaselineSpec,
    load_configs,
    load_dataset_registry,
    load_specs,
)

__all__ = [
    "DataEnvGymAdapter",
    "DataEvolverAdapter",
    "DittoAdapter",
    "ExternalBaselineAdapter",
    "ExternalBaselineConfig",
    "ExternalBaselineSpec",
    "GepaAdapter",
    "build_adapter",
    "load_configs",
    "load_dataset_registry",
    "load_specs",
]
