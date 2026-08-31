def require_positive(**fields: float) -> None:
    for name, value in fields.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}")


def require_non_negative(**fields: float) -> None:
    for name, value in fields.items():
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}")


def require_unit_interval(**fields: float) -> None:
    for name, value in fields.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1], got {value}")
