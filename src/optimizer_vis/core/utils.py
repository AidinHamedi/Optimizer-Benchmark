from typing import Any

type NestedData[T] = T | list["NestedData[T]"] | tuple["NestedData[T]", ...]


def to_tuple[T](data: NestedData[T]) -> Any:
    """Recursively converts all nested lists into tuples."""
    if isinstance(data, list):
        return tuple(to_tuple(item) for item in data)
    return data
