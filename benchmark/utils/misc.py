UNITS = ["Thousand", "Million", "Billion"]


def format_large_num(num: float) -> str:
    """Formats large numbers as readable strings"""
    unit_idx = 0

    while num >= 1000 and unit_idx < len(UNITS) - 1:
        num /= 1000.0
        unit_idx += 1

    return f"{num:.2f} {UNITS[unit_idx]}"
