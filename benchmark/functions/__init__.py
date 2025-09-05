from typing import Tuple

from .ackley import EVAL_SIZE as ACKLEY_EVAL_SIZE
from .ackley import GLOBAL_MINIMUM_LOC as ACKLEY_GLOBAL_MINIMUM_LOC
from .ackley import START_POS as ACKLEY_START_POS
from .ackley import ackley
from .cross_in_tray import EVAL_SIZE as CROSS_EVAL_SIZE
from .cross_in_tray import GLOBAL_MINIMUM_LOC as CROSS_GLOBAL_MINIMUM_LOC
from .cross_in_tray import START_POS as CROSS_START_POS
from .cross_in_tray import cross_in_tray
from .drop_wave import EVAL_SIZE as DROPWAVE_EVAL_SIZE
from .drop_wave import GLOBAL_MINIMUM_LOC as DROPWAVE_GLOBAL_MINIMUM_LOC
from .drop_wave import START_POS as DROPWAVE_START_POS
from .drop_wave import drop_wave
from .eggholder import EVAL_SIZE as EGG_EVAL_SIZE
from .eggholder import GLOBAL_MINIMUM_LOC as EGG_GLOBAL_MINIMUM_LOC
from .eggholder import START_POS as EGG_START_POS
from .eggholder import eggholder
from .gramacy_lee2d import EVAL_SIZE as GRAMACYLEE2D_EVAL_SIZE
from .gramacy_lee2d import GLOBAL_MINIMUM_LOC as GRAMACYLEE2D_GLOBAL_MINIMUM_LOC
from .gramacy_lee2d import START_POS as GRAMACYLEE2D_START_POS
from .gramacy_lee2d import gl2d
from .griewank import EVAL_SIZE as GRIEWANK_EVAL_SIZE
from .griewank import GLOBAL_MINIMUM_LOC as GRIEWANK_GLOBAL_MINIMUM_LOC
from .griewank import START_POS as GRIEWANK_START_POS
from .griewank import griewank
from .holder_table import EVAL_SIZE as HOLDER_TABLE_EVAL_SIZE
from .holder_table import GLOBAL_MINIMUM_LOC as HOLDER_TABLE_GLOBAL_MINIMUM_LOC
from .holder_table import START_POS as HOLDER_TABLE_START_POS
from .holder_table import holder_table
from .langermann import EVAL_SIZE as LANGERMANN_EVAL_SIZE
from .langermann import GLOBAL_MINIMUM_LOC as LANGERMANN_GLOBAL_MINIMUM_LOC
from .langermann import START_POS as LANGERMANN_START_POS
from .langermann import langermann
from .levy import EVAL_SIZE as LEVY_EVAL_SIZE
from .levy import GLOBAL_MINIMUM_LOC as LEVY_GLOBAL_MINIMUM_LOC
from .levy import START_POS as LEVY_START_POS
from .levy import levy
from .levy13 import EVAL_SIZE as LEVY13_EVAL_SIZE
from .levy13 import GLOBAL_MINIMUM_LOC as LEVY13_GLOBAL_MINIMUM_LOC
from .levy13 import START_POS as LEVY13_START_POS
from .levy13 import levy13
from .rastrigin import EVAL_SIZE as RASTRIGIN_EVAL_SIZE
from .rastrigin import GLOBAL_MINIMUM_LOC as RASTRIGIN_GLOBAL_MINIMUM_LOC
from .rastrigin import START_POS as RASTRIGIN_START_POS
from .rastrigin import rastrigin
from .rosenbrock import EVAL_SIZE as ROSENBROCK_EVAL_SIZE
from .rosenbrock import GLOBAL_MINIMUM_LOC as ROSENBROCK_GLOBAL_MINIMUM_LOC
from .rosenbrock import START_POS as ROSENBROCK_START_POS
from .rosenbrock import rosenbrock
from .schaffer2 import EVAL_SIZE as SCHAFFER2_EVAL_SIZE
from .schaffer2 import GLOBAL_MINIMUM_LOC as SCHAFFER2_GLOBAL_MINIMUM_LOC
from .schaffer2 import START_POS as SCHAFFER2_START_POS
from .schaffer2 import schaffer2
from .schaffer4 import EVAL_SIZE as SCHAFFER4_EVAL_SIZE
from .schaffer4 import GLOBAL_MINIMUM_LOC as SCHAFFER4_GLOBAL_MINIMUM_LOC
from .schaffer4 import START_POS as SCHAFFER4_START_POS
from .schaffer4 import schaffer4
from .shubert import EVAL_SIZE as SHUBERT_EVAL_SIZE
from .shubert import GLOBAL_MINIMUM_LOC as SHUBERT_GLOBAL_MINIMUM_LOC
from .shubert import START_POS as SHUBERT_START_POS
from .shubert import shubert
from .styblinski_tang import EVAL_SIZE as STYBLINSKI_TANG_EVAL_SIZE
from .styblinski_tang import GLOBAL_MINIMUM_LOC as STYBLINSKI_TANG_GLOBAL_MINIMUM_LOC
from .styblinski_tang import START_POS as STYBLINSKI_TANG_START_POS
from .styblinski_tang import stybtang

FUNC_DICT: dict = {
    "Ackley": {
        "func": ackley,
        "size": ACKLEY_EVAL_SIZE,
        "pos": ACKLEY_START_POS,
        "gm_pos": ACKLEY_GLOBAL_MINIMUM_LOC,
    },
    "Cross-in-Tray": {
        "func": cross_in_tray,
        "size": CROSS_EVAL_SIZE,
        "pos": CROSS_START_POS,
        "gm_pos": CROSS_GLOBAL_MINIMUM_LOC,
    },
    "Drop-Wave": {
        "func": drop_wave,
        "size": DROPWAVE_EVAL_SIZE,
        "pos": DROPWAVE_START_POS,
        "gm_pos": DROPWAVE_GLOBAL_MINIMUM_LOC,
    },
    "EggHolder": {
        "func": eggholder,
        "size": EGG_EVAL_SIZE,
        "pos": EGG_START_POS,
        "gm_pos": EGG_GLOBAL_MINIMUM_LOC,
    },
    "Gramacy-Lee 2D": {
        "func": gl2d,
        "size": GRAMACYLEE2D_EVAL_SIZE,
        "pos": GRAMACYLEE2D_START_POS,
        "gm_pos": GRAMACYLEE2D_GLOBAL_MINIMUM_LOC,
    },
    "Griewank": {
        "func": griewank,
        "size": GRIEWANK_EVAL_SIZE,
        "pos": GRIEWANK_START_POS,
        "gm_pos": GRIEWANK_GLOBAL_MINIMUM_LOC,
    },
    "Holder Table": {
        "func": holder_table,
        "size": HOLDER_TABLE_EVAL_SIZE,
        "pos": HOLDER_TABLE_START_POS,
        "gm_pos": HOLDER_TABLE_GLOBAL_MINIMUM_LOC,
    },
    "Langermann": {
        "func": langermann,
        "size": LANGERMANN_EVAL_SIZE,
        "pos": LANGERMANN_START_POS,
        "gm_pos": LANGERMANN_GLOBAL_MINIMUM_LOC,
    },
    "Levy": {
        "func": levy,
        "size": LEVY_EVAL_SIZE,
        "pos": LEVY_START_POS,
        "gm_pos": LEVY_GLOBAL_MINIMUM_LOC,
    },
    "Levy 13": {
        "func": levy13,
        "size": LEVY13_EVAL_SIZE,
        "pos": LEVY13_START_POS,
        "gm_pos": LEVY13_GLOBAL_MINIMUM_LOC,
    },
    "Rastrigin": {
        "func": rastrigin,
        "size": RASTRIGIN_EVAL_SIZE,
        "pos": RASTRIGIN_START_POS,
        "gm_pos": RASTRIGIN_GLOBAL_MINIMUM_LOC,
    },
    "Rosenbrock": {
        "func": rosenbrock,
        "size": ROSENBROCK_EVAL_SIZE,
        "pos": ROSENBROCK_START_POS,
        "gm_pos": ROSENBROCK_GLOBAL_MINIMUM_LOC,
    },
    "Schaffer 2": {
        "func": schaffer2,
        "size": SCHAFFER2_EVAL_SIZE,
        "pos": SCHAFFER2_START_POS,
        "gm_pos": SCHAFFER2_GLOBAL_MINIMUM_LOC,
    },
    "Schaffer 4": {
        "func": schaffer4,
        "size": SCHAFFER4_EVAL_SIZE,
        "pos": SCHAFFER4_START_POS,
        "gm_pos": SCHAFFER4_GLOBAL_MINIMUM_LOC,
    },
    "Shubert": {
        "func": shubert,
        "size": SHUBERT_EVAL_SIZE,
        "pos": SHUBERT_START_POS,
        "gm_pos": SHUBERT_GLOBAL_MINIMUM_LOC,
    },
    "Styblinski-Tang": {
        "func": stybtang,
        "size": STYBLINSKI_TANG_EVAL_SIZE,
        "pos": STYBLINSKI_TANG_START_POS,
        "gm_pos": STYBLINSKI_TANG_GLOBAL_MINIMUM_LOC,
    },
}


def scale_eval_size(
    eval_size: Tuple[Tuple[float, float], Tuple[float, float]], scale: float
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Scale the evaluation size by a given factor."""

    def adjust_pair(first: float, second: float, factor: float) -> Tuple[float, float]:
        if first == -second:
            return first * factor, second * factor

        sign = 1 if first >= 0 else -1
        abs_first = abs(first)
        abs_second_scaled = abs(second * factor)

        if factor > 1:
            new_mag = abs_first / factor
        else:
            new_mag = min(abs_first / factor, abs_second_scaled)

        return sign * new_mag, second * factor

    return (
        adjust_pair(eval_size[0][0], eval_size[0][1], scale),
        adjust_pair(eval_size[1][0], eval_size[1][1], scale),
    )
