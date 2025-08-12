from .ackley import EVAL_SIZE as ACKLEY_EVAL_SIZE
from .ackley import START_POS as ACKLEY_START_POS
from .ackley import ackley
from .cross_in_tray import EVAL_SIZE as CROSS_EVAL_SIZE
from .cross_in_tray import START_POS as CROSS_START_POS
from .cross_in_tray import cross_in_tray
from .eggholder import EVAL_SIZE as EGG_EVAL_SIZE
from .eggholder import START_POS as EGG_START_POS
from .eggholder import eggholder
from .langermann import EVAL_SIZE as LANGERMANN_EVAL_SIZE
from .langermann import START_POS as LANGERMANN_START_POS
from .langermann import langermann
from .levy import EVAL_SIZE as LEVY_EVAL_SIZE
from .levy import START_POS as LEVY_START_POS
from .levy import levy
from .rastrigin import EVAL_SIZE as RASTRIGIN_EVAL_SIZE
from .rastrigin import START_POS as RASTRIGIN_START_POS
from .rastrigin import rastrigin
from .schaffer2 import EVAL_SIZE as SCHAFFER2_EVAL_SIZE
from .schaffer2 import START_POS as SCHAFFER2_START_POS
from .schaffer2 import schaffer2
from .shubert import EVAL_SIZE as SHUBERT_EVAL_SIZE
from .shubert import START_POS as SHUBERT_START_POS
from .shubert import shubert
from .styblinski_tang import EVAL_SIZE as STYBLINSKI_TANG_EVAL_SIZE
from .styblinski_tang import START_POS as STYBLINSKI_TANG_START_POS
from .styblinski_tang import stybtang

FUNC_DICT: dict = {
    "Ackley": {"func": ackley, "size": ACKLEY_EVAL_SIZE, "pos": ACKLEY_START_POS},
    "Cross-in-Tray": {
        "func": cross_in_tray,
        "size": CROSS_EVAL_SIZE,
        "pos": CROSS_START_POS,
    },
    "EggHolder": {"func": eggholder, "size": EGG_EVAL_SIZE, "pos": EGG_START_POS},
    "Langermann": {
        "func": langermann,
        "size": LANGERMANN_EVAL_SIZE,
        "pos": LANGERMANN_START_POS,
    },
    "Levy": {"func": levy, "size": LEVY_EVAL_SIZE, "pos": LEVY_START_POS},
    "Rastrigin": {
        "func": rastrigin,
        "size": RASTRIGIN_EVAL_SIZE,
        "pos": RASTRIGIN_START_POS,
    },
    "Schaffer 2": {
        "func": schaffer2,
        "size": SCHAFFER2_EVAL_SIZE,
        "pos": SCHAFFER2_START_POS,
    },
    "Shubert": {"func": shubert, "size": SHUBERT_EVAL_SIZE, "pos": SHUBERT_START_POS},
    "Styblinski-Tang": {
        "func": stybtang,
        "size": STYBLINSKI_TANG_EVAL_SIZE,
        "pos": STYBLINSKI_TANG_START_POS,
    },
}
