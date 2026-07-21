from copy import deepcopy

import numpy as np

from ArcSet import ArcSet


class ArcProblem:
    """
    A basic Arc problem containing
    a list of training data (ArcSet(s))
    and a test set (ArcSet).
    """
    def __init__(self, problem_name: str, train: list[ArcSet], test: ArcSet):
        self._id = problem_name
        self._training_data: list[ArcSet] = train
        self._test: ArcSet = test

    def problem_name(self) -> str:
        """
        Returns the name of this ArcProblem.
        """
        return self._id

    def number_of_training_data_sets(self) -> int:
        """
        Returns the number of training input/output
        pairs for this test problem.
        """
        return len(self._training_data)

    def training_set(self) -> list[ArcSet]:
        """
        Returns all the training data as a list of ArcSets.
        """
        return deepcopy(self._training_data)

    def test_set(self) -> ArcSet:
        """
        Returns the test data as a dictionary
        with the keys of 'input' and 'output'
        """
        return deepcopy(self._test)


def _format_matrix(arr: np.ndarray) -> str:
    MAX_ROWS, MAX_COLS = 4, 6
    rows, cols = arr.shape

    def fmt_row(row):
        if cols <= MAX_COLS:
            return "[" + ",".join(str(int(v)) for v in row) + "]"
        return "[" + ",".join(str(int(v)) for v in row[:3]) + ",...," + str(int(row[-1])) + "]"

    if rows <= MAX_ROWS:
        row_strs = [fmt_row(arr[r]) for r in range(rows)]
    else:
        row_strs = [fmt_row(arr[0]), fmt_row(arr[1]), "[...]", fmt_row(arr[-1])]

    return f"({rows}×{cols}): [" + ",".join(row_strs) + "]"


def print_arc_tree(problem: ArcProblem) -> None:
    print(f'ArcProblem: "{problem.problem_name()}"')
    training = problem.training_set()
    n = len(training)
    print(f"├── Training [{n} pair{'s' if n != 1 else ''}]")
    for i, arc_set in enumerate(training):
        is_last = (i == n - 1)
        branch = "└──" if is_last else "├──"
        inner  = "    "               if is_last else "│   "
        print(f"│   {branch} [{i}]")
        inp = arc_set.get_input_data()
        out = arc_set.get_output_data()
        print(f"│   {inner}├── Input  {_format_matrix(inp.data())}")
        out_str = _format_matrix(out.data()) if out is not None else "None"
        print(f"│   {inner}└── Output {out_str}")
    print("└── Test")
    test = problem.test_set()
    inp = test.get_input_data()
    out = test.get_output_data()
    print(f"    ├── Input  {_format_matrix(inp.data())}")
    out_str = _format_matrix(out.data()) if out is not None else "None"
    print(f"    └── Output {out_str}")
