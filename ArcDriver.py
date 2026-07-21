import json
import os.path
import sys
import time

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ArcData import ArcData
from ArcProblem import ArcProblem
from ArcSet import ArcSet
from ArcAgent import ArcAgent
from tree_dump import write_tree_dump

def run_training_data(agent: ArcAgent, arc_problems: list[ArcProblem], timestamp: str) -> tuple[dict[ArcProblem, tuple[bool, list]], list[tuple[str, float, dict]]]:
    """
    Run each training problem with the test output included so the agent can
    test if they are getting the correct response.
    """
    train_ans_dict: dict[ArcProblem, tuple[bool, list]] = dict()
    problem_times: list[tuple[str, float, dict]] = []  # (name, ms, flags)
    for trn_problem in arc_problems:
        t0 = time.perf_counter()
        preds: list[np.ndarray] = agent.make_predictions(trn_problem)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        flags = getattr(agent, "last_flags", {})
        write_tree_dump(trn_problem.problem_name(), agent.last_test_trees, "see_outputs", timestamp)
        problem_times.append((trn_problem.problem_name(), elapsed_ms, flags))
        correct = False

        if len(preds) <= 3:
            for prediction in preds:
                answer = trn_problem.test_set().get_output_data().data()
                correct = np.array_equal(answer, prediction)
                if correct: break

        # # store the problem_set and whether it was correctly solved
        train_ans_dict[trn_problem] = (correct, preds)
        symbol = "✓" if correct else "✗"

        active_flags = [k for k in ("needsDivisionLogic", "needsTripleDivision", "hasNewLines") if flags.get(k)]
        flag_str = f"  [{', '.join(active_flags)}]" if active_flags else ""
        round_str = f"  round={flags.get('round_solved_in', '?')}"
        print(f"{symbol} {trn_problem.problem_name()}  {elapsed_ms:.1f} ms{flag_str}{round_str}")

    return train_ans_dict, problem_times

def load_arc_problems(path: str, problem_data: list[str]) -> list[ArcProblem]:
    problems: list[ArcProblem] = list()
    for problem_name in problem_data:
        with open(os.path.join(path, problem_name)) as p:
            flat_data: dict[str, dict] = json.load(p)
            # convert the data into ArcData (i.e. numpy.ndarray data)
            trn_data: list[ArcSet] = list()
            for dt in flat_data['train']:
                d_input = ArcData(np.array(dt['input']))
                d_output = ArcData(np.array(dt['output']))
                trn_set: ArcSet = ArcSet(arc_input=d_input, arc_output=d_output)
                trn_data.append(trn_set)

            tst_data: list[ArcSet] = list()
            for tst in flat_data['test']:
                t_input = ArcData(np.array(tst['input']))
                t_output = ArcData(np.array(tst['output']))
                tst_set: ArcSet = ArcSet(arc_input=t_input, arc_output=t_output)
                tst_data.append(tst_set)

            arc_problem = ArcProblem(problem_name[:-5], trn_data, tst_data[0])

            # # there should only be one test in the test data
            problems.append(arc_problem)

    return problems


if __name__ == "__main__":

    # // del - set to a problem hash (e.g. "00576224") to test only that problem, or None to run all
    # TEST_PROBLEM_HASH = "22eb0ac0"
    # TEST_PROBLEM_HASH = "25d487eb"
    # TEST_PROBLEM_HASH = "3de23699"
    # TEST_PROBLEM_HASH = "cf98881b"
    # TEST_PROBLEM_HASH = "f76d97a5"
    # TEST_PROBLEM_HASH = "f25ffba3"
    TEST_PROBLEM_HASH = None

    # Here you can use this to open other milestone data directories for running against
    #  change the MILESTONE variable below to 'B', 'C', or 'D'
    MILESTONE = os.environ.get('MILESTONE', 'C')
    milestone_path = os.path.join('..', 'Milestone B', 'Milestones', MILESTONE)
    milestone_data: list[str] = os.listdir(milestone_path)

    # // del
    if TEST_PROBLEM_HASH is not None:
        milestone_data = [f for f in milestone_data if TEST_PROBLEM_HASH in f]
        if not milestone_data:
            print(f"No problem found matching hash: {TEST_PROBLEM_HASH}")
            exit(1)

    arc_milestone_problems: list[ArcProblem] = load_arc_problems(milestone_path, milestone_data)

    # instantiate the agent once
    arc_agent: ArcAgent = ArcAgent()

    run_timestamp = time.strftime("%H%M")
    milestone_data_set, problem_times = run_training_data(arc_agent, arc_milestone_problems, run_timestamp)
    milestone_file = open('Milestone_Results.csv', 'w')
    milestone_file.write("Problem Name, Correct, Correct Answer, Prediction 1, Prediction 2, Prediction 3\n")
    for m_answer_set in milestone_data_set.keys():
        m_correct, predictions = milestone_data_set[m_answer_set]
        m_cor_ans = m_answer_set.test_set().get_output_data().data().tolist()
        milestone_file.write(f'{m_answer_set.problem_name()},'
                             f'{m_correct},'
                             f'"{m_cor_ans}",')
        if len(predictions) == 0:
            milestone_file.write("empty\n")
            continue
        for idx, pred in enumerate(predictions, 1):
            if len(predictions) == idx:
                milestone_file.write(f'"{pred.tolist()}"\n')
            else:
                milestone_file.write(f'"{pred.tolist()}",')

    milestone_file.close()
    # print summary of results: n out of m tests passed
    total = len(milestone_data_set)
    passed = sum(1 for v in milestone_data_set.values() if v[0])
    print(f"{passed} out of {total} tests passed")

    correct_problems = [p.problem_name() for p, v in milestone_data_set.items() if v[0]]
    incorrect_problems = [p.problem_name() for p, v in milestone_data_set.items() if not v[0]]

    print("\nCorrectly solved:")
    for name in correct_problems:
        print(f"  ✓ {name}")

    print("\nNot solved:")
    for name in incorrect_problems:
        print(f"  ✗ {name}")

    times_ms = [ms for _, ms, _ in problem_times]
    fastest = min(problem_times, key=lambda t: t[1])
    slowest = max(problem_times, key=lambda t: t[1])
    print(f"\nTiming (ms per problem):")
    print(f"  Average : {sum(times_ms) / len(times_ms):.1f} ms")
    print(f"  Fastest : {fastest[1]:.1f} ms  ({fastest[0]})")
    print(f"  Slowest : {slowest[1]:.1f} ms  ({slowest[0]})")
