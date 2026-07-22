import os

from ArcDriver import load_arc_problems
from ArcAgent import (
    findIfOutputsHasMoreLines,
    findIfIsDivisionCombine,
    findPossibleReflection,
    makePlanAssignments,
    performPlan,
    _shared_output_dimensions,
)
from runPlan import iter_nodes_with_paths

TEST_PROBLEM_HASH = os.environ.get("TEST_PROBLEM_HASH", "f25ffba3")
MILESTONE = os.environ.get("MILESTONE", "B")


if __name__ == "__main__":
    milestone_path = os.path.join("..", "Milestone B", "Milestones", MILESTONE)
    milestone_data = [f for f in os.listdir(milestone_path) if TEST_PROBLEM_HASH in f]
    if not milestone_data:
        print(f"No problem found matching hash: {TEST_PROBLEM_HASH}")
        exit(1)

    arc_problem = load_arc_problems(milestone_path, milestone_data)[0]

    outputHasMoreLines = findIfOutputsHasMoreLines(arc_problem)
    isDivisionCombine = findIfIsDivisionCombine(arc_problem)
    possibleReflection = findPossibleReflection(arc_problem)
    possibleBlobReflection = False

    planAssignments = makePlanAssignments(
        outputHasMoreLines,
        isDivisionCombine,
        possibleReflection,
        possibleBlobReflection,
    )

    testInputMatrix = arc_problem.test_set().get_input_data().data()
    shared_output_dims = _shared_output_dimensions(arc_problem)

    print(f"\n=== {arc_problem.problem_name()} : plans executed = {planAssignments} ===")

    for plan in planAssignments:
        tree = performPlan(testInputMatrix, plan, None, shared_output_dims)
        nodes = [
            (path, node)
            for path, node in iter_nodes_with_paths(tree)
            if node.result is not None
        ]

        print(f"\n--- plan '{plan}' : {len(nodes)} total node(s) ---")
        for path, node in nodes:
            print(f"path={path} name={node.name} tags={node.tags}")
            print(node.result)
            print()
