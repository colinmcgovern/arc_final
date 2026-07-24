import os
import time

import numpy as np
from scipy import ndimage

from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet
from runPlan import runPlan, replayPlanPath, iter_nodes_with_paths, get_node_at_path
from runTransformation import _count_lines_by_direction, _find_donuts

ARC_DEBUG = os.environ.get("ARC_DEBUG", "1") != "0"


def dbg(msg):
    if ARC_DEBUG:
        pass  # print(f"[DEBUG] {msg}")

TAG_FALLBACK_CANDIDATES_PER_PLAN = 3

_BLOB_STRUCT = np.ones((3, 3), dtype=int)

FRONTIER_SIZE_LIMIT_BEFORE_REMOVING_NON_UNIQUE = 5000

def makePlanAssignments(
    outputHasMoreLines,
    isDivisionCombine,
    possibleReflection,
    possibleBlobReflection,
    inputHasDonut,
    inputHasConsistantSize
):
    print("outputHasMoreLines", outputHasMoreLines)
    print("isDivisionCombine", isDivisionCombine)
    print("possibleReflection", possibleReflection)
    print("possibleBlobReflection", possibleBlobReflection)
    print("inputHasDonut", inputHasDonut)

    plansToExecute = []
    plansToExecute.append("blob_reflections")
    plansToExecute.append("reflections")
    plansToExecute.append("general")
    plansToExecute.append("dialate_inscribe")
    if (
        inputHasConsistantSize == True
    ):
        plansToExecute.append("make_graph")
    if (
        inputHasDonut == True
    ):
        plansToExecute.append("donut_recoloring")
    if (
        outputHasMoreLines == True
    ):
        plansToExecute.append("draw_lines_between_blobs")
        plansToExecute.append("draw_lines_drawable_directions")
    if (
        isDivisionCombine == True
    ):
        plansToExecute.append("divide_combine")
    if (
        possibleReflection == True
    ):
        plansToExecute.append("reflections")
    # if possibleBlobReflection == True:
    #     plansToExecute.append("blob_reflections")
    print("plansToExecute", plansToExecute)
    return plansToExecute


def _count_lines(matrix: np.ndarray) -> int:
    """
    Counts drawable line-runs (horizontal, vertical, and both diagonal
    directions) of length >= 3 whose color differs from the majority
    color on either side of the run.
    Ported from countLines in OLD_ArcAgent.py.
    """
    return sum(_count_lines_by_direction(matrix).values())


def findIfOutputsHasMoreLines(arc_problem: ArcProblem) -> bool:
    """
    Returns True if the training outputs contain drawable lines
    that aren't present in the corresponding inputs (i.e. the
    "draw_lines_between_blobs" / "draw_lines_drawable_directions"
    plans are worth attempting).
    """
    return any(
        _count_lines(example.get_output_data().data()) > _count_lines(example.get_input_data().data())
        for example in arc_problem.training_set()
    )


def findIfInputHasDonut(arc_problem: ArcProblem) -> bool:
    """
    Returns True if at least one training input contains a donut
    (a single-color blob that fully encloses a hole).
    """
    return any(
        len(_find_donuts(example.get_input_data().data())) > 0
        for example in arc_problem.training_set()
    )

# this takes in a arcproblem and outputs true if all of the output matrices 
# for the examples are all the same size
def findIfProblemHasConsistentOutputSize(arc_problem: ArcProblem) -> bool:
    """
    Returns True if every training example's output matrix has the
    same shape (rows and columns).
    """
    return _shared_output_dimensions(arc_problem) is not None

def _is_divider_line_set(arc_set: ArcSet) -> bool:
    input_shape = arc_set.get_input_data().shape()
    output_shape = arc_set.get_output_data().shape()
    input_rows, input_cols = input_shape[0], input_shape[1]
    output_rows, output_cols = output_shape[0], output_shape[1]
    row_halved = (abs(input_rows // 2 - output_rows) <= 1) and (input_cols == output_cols) and (output_rows > 0)
    col_halved = (input_rows == output_rows) and (abs(input_cols // 2 - output_cols) <= 1) and (output_cols > 0)
    return row_halved or col_halved


def findIfIsDivisionCombine(arc_problem: ArcProblem) -> bool:
    """
    Returns True if the training examples look like a
    divide-then-combine problem (input splits into sections that
    get logically/stacked-combined into the output).
    """
    return all(_is_divider_line_set(example) for example in arc_problem.training_set())

def _is_x_subset_of_y(input_matrix: np.ndarray, output_matrix: np.ndarray) -> bool:
    nonzero_rows, nonzero_cols = np.nonzero(input_matrix)
    if nonzero_rows.size == 0:
        return False
    r0, r1 = nonzero_rows.min(), nonzero_rows.max() + 1
    c0, c1 = nonzero_cols.min(), nonzero_cols.max() + 1
    cropped = input_matrix[r0:r1, c0:c1]

    in_rows, in_cols = cropped.shape
    out_rows, out_cols = output_matrix.shape

    if out_rows < in_rows or out_cols < in_cols:
        return False

    for r in range(out_rows - in_rows + 1):
        for c in range(out_cols - in_cols + 1):
            if np.array_equal(output_matrix[r:r + in_rows, c:c + in_cols], cropped):
                return True
    return False


def findPossibleReflection(arc_problem: ArcProblem) -> bool:
    """
    Returns True if, for every training example, the input matrix appears
    unmodified as a contiguous submatrix somewhere in the output AND at
    least one reflection (horizontal or vertical flip) of the input also
    appears as a contiguous submatrix in the output (the "reflections"
    plan).
    """
    training = arc_problem.training_set()

    if not training:
        return False

    def _example_matches(example) -> bool:
        input_matrix = example.get_input_data().data()
        output_matrix = example.get_output_data().data()
        if not _is_x_subset_of_y(input_matrix, output_matrix):
            return False
        return (
            _is_x_subset_of_y(np.fliplr(input_matrix), output_matrix)
            or _is_x_subset_of_y(np.flipud(input_matrix), output_matrix)
        )

    return all(_example_matches(example) for example in training)


def _shared_output_dimensions(arc_problem: ArcProblem):
    """
    Returns the (rows, cols) shape common to every training example's
    output, or None if the training outputs don't all share one shape.
    """
    shapes = {arc_set.get_output_data().data().shape for arc_set in arc_problem.training_set()}
    return shapes.pop() if len(shapes) == 1 else None


def _connected_component_masks(matrix: np.ndarray) -> list:
    masks = []
    for color in np.unique(matrix):
        if color == 0:
            continue
        labeled, n_features = ndimage.label((matrix == color).astype(int), structure=_BLOB_STRUCT)
        for label_id in range(1, n_features + 1):
            masks.append(labeled == label_id)
    return masks


def _has_mirrored_blob(input_matrix: np.ndarray, output_matrix: np.ndarray) -> bool:
    if input_matrix.shape != output_matrix.shape:
        return False

    input_nonzero = input_matrix != 0
    if not np.array_equal(output_matrix[input_nonzero], input_matrix[input_nonzero]):
        return False

    extra_positions = set(zip(*np.where((output_matrix != 0) & ~input_nonzero)))
    if not extra_positions:
        return False

    for blob_mask in _connected_component_masks(input_matrix):
        for flipped in (np.fliplr(blob_mask), np.flipud(blob_mask)):
            reflected_positions = set(zip(*np.where(flipped)))
            if reflected_positions and reflected_positions <= extra_positions:
                return True
    return False


def findPossibleBlobReflection(arc_problem: ArcProblem) -> bool:
    """
    Returns True if individual blobs in the input appear to be
    reflected over a line (the "blob_reflections" plan).

    Best-effort placeholder heuristic (no OLD_ArcAgent.py equivalent):
    same shape as input, every input foreground pixel preserved in the
    output, and the newly-added output pixels contain a mirrored copy
    (left-right or up-down) of at least one input blob.
    """
    training = arc_problem.training_set()
    if not training:
        return False
    return all(
        _has_mirrored_blob(example.get_input_data().data(), example.get_output_data().data())
        for example in training
    )


def performPlan(input_matrix: np.ndarray, plan: str, goal_matrix, shared_output_dims=None, frontier_size_limit=None):
    """
    Runs the named plan's transformation rounds (see plans.json) against
    a single input matrix and returns the resulting transform tree
    (the set of candidate output matrices produced by that plan, tagged
    with whatever metadata later matching steps need).

    goal_matrix is the known correct output used for MEA pruning (pass the
    training example's expected output, or None when there is no known
    goal, e.g. for test input).

    shared_output_dims is the (rows, cols) shape shared by every training
    example's output (see _shared_output_dimensions), or None if there
    isn't one.

    frontier_size_limit, if given, bounds frontier growth: once a round's
    frontier exceeds this size, all but one node per distinct transformed
    matrix are marked as dead ends (see _mark_non_unique_dead_ends).
    """
    return runPlan(input_matrix, plan, goal_matrix, shared_output_dims, frontier_size_limit)


def markMatchingOutputs(transformTreesForEveryInputMatrix):
    """
    Walks each plan's transform trees and marks which candidate nodes
    actually matched their training example's real output.
    """
    for plan_trees in transformTreesForEveryInputMatrix:
        for tree, expected_output in plan_trees:
            for _, node in iter_nodes_with_paths(tree):
                node.matched = node.result is not None and np.array_equal(node.result, expected_output)
    return transformTreesForEveryInputMatrix


def find_index_matches(transformTreesForEveryInputMatrix):
    """
    Returns, for each plan, the transform-tree node index/path that
    matched the correct output across the training examples.
    """
    index_matches = []
    for plan_trees in transformTreesForEveryInputMatrix:
        path_sets = [
            {path for path, node in iter_nodes_with_paths(tree) if node.matched}
            for tree, _ in plan_trees
        ]
        common_paths = set.intersection(*path_sets) if path_sets else set()
        index_matches.append(sorted(common_paths))
    return index_matches


def find_tag_matches(transformTreesForEveryInputMatrix):
    """
    Returns, for each plan, the transformation tags (see
    transformations.json unique_output_tags) that were shared by
    matching nodes across the training examples.
    """
    tag_matches = []
    for plan_trees in transformTreesForEveryInputMatrix:
        tag_sets = []
        for tree, _ in plan_trees:
            matched_tags = set()
            for _, node in iter_nodes_with_paths(tree):
                if node.matched:
                    matched_tags.update(f"{k}:{v}" for k, v in node.tags.items())
            tag_sets.append(matched_tags)
        common_tags = set.intersection(*tag_sets) if tag_sets else set()
        tag_matches.append(sorted(common_tags))
    return tag_matches


def _pixel_diff_count(input_matrix, result):
    """
    Returns the number of differing cells between input_matrix and result,
    or None if their shapes differ (so pixel-change can't be compared).
    """
    if input_matrix is None or result is None or input_matrix.shape != result.shape:
        return None
    return int(np.sum(input_matrix != result))


def applyAndSort(plansAppliedToTestInputMatrix, tag_matches, planAssignments) -> list[tuple[np.ndarray, dict]]:
    """
    Uses the index and tag matches found from the training examples to
    pick out the corresponding candidate matrices from the test input's
    transform trees, and returns (matrix, match_info) pairs ordered from
    highest to lowest predicted quality. match_info records which tier
    produced the candidate (index or tag) and how it matched (the shared
    path, or the shared tags).

    plansAppliedToTestInputMatrix holds one (kind, payload) pair per plan:
    kind "index" pairs with a list of (path, replayed_root) - one small,
    path-guided tree per matched path (see replayPlanPath); kind "full"
    pairs with a complete transform tree (needed for the tag-tier fallback,
    which must compare tags across many candidate nodes); kind "skip"
    means the plan had neither an index nor a tag match and was never run
    against the test input.
    """
    index_tier = []
    tag_tier = []

    for plan_idx, (kind, payload) in enumerate(plansAppliedToTestInputMatrix):
        plan = planAssignments[plan_idx]

        if kind == "index":
            for path, replayed_root in payload:
                node = get_node_at_path(replayed_root, path)
                if node is not None and node.result is not None:
                    index_tier.append((node.result, {"match_type": "index", "plan": plan, "path": path}))
            continue

        if kind == "skip":
            continue

        tree = payload  # kind == "full"
        tags_for_plan = set(tag_matches[plan_idx]) if plan_idx < len(tag_matches) else set()
        if not tags_for_plan:
            continue

        input_node = get_node_at_path(tree, ())
        input_matrix = input_node.result if input_node is not None else None

        scored = []
        for path, node in iter_nodes_with_paths(tree):
            if node.result is None:
                continue
            node_tags = {f"{k}:{v}" for k, v in node.tags.items()}
            overlap = node_tags & tags_for_plan
            if overlap:
                diff = _pixel_diff_count(input_matrix, node.result)
                scored.append((len(overlap), path, node.result, sorted(overlap), diff))
        scored.sort(key=lambda entry: (
            -entry[0],
            0 if entry[4] is not None else 1,
            -entry[4] if entry[4] is not None else 0,
            entry[1],
        ))
        for _, path, result, overlap, diff in scored[:TAG_FALLBACK_CANDIDATES_PER_PLAN]:
            tag_tier.append((result, {
                "match_type": "tag", "plan": plan, "path": path,
                "tags": overlap, "pixels_changed": diff,
            }))

    return index_tier + tag_tier

class ArcAgent:
    def __init__(self):
        """
        You may add additional variables to this init method. Be aware that it gets called only once
        and then the make_predictions method will get called several times.
        """
        pass

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:

        # Step 1 - Identify Problem Type
        outputHasMoreLines = findIfOutputsHasMoreLines(arc_problem)
        isDivisionCombine = findIfIsDivisionCombine(arc_problem)
        possibleReflection = findPossibleReflection(arc_problem)
        inputHasDonut = findIfInputHasDonut(arc_problem)
        possibleBlobReflection = False #findPossibleBlobReflection(arc_problem)
        inputHasConsistantSize = findIfProblemHasConsistentOutputSize(arc_problem)

        shared_output_dims = _shared_output_dimensions(arc_problem)

        # Step 1a - Find Line Types


        # Step 2 - Assign Plans According To Problem Type
        planAssignments = makePlanAssignments(
            outputHasMoreLines,
            isDivisionCombine,
            possibleReflection,
            possibleBlobReflection,
            inputHasDonut,
            inputHasConsistantSize
        )

        # Step 3 - Apply Plans
        step3_start = time.perf_counter()
        transformTreesForEveryInputMatrix = []
        for plan in planAssignments:

            plan_start = time.perf_counter()
            transformTreePerPlan = []

            for arc_set in arc_problem.training_set():
                inputMatrix = arc_set.get_input_data().data()
                expectedOutput = arc_set.get_output_data().data()
                transformTreePerPlan.append(
                    (performPlan(inputMatrix, plan, expectedOutput, shared_output_dims, FRONTIER_SIZE_LIMIT_BEFORE_REMOVING_NON_UNIQUE), expectedOutput)
                )

            transformTreesForEveryInputMatrix.append(transformTreePerPlan)
            dbg(f"{arc_problem.problem_name()}: [train] plan '{plan}' took {(time.perf_counter() - plan_start) * 1000:.1f} ms")

        dbg(f"{arc_problem.problem_name()}: step 3 (apply plans to training set) total {(time.perf_counter() - step3_start) * 1000:.1f} ms")

        # Step 4 - Find Matching Outputs
        step4_start = time.perf_counter()
        transformTreesForEveryInputMatrix = markMatchingOutputs(transformTreesForEveryInputMatrix)
        dbg(f"{arc_problem.problem_name()}: step 4 (mark matching outputs) took {(time.perf_counter() - step4_start) * 1000:.1f} ms")

        example_match_summaries = []
        for plan_idx, plan in enumerate(planAssignments):
            for example_idx, (tree, _) in enumerate(transformTreesForEveryInputMatrix[plan_idx]):
                nodes = list(iter_nodes_with_paths(tree))
                matched_count = sum(1 for _, node in nodes if node.matched)
                total_count = sum(1 for _, node in nodes if node.result is not None)
                example_match_summaries.append((plan, example_idx, matched_count, total_count))

        # Step 4 - Find index matches
        index_matches = find_index_matches(transformTreesForEveryInputMatrix)
        # for plan, matches in zip(planAssignments, index_matches):
        #     if matches:
        #         print(f"{arc_problem.problem_name()}: plan '{plan}' matched output for every example")

        # Step 5 - Find tag matches
        tag_matches = find_tag_matches(transformTreesForEveryInputMatrix)

        # Step 5b - Print shared tags for plans that matched every example
        for plan_idx, plan in enumerate(planAssignments):
            plan_trees = transformTreesForEveryInputMatrix[plan_idx]
            all_examples_matched = all(
                any(node.matched for _, node in iter_nodes_with_paths(tree))
                for tree, _ in plan_trees
            )
            # if all_examples_matched:
            #     print(f"{arc_problem.problem_name()}: plan '{plan}' matched every example, shared tags: {tag_matches[plan_idx]}")

        # Step 6 - Case Based Solutions (skipped for now. may implment later)

        # Step 7 - Apply Plans to Test Input
        #
        # Plans with an index match only need the single tree node their
        # matched path resolves to on the test input, so replayPlanPath
        # rebuilds just that lineage instead of the full multi-branch tree.
        # Plans with only a tag match still need a full tree (the tag-tier
        # fallback compares tags across many candidate nodes). Plans with
        # neither never get read by applyAndSort, so they're skipped
        # entirely.
        step7_start = time.perf_counter()
        testInputMatrix = arc_problem.test_set().get_input_data().data()
        plansAppliedToTestInputMatrix = []
        last_test_trees = []
        for plan_idx, plan in enumerate(planAssignments):
            plan_start = time.perf_counter()
            paths = index_matches[plan_idx] if plan_idx < len(index_matches) else []
            tags_for_plan = tag_matches[plan_idx] if plan_idx < len(tag_matches) else []

            if paths:
                replayed = [(path, replayPlanPath(testInputMatrix, plan, path, shared_output_dims)) for path in paths]
                plansAppliedToTestInputMatrix.append(("index", replayed))
                last_test_trees.extend((plan, root) for _, root in replayed)
            elif tags_for_plan:
                tree = performPlan(testInputMatrix, plan, None, shared_output_dims, FRONTIER_SIZE_LIMIT_BEFORE_REMOVING_NON_UNIQUE)
                plansAppliedToTestInputMatrix.append(("full", tree))
                last_test_trees.append((plan, tree))
            else:
                plansAppliedToTestInputMatrix.append(("skip", None))

            dbg(f"{arc_problem.problem_name()}: [test] plan '{plan}' took {(time.perf_counter() - plan_start) * 1000:.1f} ms")

        dbg(f"{arc_problem.problem_name()}: step 7 (apply plans to test input) total {(time.perf_counter() - step7_start) * 1000:.1f} ms")

        self.last_test_trees = last_test_trees

        last_train_trees_by_example = []
        for example_idx in range(len(arc_problem.training_set())):
            example_trees = [
                (plan, transformTreesForEveryInputMatrix[plan_idx][example_idx][0])
                for plan_idx, plan in enumerate(planAssignments)
            ]
            last_train_trees_by_example.append(example_trees)
        self.last_train_trees_by_example = last_train_trees_by_example

        # Step 8 - Apply top matches to plans
        step8_start = time.perf_counter()
        all_predictions_from_highest_to_lowest_quality = applyAndSort(
            plansAppliedToTestInputMatrix,
            tag_matches,
            planAssignments
        )
        dbg(f"{arc_problem.problem_name()}: step 8 (apply and sort) took {(time.perf_counter() - step8_start) * 1000:.1f} ms")

        # Step 9 - Take the top 3 unique predictions, padding with the best
        # guess if fewer than 3 unique candidates were found.
        predictions: list[np.ndarray] = []
        prediction_matches: list[dict] = []
        for candidate, match_info in all_predictions_from_highest_to_lowest_quality:
            if not any(np.array_equal(candidate, existing) for existing in predictions):
                predictions.append(candidate)
                prediction_matches.append(match_info)
            if len(predictions) == 3:
                break
        while predictions and len(predictions) < 3:
            predictions.append(predictions[-1])
            prediction_matches.append(prediction_matches[-1])

        self.last_prediction_matches = prediction_matches

        if ARC_DEBUG:
            for plan, example_idx, matched_count, total_count in example_match_summaries:
                print(f"[DEBUG] example {example_idx + 1}, plan '{plan}' has {matched_count} out of {total_count} transformed matrices matching the goal matrix")

        return predictions  # a list[np.ndarray] of size 3
