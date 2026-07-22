import json
import os
import time

import numpy as np

from runTransformation import (
    runTransformation,
    _generate_list_of_same_color_blobs,
    get_and_reset_transform_timings,
)
from PerformMea import PerformMea

ARC_DEBUG = os.environ.get("ARC_DEBUG", "1") != "0"


def dbg(msg):
    if ARC_DEBUG:
        print(f"[DEBUG] {msg}")


with open(os.path.join(os.path.dirname(__file__), "plans.json")) as _f:
    PLANS = json.load(_f)

_PLANS_BY_NAME = {plan["name"]: plan for plan in PLANS}

with open(os.path.join(os.path.dirname(__file__), "transformations.json")) as _f:
    TRANSFORMATIONS = json.load(_f)

_TRANSFORMATIONS_BY_NAME = {t["name"]: t for t in TRANSFORMATIONS}

# transformations whose outputs should not be expanded any further (see
# "has_no_children" in transformations.json)
HAS_NO_CHILDREN_TRANSFORMATIONS = {
    t["name"] for t in TRANSFORMATIONS if t.get("has_no_children", False)
}

# these transformations combine several sibling matrices together rather than
# expanding a single matrix into many candidates - see transformation_rounds
# in plans.json (e.g. "divide" then "logic_combine"/"stack_combine")
COMBINE_TRANSFORMATIONS = {"logic_combine", "stack_combine", "concat_all_combos", "reflect_over_lines"}

# transformations that need a list of coordinates as their "parameters"
# (see transformations.json); computed once per runPlan() call since there is
# no pre_info_gathering step yet
COORDINATE_TRANSFORMATIONS = {"draw_line_between_points", "draw_drawable_lines"}

# transformations that need the tree's original (pre-transformation) input
# matrix as their "parameters"
ROOT_INPUT_TRANSFORMATIONS = {"apply_original_input", "crop_to_m_n_of_input_dim"}

# transformations that need the (rows, cols) shape shared by every training
# example's output as their "parameters" (see shared_output_dims in runPlan())
SHARED_OUTPUT_DIM_TRANSFORMATIONS = {"crop_to_shared_output_dimensions"}

MAX_SECONDS_PER_PLAN = 60

# accumulated per-runPlan()-call cost of the per-node tag computation in
# _make_children, reset at the top of each runPlan() call and reported at the end
_TIME_ROOT_PATTERN = 0.0


class TransformationNode:
    def __init__(self, name, params, result, tags=None, parent=None, is_dead_end=False, has_no_children=False):
        self.name = name
        self.params = params
        self.result = result
        self.tags = tags if tags is not None else {}
        self.parent = parent
        self.children = []
        self.matched = False
        self.is_dead_end = is_dead_end
        self.has_no_children = has_no_children


def iter_nodes_with_paths(root):
    """
    Depth-first walk of the tree, yielding (path, node) for every node,
    where path is the tuple of child-indices leading from root to node
    (the root's own path is the empty tuple).
    """
    def _walk(node, path):
        yield path, node
        for i, child in enumerate(node.children):
            yield from _walk(child, path + (i,))

    yield from _walk(root, ())


def get_node_at_path(root, path):
    """
    Returns the node reached by following path (a tuple of child-indices)
    from root, or None if the path doesn't exist in this tree.
    """
    node = root
    for idx in path:
        if idx < 0 or idx >= len(node.children):
            return None
        node = node.children[idx]
    return node


def _find_root(node):
    """
    Walks node.parent up to the tree's root node (the "input" node created in
    runPlan()) and returns it.
    """
    while node.parent is not None:
        node = node.parent
    return node


def _find_important_coordinates(matrix):
    """
    Returns one representative [row, col] coordinate per same-color blob in
    matrix (deduplicated), used as the list_of_coords parameter for
    draw_line_between_points / draw_drawable_lines.
    Adapted from findImportantCoordinates in OLD_ArcAgent.py.
    """
    coords = []
    blobs = _generate_list_of_same_color_blobs([matrix])
    for blob in blobs:
        blob_rows, blob_cols = np.where(blob == 1)
        if len(blob_rows) == 0:
            continue
        min_r, max_r = int(blob_rows.min()), int(blob_rows.max())
        min_c, max_c = int(blob_cols.min()), int(blob_cols.max())
        coords.append([(min_r + max_r) // 2, (min_c + max_c) // 2])

    seen = set()
    unique_coords = []
    for c in coords:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_coords.append(c)
    return unique_coords


def _get_plan(plan_name):
    try:
        return _PLANS_BY_NAME[plan_name]
    except KeyError:
        raise ValueError(f"Unknown plan: {plan_name}")


def _normalize_transformation_output(output, tags):
    if isinstance(output, list):
        return output, tags
    return [output], [tags]


def _root_comparison_tags(root_matrix, output_matrix):
    """
    Compares output_matrix against the tree's root input matrix (the puzzle's
    original, pre-transformation input), returning tags that describe how
    much of the root's content survives in this step's output.
    """
    rows = min(root_matrix.shape[0], output_matrix.shape[0])
    cols = min(root_matrix.shape[1], output_matrix.shape[1])

    if rows == 0 or cols == 0:
        return {
            "number_of_pixels_that_match_root_input_matrix": 0,
            "not_including_0s_root_input_matrix_is_in_output": False,
            "root_input_matrix_pattern_found_anywhere_in_output": False,
        }

    root_sub = root_matrix[:rows, :cols]
    output_sub = output_matrix[:rows, :cols]
    root_nonzero_mask = root_sub != 0

    num_matching_pixels = int(np.sum((root_sub == output_sub) & root_nonzero_mask))

    if np.any(root_nonzero_mask):
        is_preserved_at_position = bool(np.all(output_sub[root_nonzero_mask] == root_sub[root_nonzero_mask]))
    else:
        is_preserved_at_position = True

    pattern_found_anywhere = _root_pattern_found_anywhere(root_matrix, output_matrix)

    return {
        "number_of_pixels_that_match_root_input_matrix": num_matching_pixels,
        "not_including_0s_root_input_matrix_is_in_output": is_preserved_at_position,
        "root_input_matrix_pattern_found_anywhere_in_output": pattern_found_anywhere,
    }


def _root_pattern_found_anywhere(root_matrix, output_matrix):
    """
    Slides root_matrix over every valid top-left offset in output_matrix,
    treating 0 in root_matrix as "don't care" (transparent), and returns
    whether any offset reproduces root_matrix's non-zero values exactly.
    """
    root_rows, root_cols = root_matrix.shape
    out_rows, out_cols = output_matrix.shape

    if root_rows > out_rows or root_cols > out_cols:
        return False

    root_nonzero_mask = root_matrix != 0
    if not np.any(root_nonzero_mask):
        return True

    windows = np.lib.stride_tricks.sliding_window_view(output_matrix, (root_rows, root_cols))
    matches = (windows == root_matrix) | ~root_nonzero_mask
    return bool(np.any(np.all(matches, axis=(-2, -1))))


def _make_children(parent_node, transformation_name, matrices, tags_list, mea_types, goal_matrix):
    global _TIME_ROOT_PATTERN
    new_nodes = []
    root_matrix = _find_root(parent_node).result
    for matrix, tags in zip(matrices, tags_list):
        t0 = time.time()
        root_tags = _root_comparison_tags(root_matrix, matrix)
        _TIME_ROOT_PATTERN += time.time() - t0
        child = TransformationNode(
            transformation_name,
            {},
            matrix,
            tags={**parent_node.tags, **tags, **root_tags, "transform": transformation_name},
            parent=parent_node,
            has_no_children=transformation_name in HAS_NO_CHILDREN_TRANSFORMATIONS,
        )
        parent_node.children.append(child)
        if goal_matrix is not None and mea_types:
            for mea_type in mea_types:
                if not PerformMea(root_matrix, parent_node.result, matrix, goal_matrix, mea_type):
                    child.is_dead_end = True
                    break
        new_nodes.append(child)
    return new_nodes


def _mark_non_unique_dead_ends(nodes):
    """
    Groups nodes by their transformed matrix (node.result) and marks every
    node after the first live one in each group as a dead end, bounding
    frontier growth when a round produces many duplicate matrices.
    Nodes already marked as dead ends are excluded from the grouping so a
    live duplicate is never killed just because an already-dead node of the
    same matrix happened to be seen first.
    """
    seen = set()
    for node in nodes:
        if node.is_dead_end:
            continue
        key = (node.result.shape, node.result.tobytes())
        if key in seen:
            node.is_dead_end = True
        else:
            seen.add(key)


def _group_by_parent(frontier):
    groups = {}
    order = []
    for node in frontier:
        key = id(node.parent)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(node)
    return [groups[key] for key in order]


def _run_one_iteration(frontier, transformation_names, mea_types, coords, goal_matrix, shared_output_dims):
    """
    Applies one round-iteration's transformation_names to frontier, returning
    the newly created child nodes. COMBINE_TRANSFORMATIONS are applied once
    per shared-parent sibling group (see _group_by_parent); every other
    transformation is applied once per node in frontier independently.
    """
    new_frontier = []
    sibling_groups = _group_by_parent(frontier)

    for transformation_name in transformation_names:
        transform_start = time.time()
        before_count = len(new_frontier)
        if transformation_name in COMBINE_TRANSFORMATIONS:
            for siblings in sibling_groups:
                if not siblings:
                    continue
                host = siblings[0]
                other_matrices = [s.result for s in siblings[1:]]
                try:
                    output, tags = runTransformation(host.result, transformation_name, other_matrices)
                except Exception:
                    continue
                matrices, tags_list = _normalize_transformation_output(output, tags)
                new_frontier.extend(_make_children(host, transformation_name, matrices, tags_list, mea_types, goal_matrix))
        else:
            for node in frontier:
                if transformation_name in COORDINATE_TRANSFORMATIONS:
                    parameters = coords
                elif transformation_name in ROOT_INPUT_TRANSFORMATIONS:
                    parameters = [_find_root(node).result]
                elif transformation_name in SHARED_OUTPUT_DIM_TRANSFORMATIONS:
                    parameters = [shared_output_dims] if shared_output_dims else []
                else:
                    parameters = []
                try:
                    output, tags = runTransformation(node.result, transformation_name, parameters)
                except Exception:
                    continue
                matrices, tags_list = _normalize_transformation_output(output, tags)
                new_frontier.extend(_make_children(node, transformation_name, matrices, tags_list, mea_types, goal_matrix))

        dbg(f"transformation '{transformation_name}' took "
            f"{(time.time() - transform_start) * 1000:.1f} ms, produced {len(new_frontier) - before_count} nodes")

    return new_frontier


def _run_rounds(frontier, transformation_rounds, coords, start_time, goal_matrix, shared_output_dims=None, frontier_size_limit=None):
    """
    Advances frontier through transformation_rounds (a plan's
    "transformation_rounds" list), building children onto whatever tree the
    frontier's nodes already belong to.
    Returns (frontier, time_exceeded) where time_exceeded is True if
    MAX_SECONDS_PER_PLAN was hit partway through.
    """
    frontier = [node for node in frontier if not node.is_dead_end and not node.has_no_children]

    for round_idx, round_spec in enumerate(transformation_rounds):
        num_times = round_spec.get("num_times", 0)
        transformation_names = round_spec.get("transformations", [])
        mea_types = round_spec.get("mea", [])

        for iteration in range(num_times):
            if time.time() - start_time > MAX_SECONDS_PER_PLAN:
                dbg(f"round {round_idx} iteration {iteration}: MAX_SECONDS_PER_PLAN ({MAX_SECONDS_PER_PLAN}s) exceeded, aborting rounds early")
                return frontier, True

            dbg(f"round {round_idx} iteration {iteration}: starting with frontier size {len(frontier)}")
            new_frontier = _run_one_iteration(frontier, transformation_names, mea_types, coords, goal_matrix, shared_output_dims)

            if frontier_size_limit is not None and len(new_frontier) > frontier_size_limit:
                _mark_non_unique_dead_ends(new_frontier)

            dead_end_count = sum(1 for node in new_frontier if node.is_dead_end)
            dbg(f"round {round_idx} iteration {iteration}: {dead_end_count} of {len(new_frontier)} nodes marked is_dead_end = True")

            frontier = [node for node in new_frontier if not node.is_dead_end and not node.has_no_children]
            if not frontier:
                return frontier, False

    return frontier, False


def runPlan(input_matrix, plan_name, goal_matrix, shared_output_dims=None, frontier_size_limit=None):
    """
    Runs the named plan (see plans.json) against input_matrix, building and
    returning the root of a TransformationNode tree of every candidate matrix
    the plan's transformation_rounds can produce. The "every_end" plan's
    transformation_rounds are then appended onto the resulting frontier, so
    every plan finishes with those rounds regardless of plan_name.

    goal_matrix is the known correct output to check the plan's progress
    against via PerformMea (see the "mea" key on individual entries of
    plan["transformation_rounds"] in plans.json), and is only available for
    training examples. Pass None when there is no known goal (e.g. when
    running against test input) to skip MEA checks entirely.

    shared_output_dims is the (rows, cols) shape shared by every training
    example's output (see SHARED_OUTPUT_DIM_TRANSFORMATIONS), or None if no
    such shared shape exists. It's the same across training and test-input
    runs of a given problem, unlike goal_matrix.

    frontier_size_limit, if given, bounds frontier growth: once a round's
    frontier exceeds this size, all but one node per distinct transformed
    matrix are marked as dead ends (see _mark_non_unique_dead_ends).
    """
    global _TIME_ROOT_PATTERN
    _TIME_ROOT_PATTERN = 0.0
    get_and_reset_transform_timings()

    plan = _get_plan(plan_name)

    root = TransformationNode("input", {}, input_matrix)
    frontier = [root]

    coord_start = time.time()
    coords = _find_important_coordinates(input_matrix)
    dbg(f"plan '{plan_name}': _find_important_coordinates took {(time.time() - coord_start) * 1000:.1f} ms, found {len(coords)} coords")

    start_time = time.time()

    rounds_start = time.time()
    frontier, time_exceeded = _run_rounds(frontier, plan.get("transformation_rounds", []), coords, start_time, goal_matrix, shared_output_dims, frontier_size_limit)
    dbg(f"plan '{plan_name}': primary rounds took {(time.time() - rounds_start) * 1000:.1f} ms, "
        f"time_exceeded={time_exceeded}, final frontier size {len(frontier)}")
    if time_exceeded or not frontier:
        _report_plan_timings(plan_name)
        return root

    every_end_start = time.time()
    every_end = _get_plan("every_end")
    _run_rounds(frontier, every_end.get("transformation_rounds", []), coords, start_time, goal_matrix, shared_output_dims, frontier_size_limit)
    dbg(f"plan '{plan_name}': every_end rounds took {(time.time() - every_end_start) * 1000:.1f} ms")

    _report_plan_timings(plan_name)

    return root


def replayPlanPath(input_matrix, plan_name, path, shared_output_dims=None):
    """
    Rebuilds only the nodes needed to reach `path` (a child-index tuple, as
    found by find_index_matches on a training tree) against input_matrix,
    instead of the full multi-branch tree runPlan() would build for every
    transformation choice at every round.

    A node's children only depend on that node's own result, so replaying
    just the single lineage of nodes leading to `path` reproduces the exact
    same node runPlan(input_matrix, plan_name, None, shared_output_dims)
    would have built at that path - except across COMBINE_TRANSFORMATIONS
    rounds (logic_combine, stack_combine, ...), which need every sibling
    produced by the previous round for the same parent. So the frontier is
    only narrowed down to the single node continuing the path when the
    *next* round-iteration doesn't combine; otherwise the full sibling
    group produced this iteration is carried forward.

    Returns the tree root; get_node_at_path(root, path) yields the replayed
    node.
    """
    plan = _get_plan(plan_name)
    root = TransformationNode("input", {}, input_matrix)

    coords = _find_important_coordinates(input_matrix)
    start_time = time.time()

    all_rounds = list(plan.get("transformation_rounds", [])) + list(_get_plan("every_end").get("transformation_rounds", []))
    flat_steps = [
        round_spec.get("transformations", [])
        for round_spec in all_rounds
        for _ in range(round_spec.get("num_times", 0))
    ]

    frontier = [root]
    path_pos = 0

    for step_idx, transformation_names in enumerate(flat_steps):
        if not frontier or path_pos >= len(path):
            break
        if time.time() - start_time > MAX_SECONDS_PER_PLAN:
            dbg(f"plan '{plan_name}' (replay): MAX_SECONDS_PER_PLAN ({MAX_SECONDS_PER_PLAN}s) exceeded, aborting early")
            break

        new_frontier = _run_one_iteration(frontier, transformation_names, [], coords, None, shared_output_dims)
        if not new_frontier:
            break

        idx = path[path_pos]
        path_pos += 1
        if idx < 0 or idx >= len(new_frontier):
            # path isn't resolvable against this input - shouldn't happen for
            # a genuine index match, stop here with whatever was built
            break

        next_names = flat_steps[step_idx + 1] if step_idx + 1 < len(flat_steps) else []
        next_is_combine = any(name in COMBINE_TRANSFORMATIONS for name in next_names)
        frontier = new_frontier if next_is_combine else [new_frontier[idx]]
        frontier = [node for node in frontier if not node.is_dead_end and not node.has_no_children]

    return root


def _report_plan_timings(plan_name):
    dbg(f"plan '{plan_name}': _root_comparison_tags total {_TIME_ROOT_PATTERN * 1000:.1f} ms")
    transform_timings = get_and_reset_transform_timings()
    if transform_timings:
        top = sorted(transform_timings.items(), key=lambda kv: kv[1][0], reverse=True)[:5]
        summary = ", ".join(f"{name}={ms:.1f}ms(x{count})" for name, (ms, count) in top)
        dbg(f"plan '{plan_name}': top transformations by time: {summary}")
