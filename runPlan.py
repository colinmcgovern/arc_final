import json
import os
import time

import numpy as np

from runTransformation import runTransformation, _generate_list_of_same_color_blobs

_PLANS_PATH = os.path.join(os.path.dirname(__file__), "plans.json")
with open(_PLANS_PATH) as f:
    PLANS = json.load(f)

# these transformations combine several sibling matrices together rather than
# expanding a single matrix into many candidates - see transformation_rounds
# in plans.json (e.g. "divide" then "logic_combine"/"stack_combine")
COMBINE_TRANSFORMATIONS = {"logic_combine", "stack_combine", "concat_all_combos", "reflect_over_lines"}

# transformations that need a list of coordinates as their "parameters"
# (see transformations.json); computed once per runPlan() call since there is
# no pre_info_gathering step yet
COORDINATE_TRANSFORMATIONS = {"draw_line_between_points", "draw_drawable_lines"}

MAX_SECONDS_PER_PLAN = 60


class TransformationNode:
    def __init__(self, name, params, result, tags=None, parent=None):
        self.name = name
        self.params = params
        self.result = result
        self.tags = tags if tags is not None else {}
        self.parent = parent
        self.children = []
        self.matched = False


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
    for plan in PLANS:
        if plan["name"] == plan_name:
            return plan
    raise ValueError(f"Unknown plan: {plan_name}")


def _normalize_transformation_output(output, tags):
    if isinstance(output, list):
        return output, tags
    return [output], [tags]


def _make_children(parent_node, transformation_name, matrices, tags_list):
    new_nodes = []
    for matrix, tags in zip(matrices, tags_list):
        child = TransformationNode(
            transformation_name,
            {},
            matrix,
            tags={**tags, "transform": transformation_name},
            parent=parent_node,
        )
        parent_node.children.append(child)
        new_nodes.append(child)
    return new_nodes


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


def _run_rounds(frontier, transformation_rounds, coords, start_time):
    """
    Advances frontier through transformation_rounds (a plan's
    "transformation_rounds" list), building children onto whatever tree the
    frontier's nodes already belong to.
    Returns (frontier, time_exceeded) where time_exceeded is True if
    MAX_SECONDS_PER_PLAN was hit partway through.
    """
    for round_spec in transformation_rounds:
        num_times = round_spec.get("num_times", 0)
        transformation_names = round_spec.get("transformations", [])

        for _ in range(num_times):
            if time.time() - start_time > MAX_SECONDS_PER_PLAN:
                return frontier, True

            new_frontier = []
            sibling_groups = _group_by_parent(frontier)

            for transformation_name in transformation_names:
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
                        new_frontier.extend(_make_children(host, transformation_name, matrices, tags_list))
                else:
                    parameters = coords if transformation_name in COORDINATE_TRANSFORMATIONS else []
                    for node in frontier:
                        try:
                            output, tags = runTransformation(node.result, transformation_name, parameters)
                        except Exception:
                            continue
                        matrices, tags_list = _normalize_transformation_output(output, tags)
                        new_frontier.extend(_make_children(node, transformation_name, matrices, tags_list))

            frontier = new_frontier
            if not frontier:
                return frontier, False

    return frontier, False


def runPlan(input_matrix, plan_name):
    """
    Runs the named plan (see plans.json) against input_matrix, building and
    returning the root of a TransformationNode tree of every candidate matrix
    the plan's transformation_rounds can produce. The "every_end" plan's
    transformation_rounds are then appended onto the resulting frontier, so
    every plan finishes with those rounds regardless of plan_name.
    """
    plan = _get_plan(plan_name)

    root = TransformationNode("input", {}, input_matrix)
    frontier = [root]

    coords = _find_important_coordinates(input_matrix)

    start_time = time.time()

    frontier, time_exceeded = _run_rounds(frontier, plan.get("transformation_rounds", []), coords, start_time)
    if time_exceeded or not frontier:
        return root

    every_end = _get_plan("every_end")
    _run_rounds(frontier, every_end.get("transformation_rounds", []), coords, start_time)

    return root
