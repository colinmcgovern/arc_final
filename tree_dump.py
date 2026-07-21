import os

from runPlan import iter_nodes_with_paths


def write_tree_dump(problem_name, plan_trees, output_dir, timestamp):
    """
    Writes the full transform tree (every node, not just leaves) for each
    (plan_name, tree_root) pair in plan_trees to
    <output_dir>/<problem_name>_<timestamp>.txt.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{problem_name}_{timestamp}.txt")

    with open(out_path, "w") as f:
        f.write(f"\n=== {problem_name} : plans executed = {[plan for plan, _ in plan_trees]} ===\n")
        for plan, tree in plan_trees:
            nodes = [
                (path, node)
                for path, node in iter_nodes_with_paths(tree)
                if node.result is not None
            ]

            f.write(f"\n--- plan '{plan}' : {len(nodes)} total node(s) ---\n")
            for path, node in nodes:
                f.write(f"path={path} name={node.name} tags={node.tags}\n")
                f.write(f"{node.result}\n")
                f.write("\n")

    return out_path
