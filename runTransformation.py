import itertools
import json
import os
from collections import Counter

import numpy as np
from scipy import ndimage

# input is a list of arugments
# the first arguemtnt is the input matrix
# the second arguement is the name of the transformation as a string
# the third is a list of parameters that will be fed into the transformation function
# there are multiple inputs
# the first output is the transformed matrix
# the second ouput is a list of tags as a dictionary for exameple: {"num_colors": 5, ...}
# tags include the following: most common color in the input matrix,
# 2nd most common color in input matrix, 3rd most common,
# most common color in output matrix, 2nd, 3rd, number of unique_colors in input,
# number of unique_color in output, does the input matrix match the
# output matrix

_TRANSFORMATIONS_PATH = os.path.join(os.path.dirname(__file__), "transformations.json")
with open(_TRANSFORMATIONS_PATH) as f:
    TRANSFORMATIONS = json.load(f)

MAX_NUM_OF_COLOR_COMBOS = 2
MAX_NUM_OF_PALETTE_ROTATION_COLORS = 5

_SCALE_PARAMS = [[2, 1], [2, 2], [1, 2], [0.5, 1], [0.5, 0.5], [1, 0.5]]
_DIVIDE_SPLIT_CONFIGS = [[1, 2], [2, 1]]
_TRIPLE_DIVIDE_SPLIT_CONFIGS = [[1, 3], [3, 1]]


# --- private helpers adapted from OLD_ArcAgent.py ---

def _generate_reflections(matrix_list):
    result = []
    for m in matrix_list:
        result.append(np.fliplr(m))
        result.append(np.flipud(m))
    return result

# rotate 90 degrees
def _generate_rotations(matrix_list):
    result = []
    for m in matrix_list:
        for k in [1,2,3]:
            result.append(np.rot90(m, k))
    return result


def _generate_list_of_same_color_blobs(matrix_list):
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    result = []
    for m in matrix_list:
        blobs = []
        for color in np.unique(m):
            mask = (m == color).astype(int)
            labeled, n_features = ndimage.label(mask, structure=struct)
            for label_id in range(1, n_features + 1):
                blobs.append((labeled == label_id).astype(int))
        blobs = blobs[:15]
        while len(blobs) < 15:
            blobs.append(np.zeros((1, 1), dtype=int))
        result.extend(blobs)
    return result


def _generate_extracted_colors(matrix_list):
    result = []
    for m in matrix_list:
        for color in range(10):
            if color in np.unique(m):
                result.append((m == color).astype(int))
            else:
                result.append(np.zeros((1, 1), dtype=int))
    return result


def _generate_split(matrix_list, split_list):
    result = []
    for m, split in zip(matrix_list, split_list):
        rows, cols = m.shape
        n_row, n_col = split
        pad_count = n_row * n_col
        if n_row == 0 or n_col == 0:
            for _ in range(pad_count):
                result.append(np.zeros((1, 1), dtype=int))
            continue
        row_h = rows // n_row
        col_w = cols // n_col
        if row_h == 0 or col_w == 0:
            for _ in range(pad_count):
                result.append(np.zeros((1, 1), dtype=int))
            continue

        row_ranges = []
        if n_row == 1:
            row_ranges = [(0, rows)]
        elif n_row == 2:
            row_ranges = [(0, row_h), (rows - row_h, rows)]
        else:
            for r in range(n_row):
                start = r * row_h
                row_ranges.append((start, start + row_h))

        col_ranges = []
        if n_col == 1:
            col_ranges = [(0, cols)]
        elif n_col == 2:
            col_ranges = [(0, col_w), (cols - col_w, cols)]
        else:
            for c in range(n_col):
                start = c * col_w
                col_ranges.append((start, start + col_w))

        for r_start, r_end in row_ranges:
            for c_start, c_end in col_ranges:
                sub = m[r_start:r_end, c_start:c_end]
                if sub.size == 0:
                    result.append(np.zeros((1, 1), dtype=int))
                else:
                    result.append(sub)
    return result


def _trace_line_pixels(r1, c1, r2, c2):
    pixels = []
    dr = r2 - r1
    dc = c2 - c1
    step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
    step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
    steps = max(abs(dr), abs(dc))
    for i in range(steps + 1):
        pixels.append((r1 + i * step_r, c1 + i * step_c))
    return pixels


def _generate_draw_line(matrix_list, corner_one_point, corner_two_point, color):
    r1, c1 = corner_one_point
    r2, c2 = corner_two_point
    dr = abs(r2 - r1)
    dc = abs(c2 - c1)
    if not (dr == 0 or dc == 0 or dr == dc):
        return [m.copy() for m in matrix_list]

    result = []
    for m in matrix_list:
        out = m.copy()
        rows, cols = out.shape
        line_pixels = _trace_line_pixels(r1, c1, r2, c2)[1:-1]
        for r, c in line_pixels:
            if 0 <= r < rows and 0 <= c < cols:
                out[r, c] = color
        result.append(out)

    return result


def _generate_eight_rays(matrix_list, source_point, color, is_through=False):
    directions = [
        (0, -1),   # left
        (-1, -1),  # up-left
        (-1, 0),   # up
        (-1, 1),   # up-right
        (0, 1),    # right
        (1, 1),    # down-right
        (1, 0),    # down
        (1, -1),   # down-left
    ]
    # When is_through is True, a line drawn from a direction is identical to the
    # line drawn from its mirrored direction (e.g. up-through == down-through),
    # so only one direction per pair is kept.
    through_directions = directions[:4]
    r, c = source_point
    result = []
    for m in matrix_list:
        rows, cols = m.shape
        if is_through:
            for dr, dc in through_directions:
                out = m.copy()
                out[r, c] = color
                for step in (1, -1):
                    rr, cc = r + dr * step, c + dc * step
                    while 0 <= rr < rows and 0 <= cc < cols:
                        out[rr, cc] = color
                        rr += dr * step
                        cc += dc * step
                result.append(out)
        else:
            for dr, dc in directions:
                out = m.copy()
                out[r, c] = color
                rr, cc = r + dr, c + dc
                while 0 <= rr < rows and 0 <= cc < cols:
                    out[rr, cc] = color
                    rr += dr
                    cc += dc
                result.append(out)
    return result


def _generate_logic_combinations(matrix_list):
    result = []
    for m1, m2 in itertools.combinations(matrix_list, 2):
        if m1.shape != m2.shape:
            continue
        b1 = m1.astype(bool)
        b2 = m2.astype(bool)
        result.append((b1 | b2).astype(int))
        result.append((b1 & b2).astype(int))
        result.append((b1 ^ b2).astype(int))
        result.append((~(b1 | b2)).astype(int))
        result.append((~(b1 & b2)).astype(int))
        result.append((~(b1 ^ b2)).astype(int))
    return result


def _stack_colored(matrix_list):
    if not matrix_list:
        return []
    result = []
    for perm in itertools.permutations(matrix_list):
        stacked = perm[0].copy().astype(int)
        for m in perm[1:]:
            mask = m != 0
            stacked[mask] = m[mask]
        result.append(stacked)
    return result


def _generate_scales(matrix_list, scale_list):
    result = []
    for m in matrix_list:
        rows, cols = m.shape
        for scale in scale_list:
            col_scale, row_scale = scale[0], scale[1]
            new_rows = rows * row_scale
            new_cols = cols * col_scale
            if (new_rows <= 0 or new_cols <= 0 or
                    abs(new_rows - round(new_rows)) > 1e-9 or
                    abs(new_cols - round(new_cols)) > 1e-9):
                result.append(np.zeros((1, 1), dtype=int))
                continue
            new_rows, new_cols = int(round(new_rows)), int(round(new_cols))
            scaled = m
            if row_scale != 1:
                if row_scale > 1:
                    rep = int(round(row_scale))
                    if rep != row_scale:
                        result.append(np.zeros((1, 1), dtype=int))
                        continue
                    scaled = np.repeat(scaled, rep, axis=0)
                else:
                    step_f = rows / new_rows
                    if abs(step_f - round(step_f)) > 1e-9:
                        result.append(np.zeros((1, 1), dtype=int))
                        continue
                    scaled = scaled[::int(round(step_f)), :]
            if col_scale != 1:
                if col_scale > 1:
                    rep = int(round(col_scale))
                    if rep != col_scale:
                        result.append(np.zeros((1, 1), dtype=int))
                        continue
                    scaled = np.repeat(scaled, rep, axis=1)
                else:
                    step_f = cols / new_cols
                    if abs(step_f - round(step_f)) > 1e-9:
                        result.append(np.zeros((1, 1), dtype=int))
                        continue
                    scaled = scaled[:, ::int(round(step_f))]
            result.append(scaled)
    return result


def _generate_cropped(matrix_list):
    result = []
    for m in matrix_list:
        nonzero_rows = np.any(m != 0, axis=1)
        nonzero_cols = np.any(m != 0, axis=0)
        if not np.any(nonzero_rows) or not np.any(nonzero_cols):
            result.append(m)
            continue
        rmin, rmax = np.where(nonzero_rows)[0][[0, -1]]
        cmin, cmax = np.where(nonzero_cols)[0][[0, -1]]
        result.append(m[rmin:rmax + 1, cmin:cmax + 1])
    return result


def _generate_block_combinations(matrix_list):
    result = []
    for m1, m2 in itertools.permutations(matrix_list, 2):
        if m1.shape[0] == m2.shape[0]:
            result.append(np.concatenate([m1, m2], axis=1))
        if m1.shape[1] == m2.shape[1]:
            result.append(np.concatenate([m1, m2], axis=0))
    return result


def _generate_all_color_combinations(matrix_list):
    result = []
    tags_list = []
    for m in matrix_list:
        present = sorted(int(c) for c in np.unique(m))
        k = len(present)
        if k == 0 or k > MAX_NUM_OF_COLOR_COMBOS:
            # print("generate all color combos skipped. num colors too high")
            continue
        # print("NOT SKIPPED!!!")
        digits = list(range(0, 10))
        for targets in itertools.product(digits, repeat=k):
            mapping = {present[i]: targets[i] for i in range(k)}
            new_m = np.zeros_like(m)
            for src, dst in mapping.items():
                new_m[m == src] = dst
            result.append(new_m)
            tags = {}
            for src, dst in mapping.items():
                tags[f"{src}_replaced_with"] = str(dst)
                tags[f"{src}_stayed_same"] = bool(src == dst)
            tags_list.append(tags)
    return result, tags_list


def _generate_palette_rotations(matrix_list):
    result = []
    for m in matrix_list:
        present = sorted(int(c) for c in np.unique(m))
        k = len(present)
        if k == 0 or k > MAX_NUM_OF_PALETTE_ROTATION_COLORS:
            continue
        identity = tuple(present)
        for perm in itertools.permutations(present):
            if perm == identity:
                continue
            mapping = {present[i]: perm[i] for i in range(k)}
            new_m = np.zeros_like(m)
            for src, dst in mapping.items():
                new_m[m == src] = dst
            result.append(new_m)
    return result


_RING_STRUCT = np.ones((3, 3), dtype=int)
_BLOB_STRUCT = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def _generate_ring_recolors(matrix_list, use_holes):
    result = []
    for m in matrix_list:
        ring = np.zeros(m.shape, dtype=bool)
        for color in np.unique(m):
            if color == 0:
                continue
            mask = (m == color)
            labeled, n_features = ndimage.label(mask, structure=_BLOB_STRUCT)
            for label_id in range(1, n_features + 1):
                blob_mask = labeled == label_id
                if use_holes:
                    region = ndimage.binary_fill_holes(blob_mask) & ~blob_mask
                    eroded = ndimage.binary_erosion(region, structure=_RING_STRUCT)
                    ring |= region & ~eroded
                else:
                    dilated = ndimage.binary_dilation(blob_mask, structure=_RING_STRUCT)
                    ring |= dilated & ~blob_mask
        for fill_color in range(10):
            out = m.copy()
            out[ring] = fill_color
            result.append(out)
    return result


# --- transformation subfunctions (see transformations.json) ---

def reflection(input_matrix: np.ndarray, parameters: list):
    """
    Reflects input_matrix left-right and up-down.
    Adapted from generate_reflections in OLD_ArcAgent.py.
    """
    return _generate_reflections([input_matrix])


def rotation(input_matrix: np.ndarray, parameters: list):
    """
    Rotates input_matrix by 90, 180, and 270 degrees.
    Adapted from generate_rotations in OLD_ArcAgent.py.
    """
    return _generate_rotations([input_matrix])


def extract_blobs(input_matrix: np.ndarray, parameters: list):
    """
    Extracts connected-component blobs of the same color (padded to 15),
    followed by a per-color mask (10, one per color 0-9).
    Adapted from generate_list_of_same_color_blobs and generate_extracted_colors
    in OLD_ArcAgent.py.
    """
    return _generate_list_of_same_color_blobs([input_matrix]) + _generate_extracted_colors([input_matrix])


def divide(input_matrix: np.ndarray, parameters: list):
    """
    Splits input_matrix using the default 1x2 and 2x1 configs.
    Adapted from generate_split in OLD_ArcAgent.py.
    """
    result = []
    for split_config in _DIVIDE_SPLIT_CONFIGS:
        result.extend(_generate_split([input_matrix], [split_config]))
    return result


def triple_divide(input_matrix: np.ndarray, parameters: list):
    """
    Splits input_matrix using the default 1x3 and 3x1 configs.
    Adapted from generate_split in OLD_ArcAgent.py.
    """
    result = []
    for split_config in _TRIPLE_DIVIDE_SPLIT_CONFIGS:
        result.extend(_generate_split([input_matrix], [split_config]))
    return result


# input is a matrix and the output is multiple matrixes with a border drawn around each blob of each color
# for example the following input:
# 000
# 010
# 000

# would become

# 111
# 111
# 111

# 222
# 212
# 222

# ...

# 999
# 919
# 999
def dialate(input_matrix: np.ndarray, parameters: list):
    """
    For each same-color blob (excluding background 0), recolors the ring of
    pixels immediately surrounding it, once per candidate color 0-9.
    """
    return _generate_ring_recolors([input_matrix], use_holes=False)

# input is a matrix and the output is multiple matrixes with a border drawn within each blob of each color
# for example the input
# 1110011111
# 1010010001
# 1110010001
# 1110010001
# 0000011111

# would become

# 1110011111
# 1110011111
# 1110011011
# 1110011111
# 0000011111

# ...

# 1110011111
# 1910019991
# 1110019091
# 1110019991
# 0000011111
def inscribe(input_matrix: np.ndarray, parameters: list):
    """
    For each same-color blob (excluding background 0), recolors the inner
    ring of any enclosed hole, once per candidate color 0-9.
    """
    return _generate_ring_recolors([input_matrix], use_holes=True)


def draw_line_between_points(input_matrix: np.ndarray, parameters: list):
    """
    Draws a line between every ordered pair of points in parameters (the coords
    list), using the color already present at the first point of each pair.
    Adapted from generate_draw_line in OLD_ArcAgent.py.
    """
    coords = parameters
    result = []
    for p1 in coords:
        for p2 in coords:
            if p1 == p2:
                continue
            color = int(input_matrix[p1[0], p1[1]])
            result.extend(_generate_draw_line([input_matrix], p1, p2, color))
    return result


def draw_drawable_lines(input_matrix: np.ndarray, parameters: list):
    """
    Draws 8-directional rays and 4 through-lines from every point in parameters
    (the coords list), using the color already present at that point.
    Adapted from generate_eight_rays in OLD_ArcAgent.py.
    """

    coords = parameters
    if len(coords) > 3:
        coords = coords[:3]
    result = []
    for point in coords:
        color = int(input_matrix[point[0], point[1]])
        result.extend(_generate_eight_rays([input_matrix], point, color, is_through=False))
        result.extend(_generate_eight_rays([input_matrix], point, color, is_through=True))
    return result


def logic_combine(input_matrix: np.ndarray, parameters: list):
    """
    Combines input_matrix with the matrices in parameters using all 6 logic
    gates (AND, OR, XOR, NOR, NAND, XNOR) pairwise.
    Adapted from generate_logic_combinations in OLD_ArcAgent.py (which only had
    NOR enabled; all 6 gates are enabled here).
    """
    matrix_list = [input_matrix] + list(parameters)
    return _generate_logic_combinations(matrix_list)


def stack_combine(input_matrix: np.ndarray, parameters: list):
    """
    Stacks input_matrix and the matrices in parameters on top of each other
    (0 treated as transparent), for every permutation.
    Adapted from stack_colored in OLD_ArcAgent.py.
    """
    matrix_list = [input_matrix] + list(parameters)
    return _stack_colored(matrix_list)


def apply_gravity(input_matrix: np.ndarray, parameters: list):
    """
    Placeholder for the "apply_gravity" transformation (see transformations.json).
    No equivalent logic exists in OLD_ArcAgent.py.
    """
    raise NotImplementedError


def scaling(input_matrix: np.ndarray, parameters: list):
    """
    Scales input_matrix using the default set of scale configs.
    Adapted from generate_scales in OLD_ArcAgent.py.
    """
    return _generate_scales([input_matrix], _SCALE_PARAMS)


def crop_to_content(input_matrix: np.ndarray, parameters: list):
    """
    Crops input_matrix down to its non-zero content.
    Adapted from generate_cropped in OLD_ArcAgent.py.
    """
    return _generate_cropped([input_matrix])[0]


def crop_one_off_each_side(input_matrix: np.ndarray, parameters: list):
    """
    Placeholder for the "crop_one_off_each_side" transformation (see transformations.json).
    No equivalent logic exists in OLD_ArcAgent.py.
    """
    raise NotImplementedError


def no_change(input_matrix: np.ndarray, parameters: list):
    """
    Returns input_matrix unchanged.
    """
    return input_matrix


def concat_all_combos(input_matrix: np.ndarray, parameters: list):
    """
    Concatenates input_matrix with the matrices in parameters, for every
    compatible ordered pair, horizontally and/or vertically.
    Adapted from generate_block_combinations in OLD_ArcAgent.py.
    """
    matrix_list = [input_matrix] + list(parameters)
    return _generate_block_combinations(matrix_list)


def crop_to_m_n_of_input_dim(input_matrix: np.ndarray, parameters: list):
    """
    Placeholder for the "crop_to_m_n_of_input_dim" transformation (see transformations.json).
    No equivalent logic exists in OLD_ArcAgent.py.
    """
    raise NotImplementedError


def make_graph(input_matrix: np.ndarray, parameters: list):
    """
    Placeholder for the "make_graph" transformation (see transformations.json).
    No equivalent logic exists in OLD_ArcAgent.py.
    """
    raise NotImplementedError


def fill_blobs(input_matrix: np.ndarray, parameters: list):
    """
    Placeholder for the "fill_blobs" transformation (see transformations.json).
    No equivalent logic exists in OLD_ArcAgent.py.
    """
    raise NotImplementedError


def recolor_donuts(input_matrix: np.ndarray, parameters: list):
    """
    Placeholder for the "recolor_donuts" transformation (see transformations.json).
    No equivalent logic exists in OLD_ArcAgent.py.
    """
    raise NotImplementedError


def reflect_over_lines(input_matrix: np.ndarray, parameters: list):
    """
    Placeholder for the "reflect_over_lines" transformation (see transformations.json).
    No equivalent logic exists in OLD_ArcAgent.py.
    """
    raise NotImplementedError


def remove_touching(input_matrix: np.ndarray, parameters: list):
    """
    Placeholder for the "remove_touching" transformation (see transformations.json).
    No equivalent logic exists in OLD_ArcAgent.py.
    """
    raise NotImplementedError


def apply_original_input(input_matrix: np.ndarray, parameters: list):
    """
    Returns input_matrix unchanged.
    """
    return input_matrix


def generate_all_color_combos(input_matrix: np.ndarray, parameters: list):
    """
    Generates all color-palette remappings of input_matrix (each present color
    independently mapped to any digit 0-9, up to MAX_NUM_OF_COLOR_COMBOS
    distinct colors present), including the identity remapping.
    Adapted from generate_all_color_combinations in OLD_ArcAgent.py.

    Returns (matrices, extra_tags) where extra_tags[i] records the color
    remapping used to produce matrices[i], e.g. {"0_replaced_with": "1"}.
    """
    return _generate_all_color_combinations([input_matrix])


# palette rotation generates all combinations of colors while keeping the same palette 
# for example the input
# 1 2
# would output
# 1 2 and 2 1
def palette_rotation(input_matrix: np.ndarray, parameters: list):
    """
    Generates every non-identity permutation of the colors present in
    input_matrix, remapping each present color to another present color
    (bijectively), up to MAX_NUM_OF_PALETTE_ROTATION_COLORS distinct colors.
    """
    return _generate_palette_rotations([input_matrix])


TRANSFORMATION_FUNCTIONS = {
    "reflection": reflection,
    "rotation": rotation,
    "extract_blobs": extract_blobs,
    "divide": divide,
    "triple_divide": triple_divide,
    "dialate": dialate,
    "inscribe": inscribe,
    "draw_line_between_points": draw_line_between_points,
    "draw_drawable_lines": draw_drawable_lines,
    "logic_combine": logic_combine,
    "stack_combine": stack_combine,
    "apply_gravity": apply_gravity,
    "scaling": scaling,
    "crop_to_content": crop_to_content,
    "crop_one_off_each_side": crop_one_off_each_side,
    "no_change": no_change,
    "concat_all_combos": concat_all_combos,
    "crop_to_m_n_of_input_dim": crop_to_m_n_of_input_dim,
    "make_graph": make_graph,
    "fill_blobs": fill_blobs,
    "recolor_donuts": recolor_donuts,
    "reflect_over_lines": reflect_over_lines,
    "remove_touching": remove_touching,
    "apply_original_input": apply_original_input,
    "generate_all_color_combos": generate_all_color_combos,
    "palette_rotation": palette_rotation,
}


def _top_color_ranks(matrix: np.ndarray, n: int = 3) -> list:
    """
    Returns up to n most common colors in matrix, ordered most->least frequent,
    padded with -1 if fewer than n unique colors are present.

    Adapted from findUniqueColors in OLD_ArcAgent.py (drops the corner_rank
    piece, which isn't needed here).
    """
    colors = [int(c) for c in np.unique(matrix)]
    counts = Counter(int(v) for v in matrix.flatten())
    ranked = sorted(colors, key=lambda c: counts[c], reverse=True)
    ranked = ranked[:n]
    while len(ranked) < n:
        ranked.append(-1)
    return ranked


def _make_tags(input_matrix: np.ndarray, output_matrix: np.ndarray) -> dict:
    input_ranks = _top_color_ranks(input_matrix)
    output_ranks = _top_color_ranks(output_matrix)

    return {
        "input_color_rank_1": input_ranks[0],
        "input_color_rank_2": input_ranks[1],
        "input_color_rank_3": input_ranks[2],
        "output_color_rank_1": output_ranks[0],
        "output_color_rank_2": output_ranks[1],
        "output_color_rank_3": output_ranks[2],
        "num_unique_colors_input": len(np.unique(input_matrix)),
        "num_unique_colors_output": len(np.unique(output_matrix)),
        "input_matches_output": bool(np.array_equal(input_matrix, output_matrix)),
    }


def runTransformation(input_matrix: np.ndarray, transformation_name: str, parameters: list):
    """
    Runs the named transformation (see transformations.json) against input_matrix
    with the given parameters, and returns (output_matrix, tags).

    If the transformation produces multiple candidate outputs, output_matrix is
    a list of matrices and tags is a parallel list of tags dicts (tags[i]
    describes output_matrix[i] vs input_matrix). Otherwise output_matrix is a
    single matrix and tags is a single dict.
    """
    if transformation_name not in TRANSFORMATION_FUNCTIONS:
        raise ValueError(f"Unknown transformation: {transformation_name}")

    result = TRANSFORMATION_FUNCTIONS[transformation_name](input_matrix, parameters)

    extra_tags = None
    if isinstance(result, tuple):
        output_matrix, extra_tags = result
    else:
        output_matrix = result

    if isinstance(output_matrix, list):
        tags = [_make_tags(input_matrix, candidate) for candidate in output_matrix]
        if extra_tags is not None:
            for tag_dict, extra in zip(tags, extra_tags):
                tag_dict.update(extra)
    else:
        tags = _make_tags(input_matrix, output_matrix)
        if extra_tags is not None:
            tags.update(extra_tags)

    return output_matrix, tags
