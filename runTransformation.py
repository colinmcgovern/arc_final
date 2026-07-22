import itertools
import os
import time
from collections import Counter

import numpy as np
from scipy import ndimage

ARC_DEBUG = os.environ.get("ARC_DEBUG", "1") != "0"


def dbg(msg):
    if ARC_DEBUG:
        print(f"[DEBUG] {msg}")


# accumulated (total_seconds, call_count) per transformation name, reset by
# get_and_reset_transform_timings() at the start/end of each runPlan() call
_TRANSFORM_TIMINGS = {}


def get_and_reset_transform_timings():
    global _TRANSFORM_TIMINGS
    timings = {name: (ms * 1000, count) for name, (ms, count) in _TRANSFORM_TIMINGS.items()}
    _TRANSFORM_TIMINGS = {}
    return timings

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


def _generate_stack_combos(matrix_list):
    result = []
    for base, overlay in itertools.permutations(matrix_list, 2):
        if base.shape != overlay.shape:
            result.append(base.copy())
            continue
        conflict = (base != 0) & (overlay != 0)
        if np.any(conflict):
            result.append(base.copy())
        else:
            combined = base.copy()
            mask = overlay != 0
            combined[mask] = overlay[mask]
            result.append(combined)
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
            lut = np.arange(10, dtype=m.dtype)
            for src, dst in mapping.items():
                lut[src] = dst
            result.append(lut[m])
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
            lut = np.arange(10, dtype=m.dtype)
            for src, dst in mapping.items():
                lut[src] = dst
            result.append(lut[m])
    return result


_RING_STRUCT = np.ones((3, 3), dtype=int)
_BLOB_STRUCT = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def _color_pixel_counts(matrix: np.ndarray) -> dict:
    counts = Counter(int(v) for v in matrix.flatten())
    counts.pop(0, None)
    return counts


def _color_blob_counts(matrix: np.ndarray) -> dict:
    counts = {}
    for color in np.unique(matrix):
        color = int(color)
        if color == 0:
            continue
        mask = (matrix == color).astype(int)
        _, n_features = ndimage.label(mask, structure=_BLOB_STRUCT)
        counts[color] = n_features
    return counts


def _build_bar_graph(color_values: dict) -> np.ndarray:
    if not color_values:
        return np.zeros((1, 1), dtype=int)
    colors = sorted(color_values)
    height = max(color_values.values())
    if height <= 0:
        return np.zeros((1, 1), dtype=int)
    graph = np.zeros((height, len(colors)), dtype=int)
    for col_idx, color in enumerate(colors):
        val = color_values[color]
        for row in range(height):
            level = height - row
            if val >= level:
                graph[row, col_idx] = color
    return graph


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


def _find_donuts(matrix: np.ndarray) -> list:
    """
    Finds "donuts": single-color 4-connected blobs (frames) that fully
    enclose a hole (per ndimage.binary_fill_holes - a hole touching the
    outer grid edge is not enclosed). A frame's enclosed interior may
    contain a mix of colors, not just background 0; color_counts and
    most_common_color only reflect those interior cells, never the frame.
    """
    donuts = []
    for color in np.unique(matrix):
        color = int(color)
        if color == 0:
            continue
        mask = (matrix == color)
        labeled, n_features = ndimage.label(mask, structure=_BLOB_STRUCT)
        for label_id in range(1, n_features + 1):
            border_mask = labeled == label_id
            interior_mask = ndimage.binary_fill_holes(border_mask) & ~border_mask
            if not interior_mask.any():
                continue

            color_counts = {c: 0 for c in range(10)}
            for c in matrix[interior_mask]:
                color_counts[int(c)] += 1
            most_common_color = max(color_counts, key=lambda c: (color_counts[c], -c))

            donuts.append({
                "border_color": color,
                "border_mask": border_mask,
                "border_coords": list(zip(*np.where(border_mask))),
                "interior_mask": interior_mask,
                "interior_coords": list(zip(*np.where(interior_mask))),
                "color_counts": color_counts,
                "most_common_color": most_common_color,
            })
    return donuts


# --- transformation subfunctions (see transformations.json) ---

def reflection(input_matrix: np.ndarray, parameters: list):
    """
    Reflects input_matrix left-right and up-down.
    Adapted from generate_reflections in OLD_ArcAgent.py.draw_line_between_points
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

# This function recolors only one donut at a time from in input matrix
# and outputs the new color for each of the possible colors
# if there is more than one donut then only recolor one at a time 
# ensure that tags capture if the donut is recolored to the most common
# color within the donut and other donut related fields
# example 1
# input
# 1 1 1 0 1
# 1 0 1 0 1
# 1 1 1 0 0

# output
# 1 1 1 0 1
# 1 0 1 0 1
# 1 1 1 0 0

# 2 2 2 0 1
# 2 0 2 0 1
# 2 2 2 0 0

# ...

# 9 9 9 0 1
# 9 0 9 0 1
# 9 9 9 0 0
# example 2
# input
# 1 1 1 0 1 1 1
# 1 0 1 0 1 0 1
# 1 1 1 0 1 1 1

# output
# 1 1 1 0 1 1 1
# 1 0 1 0 1 0 1
# 1 1 1 0 1 1 1

# ...

# 9 9 9 0 1 1 1
# 9 0 9 0 1 0 1
# 9 9 9 0 1 1 1

# 1 1 1 0 2 2 2
# 1 0 1 0 2 0 2
# 1 1 1 0 2 2 2
# ...
# 1 1 1 0 9 9 9
# 1 0 1 0 9 0 9
# 1 1 1 0 9 9 9
def recolor_a_donut_frame(input_matrix: np.ndarray, parameters: list):
    """
    For each donut (a single-color frame blob fully enclosing a hole, see
    _find_donuts), recolors just that donut's frame, once per candidate
    color 0-9, leaving any other donuts untouched. Tags note the donut's
    original/new color and whether the new color matches the most common
    color within the donut's interior.
    """
    donuts = _find_donuts(input_matrix)
    result = []
    tags_list = []
    for i, donut in enumerate(donuts):
        for new_color in range(10):
            out = input_matrix.copy()
            out[donut["border_mask"]] = new_color
            result.append(out)
            tags_list.append({
                "donut_original_color": donut["border_color"],
                "donut_new_color": new_color,
                "donut_most_common_interior_color": donut["most_common_color"],
                "recolored_to_most_common_color": bool(new_color == donut["most_common_color"]),
                "num_donuts_in_matrix": len(donuts),
                "donut_index": i,
            })
    return result, tags_list

# This function the interior of a donut, one at a time
# Ensure that tags capture if the donut is recolored to the most common
# color in the frame before recoloring, fill color matches frame color, and other donut related information
# example 1

# input
# 1 1 1 1
# 1 0 0 1
# 1 0 2 1
# 1 1 1 1

# outputs
# 1 1 1 1
# 1 0 0 1
# 1 0 0 1
# 1 1 1 1

# ...

# 1 1 1 1
# 1 9 9 1
# 1 9 9 1
# 1 1 1 1

# example 2

# input
# 1 1 1 0 2 2 2
# 1 0 1 0 2 0 2
# 1 1 1 0 2 2 2

# outputs
# 1 1 1 0 2 2 2
# 1 0 1 0 2 0 2
# 1 1 1 0 2 2 2

# 1 1 1 0 2 2 2
# 1 1 1 0 2 0 2
# 1 1 1 0 2 2 2
# ...
# 1 1 1 0 2 2 2
# 1 9 1 0 2 0 2
# 1 1 1 0 2 2 2

# 1 1 1 0 2 2 2
# 1 0 1 0 2 0 2
# 1 1 1 0 2 2 2
# ...
# 1 1 1 0 2 2 2
# 1 0 1 0 2 9 2
# 1 1 1 0 2 2 2
def recolor_a_donut_interior(input_matrix: np.ndarray, parameters: list):
    """
    For each donut (see _find_donuts), recolors just that donut's interior,
    once per candidate color 0-9, leaving the frame and any other donuts
    untouched. Tags note the donut's frame color, the interior's most common
    color before recoloring, the new fill color, whether the fill matches
    the most common interior color, and whether it matches the frame color.
    """
    donuts = _find_donuts(input_matrix)
    result = []
    tags_list = []
    for i, donut in enumerate(donuts):
        for new_color in range(10):
            out = input_matrix.copy()
            out[donut["interior_mask"]] = new_color
            result.append(out)
            tags_list.append({
                "donut_frame_color": donut["border_color"],
                "donut_new_interior_color": new_color,
                "donut_most_common_interior_color": donut["most_common_color"],
                "recolored_to_most_common_color": bool(new_color == donut["most_common_color"]),
                "fill_color_matches_frame_color": bool(new_color == donut["border_color"]),
                "num_donuts_in_matrix": len(donuts),
                "donut_index": i,
            })
    return result, tags_list


def draw_line_between_points(input_matrix: np.ndarray, parameters: list):
    """
    Draws a line between every ordered pair of points in parameters (the coords
    list) whose source and destination points share the same original color,
    using that shared color. Tags each result with
    source_meets_destination_color (always True, since mismatched pairs are
    skipped).
    Adapted from generate_draw_line in OLD_ArcAgent.py.
    """
    coords = parameters
    result = []
    tags_list = []
    for p1 in coords:
        for p2 in coords:
            if p1 == p2:
                continue
            color = int(input_matrix[p1[0], p1[1]])
            dest_color = int(input_matrix[p2[0], p2[1]])
            if color != dest_color:
                continue
            lines = _generate_draw_line([input_matrix], p1, p2, color)
            result.extend(lines)
            tags_list.extend({"source_meets_destination_color": True} for _ in lines)
    return result, tags_list


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

# stack combine takes combines two matrixes and stacks them on top of each other
# 0 is treated as the background. if the two matrixes don't stack perfectly 
# then only return the first matrix
# example 1 
# input a 
# 1 1 1 0
# 1 0 1 0
# 1 1 1 0
# input b
# 0 0 0 0
# 0 2 0 0 
# 0 0 0 0 
# outputs
# 1 1 1 0
# 1 2 1 0
# 1 1 1 0
# example 1 
# input a 
# 1 1 1 0
# 1 0 1 0
# 1 1 1 0
# input b
# 0 0 0 0
# 0 2 2 0 
# 0 0 0 0 
# outputs
# 1 1 1 0
# 1 0 1 0
# 1 1 1 0
def stack_combine(input_matrix: np.ndarray, parameters: list):
    """
    Stacks input_matrix and the matrices in parameters on top of each other
    (0 treated as transparent). For each ordered pair drawn from
    [input_matrix] + parameters, overlays the second matrix's non-zero
    pixels onto the first wherever the first is zero; if any non-zero
    pixels from both matrices land on the same cell, the pair doesn't
    "stack perfectly" and the first matrix is returned unchanged for that
    pair.
    """
    matrix_list = [input_matrix] + list(parameters)
    return _generate_stack_combos(matrix_list)


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

# this takes the input matrix and crops one row and column off of the input matrix
# for example the input
# 0 1 2
# 1 2 3
# becomes the outputs
# 0 1
# 1 2
# and
# 1 2
# 2 3
# and
# 0 1 2
# and 
# 1 2 3
def crop_one_off_each_side(input_matrix: np.ndarray, parameters: list):
    """
    Crops one row/column off a single side of input_matrix at a time (left,
    right, top, bottom), producing up to 4 outputs. A side is skipped if
    cropping it would leave an empty matrix.
    """
    rows, cols = input_matrix.shape
    result = []
    if cols > 1:
        result.append(input_matrix[:, :-1])
        result.append(input_matrix[:, 1:])
    if rows > 1:
        result.append(input_matrix[:-1, :])
        result.append(input_matrix[1:, :])
    return result


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

# this crops the matrix based on the dimenstions of the original root matrix
# the root matrix is inputted via the parameters
# example 1
# root matrix:
# 1 1
# 1 1
# input matrix
# 1 2 3 4
# 1 2 3 4
# 1 2 3 4
# 1 2 3 4
# # 
# 1 2
# 1 2

# 3 4 
# 3 4

# 1 2
# 1 2

# 1 2
# 1 2

# 3 4
# 3 4

# 1 2 3 4
# 1 2 3 4
# 1 2 3 4
# 1 2 3 4

# example 2
# # root matrix:
# 1
# input matrix
# 1 2 3
# 1 2 3
# 1 2 3
# outputs
# 1 

# 2

# 3 

# 1 

# 2

# 3 

# 1 

# 2 

# 3

# 1 2
# 1 2

# 2 3
# 2 3

# 1 2
# 1 2

# 2 3
# 2 3

# 1 2 3
# 1 2 3
# 1 2 3
def crop_to_m_n_of_input_dim(input_matrix: np.ndarray, parameters: list):
    """
    Crops input_matrix into a grid of tiles sized as multiples of the root
    matrix's shape (parameters[0]). For each k from 1 up to
    min(input_rows // root_rows, input_cols // root_cols), tiles input_matrix
    into a grid of (root_rows*k, root_cols*k)-sized pieces (skipping any k
    whose tile size doesn't evenly divide input_matrix's shape), returning
    every tile across every valid k as a separate output. The last k
    (min ratio) always yields the whole input_matrix as a single tile.
    """
    if not parameters:
        return input_matrix

    root_matrix = parameters[0]
    root_rows, root_cols = root_matrix.shape
    input_rows, input_cols = input_matrix.shape
    if root_rows <= 0 or root_cols <= 0:
        return input_matrix

    row_ratio = input_rows // root_rows
    col_ratio = input_cols // root_cols
    if row_ratio == 0 or col_ratio == 0:
        return input_matrix

    result = []
    for k in range(1, min(row_ratio, col_ratio) + 1):
        tile_h = root_rows * k
        tile_w = root_cols * k
        if input_rows % tile_h != 0 or input_cols % tile_w != 0:
            continue
        for r_start in range(0, input_rows, tile_h):
            for c_start in range(0, input_cols, tile_w):
                result.append(input_matrix[r_start:r_start + tile_h, c_start:c_start + tile_w])

    return result if result else input_matrix

# this function takes in the input matrix and makes bar graphs according to aspects
# about the matrix. 
# the graphs it must make include: num_of_each_color, num_blobs of each count_colors
# example input:
# 0 0 1 1 0 2
# 2 0 0 3 0 0
# # would output:
# 1 2 0
# 1 2 3
# and
# 0 2 0
# 1 2 3
def make_graph(input_matrix: np.ndarray, parameters: list):
    """
    Builds two bar-graph matrices from input_matrix: one showing pixel count
    per non-background color, one showing blob count per non-background
    color. Columns are sorted by color ascending; bars fill bottom-up.
    """
    pixel_counts = _color_pixel_counts(input_matrix)
    blob_counts = _color_blob_counts(input_matrix)
    return [_build_bar_graph(pixel_counts), _build_bar_graph(blob_counts)]

# This crops the input matrix to the shared dimensions of the output
# matrix for each of the examples
# if there are no shared dimensions of the output matrix then skip this transformation
# example 1
# Lets say the shared output matrix is size 2x2
# input
# 1 2 3
# 1 2 3
# 4 4 4

# outputs
# 1 2
# 1 2

# 2 3
# 2 3

# 1 2
# 4 4

# 2 3
# 4 4
def crop_to_shared_output_dimensions(input_matrix: np.ndarray, parameters: list):
    """
    Crops input_matrix into every overlapping (rows, cols)-sized window, where
    (rows, cols) = parameters[0] is the shape shared by every training
    example's output. Slides the window across all valid row/col offsets
    (stride 1), returning each crop as a separate candidate output. Returns
    input_matrix unchanged if parameters is empty (no shared output shape
    was found) or the shared shape doesn't fit inside input_matrix.
    """
    if not parameters or not parameters[0]:
        return input_matrix

    tile_h, tile_w = parameters[0]
    input_rows, input_cols = input_matrix.shape
    if tile_h <= 0 or tile_w <= 0 or tile_h > input_rows or tile_w > input_cols:
        return input_matrix

    result = []
    for r_start in range(input_rows - tile_h + 1):
        for c_start in range(input_cols - tile_w + 1):
            result.append(input_matrix[r_start:r_start + tile_h, c_start:c_start + tile_w])

    return result if result else input_matrix


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


# this outputs a matrix where all of the colors that touch are removed
# example 1
# input:
# 0 1 2
# 0 1 2
# would output:
# 0 1 0
# 0 1 0
# and
# 0 0 2
# 0 0 2

# example 2
# input:
# 0 1 2 0
# 0 1 0 0
# would output:
# 0 0 2 0
# 0 1 0 0
# and
# 0 1 0 0
# 0 1 0 0

# example 3
# input:
# 0 1 0 2
# 0 1 0 2
# would output:
# 0 1 0 2
# 0 1 0 2
def remove_touching(input_matrix: np.ndarray, parameters: list):
    """
    For each non-background (non-zero) color present in input_matrix, builds
    an output where that color's pixels touching (4-connected) a
    differently-colored non-background pixel are zeroed out; pixels of other
    colors are left unchanged. Colors with no touching pixels are skipped.
    Returns input_matrix unchanged if no color touches a different color
    anywhere in the matrix.
    """
    colors = [c for c in np.unique(input_matrix) if c != 0]
    padded = np.pad(input_matrix, 1, mode="constant", constant_values=0)
    neighbors = [
        padded[:-2, 1:-1],
        padded[2:, 1:-1],
        padded[1:-1, :-2],
        padded[1:-1, 2:],
    ]

    result = []
    for color in colors:
        color_mask = input_matrix == color
        touches_other = np.zeros_like(color_mask)
        for neighbor in neighbors:
            touches_other |= color_mask & (neighbor != 0) & (neighbor != color)
        if not np.any(touches_other):
            continue
        output = input_matrix.copy()
        output[touches_other] = 0
        result.append(output)

    return result if result else input_matrix

# this function adds the original root input matrix to the output matrix 
# it assumes that the 0 color is the background color and should be treated as transparent
# example 
# root matrix:
# 0 0 0 
# 0 1 1
# input matrix:
# 2 2 2
# 2 2 2
# output:
# 2 2 2 
# 2 1 1
def apply_original_input(input_matrix: np.ndarray, parameters: list):
    """
    Overlays the root matrix (parameters[0]) on top of input_matrix, treating
    0 in the root matrix as transparent so input_matrix shows through there.
    Returns input_matrix unchanged if no root matrix was supplied or its shape
    doesn't match input_matrix's.
    """
    if not parameters:
        return input_matrix

    root_matrix = parameters[0]
    if root_matrix.shape != input_matrix.shape:
        return input_matrix

    result = input_matrix.copy()
    mask = root_matrix != 0
    result[mask] = root_matrix[mask]
    return result


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
    "crop_to_shared_output_dimensions": crop_to_shared_output_dimensions,
    "make_graph": make_graph,
    "fill_blobs": fill_blobs,
    "recolor_donuts": recolor_donuts,
    "recolor_a_donut_frame": recolor_a_donut_frame,
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


def _count_lines_by_direction(matrix: np.ndarray) -> dict:
    """
    Counts drawable line-runs (horizontal, vertical, and both diagonal
    directions) of length >= 3 whose color differs from the majority
    color on either side of the run, broken down per direction.
    Ported from countLines in OLD_ArcAgent.py (see also _count_lines in
    ArcAgent.py, which sums this dict's values).
    """
    def majority_color(counts, total):
        if not counts:
            return None
        color, cnt = counts.most_common(1)[0]
        return color if cnt * 2 > total else None

    def count_runs(seq):
        seq = list(seq)
        n_total = len(seq)
        n, i = 0, 0
        # left_counts/right_counts are kept in sync with Counter(seq[:i]) and
        # Counter(seq[i:]) respectively, so majority_color never has to
        # rescan a slice from scratch (each seq[:i]/seq[j:] used to be
        # rebuilt into a fresh Counter on every run, making this O(n^2) per
        # line for lines with many short runs).
        left_counts = Counter()
        right_counts = Counter(seq)
        while i < n_total:
            if seq[i] != 0:
                j = i + 1
                while j < n_total and seq[j] == seq[i]:
                    j += 1
                line_color = seq[i]
                run_len = j - i
                lc = majority_color(left_counts, i)
                right_counts[line_color] -= run_len
                if right_counts[line_color] <= 0:
                    del right_counts[line_color]
                rc = majority_color(right_counts, n_total - j)
                if lc is not None and rc is not None:
                    bg_ok = lc == rc and lc != line_color
                elif lc is not None:
                    bg_ok = lc != line_color
                elif rc is not None:
                    bg_ok = rc != line_color
                else:
                    bg_ok = True
                if run_len >= 3 and bg_ok:
                    n += 1
                left_counts[line_color] += run_len
                i = j
            else:
                left_counts[0] += 1
                right_counts[0] -= 1
                if right_counts[0] <= 0:
                    del right_counts[0]
                i += 1
        return n

    matrix = np.array(matrix)
    rows, cols = matrix.shape
    horizontal = sum(count_runs(matrix[r, :]) for r in range(rows))
    vertical = sum(count_runs(matrix[:, c]) for c in range(cols))
    diagonal_tl_br = sum(count_runs(np.diag(matrix, d)) for d in range(-(rows - 1), cols))
    flipped = np.fliplr(matrix)
    diagonal_tr_bl = sum(count_runs(np.diag(flipped, d)) for d in range(-(rows - 1), cols))
    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "diagonal_tl_br": diagonal_tl_br,
        "diagonal_tr_bl": diagonal_tr_bl,
    }


def _count_pixels_touching_border(matrix: np.ndarray) -> int:
    """
    Counts non-zero cells that lie on the outer border (first/last row or
    first/last column) of matrix.
    """
    rows, cols = matrix.shape
    border_mask = np.zeros((rows, cols), dtype=bool)
    border_mask[0, :] = True
    border_mask[rows - 1, :] = True
    border_mask[:, 0] = True
    border_mask[:, cols - 1] = True
    return int(np.sum((matrix != 0) & border_mask))


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
        "num_pixels_touching_border": _count_pixels_touching_border(output_matrix),
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

    t0 = time.time()
    result = TRANSFORMATION_FUNCTIONS[transformation_name](input_matrix, parameters)
    elapsed = time.time() - t0
    prev_ms, prev_count = _TRANSFORM_TIMINGS.get(transformation_name, (0.0, 0))
    _TRANSFORM_TIMINGS[transformation_name] = (prev_ms + elapsed, prev_count + 1)

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
