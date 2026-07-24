import numpy as np
from collections import Counter

from ArcProblem import ArcProblem


def _background_color(matrix):
    return Counter(int(v) for v in np.asarray(matrix).flatten()).most_common(1)[0][0]


# this finds if the matrix has dotted lines in it
# lines can be horizontal, vertical, or 45 degrees
# the dotted line has to be only one color
# example 1

# input
# 0 0
# 0 0

# output
# False

# example 2 

# input 
# 1 0 0 0 0 1
# 1 2 2 2 2 1
# 1 0 0 0 0 1

# output
# false 

# example 3

# input
# 1 0 0 0 0
# 0 0 0 0 0
# 0 0 1 0 0
# 0 0 0 0 0
# 0 0 0 0 1

# output
# True

# example 4

# input 
# 1 0 0 0 0 0 0 0 1
# 1 2 0 2 0 2 0 2 1
# 1 0 0 0 0 0 0 0 1

# output
# True

def hasDottedLine(matrix):
    matrix = np.asarray(matrix)
    rows, cols = matrix.shape
    bg = _background_color(matrix)
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    coords_by_color = {}
    for r in range(rows):
        for c in range(cols):
            color = int(matrix[r, c])
            if color != bg:
                coords_by_color.setdefault(color, set()).add((r, c))

    for coords in coords_by_color.values():
        for start in coords:
            for dr, dc in directions:
                max_step = max(rows, cols)
                for gap in range(1, max_step):
                    step = gap + 1
                    count = 1
                    r, c = start
                    while True:
                        nr, nc = r + dr * step, c + dc * step
                        if not (0 <= nr < rows and 0 <= nc < cols):
                            break
                        if (nr, nc) not in coords:
                            break
                        between_clear = True
                        for g in range(1, step):
                            br, bc = r + dr * g, c + dc * g
                            if int(matrix[br, bc]) != bg:
                                between_clear = False
                                break
                        if not between_clear:
                            break
                        count += 1
                        r, c = nr, nc
                    if count >= 3:
                        return True
    return False


def findIfAllInputsHaveDottedLine(arc_problem: ArcProblem) -> bool:
    """
    Returns True if every training input contains a dotted line.
    """
    return all(
        hasDottedLine(example.get_input_data().data())
        for example in arc_problem.training_set()
    )


# this finds if the input matrix has a curved line in it
# a curved line is a line of one color that contains both a vertical or horizontal
# and a 45 degree line

# example 1

# input
# 0 0
# 0 0

# output
# False

# example 2

# input
# 0 0 0 0
# 1 1 1 1
# 0 0 0 0 

# output
# False

# example 3

# input 

# 1 0 0 0 0 0
# 0 1 0 0 0 0
# 0 0 1 0 0 0 
# 0 0 0 1 1 1

# output
# True

# example 4

# input 
# 1 1 1 1 1 1
# 0 2 0 0 0 0 
# 0 0 2 0 0 0
# 0 0 0 2 0 0
# 0 5 0 0 2 0
# 0 0 0 0 2 0
# 5 0 0 0 2 0
# 0 0 0 0 2 0
# 1 1 1 1 1 1

# output
# True

def hasCurvedLine(matrix):
    matrix = np.asarray(matrix)
    rows, cols = matrix.shape
    bg = _background_color(matrix)
    orthogonal = {(0, 1), (0, -1), (1, 0), (-1, 0)}
    diagonal = {(1, 1), (1, -1), (-1, 1), (-1, -1)}

    coords_by_color = {}
    for r in range(rows):
        for c in range(cols):
            color = int(matrix[r, c])
            if color != bg:
                coords_by_color.setdefault(color, set()).add((r, c))

    for coords in coords_by_color.values():
        visited = set()
        for start in coords:
            if start in visited:
                continue
            component = set()
            stack = [start]
            visited.add(start)
            while stack:
                r, c = stack.pop()
                component.add((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        neighbor = (r + dr, c + dc)
                        if neighbor in coords and neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)

            if len(component) < 3:
                continue

            has_orthogonal = False
            has_diagonal = False
            is_line_shaped = True
            for r, c in component:
                degree = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        neighbor = (r + dr, c + dc)
                        if neighbor in component:
                            degree += 1
                            if (dr, dc) in orthogonal:
                                has_orthogonal = True
                            elif (dr, dc) in diagonal:
                                has_diagonal = True
                if degree > 2:
                    is_line_shaped = False
                    break

            if is_line_shaped and has_orthogonal and has_diagonal:
                return True
    return False


def findIfAllInputsHaveCurvedLine(arc_problem: ArcProblem) -> bool:
    """
    Returns True if every training input contains a curved line.
    """
    return all(
        hasCurvedLine(example.get_input_data().data())
        for example in arc_problem.training_set()
    )


# this finds if the input matrix has a double color line

# example 1

# input
# 0 0
# 0 0

# output
# False

# example 2

# input
# 0 0 0 0
# 1 1 1 1
# 0 0 0 0 

# output
# False

# example 3

# input
# 0 0 0 0 0 0
# 1 1 2 0 2 1
# 0 0 0 0 0 0 

# output
# true

# example 4

# input 
# 0 0 0 0 9
# 0 0 0 8 0
# 0 0 8 0 0
# 0 9 0 0 0
# 1 1 1 1 1

# output
# true

def _has_double_color_run(seq, bg):
    run = []
    for value in list(seq) + [bg]:
        value = int(value)
        if value != bg:
            run.append(value)
            continue
        if len(run) >= 3 and len(set(run)) >= 2:
            return True
        run = []
    return False


def hasDoubleColorLine(matrix):
    matrix = np.asarray(matrix)
    rows, cols = matrix.shape
    bg = _background_color(matrix)

    for r in range(rows):
        if _has_double_color_run(matrix[r, :], bg):
            return True
    for c in range(cols):
        if _has_double_color_run(matrix[:, c], bg):
            return True

    flipped = np.fliplr(matrix)
    for d in range(-(rows - 1), cols):
        if _has_double_color_run(np.diag(matrix, d), bg):
            return True
        if _has_double_color_run(np.diag(flipped, d), bg):
            return True
    return False


def findIfAllInputsHaveDoubleColorLine(arc_problem: ArcProblem) -> bool:
    """
    Returns True if every training input contains a double-color line.
    """
    return all(
        hasDoubleColorLine(example.get_input_data().data())
        for example in arc_problem.training_set()
    )