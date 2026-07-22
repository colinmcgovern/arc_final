import numpy as np


def PerformMea(initial_matrix, current_matrix, proposed_matrix, goal_matrix, mea_type):

    passesMea = False
    
    # proposed matrix must be a subset of the goal_matrix
    # When searching the stride is the size of the initial_matrix
    # example 1 
    # initial_matrix:
    # 0 1
    # 0 1
    # # proposed_matrix:
    # 0 1
    # 0 1
    # # goal_matrix
    # 0 1 0 1
    # 0 1 0 1
    # # output: True
    # #
    # # example 2 
    # initial_matrix:
    # 0 1
    # 0 1
    # # proposed_matrix:
    # 0 1
    # 0 1
    # # goal_matrix
    # 2 2 2 2
    # 2 2 2 2
    # # output: False
    # #
    # # example 3 
    # initial_matrix:
    # 0 1
    # 0 1
    # # proposed_matrix:
    # 1 1
    # 1 1
    # # goal_matrix
    # 0 1 1 0
    # 0 1 1 0
    # # output: False
    # #
    # # example 4 
    # initial_matrix:
    # 1
    # 1
    # # proposed_matrix:
    # 1 1
    # 1 1
    # # goal_matrix
    # 0 1 1 0
    # 0 1 1 0
    # # output: True
    # #
    # # example 5 
    # initial_matrix:
    # 1 1
    # 1 1
    # # proposed_matrix:
    # 1 1 1 1
    # 1 1 1 1
    # # goal_matrix
    # 0 0 0 0
    # 0 0 0 0
    # 1 1 1 1
    # 1 1 1 1
    # 0 0 0 0
    # 0 0 0 0
    # # output: True
    # #
    # # example 6 
    # initial_matrix:
    # 1 1 1 1 
    # 1 1 1 1 
    # # proposed_matrix:
    # 1 1 1 1
    # 1 1 1 1
    # # goal_matrix
    # 1 1
    # 1 1
    # # output: False
    # #
    # # example 6 
    # initial_matrix:
    # 1
    # 1 
    # # proposed_matrix:
    # 2 2
    # 2 2
    # # goal_matrix
    # 1 2
    # 1 2
    # output: False
    if(mea_type == "in_progress_always_subset_of_goal"):
        unit_rows, unit_cols = initial_matrix.shape
        goal_rows, goal_cols = goal_matrix.shape
        proposed_rows, proposed_cols = proposed_matrix.shape

        for row_offset in range(0, goal_rows, unit_rows):
            for col_offset in range(0, goal_cols, unit_cols):
                if row_offset + proposed_rows > goal_rows or col_offset + proposed_cols > goal_cols:
                    continue

                goal_overlap = goal_matrix[row_offset:row_offset + proposed_rows,
                                           col_offset:col_offset + proposed_cols]

                if np.array_equal(proposed_matrix, goal_overlap):
                    passesMea = True
                    break
            if passesMea:
                break

    # This works the same as in_progress_always_subset_of_goal except
    # the sliding window can go out of bounds, so long as the matrix size in the initial matrix
    # is being checked in the proposed matrix vs the goal matrix. assume all out of bounds pixels are correct 
    # # example 1 
    # initial_matrix:
    # 1
    # 1 
    # # proposed_matrix:
    # 2 2
    # 2 2
    # # goal_matrix
    # 1 2
    # 1 2
    # output: True
    # # example 2
    # initial_matrix:
    # 1 1
    # 1 1
    # # proposed_matrix:
    # 2 2
    # 2 2
    # # goal_matrix
    # 1 2
    # 1 2
    # output: False
    # # example 3 
    # initial_matrix:
    # 1 1
    # 1 1
    # # proposed_matrix:
    # 2 2 2
    # 2 2 2
    # # goal_matrix
    # 1 1 1
    # 1 2 2
    # 1 2 2
    # output: True
    if(mea_type == "in_progress_always_subset_of_goal_with_oob"):
        unit_rows, unit_cols = initial_matrix.shape
        goal_rows, goal_cols = goal_matrix.shape
        proposed_rows, proposed_cols = proposed_matrix.shape

        for row_offset in range(-(proposed_rows - 1), goal_rows):
            for col_offset in range(-(proposed_cols - 1), goal_cols):
                row_start_in = max(row_offset, 0)
                row_end_in = min(row_offset + proposed_rows, goal_rows)
                col_start_in = max(col_offset, 0)
                col_end_in = min(col_offset + proposed_cols, goal_cols)

                in_rows = row_end_in - row_start_in
                in_cols = col_end_in - col_start_in

                # require at least a full unit-sized block in bounds to compare
                if in_rows < unit_rows or in_cols < unit_cols:
                    continue

                goal_overlap = goal_matrix[row_start_in:row_end_in, col_start_in:col_end_in]
                proposed_overlap = proposed_matrix[row_start_in - row_offset:row_end_in - row_offset,
                                                    col_start_in - col_offset:col_end_in - col_offset]

                if np.array_equal(goal_overlap, proposed_overlap):
                    passesMea = True
                    break
            if passesMea:
                break

    if(mea_type == "in_progress_can_only_add_non_background"):
        passesMea = True

    if(mea_type == "always_gets_closer_to_goal"):
        passesMea = True

    return passesMea