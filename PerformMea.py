import numpy as np


def PerformMea(initial_matrix, current_matrix, proposed_matrix, goal_matrix, mea_type):

    passesMea = False
    
    # proposed matrix must be a subset of the goal_matrix
    # When searching the stride is the size of the initial_matrix
    # If the proposed_matrix is larger than the goal_matrix, then assume the 
    # tiles that are outside the goal_matrix area is correct
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
    # # output: True
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
    # output: True
    if(mea_type == "in_progress_always_subset_of_goal"):
        unit_rows, unit_cols = initial_matrix.shape
        goal_rows, goal_cols = goal_matrix.shape
        proposed_rows, proposed_cols = proposed_matrix.shape

        for row_offset in range(0, goal_rows, unit_rows):
            for col_offset in range(0, goal_cols, unit_cols):
                overlap_rows = min(proposed_rows, goal_rows - row_offset)
                overlap_cols = min(proposed_cols, goal_cols - col_offset)

                proposed_overlap = proposed_matrix[:overlap_rows, :overlap_cols]
                goal_overlap = goal_matrix[row_offset:row_offset + overlap_rows,
                                           col_offset:col_offset + overlap_cols]

                if np.array_equal(proposed_overlap, goal_overlap):
                    passesMea = True
                    break
            if passesMea:
                break

    if(mea_type == "in_progress_can_only_add_non_background"):
        passesMea = False

    if(mea_type == "always_gets_closer_to_goal"):
        passesMea = False

    return passesMea