from phases.phase_05_solver.solver import solve_board


def test_solver_solves_known_board():
    board = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]
    success, solved, message = solve_board(board)
    assert success, message
    assert solved[0] == [5, 3, 4, 6, 7, 8, 9, 1, 2]


def test_solver_rejects_duplicate_row():
    board = [[0 for _ in range(9)] for _ in range(9)]
    board[0][0] = 5
    board[0][1] = 5
    success, solved, message = solve_board(board)
    assert not success
    assert solved is None
    assert "Duplicate" in message
