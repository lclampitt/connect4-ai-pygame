import numpy as np
import random
import math
import time

# --- CONSTANTS ---
ROW_COUNT = 6
COLUMN_COUNT = 7
PLAYER_PIECE = 1  # The Opponent (Random/Greedy/Positional)
AI_PIECE = 2      # Your Main AI (Depth 4)
EMPTY = 0
WINDOW_LENGTH = 4

# --- CORE GAME LOGIC (Matches your main game) ---
def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT))

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[ROW_COUNT - 1][col] == 0

def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r

def winning_move(board, piece):
    # Check horizontal
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True
    # Check vertical
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True
    # Check positive diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True
    # Check negative diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True
    return False

def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0

# --- YOUR AI LOGIC (Minimax + Heuristics) ---
def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE:
        opp_piece = AI_PIECE

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4 

    return score

def score_position(board, piece):
    score = 0
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r,:])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:,c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score

def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return (None, 100000000000000)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -10000000000000)
            else: # Game is over
                return (None, 0)
        else: # Depth is zero
            return (None, score_position(board, AI_PIECE))

    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value
    else: 
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value

# --- 3 OPPONENT AGENTS ---

def random_agent(board):
    """Level 1: Moves completely randomly."""
    valid_locations = get_valid_locations(board)
    return random.choice(valid_locations)

def greedy_agent(board):
    """Level 2: Blocks losses and takes wins. Random otherwise."""
    valid_locations = get_valid_locations(board)
    
    # Check for immediate win
    for col in valid_locations:
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        drop_piece(temp_board, row, col, PLAYER_PIECE)
        if winning_move(temp_board, PLAYER_PIECE):
            return col

    # Check for immediate block
    for col in valid_locations:
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        drop_piece(temp_board, row, col, AI_PIECE)
        if winning_move(temp_board, AI_PIECE):
            return col
            
    return random.choice(valid_locations)

def positional_agent(board):
    """
    Level 3 (Upgraded): Uses Minimax with Depth 2.
    It looks at its move AND the opponent's best response.
    It tries to minimize the score (since it plays as PLAYER_PIECE).
    """
    # Call minimax with depth=2
    # maximizingPlayer=False because this agent IS the Player (Piece 1)
    col, score = minimax(board, 2, -math.inf, math.inf, False)
    return col

# --- TOURNAMENT LOOP ---
def run_tournament(opponent_func, opponent_name, num_games):
    ai_wins = 0
    opp_wins = 0
    draws = 0
    
    print(f"\nStarting tournament against {opponent_name} ({num_games} games)...")
    
    start_time = time.time()
    
    for game in range(num_games):
        board = create_board()
        game_over = False
        turn = random.randint(0, 1) 
        
        while not game_over:
            if turn == 0: # Opponent Turn (Player Piece)
                col = opponent_func(board)
                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, PLAYER_PIECE)
                    if winning_move(board, PLAYER_PIECE):
                        opp_wins += 1
                        game_over = True
                    turn = 1
                    
            else: # Your AI Turn (AI Piece)
                # Depth 4 is standard for Main AI
                col, score = minimax(board, 4, -math.inf, math.inf, True)
                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, AI_PIECE)
                    if winning_move(board, AI_PIECE):
                        ai_wins += 1
                        game_over = True
                    turn = 0
            
            if not game_over and len(get_valid_locations(board)) == 0:
                draws += 1
                game_over = True

    total_time = time.time() - start_time
    print(f"--- Results vs {opponent_name} ---")
    print(f"Your AI Wins: {ai_wins}")
    print(f"Opponent Wins: {opp_wins}")
    print(f"Draws: {draws}")
    print(f"Win Rate: {((ai_wins / num_games) * 100):.1f}%")
    print(f"Total Time: {total_time:.2f}s")
    print("-" * 30)

if __name__ == "__main__":
    # 1. Random (Easy)
    run_tournament(random_agent, "Random Agent", num_games=10000)
    
    # 2. Greedy (Medium)
    run_tournament(greedy_agent, "Greedy Agent", num_games=10000)

    # 3. Positional (Hard - Depth 2 AI)
    run_tournament(positional_agent, "Positional Agent (Depth 2)", num_games=10000)