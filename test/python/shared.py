from browser import window, html

# Board Constants
BOARD_SIZE = 6
CELL_SIZE = 60
PIECE_PADDING = 5  # The padding used when drawing pieces (x + 5, y + 5 in the render code)

# Player Constants
PLAYER_0 = 0
PLAYER_1 = 1

# Piece Types
KITTEN = 0
CAT = 1

# Action Types
ACTION_PLACE = 0
ACTION_REMOVE = 1

# Game States
STATE_NOT_STARTED = "not_started"
STATE_IN_PROGRESS = "in_progress"
STATE_GAME_OVER = "game_over"

# Image Assets
class GameImages:
    def __init__(self):
        self.images = {
            'kitten_a': window.Image.new(),
            'cat_a': window.Image.new(),
            'kitten_b': window.Image.new(),
            'cat_b': window.Image.new()
        }
        # Set image sources
        self.images['kitten_a'].src = "../images/kitten_a.png"
        self.images['cat_a'].src = "../images/cat_a.png"
        self.images['kitten_b'].src = "../images/kitten_b.png"
        self.images['cat_b'].src = "../images/cat_b.png"

# Utility Functions
def get_coordinate_notation(row: int, col: int) -> str:
    """Convert zero-based row, col to coordinate notation (e.g., 0,0 -> A1)"""
    return f"{chr(65 + col)}{row + 1}"

def get_grid_position(x: float, y: float) -> tuple[int, int]:
    """Convert pixel coordinates to grid position"""
    row = int(y // CELL_SIZE)
    col = int(x // CELL_SIZE)
    return row, col

def is_valid_position(row: int, col: int) -> bool:
    """Check if a grid position is within bounds"""
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

def create_piece_image(player: int, is_cat: bool, size: int = 24) -> html.IMG:
    """Create an HTML image element for a piece
    
    Args:
        player: Player number (0 or 1)
        is_cat: Whether the piece is a cat (True) or kitten (False)
        size: Size of the image in pixels
    
    Returns:
        html.IMG: The created image element
    """
    piece_type = "cat" if is_cat else "kitten"
    player_letter = "a" if player == 0 else "b"
    return html.IMG(src=f"{piece_type}_{player_letter}.png", width=size, height=size)

def deep_copy_board(board: list) -> list:
    """Create a deep copy of the game board
    
    Args:
        board: The game board to copy
    
    Returns:
        list: A deep copy of the board
    """
    return [row[:] for row in board]

def get_player_piece_type(piece: list) -> tuple[int, bool]:
    """Get the player number and piece type from a piece array
    
    Args:
        piece: The piece array [p0, p1, is_cat]
    
    Returns:
        tuple: (player_number, is_cat)
    """
    if piece[0] == 1:
        return PLAYER_0, piece[2] == 1
    elif piece[1] == 1:
        return PLAYER_1, piece[2] == 1
    return None, None

# API Endpoints
class APIEndpoints:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        
    @property
    def new_game(self) -> str:
        """URL for creating a new game"""
        return f"{self.base_url}/api/games/boop/new"
    
    @property
    def make_move(self) -> str:
        """URL for making a move"""
        return f"{self.base_url}/api/games/boop/move"

# Error Messages
class GameErrors:
    GAME_NOT_STARTED = "Game not started"
    INVALID_MOVE = "Invalid move"
    AI_THINKING = "AI is thinking..."
    START_ERROR = "Failed to start game: {}"
    MOVE_ERROR = "Move error: {}"
    NETWORK_ERROR = "Network error: {}"

# Success Messages
class GameMessages:
    GAME_STARTED = "Game started successfully"
    MOVE_MADE = "Move completed successfully"
    WAITING_FOR_OPPONENT = "Waiting for opponent..."
    GAME_OVER = "Game Over! {} wins!"

# Type Hints (simplified for Brython compatibility)
from typing import Dict, List, Optional, Union

# Instead of TypedDict, use simple type comments or basic annotations
GameStock = Dict[str, int]  # {'kitten': int, 'cat': int}

GameState = Dict[str, Union[List[List[List[int]]], int, Dict[str, GameStock], str]]

GameAction = Dict[str, Union[int, int, int, int]]  # {action_type: int, row: int, col: int, piece_type: int}

ServerResponse = Dict[str, Union[str, 'GameState', str, Optional[bool]]]
