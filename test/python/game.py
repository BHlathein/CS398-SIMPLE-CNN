from browser import document, ajax, html, timer
import json
from typing import Union, List, Optional, Tuple, Dict, Any
from shared import (
    BOARD_SIZE, CELL_SIZE, PIECE_PADDING,
    PLAYER_0, PLAYER_1,
    KITTEN, CAT,
    ACTION_PLACE, ACTION_REMOVE,
    GameImages, APIEndpoints,
    GameErrors, GameMessages,
    get_coordinate_notation, is_valid_position,
    deep_copy_board, get_player_piece_type,
    get_grid_position, create_piece_image
)
from menu import RulesModal

class GameBoard:
    def __init__(self):
        self.game_id: Optional[str] = None
        self.state: Dict[str, Any] = {}
        self.move_count: int = 0
        self.move_log: List[html.DIV] = []
        self.images = GameImages()
        self.api = APIEndpoints(document['server-url'].value)
        self.hover_cell = None
        self.is_paused = False
        self.pause_btn = document["pause-btn"]
        self.pause_btn.bind("click", self.toggle_pause)

    def check_three_in_a_row(self, board: List[List[Any]], player: int) -> Optional[List[Tuple[int, int]]]:
        """Check for three kittens in a row that should be promoted"""
        def is_player_kitten(piece: List[int]) -> bool:
            """Helper function to check if a piece is a kitten belonging to the player"""
            return (piece is not None and 
                    ((player == PLAYER_0 and piece[0] == 1) or 
                     (player == PLAYER_1 and piece[1] == 1)) and
                    piece[2] == KITTEN)
        
        # Check horizontal lines
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE - 2):
                if (is_player_kitten(board[row][col]) and
                    is_player_kitten(board[row][col + 1]) and
                    is_player_kitten(board[row][col + 2])):
                    return [(row, col), (row, col + 1), (row, col + 2)]

        # Check vertical lines
        for row in range(BOARD_SIZE - 2):
            for col in range(BOARD_SIZE):
                if (is_player_kitten(board[row][col]) and
                    is_player_kitten(board[row + 1][col]) and
                    is_player_kitten(board[row + 2][col])):
                    return [(row, col), (row + 1, col), (row + 2, col)]

        # Check diagonal (top-left to bottom-right)
        for row in range(BOARD_SIZE - 2):
            for col in range(BOARD_SIZE - 2):
                if (is_player_kitten(board[row][col]) and
                    is_player_kitten(board[row + 1][col + 1]) and
                    is_player_kitten(board[row + 2][col + 2])):
                    return [(row, col), (row + 1, col + 1), (row + 2, col + 2)]

        # Check diagonal (top-right to bottom-left)
        for row in range(BOARD_SIZE - 2):
            for col in range(2, BOARD_SIZE):
                if (is_player_kitten(board[row][col]) and
                    is_player_kitten(board[row + 1][col - 1]) and
                    is_player_kitten(board[row + 2][col - 2])):
                    return [(row, col), (row + 1, col - 1), (row + 2, col - 2)]

        return None

    def track_boops(self, row: int, col: int, board_before: List[List[Any]], board_after: List[List[Any]]) -> List[Dict[str, Any]]:
        """Track pieces that were booped by comparing board states"""
        booped_pieces = []
        
        # Check all adjacent positions (including diagonals)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                    
                r, c = row + dr, col + dc
                if is_valid_position(r, c):
                    # Check if there was actually a piece here before
                    piece_before = board_before[r][c]
                    if piece_before is None or (piece_before[0] == 0 and piece_before[1] == 0):
                        continue  # No piece was here before
                    
                    piece_after = board_after[r][c]
                    # Only track if the piece actually moved
                    if piece_before != piece_after:
                        new_r, new_c = r + dr, c + dc
                        player, is_cat = get_player_piece_type(piece_before)
                        
                        # Check if piece was booped off the board
                        if not is_valid_position(new_r, new_c):
                            booped_pieces.append({
                                'from_pos': (r, c),
                                'to_pos': None,
                                'piece_type': 'Cat' if is_cat else 'Kitten',
                                'player': player
                            })
                        else:
                            # Check if piece moved to the expected position
                            piece_at_new_pos = board_after[new_r][new_c]
                            if (piece_at_new_pos is not None and 
                                ((piece_before[0] == 1 and piece_at_new_pos[0] == 1) or 
                                 (piece_before[1] == 1 and piece_at_new_pos[1] == 1))):
                                booped_pieces.append({
                                    'from_pos': (r, c),
                                    'to_pos': (new_r, new_c),
                                    'piece_type': 'Cat' if is_cat else 'Kitten',
                                    'player': player
                                })
        
        return booped_pieces

    def track_promotions(self, board_before: List[List[Any]], board_after: List[List[Any]], current_player: int) -> Optional[List[Tuple[int, int]]]:
        """Track kitten promotions"""
        promotions = self.check_three_in_a_row(board_after, current_player)
        
        if promotions:
            # Verify these positions changed from kittens to cats or empty
            all_promoted = True
            for row, col in promotions:
                piece_before = board_before[row][col]
                piece_after = board_after[row][col]
                
                # Check if the piece was a kitten before and is now either gone or a cat
                if not (piece_before is not None and piece_before[2] == KITTEN and  # was a kitten
                       (piece_after is None or piece_after[2] == CAT)):  # is now gone or a cat
                    all_promoted = False
                    break
            
            if all_promoted:
                return promotions
        
        return None

    def add_move_to_log(self, player: int, action_type: int, row: int, col: int, 
                        piece_type: int, board_before: List[List[Any]], board_after: List[List[Any]]) -> None:
        """Add a move to the move log with proper formatting"""
        self.move_count += 1
        
        # Create move entry
        move_entry = html.DIV(Class="move-log-entry")
        
        # Add move number
        move_number = html.SPAN(f"{self.move_count}.", Class="move-number")
        move_entry <= move_number
        
        # Basic move description
        piece_name = "Cat" if piece_type == CAT else "Kitten"
        if action_type == ACTION_PLACE:
            action_desc = f"Player {player} places {piece_name} at {get_coordinate_notation(row, col)}"
        else:
            action_desc = f"Player {player} removes {piece_name} from {get_coordinate_notation(row, col)}"
        
        move_desc = html.SPAN(action_desc, Class=f"move-description player-{player}-move")
        move_entry <= move_desc
        
        effects_container = html.DIV(Class="move-effects")
        effects_added = False
        
        # Track and add boop effects
        if action_type == ACTION_PLACE:
            booped_pieces = self.track_boops(row, col, board_before, board_after)
            if booped_pieces:
                effects_added = True
                for piece in booped_pieces:
                    from_coord = get_coordinate_notation(*piece['from_pos'])
                    effect_text = f"→ {piece['piece_type']} at {from_coord} "
                    if piece['to_pos'] is None:
                        effect_text += "was booped off the board"
                    else:
                        to_coord = get_coordinate_notation(*piece['to_pos'])
                        effect_text += f"was booped to {to_coord}"
                    
                    effect = html.DIV(
                        effect_text, 
                        Class=f"special-move boop player-{piece['player']}-boop"
                    )
                    effects_container <= effect
        
        # Track and add promotion effects
        promotions = self.track_promotions(board_before, board_after, player)
        if promotions:
            effects_added = True
            promotion_coords = [get_coordinate_notation(r, c) for r, c in promotions]
            promotion_text = f"→ Kittens at {', '.join(promotion_coords)} promoted to Cat"
            promotion_effect = html.DIV(
                promotion_text,
                Class=f"special-move promotion player-{player}-promotion"
            )
            effects_container <= promotion_effect
        
        # Only add effects container if there were effects
        if effects_added:
            move_entry <= effects_container
        
        # Insert at the beginning of the move log
        self.move_log.insert(0, move_entry)
        self.update_move_log_display()

    def update_move_log_display(self) -> None:
        """Update the move log display"""
        move_log_div = document["move-log"]
        move_log_div.clear()
        # No need to reverse the list since we're already inserting at the beginning
        for entry in self.move_log:
            move_log_div <= entry

    def clear_move_log(self) -> None:
        """Clear the move log"""
        self.move_log = []
        self.move_count = 0
        self.update_move_log_display()

    def update_inventory_display(self) -> None:
        """Update the inventory display with current stock numbers"""
        if 'stock' in self.state:
            for player in [0, 1]:
                stock = self.state['stock'][str(player)]
                # Update kitten count
                kitten_span = document[f"p{player}-kittens"]
                kitten_span.text = str(stock['kitten'])
                # Update cat count
                cat_span = document[f"p{player}-cats"]
                cat_span.text = str(stock['cat'])

    def render_board(self) -> None:
        """Render the game board"""
        canvas = document["game-canvas"]
        ctx = canvas.getContext("2d")
        
        # Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        
        # Draw grid
        ctx.strokeStyle = "#000"
        ctx.lineWidth = 1
        for i in range(BOARD_SIZE + 1):
            # Vertical lines
            ctx.beginPath()
            ctx.moveTo(i * CELL_SIZE, 0)
            ctx.lineTo(i * CELL_SIZE, BOARD_SIZE * CELL_SIZE)
            ctx.stroke()
            
            # Horizontal lines
            ctx.beginPath()
            ctx.moveTo(0, i * CELL_SIZE)
            ctx.lineTo(BOARD_SIZE * CELL_SIZE, i * CELL_SIZE)
            ctx.stroke()
        
        # Draw hover effect
        if self.hover_cell is not None:
            row, col = self.hover_cell
            ctx.fillStyle = "rgba(200, 200, 200, 0.5)"
            ctx.fillRect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        
        # Draw pieces
        if 'board' in self.state:
            board = self.state['board']
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    piece = board[row][col]
                    if piece is not None and (piece[0] == 1 or piece[1] == 1):
                        # Determine which player's piece it is
                        player = PLAYER_0 if piece[0] == 1 else PLAYER_1
                        # Determine if it's a cat or kitten
                        is_cat = piece[2] == CAT
                        
                        # Get the correct image
                        img_key = f"{'cat' if is_cat else 'kitten'}_{'a' if player == PLAYER_0 else 'b'}"
                        img = self.images.images[img_key]
                        
                        # Draw the image
                        x = col * CELL_SIZE + PIECE_PADDING
                        y = row * CELL_SIZE + PIECE_PADDING
                        size = CELL_SIZE - 2 * PIECE_PADDING
                        ctx.drawImage(img, x, y, size, size)
        
        # Update inventory display
        self.update_inventory_display()

    def handle_start_response(self, req: ajax.Ajax) -> None:
        """Handle the response from starting a new game"""
        try:
            response = json.loads(req.text)
            self.game_id = response["game_id"]
            self.state = response["state"]
            
            # Update status with the first player's turn
            current_player = self.state["current_player"]
            if (current_player < len(self.state.get('players', [])) and
                self.state['players'][current_player] == "ai"):
                self.update_turn_status(GameErrors.AI_THINKING)
                timer.set_timeout(self.make_ai_move, 1000)  # Start AI move if AI goes first
            else:
                self.update_turn_status(f"Player {current_player}'s turn")
            
            self.clear_move_log()
            self.render_board()
        except Exception as e:
            self.update_turn_status(GameErrors.START_ERROR.format(str(e)))

    def handle_move_response(self, req: ajax.Ajax) -> None:
        """Handle the response from making a move"""
        try:
            response = json.loads(req.text)
            self.state = response["state"]
            self.render_board()
            
            if response.get("game_over", False):
                self.game_id = None
        except Exception as e:
            self.update_turn_status(GameErrors.MOVE_ERROR.format(str(e)))

    def toggle_pause(self, ev: Any) -> None:
        """Toggle the pause state for AI moves"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.text = "Resume AI"
            self.pause_btn.classList.add("paused")
        else:
            self.pause_btn.text = "Pause AI"
            self.pause_btn.classList.remove("paused")
            # If it's currently an AI's turn, resume play
            current_player = self.state.get('current_player')
            if (current_player is not None and 
                current_player < len(self.state.get('players', [])) and
                self.state['players'][current_player] == "ai"):
                timer.set_timeout(self.make_ai_move, 1000)

    def start_game(self, ev: Any) -> None:
        """Start a new game"""
        p0_type = document["player-0-type"].value
        p1_type = document["player-1-type"].value
        
        # Show/hide pause button based on game type
        if p0_type == "ai" and p1_type == "ai":
            self.pause_btn.style.display = "block"
            self.pause_btn.text = "Pause AI"
            self.pause_btn.classList.remove("paused")
            self.is_paused = False
        else:
            self.pause_btn.style.display = "none"
        
        req = ajax.Ajax()
        req.bind('complete', self.handle_start_response)
        req.open('POST', self.api.new_game, True)
        req.set_header('content-type', 'application/json')
        req.send(json.dumps({"players": [p0_type, p1_type]}))

    def send_move(self, ev: Any) -> None:
        """Send a move to the server"""
        if not self.game_id:
            self.update_turn_status(GameErrors.GAME_NOT_STARTED)
            return
            
        # Get click coordinates relative to canvas
        canvas = document["game-canvas"]
        rect = canvas.getBoundingClientRect()
        
        # Get the position relative to the canvas
        x = ev.clientX - rect.left
        y = ev.clientY - rect.top
        
        # Convert to grid position
        row, col = get_grid_position(x, y)
        
        if not is_valid_position(row, col):
            self.update_turn_status(GameErrors.INVALID_MOVE)
            return
            
        # Get the selected piece type
        piece_type = int(document["piece-type"].value)
        
        # Store board state before move
        board_before = deep_copy_board(self.state['board'])
        
        # Send move to server
        req = ajax.Ajax()
        req.bind('complete', lambda req: self.handle_move_complete(req, ACTION_PLACE, row, col, piece_type, board_before))
        req.open('POST', self.api.make_move, True)
        req.set_header('content-type', 'application/json')
        req.send(json.dumps({
            "game_id": self.game_id,
            "action": [ACTION_PLACE, row, col, piece_type]
        }))
        
        # Don't update status here - let handle_move_complete handle it

    def handle_move_complete(self, req: ajax.Ajax, action_type: int, row: int, col: int, 
                           piece_type: int, board_before: List[List[Any]]) -> None:
        """Handle completion of a move, including updating the move log"""
        try:
            response = json.loads(req.text)
            
            if not response.get("game_over", False):
                # Add move to log
                self.add_move_to_log(
                    self.state['current_player'],
                    action_type,
                    row,
                    col,
                    piece_type,
                    board_before,
                    response['state']['board']
                )
            
            # Update game state
            self.state = response["state"]
            self.render_board()
            
            # Update turn status based on current player type
            if response.get("game_over", False):
                self.game_id = None
                self.update_turn_status(response.get("status", "Game Over!"))
                self.pause_btn.style.display = "none"
            else:
                current_player = self.state['current_player']
                if (current_player < len(self.state.get('players', [])) and
                    self.state['players'][current_player] == "ai"):
                    # If it's AI's turn, trigger move without showing thinking message
                    self.update_turn_status(f"Player {current_player}'s turn")
                    timer.set_timeout(self.make_ai_move, 1000)  # 1 second delay for visual effect
                else:
                    # If it's a human player's turn, show whose turn it is
                    self.update_turn_status(f"Player {current_player}'s turn")
            
        except Exception as e:
            self.update_turn_status(GameErrors.MOVE_ERROR.format(str(e)))

    def make_ai_move(self) -> None:
        """Make an AI move"""
        if self.is_paused:
            return
            
        req = ajax.Ajax()
        req.bind('complete', lambda req: self.handle_ai_move(req))
        req.open('POST', self.api.make_move, True)
        req.set_header('content-type', 'application/json')
        req.send(json.dumps({
            "game_id": self.game_id,
            "action": None  # None action triggers AI move
        }))

    def handle_ai_move(self, req: ajax.Ajax) -> None:
        """Handle AI move response"""
        try:
            response = json.loads(req.text)
            
            # Log the AI move if move details are provided
            if "ai_move" in response:
                move = response["ai_move"]
                self.add_move_to_log(
                    self.state['current_player'],  # Current player before state update
                    move["action_type"],
                    move["row"],
                    move["col"],
                    move["piece_type"],
                    move["board_before"],
                    response["state"]["board"]
                )
            
            # Update game state
            self.state = response["state"]
            self.render_board()
            
            if response.get("game_over", False):
                self.game_id = None
                self.update_turn_status(response.get("status", "Game Over!"))
                self.pause_btn.style.display = "none"
            else:
                current_player = self.state['current_player']
                if (not self.is_paused and 
                    current_player < len(self.state.get('players', [])) and
                    self.state['players'][current_player] == "ai"):
                    # If it's AI's turn and not paused, trigger the next AI move
                    self.update_turn_status(f"Player {current_player}'s turn")
                    timer.set_timeout(self.make_ai_move, 1000)
                else:
                    self.update_turn_status(f"Player {current_player}'s turn")
            
        except Exception as e:
            self.update_turn_status(GameErrors.MOVE_ERROR.format(str(e)))

    def handle_mouse_move(self, ev: Any) -> None:
        """Handle mouse movement over the canvas"""
        if not self.game_id:
            return
            
        canvas = document["game-canvas"]
        rect = canvas.getBoundingClientRect()
        
        # Get the position relative to the canvas
        x = ev.clientX - rect.left
        y = ev.clientY - rect.top
        
        # Convert to grid position
        row, col = get_grid_position(x, y)
        
        # Update hover state and redraw
        if is_valid_position(row, col):
            coord = get_coordinate_notation(row, col)
            document["position-status"].text = f"Position: {coord}"
            if self.hover_cell != (row, col):
                self.hover_cell = (row, col)
                self.render_board()
        else:
            document["position-status"].text = "Position: --"
            if self.hover_cell is not None:
                self.hover_cell = None
                self.render_board()

    def update_turn_status(self, message: str) -> None:
        """Update the turn status display"""
        document["turn-status"].text = message

def init_game() -> None:
    """Initialize the game"""
    game = GameBoard()
    
    # Initialize rules modal
    rules = RulesModal()
    
    # Set up coordinate labels
    horizontal_labels = document.select_one(".coordinate-labels.horizontal")
    vertical_labels = document.select_one(".coordinate-labels.vertical")
    
    # Clear existing labels
    horizontal_labels.clear()
    vertical_labels.clear()
    
    # Add horizontal labels (A-F)
    for i in range(BOARD_SIZE):
        label = html.DIV(chr(65 + i), Class="coordinate-label")
        label.style.gridColumn = str(i + 2)
        horizontal_labels <= label
    
    # Add vertical labels (1-6)
    for i in range(BOARD_SIZE):
        label = html.DIV(str(i + 1), Class="coordinate-label")
        label.style.gridRow = str(i + 2)
        vertical_labels <= label
    
    # Bind event handlers
    document["game-canvas"].bind("click", game.send_move)
    document["game-canvas"].bind("mousemove", game.handle_mouse_move)
    
    # Start game automatically if player types are set
    game.start_game(None)

# Make init_game available to the window object
from browser import window
window.init_game = init_game
