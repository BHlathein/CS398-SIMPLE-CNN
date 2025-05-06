# TODO: change to CNN policy! If you want to later use a CNN policy, you'll need to change the observation format to channel-first (5, 6, 6) and adjust the policy accordingly.

import gymnasium as gym
import numpy as np
import random

class Player:
    def __init__(self, id, token):
        self.id = id
        self.token = token
        self.stock = {'kitten': 8, 'cat': 0}
        self.placed = {'kitten': 0, 'cat': 0}

class Kitten:
    def __init__(self, player, is_cat=False):
        self.player = player
        self.is_cat = is_cat

    def symbol(self):
        return ['b', 'w'][self.player] if not self.is_cat else ['B', 'W'][self.player]

class BoopEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.rows, self.cols = 6, 6
        self.grid_shape = (self.rows, self.cols)
        self.action_space = gym.spaces.MultiDiscrete([2, self.rows, self.cols, 2])  # action_type, row, col, piece_type
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.rows, self.cols, 5), dtype=np.float32)
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.full(self.grid_shape, None)
        self.players = [Player(0, 'K'), Player(1, 'O')]
        self.current_player_num = 0
        self.done = False
        self.turns_taken = 0
        return self.observation.astype(np.float32), {}

    @property
    def observation(self):
        pos0 = np.array([[1 if isinstance(c, Kitten) and c.player == 0 else 0 for c in row] for row in self.board])
        pos1 = np.array([[1 if isinstance(c, Kitten) and c.player == 1 else 0 for c in row] for row in self.board])
        cats = np.array([[1 if isinstance(c, Kitten) and c.is_cat else 0 for c in row] for row in self.board])

        # Encode stock for each player in all cells
        stock_p0 = np.full(self.grid_shape, (self.players[0].stock['kitten'] + self.players[0].stock['cat']) / 8.0)
        stock_p1 = np.full(self.grid_shape, (self.players[1].stock['kitten'] + self.players[1].stock['cat']) / 8.0)

        return np.stack([pos0, pos1, cats, stock_p0, stock_p1], axis=-1)

    def get_state(self):
        return {
            "board": self.observation.tolist(),
            "stock": {
                "0": self.players[0].stock,
                "1": self.players[1].stock
            },
            "current_player": self.current_player_num,
            "players": []  # This will be filled by the server
        }

    def is_legal(self, action):
        action_type, row, col, piece_type = action
        player = self.players[self.current_player_num]

        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False

        if action_type == 0:
            # Placement action
            if self.board[row, col] is not None:
                return False
            if piece_type == 0 and player.stock['kitten'] == 0:
                return False
            if piece_type == 1 and player.stock['cat'] == 0:
                return False
            return True

        elif action_type == 1:
            # Graduation/removal action

            if sum(player.placed.values()) < 8:
                return False  # can't graduate/remove unless board is full
            
            piece = self.board[row, col]
            if piece is None or not isinstance(piece, Kitten) or piece.player != self.current_player_num:
                return False

            if piece_type == 0 and not piece.is_cat and player.stock['cat'] == 0:
                return True  # graduate kitten to cat

            if piece_type == 1 and piece.is_cat and player.stock['cat'] == 0:
                return True  # remove cat to stock

            return False

    def legal_actions(self):
        legal = []
        for r in range(self.rows):
            for c in range(self.cols):
                for t in [0, 1]:
                    for a in [0, 1]:  # both action types
                        if self.is_legal((a, r, c, t)):
                            legal.append((a, r, c, t))
        return legal

    def boop_adjacent(self, row, col):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c]:
                    booper = self.board[row, col]
                    boopee = self.board[r, c]
                    if booper.is_cat or not boopee.is_cat:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            if self.board[nr, nc] is None:
                                self.board[nr, nc] = self.board[r, c]
                                self.board[r, c] = None
                        else:
                            # Piece falls off the board — decrement placed count!
                            piece = self.board[r, c]
                            if isinstance(piece, Kitten):
                                owner = self.players[piece.player]
                                kind = 'cat' if piece.is_cat else 'kitten'
                                owner.placed[kind] -= 1
                                owner.stock[kind] += 1
                            self.board[r, c] = None


    def find_three_in_a_row(self, player, only_cats):
        dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
        matches = []
        for r in range(self.rows):
            for c in range(self.cols):
                if not isinstance(self.board[r, c], Kitten) or self.board[r, c].player != player:
                    continue
                if only_cats and not self.board[r, c].is_cat:
                    continue
                for dr, dc in dirs:
                    positions = [(r + i * dr, c + i * dc) for i in range(3)]
                    if all(0 <= pr < self.rows and 0 <= pc < self.cols and
                           isinstance(self.board[pr, pc], Kitten) and
                           self.board[pr, pc].player == player and
                           (not only_cats or self.board[pr, pc].is_cat)
                           for pr, pc in positions):
                        matches.extend(positions)
        return matches if matches else None

    
    def step(self, action):
        reward = 0.0

        player = self.players[self.current_player_num]

        """if not self.is_legal(action):
            legal_actions = self.legal_actions()
            if legal_actions:
                action = random.choice(legal_actions)
            else:
                # No legal actions exist: game ends (very rare edge case)
                return self.observation.astype(np.float32), 0.0, True, False, {"reason": "no_legal_moves"}
        """    
        action_type, row, col, piece_type = action

        if action_type == 0:
            # Place piece
            piece_name = 'kitten' if piece_type == 0 else 'cat'
            player.stock[piece_name] -= 1
            player.placed[piece_name] += 1
            self.board[row, col] = Kitten(self.current_player_num, is_cat=(piece_type == 1))

            self.boop_adjacent(row, col)

            # Check for promotion
            to_promote = self.find_three_in_a_row(self.current_player_num, only_cats=False)
            if to_promote:
                for r, c in to_promote:
                    piece = self.board[r, c]
                    if piece and not piece.is_cat:
                        player.placed['kitten'] -= 1
                        self.board[r, c] = None
                        player.stock['cat'] += 1
                        reward += 0.1

            winning = self.find_three_in_a_row(self.current_player_num, only_cats=True)
            if winning:
                return self.observation.astype(np.float32), 1.0, True, False, {"winning_positions": winning}

            self.current_player_num = 1 - self.current_player_num
            return self.observation.astype(np.float32), reward, False, False, {}

        elif action_type == 1:
            # Graduation/removal action (already legal)
            piece = self.board[row, col]
            if not piece.is_cat:
                self.board[row, col] = None
                player.placed['kitten'] -= 1
                player.stock['cat'] += 1
            else:
                self.board[row, col] = None
                player.placed['cat'] -= 1
                player.stock['cat'] += 1

            return self.observation.astype(np.float32), 0.1, False, False, {"status": "Graduated or removed piece"}


    def render(self):
        symbols = lambda k: '.' if k is None else k.symbol()
        print("\n".join(" ".join(symbols(c) for c in row) for row in self.board))
        print(f"Current player: {self.current_player_num}")

