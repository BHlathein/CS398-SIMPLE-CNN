from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from stable_baselines3 import PPO, DQN
import numpy as np
import uuid
from typing import Optional

from boop_env import BoopEnv

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()            # Logs to the terminal
    ]
)
logger = logging.getLogger(__name__)

# === FastAPI setup ===
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Models ===
games = {
    "tictactoe": {'env': None,
                  'model': "tictactoe_dqn" }, # DQN.load
    "boop": {'env': BoopEnv,
             'model': PPO.load("ppo_boop_v0") } # ppo_boop/gen_02/v8
    }

# === Session store ===
game_sessions = {}

# === Init new game ===
class NewGameRequest(BaseModel):
    players: list[str]

@app.post("/api/games/{game}/new")
def new_game(game: str, config: NewGameRequest):
    if game not in games:
        raise HTTPException(status_code=404, detail=f"Game '{game}' not supported")

    game_id = str(uuid.uuid4())
    env = games[game]['env']()
    env.reset()

    # Store player types in the session
    game_sessions[game_id] = {
        "env": env,
        "players": config.players
    }
    
    # Get the initial state
    state = env.get_state()
    # Add player types to the state
    state["players"] = config.players
    
    # Set initial status
    current_player = state["current_player"]
    status = f"Player {current_player}'s turn"
    if config.players[current_player] == "ai":
        status = "AI is thinking..."
        
    return {"game_id": game_id, "state": state, "status": status}

# === Game move ===
class MoveRequest(BaseModel):
    game_id: str
    action: Optional[list[int]]

@app.post("/api/games/{game}/move")
def make_move(game: str, request: MoveRequest):
    game_id = request.game_id
    action = request.action

    if game_id not in game_sessions:
        raise HTTPException(status_code=400, detail="Invalid game ID")

    session = game_sessions[game_id]
    env = session["env"]
    players = session["players"]

    if action is None:  # if sends empty action, handle as AI move
        current_player = env.current_player_num
        if players[current_player] != "ai":
            return {
                "state": env.get_state(),
                "status": "It's not the AI's turn.",
                "game_over": False
            }

        legal = env.legal_actions()
        obs = env.observation.reshape(1, *env.observation.shape)
        model = games[game]["model"]
        ai_action, _ = model.predict(obs, deterministic=True)
        
        # Convert numpy array to list of regular Python integers
        if isinstance(ai_action, np.ndarray):
            ai_action = [int(x) for x in ai_action[0]]
        else:
            ai_action = [int(x) for x in ai_action]
            
        ai_action = tuple(ai_action)

        if ai_action not in legal:
            ai_action = legal[np.random.choice(len(legal))]

        # Store board state before AI move
        board_before = env.get_state()["board"]
        
        obs, reward, terminated, truncated, info = env.step(ai_action)

        # Convert AI move details to regular Python types
        action_type, row, col, piece_type = [int(x) for x in ai_action]
        
        # Get updated state and ensure all numpy values are converted
        state = env.get_state()
        state["players"] = players

        return {
            "state": state,
            "status": f"Player {env.current_player_num}'s turn.",
            "game_over": terminated or truncated,
            "ai_move": {
                "action_type": action_type,
                "row": row,
                "col": col,
                "piece_type": piece_type,
                "board_before": board_before
            }
        }

    else:
        if tuple(action) not in env.legal_actions():
            return {
                "state": env.get_state(),
                "status": "Invalid action! Try again.",
                "game_over": False
            }

        obs, reward, terminated, truncated, info = env.step(action)

    # When returning state, include player types
    state = env.get_state()
    state["players"] = players
    
    if terminated or truncated:
        del game_sessions[game_id]
        return {
            "state": state,
            "status": f"Game over! Player {env.current_player_num} wins!",
            "game_over": True
        }

    return {
        "state": state,
        "status": f"Player {env.current_player_num}'s turn.",
        "game_over": False
    }

# === Static file serving ===
app.mount("/python", StaticFiles(directory="test/python"), name="python")
app.mount("/css", StaticFiles(directory="test/css"), name="css")
app.mount("/images", StaticFiles(directory="test/images"), name="images")
app.mount("/", StaticFiles(directory="test/html", html=True), name="html")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8001, log_level="debug")
