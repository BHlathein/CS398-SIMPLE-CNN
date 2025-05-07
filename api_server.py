from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from stable_baselines3 import PPO
import numpy as np
import uuid
from typing import Optional, List, Tuple, Union

from boop_env import BoopEnv

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()           # Logs to the terminal
    ]
)
logger = logging.getLogger(__name__)

# === FastAPI setup ===
app = FastAPI()

# === Models ===
# Load models at startup
try:
    boop_model = PPO.load("ppo_boop_cnn_v0")
    logger.info("Successfully loaded Boop AI model")
except Exception as e:
    logger.error(f"Failed to load Boop model: {e}")
    boop_model = None

games = {
    "boop": {'env': BoopEnv, 'model': boop_model}
}

# === Session store ===
game_sessions = {}

# === Init new game ===
class NewGameRequest(BaseModel):
    players: List[str]

@app.post("/api/games/{game}/new")
def new_game(game: str, config: NewGameRequest):
    if game not in games:
        raise HTTPException(status_code=404, detail=f"Game '{game}' not supported")
    
    game_id = str(uuid.uuid4())
    env = games[game]['env']()
    obs, _ = env.reset()  # Get observation from reset
    
    game_sessions[game_id] = {
        "env": env,
        "players": config.players
    }
    state = env.get_state()
    return {"game_id": game_id, "state": state, "status": "Game started"}

# === Game move ===
class MoveRequest(BaseModel):
    game_id: str
    action: Optional[List[int]] = None

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
        if not legal:
            return {
                "state": env.get_state(),
                "status": "No legal actions available",
                "game_over": True
            }
        
        try:
            # Get the model's prediction
            obs = env.get_observation()
            obs_array = np.expand_dims(obs, axis=0)  # Add batch dimension for model
            model = games[game]["model"]
            
            if model is None:
                # Fallback to random action if model not loaded
                ai_action = np.array(legal[np.random.choice(len(legal))])
            else:
                # Get model prediction
                ai_action, _ = model.predict(obs_array, deterministic=True)
                
                # Convert to tuple for comparison with legal actions
                if isinstance(ai_action, np.ndarray):
                    if ai_action.ndim > 1:
                        ai_action = ai_action[0]  # Take first element if batched
                
                # Check if action is legal, fallback to random if not
                ai_action_tuple = tuple(int(x) for x in ai_action)
                if ai_action_tuple not in legal:
                    ai_action = np.array(legal[np.random.choice(len(legal))])
            
            obs, reward, terminated, truncated, info = env.step(ai_action)
            
        except Exception as e:
            logger.error(f"Error during AI move: {e}")
            # Fallback to random action
            action = legal[np.random.choice(len(legal))]
            obs, reward, terminated, truncated, info = env.step(action)
            
    else:
        # Convert action to tuple for comparison
        action_tuple = tuple(action)
        legal_actions = env.legal_actions()
        
        if action_tuple not in legal_actions:
            return {
                "state": env.get_state(),
                "status": "Invalid action! Try again.",
                "game_over": False
            }
        
        obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        winner = 1 - env.current_player_num  # Previous player is the winner
        del game_sessions[game_id]
        return {
            "state": env.get_state(),
            "status": f"Game over! Player {winner} wins!",
            "game_over": True
        }
    
    return {
        "state": env.get_state(),
        "status": f"Player {env.current_player_num}'s turn.",
        "game_over": False
    }

# === Static file serving ===
app.mount("/", StaticFiles(directory="static", html=True), name="static")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8001, log_level="debug")
