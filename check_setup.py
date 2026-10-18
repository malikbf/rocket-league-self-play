
import torch
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym_ppo import Learner

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

engine = RocketSimEngine()

print("RocketSim initialized successfully")
print("RLGym-PPO imported successfully")
