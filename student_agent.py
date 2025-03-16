# Remember to adjust your student ID in meta.xml
from DQN import DQNAgent
from Costum_env import Costumenv
from agent_util import obs_reshape


model_path = "agents/survive"

env = Costumenv()
obs, _ = env.reset()
state_dim = len(obs_reshape(obs))
action_dim = 6
AGENT = DQNAgent(state_dim, action_dim, model_path=model_path)
AGENT.load_model()


def get_action(obs):
    # use DQN
    return AGENT.select_action(obs_reshape(obs))
