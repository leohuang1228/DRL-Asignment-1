# Remember to adjust your student ID in meta.xml
from Qtable_agent import QLearningAgent
from Costum_env import Costumenv
from agent_util import obs_only_obstacle


model_path = "q_tables/walk_only"

env = Costumenv()
obs, _ = env.reset()
state_dim = len(obs_only_obstacle(obs))
action_dim = 6
AGENT = QLearningAgent(action_dim)
AGENT.load_q_table(model_path)


def get_action(obs):
    # use DQN
    return AGENT.choose_best_action(obs_only_obstacle(obs))
