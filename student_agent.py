# Remember to adjust your student ID in meta.xml
from DQN import DQNAgent


model_path = "agents/model_test"

state_dim = 16
action_dim = 6
AGENT = DQNAgent(state_dim, action_dim, model_path=model_path)


def get_action(obs):
    # use DQN
    return AGENT.select_action(obs)
