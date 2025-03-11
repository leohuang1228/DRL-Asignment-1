# Remember to adjust your student ID in meta.xml
from DQN import DQNAgent
from agent_util import obs_reshape


model_path = "agents/model_reshape_agent_lower_update"

state_dim = 16
action_dim = 6
AGENT = DQNAgent(state_dim, action_dim, model_path=model_path)
AGENT.load_model()


def get_action(obs):
    # use DQN
    return AGENT.select_action(obs_reshape(obs))
