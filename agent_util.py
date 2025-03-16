def obs_reshape(obs):
    max_x = max(obs[2], obs[4], obs[6], obs[8])
    max_y = max(obs[3], obs[5], obs[7], obs[9])
    if max_x == 0:
        agent_x1 = 1
    else:
        agent_x1 = obs[0] / max_x
    if max_y == 0:
        agent_y1 = 1
    else:
        agent_y1 = obs[1] / max_y
    return (agent_x1, agent_y1, obs[10], obs[11], obs[12], obs[13])
