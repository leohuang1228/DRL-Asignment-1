def obs_reshape(obs):
    max_value = max(obs)
    if max_value == 0:
        max_value = 1
    return (obs[0] / max_value, obs[1] / max_value, 1 - obs[0] / max_value, 1 - obs[1] / max_value, obs[10], obs[11], obs[12], obs[13])
