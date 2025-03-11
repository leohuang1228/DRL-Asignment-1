def obs_reshape(obs):
    return (obs[0] / obs[5], obs[1] / obs[5], 1 - obs[0] / obs[5], 1 - obs[1] / obs[5], obs[10], obs[11], obs[12], obs[13])
