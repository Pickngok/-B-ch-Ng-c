import numpy as np

def generate_hmm_sequence(n_steps=100, n_states=3, n_obs=4):
    np.random.seed(42)
    start_prob = np.random.dirichlet(np.ones(n_states))
    trans_prob = np.random.dirichlet(np.ones(n_states), size=n_states)
    emit_prob = np.random.dirichlet(np.ones(n_obs), size=n_states)
    states = [np.random.choice(n_states, p=start_prob)]
    observations = [np.random.choice(n_obs, p=emit_prob[states[0]])]
    for _ in range(1, n_steps):
        states.append(np.random.choice(n_states, p=trans_prob[states[-1]]))
        observations.append(np.random.choice(n_obs, p=emit_prob[states[-1]]))
    return np.array(states), np.array(observations)