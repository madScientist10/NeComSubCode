import numpy as np

def epsilon_greedy_policy(Q, epsilon, nA):
    """
    Epsilon-greedy policy.
    
    Input:
        Q: Dictionary that maps from state -> action-values.
        epsilon: Probability of selecting a random action.
        nA: Number of actions in the environment.
        
    Output:
        Probabilities for each action.
    """
    
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn
    
        
        