import numpy as np

def actionSelect(action, config):
    if action == 'egreedy':
        indPicked = greedyActionSelection(config)
    elif action == 'ucb':
        pass

    return indPicked

def greedyActionSelection(config):
    eps = config['eps']
    isGreedy = rollGreedy(eps)

    indPicked = greedyArm(config) if isGreedy else randomArm(config)
    return indPicked


def rollGreedy(eps):
    return True if np.random.rand(1)[0] <= eps else False


# Action to take if making a greedy move
def greedyArm(config):
    # Pick the indices from the estimated distribution with the highest values
    # If more than one of the same highest value, pick one at random
    # Take the arm with a probability corresponding to the true distribution
    trueDistrib = config['trueDistrib']
    estDistrib = config['estDistrib']
    t = config['t']
    maxInds = np.argwhere(estDistrib == np.amax(estDistrib))
    if len(maxInds) > 1:
        indPicked = np.squeeze(maxInds[np.random.randint(0, len(maxInds))])
        armPicked = trueDistrib[indPicked]
    else:
        indPicked = maxInds[0][0]
        armPicked = trueDistrib[indPicked]

    return indPicked


# Action to take if making a random move
def randomArm(config):
    # Pick an index at random.
    estDistrib = config['estDistrib']
    indPicked = np.random.randint(0, len(estDistrib))
    return indPicked






