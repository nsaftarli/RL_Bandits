import numpy as np

# Update estimated probabilities
def updateProbs(reward, table, n):
    table += (1/n) * (reward - table)
    return table

def rollReward(config):
    nArms = config['nArms']
    armPulled = config['armPulled']
    trueDistrib = config['trueDistrib']
    prob = trueDistrib[armPulled]
    # Roll a number that samples from the uniform distribution [0,1)
    # If the number is smaller than the probability, return a reward
    rewardTable = np.zeros((nArms,))
    roll = np.random.rand(1)[0]
    if roll <= prob:
        rewardTable[armPulled] = 1
    return rewardTable

def updateAutomatonSuccess(armPulled, probs, lr):
    left = probs[:armPulled]
    right = probs[armPulled + 1:]
    p = probs[armPulled:armPulled + 1]
    p += lr * (1 - p)
    left = (1 - lr) * left
    right = (1 - lr) * right
    return np.concatenate([left, p, right], axis=0)

def updateAutomatonFailure(armPulled, probs, lr):
    k = len(probs)
    incVal = lr / (k-1)
    left = probs[:armPulled]
    right = probs[armPulled + 1:]
    p = probs[armPulled:armPulled + 1]
    p = (1 - lr) * p
    left = incVal + (1 - lr) * left
    right = incVal + (1 - lr) * right
    return np.concatenate([left, p, right], axis=0)