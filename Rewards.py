import numpy as np

# Update estimated probabilities
def updateProbs(reward, table, n):
    table += (1/n) * (reward - table)
    return table

def rollReward(config):
    nArms = config['nArms']
    armPulled = config['armPulled']
    prob = config['prob']
    trueDistrib = config['trueDistrib']
    # Roll a number that samples from the uniform distribution [0,1)
    # If the number is smaller than the probability, return a reward
    rewTable = np.zeros((nArms,))
    roll = np.random.rand(1)[0]
    if roll <= prob:
        rewTable[ind] = 1
    return rewTable