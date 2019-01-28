import numpy as np
import matplotlib.pyplot as plt


def main():
    nArms = 10
    eps = 0.9
    totalrew = 0
    trueDistrib = setEnv(nArms)
    # trueDistrib = [0.20667192, 0.21488821, 0.60482498, 0.30061569, 0.44926704, 0.98593824, 0.3833595,  0.98111571, 0.99304015, 0.96186659]
    estDistrib = np.full((nArms,), 1/nArms)
    print(trueDistrib)
    for i in range(1, 100000):
        isGreedy = rollGreedy(eps)
        estDistrib = greedyArm(trueDistrib, estDistrib, i) if isGreedy else randomArm(trueDistrib, estDistrib, i)

    print(estDistrib)
    plotHistograms(trueDistrib, estDistrib)


# Generate a true probability distribution of nArm bandits
def setEnv(nArms):
    trueDist = np.random.rand(nArms,)
    return trueDist


def averageReward(currRew, currAvg, n):
    return currAvg + (1/n) * (currRew - currAvg)

def rollReward(nArms, ind, prob):
    # Roll a number that samples from the uniform distribution [0,1)
    # If the number is smaller than the probability, return a reward
    rewTable = np.zeros((nArms,))
    roll = np.random.rand(1)[0]
    if roll <= prob:
        rewTable[ind] = 1
    return rewTable


# Update estimated probabilities
def updateProbs(reward, table, n):
    # table[ind] += (1/n) * (reward - table[ind])
    table += (1/n) * (reward - table)
    return table

# Pick whether or not the move made is greedy
def rollGreedy(e):
    return True if np.random.rand(1)[0] <= e else False


# Action to take if making a greedy move
def greedyArm(trueDistrib, estDistrib, n):
    # Pick the indices from the estimated distribution with the highest values
    # If more than one of the same highest value, pick one at random
    # Take the arm with a probability corresponding to the true distribution
    maxInds = np.argwhere(estDistrib == np.amax(estDistrib))
    if len(maxInds) > 1:
        indPicked = maxInds[np.random.randint(0, len(maxInds))]
        armPicked = trueDistrib[np.squeeze(indPicked)]
    else:
        indPicked = maxInds[0][0]
        armPicked = trueDistrib[indPicked]
    # Check whether or not the action gets a reward. Update estimated table.
    rewardVal = rollReward(len(estDistrib), indPicked, armPicked)
    estDistrib = updateProbs(rewardVal, estDistrib, n)
    return estDistrib

# Action to take if making a random move
def randomArm(trueDistrib, estDistrib, n):
    # Pick an index at random. Take the true probability corresponding to it.
    indPicked = np.random.randint(0, len(estDistrib))
    armPicked = trueDistrib[indPicked]
    rewardVal = rollReward(len(estDistrib), indPicked, armPicked)
    estDistrib = updateProbs(rewardVal, estDistrib, n)
    return estDistrib

def actionSelect():
    estDistrib + c * np.sqrt(np.log(t) / numActions)


def plotHistograms(true, est):
    plt.subplot(1, 2, 1)
    plt.title("True Distribution")
    plt.plot(true)
    plt.subplot(1, 2, 2)
    plt.title("Estimated Distribution")
    plt.plot(est)
    plt.show()


if __name__ == '__main__':
    main()
