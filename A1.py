import numpy as np
import matplotlib.pyplot as plt



def setEnv(nArms):
    trueDist = np.random.rand(nArms,)
    return trueDist


def averageReward(currRew, currAvg, n):
    return currAvg + (1/n) * (currRew - currAvg)

def rollReward(prob):
    # Roll a number that samples from the uniform distribution [0,1)
    # If the number is smaller than the probability, return a reward
    roll = np.random.rand(1)[0]
    if roll <= prob:
        return 1
    else:
        return 0

def updateProbs(ind, reward, table, n):
    table[ind] += (1/n) * (reward - table[ind])
    return table


def pickArm(trueDistrib, estDistrib, eps, n):
    # Pick with probability eps whether or not to make the greedy move
    greedy = True
    numGen = np.random.rand(1)[0]
    if numGen < eps:
        greedy = False

    # If it's the greedy move, pick the highest value in the estimated probability table
    if greedy:
        # Greedy move
        maxInds = np.argwhere(estDistrib == np.amax(estDistrib))
        # print(estDistrib)
        if len(maxInds) > 1:
            indPicked = np.random.randint(0, len(maxInds) - 1)
            armPicked = trueDistrib[indPicked]
        else:
            indPicked = maxInds[0][0]
            armPicked = trueDistrib[indPicked]
    # Otherwise, pick a value at random
    else:
        indPicked = np.random.randint(0, len(estDistrib) - 1)
        armPicked = trueDistrib[indPicked]

    # Given the arm, roll a reward for the arm
    rewardVal = rollReward(armPicked)
    estDistrib = updateProbs(indPicked, rewardVal, estDistrib, n)
    return estDistrib


nArms = 10
eps = 0.1
totalrew = 0
trueDistrib = setEnv(nArms)
trueDistrib = [0.20667192, 0.21488821, 0.60482498, 0.30061569, 0.44926704, 0.98593824, 0.3833595,  0.98111571, 0.99304015, 0.96186659]
estDistrib = np.full((nArms,), 1/nArms)
print(trueDistrib)
# print(estDistrib)
for i in range(1, 100000):
    estDistrib = pickArm(trueDistrib, estDistrib, eps, i)
    # totalrew += rew
    # print(rew)
print(estDistrib)





