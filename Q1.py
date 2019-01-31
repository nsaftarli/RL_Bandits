import numpy as np
import matplotlib.pyplot as plt
from ActionSelect import actionSelect
from Rewards import rollReward, updateProbs

def main():
    iterations = 100000
    config = {}
    c = 2
    nArms = 10
    nTimesChosen = np.zeros((10,))
    trueDistrib = np.random.rand(nArms,)
    estDistrib = np.full((nArms,), 1/nArms)
    optimalAction = np.argmax(trueDistrib)
    nTimesOptimal = 0
    averageReward = 0
    optimalHistory = []
    rewardHistory = []

    config['c'] = c
    config['nArms'] = nArms
    config['trueDistrib'] = trueDistrib
    config['estDistrib'] = estDistrib
    config['nTimesChosen'] = nTimesChosen

    for t in range(1, iterations):
        config['t'] = t
        if t <= nArms:
            armPulled = t-1
            nTimesChosen[t-1] += 1
            config['nTimesChosen'] = nTimesChosen
        else:
            armPulled, nTimesChosen = actionSelect('ucb', config)
            config['nTimesChosen'] = nTimesChosen

        if armPulled == optimalAction:
            nTimesOptimal += 1

        if t % 100 == 0:
            optPercent = (nTimesOptimal/t) * 100
            optimalHistory.append(optPercent)
            rewardHistory.append(averageReward)
            print("Optimal choice was made " + str(optPercent) + "% of the time")
            print("Average Reward is: " + str(averageReward))
        config['armPulled'] = armPulled
        rewardTable = rollReward(config)
        reward = np.amax(rewardTable)
        averageReward = averageReward * (t-1)/t + reward/t
        estDistrib = updateProbs(rewardTable, estDistrib, t)
        config['estDistrib'] = estDistrib

    plotHistograms(trueDistrib, estDistrib)
    plotOptimal(optimalHistory)
    plotReward(rewardHistory)

def plotHistograms(true, est):
    plt.suptitle("Real and Predicted Probability Distribution for 10-Armed Bandit Problem Using UCB Action Selection", fontsize=14)
    plt.subplot(1, 2, 1)
    plt.title("True Distribution")
    plt.plot(true)
    plt.subplot(1, 2, 2)
    plt.title("Estimated Distribution")
    plt.plot(est)
    plt.show()

def plotOptimal(hist):
    plt.suptitle("Frequency of Optimal Actions for 10-Armed Bandit Problem using UCB Action Selection", fontsize=14)
    plt.plot(hist)
    plt.show()


def plotReward(hist):
    plt.suptitle("Average Reward Over Time for 10-Armed Bandit Problem using Epsilon-Greedy Action Selection", fontsize=14)
    plt.plot(hist)
    plt.show()



if __name__ == '__main__':
    main()