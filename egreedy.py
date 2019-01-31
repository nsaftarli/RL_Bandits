import numpy as np
import matplotlib.pyplot as plt
from ActionSelect import actionSelect
from Rewards import rollReward, updateProbs

def main():
    config = {}
    nArms = 10
    eps = 0.9
    totalrew = 0
    trueDistrib = np.random.rand(nArms,)
    estDistrib = np.full((nArms,), 1/nArms)
    optimalAction = np.argmax(trueDistrib)
    nTimesOptimal = 0
    optimalHistory = []

    config['nArms'] = nArms
    config['eps'] = eps
    config['totalrew'] = totalrew
    config['trueDistrib'] = trueDistrib
    config['estDistrib'] = estDistrib
    

    for t in range(1, 100000):
        config['t'] = t
        armPulled = actionSelect('egreedy', config)
        if armPulled == optimalAction:
            nTimesOptimal += 1

        if t % 100 == 0:
            optPercent = (nTimesOptimal/t) * 100
            optimalHistory.append(optPercent)
            print("Optimal choice was made " + str(optPercent) + "% of the time")

        config['armPulled'] = armPulled
        rewardTable = rollReward(config)
        estDistrib = updateProbs(rewardTable, estDistrib, t)
        config['estDistrib'] = estDistrib

    plotHistograms(trueDistrib, estDistrib)
    plotOptimal(optimalHistory)


def plotHistograms(true, est):
    plt.suptitle("Real and Predicted Probability Distribution for 10-Armed Bandit Problem Using Epsilon-Greedy Action Selection", fontsize=14)
    plt.subplot(1, 2, 1)
    plt.title("True Distribution")
    plt.plot(true)
    plt.subplot(1, 2, 2)
    plt.title("Estimated Distribution")
    plt.plot(est)
    plt.show()

def plotOptimal(hist):
    plt.suptitle("Frequency of Optimal Actions for 10-Armed Bandit Problem using Epsilon-Greedy Action Selection", fontsize=14)
    plt.plot(hist)
    plt.show()

if __name__ == '__main__':
    main()