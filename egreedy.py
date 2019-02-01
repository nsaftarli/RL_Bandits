import numpy as np
import matplotlib.pyplot as plt
import argparse
from ActionSelect import actionSelect
from Rewards import rollReward, updateProbs

def main(iterations):
    iterations = iterations
    config = {}
    nArms = 10
    eps = 0.9
    totalrew = 0
    trueDistrib = np.random.rand(nArms,)
    estDistrib = np.full((nArms,), 1/nArms)
    optimalAction = np.argmax(trueDistrib)
    nTimesOptimal = 0
    optimalHistory = []
    rewardHistory = []
    averageReward = 0

    config['nArms'] = nArms
    config['eps'] = eps
    config['totalrew'] = totalrew
    config['trueDistrib'] = trueDistrib
    config['estDistrib'] = estDistrib
    

    for t in range(1, iterations):
        config['t'] = t
        armPulled = actionSelect('egreedy', config)
        if armPulled == optimalAction:
            nTimesOptimal += 1

        if t % 100 == 0:
            optPercent = (nTimesOptimal/t) * 100
            optimalHistory.append(optPercent)
            rewardHistory.append(averageReward)
            print("Optimal choice was made " + str(optPercent) + "% of the time")
            print("Average reward is: " + str(averageReward))

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

def plotReward(hist):
    plt.suptitle("Average Reward Over Time for 10-Armed Bandit Problem using Epsilon-Greedy Action Selection", fontsize=14)
    plt.plot(hist)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iterations', default=100000, type=int)
    args = parser.parse_args()

    iterations = args.iterations
    main(iterations)