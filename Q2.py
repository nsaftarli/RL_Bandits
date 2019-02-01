import numpy as np
import matplotlib.pyplot as plt
import argparse
from ActionSelect import actionSelect
from Rewards import *


def main(iterations, automaton, learning_rate):
    iterations = iterations
    config = {}
    nArms = 10
    lr = learning_rate
    automatonType = automaton

    nTimesChosen = np.zeros((10,))
    trueDistrib = np.random.rand(nArms,)
    print(trueDistrib)
    # trueDistrib = [0.20667192, 0.21488821, 0.60482498, 0.30061569, 0.44926704, 0.98593824, 0.3833595,  0.98111571, 0.99304015, 0.96186659]
    estDistrib = np.full((nArms,), 1/nArms)
    probs = np.full((nArms,), 1/nArms)
    # probs = np.linspace(0, 1, 11)[:-1]
    optimalAction = np.argmax(trueDistrib)
    # optimalAction = 8
    nTimesOptimal = 0
    averageReward = 0
    optimalHistory = []
    rewardHistory = []

    config['nArms'] = nArms
    config['trueDistrib'] = trueDistrib
    config['estDistrib'] = estDistrib
    config['nTimesChosen'] = nTimesChosen
    config['probs'] = probs

    for t in range(1, iterations):
        config['t'] = t
        armPulled = actionSelect('automaton', config)
        config['armPulled'] = armPulled
        rewardTable = rollReward(config)
        reward = np.amax(rewardTable)
        averageReward = averageReward * (t-1)/t + reward/t

        if armPulled == optimalAction:
            nTimesOptimal += 1
            # print(nTimesOptimal)
        if np.amax(rewardTable) == 1:
            probs = updateAutomatonSuccess(armPulled, config['probs'], lr)
            config['probs'] = probs
        else:
            if automatonType == 'linear':
                pass
            else:
                probs = updateAutomatonFailure(armPulled, config['probs'], lr)
                config['probs'] = probs

        if t % 100 == 0:
            optPercent = (nTimesOptimal/t) * 100
            optimalHistory.append(optPercent)
            rewardHistory.append(averageReward)
            print("Optimal choice was made " + str(optPercent) + "% of the time")
            print("Average Reward is: " + str(averageReward))

    plotOptimal(optimalHistory)
    plotReward(rewardHistory)
    plotHistograms(trueDistrib, probs)


def plotHistograms(true, est):
    plt.suptitle("True Reward Probability Distribution for 10-Armed Bandit Problem and i-th Arm Choice Probability Distribution Using Linear Learning Automata", fontsize=10)
    plt.subplot(1, 2, 1)
    plt.title("True Reward Distribution")
    plt.plot(true)
    plt.subplot(1, 2, 2)
    plt.title("i-th Arm Choice Probability Distribution")
    plt.plot(est)
    plt.show()

def plotOptimal(hist):
    plt.suptitle("Frequency of Optimal Actions for 10-Armed Bandit Problem using Linear Learning Automata", fontsize=14)
    plt.plot(hist)
    plt.show()


def plotReward(hist):
    plt.suptitle("Average Reward Over Time for 10-Armed Bandit Problem using Linear Learning Automata", fontsize=14)
    plt.plot(hist)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iterations', default=100000, type=int)
    parser.add_argument('-a', '--automaton', default='linear')
    parser.add_argument('-lr', '--learning_rate', default=0.1, type=float)
    args = parser.parse_args()

    iterations = args.iterations
    automaton = args.automaton
    learning_rate = args.learning_rate
    main(iterations, automaton, learning_rate)











