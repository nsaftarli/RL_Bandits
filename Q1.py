import numpy as np
import matplotlib.pyplot as plt
from ActionSelect import actionSelect
from Rewards import rollReward, updateProbs

def main():
    config = {}
    c = 2
    nArms = 10
    nTimesChosen = np.zeros((10,))
    trueDistrib = np.random.rand(nArms,)
    estDistrib = np.full((nArms,), 1/nArms)

    config['c'] = c
    config['nArms'] = nArms
    config['trueDistrib'] = trueDistrib
    config['estDistrib'] = estDistrib
    config['nTimesChosen'] = nTimesChosen

    for t in range(1, 100000):
        config['t'] = t
        if t <= nArms:
            armPulled = t-1
            nTimesChosen[t-1] += 1
            config['nTimesChosen'] = nTimesChosen
        else:
            armPulled, nTimesChosen = actionSelect('ucb', config)
            config['nTimesChosen'] = nTimesChosen

        config['armPulled'] = armPulled
        rewardTable = rollReward(config)
        estDistrib = updateProbs(rewardTable, estDistrib, t)
        config['estDistrib'] = estDistrib

    plotHistograms(trueDistrib, estDistrib)

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