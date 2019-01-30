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

    config['nArms'] = nArms
    config['eps'] = eps
    config['totalrew'] = totalrew
    config['trueDistrib'] = trueDistrib
    config['estDistrib'] = estDistrib

    for t in range(1, 10000):
        config['t'] = t
        armPulled = actionSelect('egreedy', config)
        config['armPulled'] = armPulled
        rewardTable = rollReward(config)
        estDistrib = updateProbs(rewardTable, estDistrib, t)

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