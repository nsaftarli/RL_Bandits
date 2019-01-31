import numpy as np
import matplotlib.pyplot as plt
from ActionSelect import actionSelect
from Rewards import rollReward, updateProbs


def main():
    iterations = 100000
    config = {}
    nArms = 10
    lr = 0.1

    nTimesChosen = np.zeros((10,))
    trueDistrib = np.random.rand(nArms,)
    estDistrib = np.full((nArms,), 1/nArms)
    automaton = np.full((nArms,), 1/nArms)
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