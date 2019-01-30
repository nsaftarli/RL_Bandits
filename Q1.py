from ActionSelect import actionSelect
import numpy as np

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
        print(armPulled)

if __name__ == '__main__':
    main()