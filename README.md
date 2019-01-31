# Reinforcement Learning - Assignment 1

## Nariman Saftarli - nsaftarli@ryerson.ca - 500615448

### Instructions for running

#### Requirements/Dependencies:

* Python 3.6
* Numpy
* Matplotlib

#### Instructions for running

* The UCB algorithm is implemented in Q1.py. I've also implemented the epsilon-greedy algorithm in egreedy.py. Either of these can be run with `python3 [filename].py`. By default, they each run for 100,000 steps. To make the runs shorter, the `iterations` variable (first variable specified) can be changed. The algorithm will run for the specified number of iterations and every 100 steps will print the optimal move percentage and average reward. Once the program finishes running, the probability distributions (estimated and real), optimal move percentages over time, and average rewards over time will be plotted using MatPlotLib's PyPlot library.

