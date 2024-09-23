import matplotlib.pyplot as plt
from numpy.random import random
from pylab import *
num_particles = 1000  # Renamed N
half_life = 3.053 * 60  # Renamed tau
decay_constant = log(2) / half_life  # Renamed mu

random_values = random(num_particles)  # Renamed z

decay_times = -1 / decay_constant * log(1 - random_values)  # Renamed t_dec
decay_times = sort(decay_times)
num_decayed_particles = arange(1, num_particles + 1)  # Renamed decayed

num_survived_particles = -num_decayed_particles + num_particles  # Renamed surrived

plt.plot(decay_times, num_survived_particles)
plt.title("the exponentional decay of TI")
plt.legend()
plt.xlabel('time, sec')
plt.ylabel('Number')
plt.savefig("figq3.png")
plt.show()