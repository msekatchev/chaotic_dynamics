import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import odeint
import matplotlib.cm as cm

# set up the figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# define the Lorentz attractor equations
def lorentz_deriv(state, t0, sigma, rho, beta):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

# set the initial states and time step
N = 50
states0 = [[np.random.rand(1)[0]*100, np.random.rand(1)[0]*100, np.random.rand(1)[0]*100] for _ in range(N)]
t = np.arange(0.0, 40.0, 0.01)

# set the parameters for the Lorentz attractor
sigma, rho, beta = 10.0, 100, 8.0 / 3.0

# solve the differential equations for each particle
states = [odeint(lorentz_deriv, state0, t, (sigma, rho, beta)) for state0 in states0]

# reshape the states arrays to have the correct shape
states = [state.T for state in states]

# set up the 3D scatter plot
scats = [ax.scatter(state[0], state[1], state[2], c='b', edgecolor='none', alpha=0.5) for state in states]
#colors = cm.jet(np.linspace(0, 1, N))
#scats = [ax.scatter(state[0], state[1], state[2], c=color, edgecolor='none', alpha=0.5) for state, color in zip(states, colors)]


# set the axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# create the animation function
def animate(i):
    for scat, state in zip(scats, states):
        scat._offsets3d = (state[0,[i]], state[1,[i]], state[2,[i]])
    return scats

# create the animation
anim = animation.FuncAnimation(fig, animate, frames=np.arange(0, t.size), interval=20, blit=False)

# show the plot
plt.show()
