import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, rc
from IPython.display import HTML

class TriangularWave():
    """
    This class implementation is for problem 16.14 in J.R. Taylor's
    Classical Mechanics.
    
    Parameters
    ----------
    x_pts : array of floats
        x positions of the wave 
    delta_t : float
        time step
    c : float
        speed of the wave
    L : length of the string

    Methods
    -------
    B_coeff(m):
        Returns the solution to B coefficient of the waves Fourier series. 
    k(self, n)
        Returns the wave number given n = 2*m + 1.
    u_triangular(self, t)
        Returns the wave from the sum of Fourier components.
    
    """
    def __init__(self, x_pts, L=1., c_wave=1., m_max=20, num_terms=20):
        """Initialize all parametrs for this class."""
        self.x_pts = x_pts
        self.L = L
        self.c = c_wave
        self.m_max = m_max
        self.num_terms = num_terms
    
    def B_coeff(self, m):
        """Fourier coefficient for the n = 2*m + 1 term in the expansion.
        """
        # if n % 2 == 1:   # n is odd
        #     m = (n - 1)/2
        #     return (-1)**m * 8. / (n * np.pi)**2 
        # else:  # n is even
        #     return 0.
        n = 2.*m + 1
        return (-1)**m * 8. / (n * np.pi)**2

    def k(self, m):
        """Wave number for n = 2*m + 1.
        """
        return (2.*m + 1.) * np.pi / self.L

    def u_triangular(self, t):
        """Returns the wave from the sum of Fourier components.  
        """
        y_pts = np.zeros(len(self.x_pts))  # define y_pts as the same size as x_pts
        for m in np.arange(0, self.num_terms):
            y_pts += self.B_coeff(m) * np.sin(self.k(m) * self.x_pts) \
                      * np.cos(self.k(m) * self.c * t) 
        return y_pts

### Plotting ####

L = 1.
num_n = 40
m_max = 20
c_wave = 1
omega_1 = np.pi * c_wave / L
tau = 2.*np.pi / omega_1

# Set up the array of x points (whatever looks good)
x_min = 0.
x_max = L + num_n
delta_x = 0.01
x_pts = np.arange(x_min, x_max, delta_x)

# create instance for the class
u_triangular_1 = TriangularWave(x_pts, num_n, c_wave, L)
u_triangular_2 = TriangularWave(x_pts, num_n/8., c_wave, L)

# Make a figure showing the initial wave.
t_now = 0.

fig = plt.figure(figsize=(6,4), num='Standing wave')
ax = fig.add_subplot(1,1,1)
ax.set_xlim(x_min, x_max + 1.)
gap = 0.1
ax.set_ylim(-1. - gap, 1. + gap)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$u(x, t=0)$')
ax.set_title(rf'$t = {t_now:.1f}$')

line, = ax.plot(x_pts, 
                u_triangular_1.u_triangular(t_now), 
                color='blue', lw=2)

line2, = ax.plot(x_pts, 
                u_triangular_2.u_triangular(t_now), 
                color='red', lw=2)
plt.show()
fig.tight_layout()












# t_array = tau * np.arange(0., 1.125, .125)

# fig_array = plt.figure(figsize=(12,12), num='Standing wave')

# for i, t_now in enumerate(t_array): 
#     ax_array = fig_array.add_subplot(3, 3, i+1)
#     ax_array.set_xlim(x_min, x_max)
#     gap = 0.1
#     ax_array.set_ylim(-1. - gap, 1. + gap)
#     ax_array.set_xlabel(r'$x$')
#     ax_array.set_ylabel(r'$u(x, t)$')
#     ax_array.set_title(rf'$t = {t_now/tau:.3f}\tau$')

#     ax_array.plot(x_pts, 
#                   u_triangular(x_pts, t_now, m_max, c_wave, L), 
#                   color='blue', lw=2)

# fig_array.tight_layout()
# fig_array.savefig('Taylor_Problem_16p14.png', 
#                    bbox_inches='tight')  




# # Set up the t mesh for the animation.  The maximum value of t shown in
# #  the movie will be t_min + delta_t * frame_number
# t_min = 0.   # You can make this negative to see what happens before t=0!
# t_max = 2.*tau
# delta_t = t_max / 100.
# t_pts = np.arange(t_min, t_max + delta_t, delta_t)





