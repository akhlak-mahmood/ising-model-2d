#!/usr/bin/env python3

""" 2D Ising simulator using Metropolis algorithm 
    Author: Akhlak Mahmood
    License: MIT
    Last update: April 18, 2019
"""

## Import modules
# -------------------------------------------
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from tqdm import tqdm # fancy python progress bar (> pip install tqdm)

## Define constants
# -------------------------------------------
J = 1

## Setup plot
# -------------------------------------------
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["font.size"] = 12


## 2D Ising Lattice class
# -------------------------------------------

class IsingModel:
    
    def __init__(self, sqrsize=4, initial_spins='r'):
        """ Initialize a 2D square Ising lattice with the specified initial spin 
        
        Args
        ----
        sqrsize (int) : size of the lattice
        initial_spins (1 or -1 or 'r') : set spins to 1 or -1 or random if 'r'

        """
        self.sqr = sqrsize*sqrsize
        self.size = sqrsize
        self._magvalue = np.zeros(1)
        self.encoder = FFMpegWriter = animation.writers['ffmpeg']
        self.writer = None
        
        # build the system
        if initial_spins == 'r':
            # random spins
            system = np.random.randint(0, 1+1, (self.size, self.size))
            # flip to -1 if an item is a zero
            system[system==0] = -1
        else:
            # fixed spins
            system = np.ones((self.size, self.size)) * initial_spins 
            
        self.system = system
        
    def _bc(self, x):
        """ Apply pbc condition to lattice site abscissa or ordinate x """
        if x+1 > self.size-1:
            # wrap back to 0
            return 0
        elif x-1 < 0:
            # wrap back to far side
            return self.size-1
        else:
            # inside box
            return x
        
    def _deltaH(self, i, j, B=0.0):
        """ Compute delta H for a single spin site i, j using pbc.
            Include external field contribution if any.
        """
        return -2 * self.system[i,j] * (
            B   + self.system[self._bc(i-1), j]
                + self.system[self._bc(i+1), j] 
                + self.system[i, self._bc(j-1)] 
                + self.system[i, self._bc(j+1)])
    
    def plot_system(self, saveto=None):
        """ Plot the current state of the system """
        plt.close('all')
        plt.ion()
        plt.imshow(self.system, interpolation='nearest')
        plt.xlim(0, self.size)
        plt.ylim(0, self.size)
        if saveto:
            plt.savefig(saveto)
        plt.show()
        
    def plot_magnetization(self, saveto=None):
        """ Plot magnetization vs time for the last run """
        plt.close('all')
        plt.plot(self._magvalue)
        plt.xlabel('time')
        plt.ylabel('average magnetization')
        plt.grid()
        if saveto:
            plt.savefig(saveto)
        plt.show()
        
    @property
    def magnetization(self):
        """ Return average magnetization of the system """
        return np.sum(self.system) / self.sqr
        
    def run(self, T, stepcount, B=0.0):
        """ Do a Metropolis sweep on the system for stepcount steps
        
        Use run() if you need to do multiple MC sweeps as fast as possible.
        Use runMovie() if you need to run single a MC sweep and save as a movie.

        Args
        ----
        T (float) : temperature of the run

        stepcount (int) : number of Metropolis iterations
        
        B (float) : external field, default 0.0
        
        Return
        ------
        A list of magnetization values at each time step
        
        """
        
        self._magvalue = np.zeros(1)
        self._magvalue[0] = self.magnetization

        # 15 initial MC steps as a heat up process
        for step in range(stepcount+15):
            # choose a random site
            i, j = np.random.randint(0, self.size, 2)
            
            # find the energy change if the site was flipped
            delE = -self._deltaH(i, j, B)

            # accept if favorable
            if delE <= 0. or np.random.rand() < np.exp(- delE / T):
                self.system[i][j] *= -1
        
            # a list of magnetizations
            self._magvalue = np.concatenate([self._magvalue, [self.magnetization]])
            
        # discard the initial 15 steps
        self._magvalue = self._magvalue[15:]
        return self._magvalue
    
    def save_movie(self, filename='ising.mp4'):
        """ Initialize or save current frame to the movie file """
        # initialize movie writer
        if self.writer is None:
            print('saving movie %s' %filename)
            self.writer = self.encoder(fps=10)
            plt.close('all')
            plt.ion()
            fig = plt.figure()
            self.writer.setup(fig, filename, 100)
            
        # grab the current plot as movie frame
        else:
            img = plt.imshow(self.system, interpolation='nearest')
            self.writer.grab_frame()
            img.remove()
        
    def finish_movie(self):
        """ Make sure this method is called when finished with the movie """
        if self.writer:
            self.writer.finish()
            self.writer = None
            plt.close('all')
            print('movie saved')
        
    def runMovie(self, T, stepcount, B=0.0, nmov=100):
        """ Run Metropolis for stepcount steps and save movie frames
        with a progress bar. 
        
        finish_movie() is called automatically at the end.
        
        Args
        ----
        T (float) : temperature 
        
        stepcout (int) : number of metropolis steps
        
        B (float) : external magnetic field
        
        nmov (int) : save movie frames every nmov steps, specify false
        if not to save any (for speedup reasons).
        
        """
        self._magvalue = np.zeros(1)
        self._magvalue[0] = self.magnetization

        sys.stdout.flush()
        for step in tqdm(range(stepcount)):
            i, j = np.random.randint(0, self.size, 2)
            delE = -self._deltaH(i, j, B)
            if delE <= 0. or np.random.rand() < np.exp(-delE/T):
                self.system[i][j] *= -1

            if nmov and step % nmov == 0:
                self.save_movie()

            # a list of magnetizations
            self._magvalue = np.concatenate([self._magvalue, [self.magnetization]])
                
        tqdm.write("Net Magnetization: {:.2f}".format(self.magnetization))
        self.finish_movie()
        
def make_droplet(lattice, radius, spin):
    """ Choose a circular area at the center and set specified spin """
    size = lattice.size
    if radius < size / 2:
        # center of the system
        x, y = int(size/2), int(size/2)

        # sweep over the sites
        for i in range(size):
            for j in range(size):
                # measure distance from center
                d = np.sqrt((x - i)**2 + (y - j)**2)
                if d <= radius:
                    lattice.system[i][j] = spin
    return lattice


def run():
    """ Run for magnetization as a function of Temperature and External Field """
    global lattice, size
    size = 40

    print(" -- reverse T, -ve B field -- ")
    # --------------------------------------------------------    
    lattice = IsingModel(size)
    lattice.save_movie('cooling-negativeB.mp4')
    
    T = np.flip(np.linspace(0.1, 4.0, 20))
    M0 = np.zeros(len(T))

    sys.stdout.flush()
    for i, t in enumerate(tqdm(T)):
        M0[i] = np.mean(lattice.run(t, 100 * size * size, -0.1))
        lattice.save_movie()

    lattice.finish_movie()
    plt.plot(T, M0)
    plt.show()

    print(" -- field dependence, T = 2.0 -- ")
    # --------------------------------------------------------
    lattice = IsingModel(size)
    lattice.save_movie('hysteresis.mp4')

    B = np.linspace(-0.225, 0.225, 20)
    M0 = np.zeros(len(B))
    
    sys.stdout.flush()
    for i, b in enumerate(tqdm(B)):
        M0[i] = np.mean(lattice.run(2.0, 100 * size * size, b))
        lattice.save_movie()
    
    lattice.finish_movie()
    plt.plot(B, M0)
    plt.show()
    
def main():
    """ Nucleation run """

    size = 40
    lattice = IsingModel(size, 1)
    lattice.plot_system('initial-system.png')
    lattice.save_movie('no-droplet.mp4')
    lattice.runMovie(1.5, 250*size*size, -0.30, False)
    lattice.plot_system('no-droplet-final-system.png')
    lattice.plot_magnetization('no-droplet-magnetization.png')

    r = 10 #run number
    lattice = make_droplet(lattice, 5.0, -1)
    lattice.plot_system('droplet-system-%s.png' %r)
    lattice.save_movie('droplet-%s.mp4' %r)
    lattice.runMovie(1.5, 1000*size*size, -0.15, False)
    lattice.plot_system('droplet-final-system-%s.png' %r)
    lattice.plot_magnetization('droplet-magnetization-%s.png' %r)

if __name__ == '__main__':
    main()
