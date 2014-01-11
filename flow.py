#!/usr/bin/env python -u
"""Demonstrates how to load geometry from an external file.

The external file is a Boolean numpy array in the .npy format.
Nodes marked as True indicate walls.

In order to generate a .npy file from an STL geometry, check
out utils/voxelizer.

The sample file pipe.npy was generated using:
  a = np.zeros((128, 41, 41), dtype=np.bool)
  hz, hy, hx = np.mgrid[0:41, 0:41, 0:128]
  a[(hz - 20)**2 + (hy - 20)**2 >
    (19.3 * (0.8 + 0.2 * np.sin(2 * pi * hx / 128.0)))**2] = True
"""

import os
import numpy as np

from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTFullBBWall
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim, LBForcedSim

import pygame

def render_barriers(barriers, grid_size):
    '''
        Renders a set of barriers on a grid using pygame
        sets the GLOBAL wall_map (2D numpy array of booleans)

    '''
    window = pygame.Surface(grid_size)
    window.fill(pygame.Color(0, 0, 0))

    for p1, p2 in barriers:
        pygame.draw.line(window, pygame.Color(255, 255, 255), p1, p2, 2)

    wm = pygame.surfarray.array2d(window).transpose()
    wm = np.greater(wm, 0)

    global wall_map, size
    size = grid_size
    wall_map = wm

class ExternalSubdomain(Subdomain2D):
    def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = 1.0

    def boundary_conditions(self, hx, hy):
        if hasattr(self.config, '_wall_map'):
            partial_wall_map = self.select_subdomain(
                self.config._wall_map, hx, hy)
            self.set_node(partial_wall_map, NTFullBBWall)

    # Only used with node_addressing = 'indirect'.
    def load_active_node_map(self, hx, hy):
        partial_wall_map = self.select_subdomain(
            self.config._wall_map, hx, hy)
        self.set_active_node_map_from_wall_map(partial_wall_map)


class ExternalSimulation(LBFluidSim, LBForcedSim):
    subdomain = ExternalSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'max_iters': 50000,
            'visc': 0.05,
            'periodic_x': False,
            'periodic_y': False})

    @classmethod
    def modify_config(cls, config):
        '''
        if config.geometry == 'pipe.npy':
            # Look for the geometry file in the same directory where
            # external_geometry.py is located.
            config.geometry = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), config.geometry)

        # Override lattice size based on the geometry file.
        wall_map = np.load(config.geometry)
        '''
        config.lat_nx, config.lat_ny = size
        #cx = config.lat_nx/4; cy=config.lat_ny/2; r=config.lat_ny/9
        #wall_map = np.fromfunction(lambda x,y: (x-cx)**2+(y-cy)**2<r**2, (config.lat_nx,config.lat_ny))

        # Add nodes corresponding to ghosts. Assumes an envelope size of 1.
        wm = np.pad(wall_map, (1, 1), 'constant', constant_values=True)
        config._wall_map = wm

    def __init__(self, config):        
        super(ExternalSimulation, self).__init__(config)
        self.add_body_force((1e-5, 0))

    def after_main_loop(self, runner):
        '''
            Save velocity as .npy
        '''
        s = runner._sim
        velocity = np.array([s.vx, s.vy])
        # File detour (global was acting weird)
        print velocity
        np.save('velocity', velocity)

def run_simulation():
    LBSimulationController(ExternalSimulation).run()
    return True

if __name__ == '__main__':
    size = (1200, 400)

    barriers = [
        # Some channel walls
        [[50, size[1] - 50], [200, 280]],
        [[50, 50], [200, 120]],

        # Two layer
        [[200, 280], [450, 280]],
        [[200, 120], [450, 120]],

        # Constriction
        [[450, 280], [600, 240]],
        [[450, 120], [600, 160]],

        # Alternation
        [[600, 240], [1000, 250]],
        [[600, 160], [1000, 150]]
    ]

    render_barriers(barriers, size)
    run_simulation()

