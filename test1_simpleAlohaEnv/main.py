#@title All `dm_control` imports required for this tutorial

# The basic mujoco wrapper.
from dm_control import mujoco

# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

# PyMJCF
from dm_control import mjcf

# Composer high level imports
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer import variation

# Imports for Composer tutorial example
from dm_control.composer.variation import distributions
from dm_control.composer.variation import noises
from dm_control.locomotion.arenas import floors

# Control Suite
from dm_control import suite

# Run through corridor example
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.arenas import corridors as corridor_arenas
from dm_control.locomotion.tasks import corridors as corridor_tasks

# Soccer
from dm_control.locomotion import soccer

# Manipulation
from dm_control import manipulation

#@title Other imports and helper functions

# General
import copy
import os
import itertools
from IPython.display import clear_output
import numpy as np

# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import PIL.Image

#
from sim_env import make_sim_env,BOX_POSE
from constants import FPS
from utils import sample_box_pose
#-----------------------------------------------------------------------------------------------
task_name = 'sim_transfer_cube'
onscreen_cam = 'angle'
while True:
    DT = 1 / FPS
    env = make_sim_env(task_name)
    
    BOX_POSE[0] = sample_box_pose() # used in sim reset
    env.reset()
    ax = plt.subplot()
    plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
    plt.show()
    plt.pause(DT)