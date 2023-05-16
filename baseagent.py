from pysc2.agents import base_agent
from pysc2.lib import actions, features
from pysc2.env import sc2_env, run_loop
from absl import app

import numpy as np
import random

class RandomAgent(base_agent.BaseAgent):
    def step(self, obs):
        super(RandomAgent, self).step(obs)
        
        # get a list of all possible actions
        action_id = np.random.choice(obs.observation.available_actions)

        # if the action is to select a unit, we need to provide a parameter: the location of a unit
        if action_id == actions.FUNCTIONS.select_point.id:
            unit_type = obs.observation.feature_screen.unit_type
            unit_y, unit_x = (unit_type == features.NEUTRAL_MINERAL_FIELD).nonzero()

            # if we have any unit in the observation space, select it
            if unit_y.any():
                target = [int(unit_x.mean()), int(unit_y.mean())]
                return actions.FUNCTIONS.select_point("select", target)
        
        # else, return a no-operation action
        return actions.FUNCTIONS.no_op()
