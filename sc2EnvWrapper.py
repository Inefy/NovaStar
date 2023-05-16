class SC2EnvWrapper:
    def __init__(self, env):
        self.env = env

    def reset(self):
        obs = self.env.reset()
        return obs.observation.feature_screen.unit_type

    def step(self, action):
        if action == 0:
            sc2_action = actions.FUNCTIONS.no_op()
        else:
            unit_type = obs.observation.feature_screen.unit_type
            unit_y, unit_x = (unit_type != 0).nonzero()
            target = [int(unit_x.mean()), int(unit_y.mean())]
            sc2_action = actions.FUNCTIONS.select_point("select", target)
        obs = self.env.step([sc2_action])
        next_state = obs.observation.feature_screen.unit_type
        reward = self.calculate_reward(obs)
        done = obs.last()
        return next_state, reward, done, {}

    def calculate_reward(self, obs):
    reward = 0

    # Reward for defeating enemy units
    enemy_units = self.previous_obs.observation.raw_units.enemy
    current_enemy_units = obs.observation.raw_units.enemy
    reward += len(enemy_units) - len(current_enemy_units)

    # Penalty for losing own units
    own_units = self.previous_obs.observation.raw_units.own
    current_own_units = obs.observation.raw_units.own
    reward -= len(own_units) - len(current_own_units)

    self.previous_obs = obs
    return reward