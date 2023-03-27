import numpy as np

from spriteworld import environment
from spriteworld import renderers


class RandomAgent(object):
  """Agent that takes random actions."""

  def __init__(self, env):
    """Construct random agent."""
    self._env = env

  def step(self, timestep):
    # observation is a dictionary with renderer outputs to be used for training
    observation = timestep.observation
    action = self._env.action_space.sample()
    return observation, action


def main(argv):
    env = environment.Environment(**config)
    agent = RandomAgent(env)

    # Loop over episodes, logging success and mean reward per episode
    for episode in range(FLAGS.num_episodes):
        timestep = env.reset()
        rewards = []
        while not timestep.last():
            action = agent.step(timestep)
            timestep = env.step(action)
            rewards.append(timestep.reward)
            logging.info('Episode %d: Success = %r, Reward = %s.', episode,
                        timestep.observation['success'], np.nanmean(rewards))