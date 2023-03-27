from absl import app
from absl import flags
from absl import logging
import importlib
import numpy as np
from spriteworld import environment
from spriteworld import renderers
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes', 100, 'Number of training episodes.')
flags.DEFINE_string('config',
                    'spriteworld.configs.cobra.goal_finding_new_position',
                    'Module name of task config to use.')
flags.DEFINE_string('mode', 'train', 'Task mode, "train" or "test"]')

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

from PIL import Image

def main(argv):
    config = importlib.import_module(FLAGS.config)
    config = config.get_config(FLAGS.mode)
    config['renderers']['success'] = renderers.Success()  # Used for logging
    env = environment.Environment(**config)
    agent = RandomAgent(env)

    # Loop over episodes, logging success and mean reward per episode
    images = []
    for episode in tqdm(range(200000)):
        timestep = env.reset()
        rewards = []
        observation, action = agent.step(timestep)
        timestep = env.step(action)
        image = observation["image"]
        images.append(image)
        rewards.append(timestep.reward)
        # logging.info('Episode %d: Success = %r, Reward = %s.', episode,
        #             timestep.observation['success'], np.nanmean(rewards))

    images = np.array(images)
    np.save('./datasets/spriteworld_200k.npy', images)

if __name__ == '__main__':
    app.run(main)