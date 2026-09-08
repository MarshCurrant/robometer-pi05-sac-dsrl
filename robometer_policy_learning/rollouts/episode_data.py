import numpy as np


class EpisodeData:
    """Helper class to store data for a single episode."""

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []
        self.truncateds = []

    def add_transition(
        self,
        obs,
        action,
        reward,
        next_obs,
        done,
        truncated,
    ):
        """Adds a single transition to the episode."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_obs)
        self.dones.append(done)
        self.truncateds.append(truncated)

    def is_empty(self):
        """Checks if the episode data is empty."""
        return len(self.observations) == 0

    def __len__(self):
        return len(self.observations)
