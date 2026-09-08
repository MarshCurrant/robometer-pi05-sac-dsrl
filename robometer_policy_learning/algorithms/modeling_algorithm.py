import abc
import copy
import json
import os
from typing import List

import torch
import numpy as np


from robometer_policy_learning.algorithms.configuration_algorithm import BaseAlgorithmConfig
from robometer_policy_learning.utils.network_utils import CriticEnsemble


def get_value(x):
    if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        return x.item()
    else:
        return x


class BaseAlgorithm(abc.ABC):
    """
    A base algorithm for all algorithms used to train policies.
    """

    def __init__(self, config: BaseAlgorithmConfig):
        self.config = config
        self.component_names = []
        self.step_counter = 0
        self.logger = config.logger

    @abc.abstractmethod
    def train_step(self, batch: torch.Tensor, logging_prefix: str = None, rollout_step: int = None):
        """
        A single training step for the algorithm.
        """
        pass

    def copy_components(self, other: "BaseAlgorithm", components: List[str] = None):
        """
        Copy the components of the algorithm from another algorithm.
        If components is None, copy all components from self.component_names.
        """
        if components is None:
            components = self.component_names

        for component in components:
            if hasattr(other, component):
                setattr(self, component, getattr(other, component))
            else:
                print(f"NOTE: Component {component} not found in other algorithm")

    def _create_critic_ensemble(self, base_critic, num_critics):
        """Create an ensemble of critics from a single base critic."""
        critics = []

        def reset_parameters(module):
            for child in module.children():
                reset_parameters(child)

            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        for i in range(num_critics):
            # Create a copy that shares the heavy feature extractor, but has its own output head
            critic_copy = copy.deepcopy(base_critic)
            # Reset parameters for diversity
            reset_parameters(critic_copy)
            if self.pooled_critic_features:
                # Share transformer-specific heavy parts when available; MLP critics won't have these
                if hasattr(base_critic, "obs_feature_extractor"):
                    critic_copy.obs_feature_extractor = base_critic.obs_feature_extractor
                    critic_copy.position_embedding = base_critic.position_embedding
                    critic_copy.transformer_encoder = base_critic.transformer_encoder
                    critic_copy.action_embedding = base_critic.action_embedding
                    critic_copy.action_projection = getattr(base_critic, "action_projection", None)
                    critic_copy.obs_projection = getattr(base_critic, "obs_projection", None)
                    critic_copy.input_norm = getattr(base_critic, "input_norm", None)
                # share mlp parts
                if hasattr(base_critic, "obs_featurizer"):
                    critic_copy.obs_featurizer = base_critic.obs_featurizer
                    critic_copy.output_mlp = base_critic.output_mlp
            critics.append(critic_copy)
        return CriticEnsemble(critics)

    def save(self, save_dir: str):
        # go through all the components and save them
        for component in self.component_names:
            if hasattr(self, component):
                component_obj = getattr(self, component)
                file_path = os.path.join(save_dir, f"{component}.pt")
                torch.save(component_obj, file_path)

        # Save the training state (like the step or log_ent_coef)
        with open(os.path.join(save_dir, "training_state.json"), "w") as f:
            training_state = {
                "step": self.step_counter,
                "log_ent_coef": get_value(self.log_ent_coef) if hasattr(self, "log_ent_coef") else None,
                "n_updates": getattr(self, "_n_updates", None),
            }
            json.dump(training_state, f)

    def load(self, load_dir: str):
        # go through all the components and load them
        for component in self.component_names:
            if hasattr(self, component):
                file_path = os.path.join(load_dir, f"{component}.pt")
                # Load the component
                loaded_component = torch.load(file_path, map_location="cpu", weights_only=False)

                current_component = getattr(self, component)
                # Keep the objects created by the current runtime. Replacing a
                # model and its separately deserialized optimizer breaks their
                # shared parameter identity and silently stops optimization.
                if current_component is None:
                    setattr(self, component, loaded_component)
                elif hasattr(current_component, "load_state_dict") and hasattr(
                    loaded_component, "state_dict"
                ):
                    current_component.load_state_dict(loaded_component.state_dict())
                elif torch.is_tensor(current_component) and torch.is_tensor(
                    loaded_component
                ):
                    current_component.data.copy_(
                        loaded_component.to(
                            device=current_component.device,
                            dtype=current_component.dtype,
                        )
                    )
                else:
                    setattr(self, component, loaded_component)

        with open(os.path.join(load_dir, "training_state.json"), "r") as f:
            training_state = json.load(f)
            self.step_counter = training_state["step"]
            if training_state.get("n_updates") is not None and hasattr(
                self, "_n_updates"
            ):
                self._n_updates = int(training_state["n_updates"])
            if training_state["log_ent_coef"] is not None and hasattr(self, "log_ent_coef"):
                self.log_ent_coef.data.copy_(
                    torch.as_tensor(
                        training_state["log_ent_coef"],
                        device=self.log_ent_coef.device,
                        dtype=self.log_ent_coef.dtype,
                    )
                )
