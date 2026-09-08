"""Register all configs with Hydra's ConfigStore."""

from hydra.core.config_store import ConfigStore
from robometer_policy_learning.configs.configs import TrainConfig, DSRLConfig, PolicyConfig, ValueFunctionConfig, RewardModelConfig
from robometer_policy_learning.algorithms.sac.configuration_sac import SACConfig
from robometer_policy_learning.algorithms.iql.configuration_iql import IQLConfig
from robometer_policy_learning.algorithms.bc.configuration_bc import BCConfig


def register_configs():
    """Register all configs with Hydra's ConfigStore."""
    cs = ConfigStore.instance()

    cs.store(name="config", node=TrainConfig)
    cs.store(name="dsrl_config", node=DSRLConfig)

    # Register algorithm configs as a group
    cs.store(group="algorithm", name="sac", node=SACConfig)
    cs.store(group="algorithm", name="iql", node=IQLConfig)
    cs.store(group="algorithm", name="bc", node=BCConfig)

    # Register policy configs as a group
    cs.store(group="policy", name="mlp", node=PolicyConfig)
    cs.store(group="policy", name="rnn", node=PolicyConfig)
    cs.store(group="policy", name="transformer", node=PolicyConfig)

    # Register value function configs as a group
    cs.store(group="value_function", name="mlp", node=ValueFunctionConfig)
    cs.store(group="value_function", name="rnn", node=ValueFunctionConfig)
    cs.store(group="value_function", name="transformer", node=ValueFunctionConfig)
