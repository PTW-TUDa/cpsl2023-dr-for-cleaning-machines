import math

import numpy as np
from eta_utility import get_logger
from eta_utility.eta_x.common import episode_results_path
from eta_utility.eta_x.envs import BaseEnvLive, StateConfig, StateVar
from eta_utility.util import csv_export


log = get_logger("eta_x.envs")


class ConnectionKEA(BaseEnvLive):
    """Environment for the connection to the cleaning machine.

    :param env_id: Identification for the environment, useful when creating multiple environments.
    :param config_run: Configuration of the optimization run.
    :param seed: Random seed to use for generating random numbers in this environment.
        (default: None / create random seed).
    :param verbose: Verbosity to use for logging.
    :param callback: callback which should be called after each episode.
    :param scenario_time_begin: Beginning time of the scenario.
    :param scenario_time_end: Ending time of the scenario.
    :param episode_duration: Duration of the episode in seconds.
    :param sampling_time: Duration of a single time sample / time step in seconds.
    """

    version = "v0.2"
    description = "Environment for connection to a single chamber cleaning machine."
    config_name = "connection_KEA"

    def __init__(
        self,
        env_id,
        config_run,
        seed=None,
        verbose=2,
        callback=None,
        *,
        scenario_time_begin,
        scenario_time_end,
        episode_duration,
        sampling_time,
        **kwargs,
    ):
        # Instantiate BaseEnvLive
        super().__init__(
            env_id=env_id,
            config_run=config_run,
            seed=seed,
            verbose=verbose,
            callback=callback,
            scenario_time_begin=scenario_time_begin,
            scenario_time_end=scenario_time_end,
            episode_duration=episode_duration,
            sampling_time=sampling_time,
            **kwargs,
        )

        self.state_config = StateConfig(
            StateVar(name="nKEAOperatingState", ext_id="CM.nKEAOperatingState", is_ext_output=True),
            StateVar(name="nInterruptedStep", ext_id="CM.nInterruptedStep", is_ext_output=True),
            StateVar(name="tDurationSprayCleaning", ext_id="CM.fDurationSprayCleaning", is_ext_output=True),
            StateVar(name="tDurationDripping", ext_id="CM.fDurationDripping", is_ext_output=True),
            StateVar(name="tDurationSuction", ext_id="CM.fDurationSuction", is_ext_output=True),
            StateVar(name="tDurationBlowing", ext_id="CM.fDurationBlowing", is_ext_output=True),
            StateVar(name="fInterruptionCountdown", ext_id="CM.fInterruptionCountdown", is_ext_output=True),
            # Observartions (interact id is int, defined by order of observations)
            StateVar(name="n_start", is_agent_observation=True),  # interact_id = 0
            StateVar(
                name="p_heat", ext_id="CM.fTankHeaterNominalLoad", is_ext_output=True, is_agent_observation=True
            ),  # interact_id = 1
            StateVar(
                name="p_int", ext_id="CM.fPowerConsumptionOperational", is_ext_output=True, is_agent_observation=True
            ),  # interact_id = 2
            StateVar(
                name="p_clean", ext_id="CM.fPowerConsumptionCleaning", is_ext_output=True, is_agent_observation=True
            ),  # interact_id = 3
            StateVar(
                name="p_dry", ext_id="CM.fPowerConsumptionDrying", is_ext_output=True, is_agent_observation=True
            ),  # interact_id = 4
            StateVar(name="durationStart", is_agent_observation=True),  # interact_id = 5
            StateVar(name="durationCleaning", is_agent_observation=True),  # interact_id = 6
            StateVar(
                name="durationDrying",
                ext_id="CM.fDurationConvectionDrying",
                is_ext_output=True,
                is_agent_observation=True,
            ),  # interact_id = 7
            StateVar(
                name="temp_tank", ext_id="CM.fTankTemperature", is_ext_output=True, is_agent_observation=True
            ),  # interact_id = 8
            StateVar(
                name="tankTemperatureMin",
                ext_id="CM.fTemperatureLimitLowerWorking",
                is_ext_output=True,
                is_agent_observation=True,
            ),  # interact_id = 9
            StateVar(
                name="tankTemperatureMax",
                ext_id="CM.fTemperatureLimitUpperWorking",
                is_ext_output=True,
                is_agent_observation=True,
            ),  # interact_id = 10
            StateVar(
                name="c_pfluid",
                ext_id="CM.fCleaningFluidSpecificHeatCapacity",
                is_ext_output=True,
                is_agent_observation=True,
            ),  # interact_id = 11
            StateVar(
                name="V_tank", ext_id="CM.fTankVolume", is_ext_output=True, is_agent_observation=True
            ),  # interact_id = 12
            StateVar(
                name="rho_fluid", ext_id="CM.fCleaningFluidDensity", is_ext_output=True, is_agent_observation=True
            ),  # interact_id = 13
            StateVar(
                name="n_wp", ext_id="CM.nWorkpieces", is_ext_output=True, is_agent_observation=True
            ),  # interact_id = 14
            StateVar(
                name="m_wp", ext_id="CM.fMassWorkpiece", is_ext_output=True, is_agent_observation=True
            ),  # interact_id = 15
            StateVar(
                name="t_environment", ext_id="CM.fHallTemperature", is_ext_output=True, is_agent_observation=True
            ),  # interact_id = 16
            # Actions
            StateVar(name="set_interruption", ext_id="CM.bInterruptProcess", is_ext_input=True, is_agent_action=True),
            StateVar(
                name="set_tankheater",
                ext_id="CM.bTankHeaterSetStatusOnAlgorithm",
                is_ext_input=True,
                is_agent_action=True,
            ),
        )

        self.action_space, self.observation_space = self.state_config.continuous_spaces()
        self._init_live_connector()

    def step(self, action):
        """Perform one time step and return its results. This is called for every event or for every time step during
        the optimization run. It should utilize the actions as supplied by the agent to determine
        the new state of the environment. The method must return a four-tuple of observations, rewards, dones, info.

        :param action: Actions to perform in the environment.
        :return: The return value represents the state of the environment after the step was performed.
        """
        self._actions_valid(action)

        assert self.state_config is not None, "Set state_config before calling step function."

        self.n_steps += 1
        self._create_new_state(self.additional_state)

        # Preparation for the setting of the actions, store actions
        node_in = {}
        # Set actions in the opc ua server and read out the observations
        for idx, name in enumerate(self.state_config.actions):
            self.state[name] = action[idx]
            node_in.update({str(self.state_config.map_ext_ids[name]): action[idx]})

        # Update scenario data, do one time step in the live connector and store the results.
        self.state.update(self.get_scenario_state())

        results = self.live_connector.step(node_in)

        self.state = {name: results[str(self.state_config.map_ext_ids[name])] for name in self.state_config.ext_outputs}
        self.state.update(self.get_scenario_state())
        self._calculate_times()
        self.state_log.append(self.state)

        # Print the temperature value
        log.info(f"Current temperature in tank: {self.state['temp_tank']} °C")

        return self._observations(), 0, self._done(), {}

    def reset(self):
        """Reset the environment. This is called after each episode is completed and should be used to reset the
        state of the environment such that simulation of a new episode can begin.

        :return: The return value represents the observations (state) of the environment before the first
                 step is performed
        """
        assert self.state_config is not None, "Set state_config before calling reset function."
        self._reset_state()
        self._init_live_connector()

        self.state = {} if self.additional_state is None else self.additional_state

        # Read out and store start conditions
        results = self.live_connector.read(*self.state_config.map_ext_ids.values())
        self.state.update(
            {name: results[str(self.state_config.map_ext_ids[name])] for name in self.state_config.ext_outputs}
        )
        self.state.update(self.get_scenario_state())

        self._calculate_times()
        observations = self._observations()

        # Print the temperature value
        log.info(f"Current temperature in tank: {self.state['temp_tank']} °C")

        self.state_log.append(self.state)
        return observations

    def _calculate_times(self):
        """Calculate duration of the cleaning process"""
        self.state["durationCleaning"] = (
            self.state["tDurationSprayCleaning"]
            + self.state["tDurationDripping"]
            + self.state["tDurationSuction"]
            + self.state["tDurationBlowing"]
        ) // self.sampling_time

        self.state["durationStart"] = math.floor(self.state["fInterruptionCountdown"] / self.sampling_time)

        self.state["durationDrying"] //= self.sampling_time
        # Calculate current time step
        if self.state["nKEAOperatingState"] in {3, 4, 5, 6}:
            self.state["n_start"] = 2
        elif self.state["nKEAOperatingState"] in {7, 8}:
            self.state["n_start"] = 4
        elif self.state["nKEAOperatingState"] == 9:
            if self.state["nInterruptedStep"] == 2:
                self.state["n_start"] = 1
            else:
                self.state["n_start"] = 3
        else:
            self.state["n_start"] = 5
        self.state["n_start"] = int(self.state["n_start"])

    def render(self, mode: str = "human"):
        """Render the episode results. Outputs a CSV file with the results.

        :param mode: Unused in this environment.
        """
        csv_export(
            path=episode_results_path(self.path_results, self.run_name, self.n_episodes, self.env_id),
            data=self.state_log,
        )
