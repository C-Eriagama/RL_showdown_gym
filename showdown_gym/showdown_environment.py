import os
import time
from typing import Any, Dict
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.status import Status

import numpy as np
from poke_env import (
    AccountConfiguration,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.battle import AbstractBattle
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.environment.singles_env import ObsType
from poke_env.player.player import Player

from showdown_gym.base_environment import BaseShowdownEnv
import math

class ShowdownEnvironment(BaseShowdownEnv):

    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        account_name_one: str = "train_one",
        account_name_two: str = "train_two",
        team: str | None = None,
    ):        
        
        self.allowable_moves = [0,1,2,3,4,5,6,7,8,9] # showdown move IDs (switches, moves)

        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        self.rl_agent = account_name_one

    def _get_action_size(self) -> int | None:
        """
        None just uses the default number of actions as laid out in process_action - 26 actions.

        This defines the size of the action space for the agent - e.g. the output of the RL agent.

        This should return the number of actions you wish to use if not using the default action scheme.
        """
        return len(self.allowable_moves)  # Return None if action size is default

    def process_action(self, action: np.int64) -> np.int64:
        """
        Returns the np.int64 relative to the given action.

        The action mapping is as follows:
        action = -2: default
        action = -1: forfeit
        0 <= action <= 5: switch
        6 <= action <= 9: move
        10 <= action <= 13: move and mega evolve
        14 <= action <= 17: move and z-move
        18 <= action <= 21: move and dynamax
        22 <= action <= 25: move and terastallize

        :param action: The action to take.
        :type action: int64

        :return: The battle order ID for the given action in context of the current battle.
        :rtype: np.Int64
        """
        return np.int64(self.allowable_moves[action])

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()

        # Add any additional information you want to include in the info dictionary that is saved in logs
        # For example, you can add the win status

        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won

        return info

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Calculates the reward based on the changes in state of the battle.

        You need to implement this method to define how the reward is calculated

        Args:
            battle (AbstractBattle): The current battle instance containing information
                about the player's team and the opponent's team from the player's perspective.
            prior_battle (AbstractBattle): The prior battle instance to compare against.
        Returns:
            float: The calculated reward based on the change in state of the battle.
        """

        # If win/lose, give a big reward/penalty
        if battle.won:
            return 25.0
        elif battle.lost:
            return -10.0

        prior_battle = self._get_prior_battle(battle)

        reward = 0.0

        health_team = [mon.current_hp_fraction for mon in battle.team.values()]
        health_opponent = [mon.current_hp_fraction for mon in battle.opponent_team.values()]
        prior_health_team = [mon.current_hp_fraction for mon in prior_battle.team.values()] if prior_battle is not None else []
        prior_health_opponent = [mon.current_hp_fraction for mon in prior_battle.opponent_team.values()] if prior_battle is not None else []

        # If the opponent has less than 6 Pokémon, fill the missing values with 1.0 (fraction of health)
        if len(health_opponent) < len(health_team):
            health_opponent.extend([1.0] * (len(health_team) - len(health_opponent)))
        if len(prior_health_opponent) < len(health_team):
            prior_health_opponent.extend([1.0] * (len(health_team) - len(prior_health_opponent)))

        # Calculate the damage done to both teams since the last turn
        diff_health_team = np.sum(np.array(prior_health_team) - np.array(health_team))
        diff_health_opponent = np.sum(np.array(prior_health_opponent) - np.array(health_opponent))


        # Reward for reducing the opponent's health (up to +1 per turn)
        if diff_health_opponent > 0.5:
            reward += 2.0 * diff_health_opponent  # Bonus for doing a lot of damage in one turn
        elif diff_health_opponent > 0: # do not accidentally penalise for opponent healing
            reward += diff_health_opponent


        # Penalty for losing a lot of health (up to -1 per turn)
        if diff_health_team > 0.4:
            reward -= diff_health_team

        
        # Reward for knocking out an opponent's Pokémon
        prior_knocked_out_opponents = sum(1 for mon_hp in prior_health_opponent if mon_hp == 0) if prior_battle is not None else 0
        current_knocked_out_opponents = sum(1 for mon_hp in health_opponent if mon_hp == 0)
        if current_knocked_out_opponents > prior_knocked_out_opponents:
            reward += 3.0

        return reward

    def _observation_size(self) -> int:
        """
        Returns the size of the observation size to create the observation space for all possible agents in the environment.

        You need to set obvervation size to the number of features you want to include in the observation.
        Annoyingly, you need to set this manually based on the features you want to include in the observation from emded_battle.

        Returns:
            int: The size of the observation space.
        """

        # Simply change this number to the number of features you want to include in the observation from embed_battle.
        # If you find a way to automate this, please let me know!
        return 213

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Embeds the current state of a Pokémon battle into a numerical vector representation.
        This method generates a feature vector that represents the current state of the battle,
        this is used by the agent to make decisions.

        You need to implement this method to define how the battle state is represented.

        Args:
            battle (AbstractBattle): The current battle instance containing information about
                the player's team and the opponent's team.
        Returns:
            np.float32: A 1D numpy array containing the state you want the agent to observe.
        """

        my_team = list(battle.team.values())

        health_team = [mon.current_hp_fraction for mon in battle.team.values()]
        health_opponent = [
            mon.current_hp_fraction for mon in battle.opponent_team.values()
        ]

        # Ensure health_opponent has 6 components, filling missing values with 1.0 (fraction of health)
        if len(health_opponent) < len(health_team):
            health_opponent.extend([1.0] * (len(health_team) - len(health_opponent)))

        # Multiply hp by 10, and round up to nearest integer (bucket health into 11 segments)
        health_team = [math.ceil(hp * 10) for hp in health_team]
        health_opponent = [math.ceil(hp * 10) for hp in health_opponent]

        # Calculate move effectiveness for available moves
        move_effectiveness = []
        for move in battle.active_pokemon.moves if battle.active_pokemon else []:
            if move not in battle.available_moves:
                move_effectiveness.append(0.0)  # Move not available
                continue
            if move is not None and battle.opponent_active_pokemon is not None:
                effectiveness = battle.opponent_active_pokemon.damage_multiplier(move)
                move_effectiveness.append(effectiveness)
            else:
                move_effectiveness.append(1.0)  # Neutral effectiveness if type is unknown

        if len(move_effectiveness) < 4:
            move_effectiveness.extend([1.0] * (4 - len(move_effectiveness)))

        # Move categories (Physical, Special, Status) for available moves - one hot encoded
        move_categories = [0.0]*(4*3)
        for move in battle.active_pokemon.moves if battle.active_pokemon else []:
            if move in battle.available_moves:
                if move.category == MoveCategory.PHYSICAL:
                    move_categories[battle.active_pokemon.moves.index(move)*3 + 0] = 1.0
                elif move.category == MoveCategory.SPECIAL:
                    move_categories[battle.active_pokemon.moves.index(move)*3 + 1] = 1.0
                elif move.category == MoveCategory.STATUS:
                    move_categories[battle.active_pokemon.moves.index(move)*3 + 2] = 1.0

        # Active stat boosts
        relevant_stats = ['atk', 'spa', 'spe', 'def', 'spd']
        active_stat_boosts = [0.0]*5
        for stat in relevant_stats:
            active_stat_boosts[relevant_stats.index(stat)] = battle.active_pokemon.boosts[stat] if battle.active_pokemon is not None else 0.0

        # one hot encoding of my types
        my_types = []
        if battle.active_pokemon is not None:
            for poke_type in PokemonType:
                if poke_type in battle.active_pokemon.types:
                    my_types.append(1.0)
                else:
                    my_types.append(0.0)
        if len(my_types) < 20:
            my_types.extend([0.0] * (20 - len(my_types)))

        # one hot encoding of opponent types
        opponent_types = []
        if battle.opponent_active_pokemon is not None:
            for poke_type in PokemonType:
                if poke_type in battle.opponent_active_pokemon.types:
                    opponent_types.append(1.0)
                else:
                    opponent_types.append(0.0)
        if len(opponent_types) < 20:
            opponent_types.extend([0.0] * (20 - len(opponent_types)))


        # is switch available
        available_switches = [0.0]*6
        for mon in my_team:
            if mon in battle.available_switches:
                available_switches[my_team.index(mon)] = 1.0

        # switch types
        switch_types = [0.0]*120
        for mon in battle.available_switches:
            for poke_type in PokemonType:
                if poke_type in mon.types:
                    switch_types[(my_team.index(mon)*20) + poke_type.value - 1] = 1.0

        # active status condition
        active_status_effect = [0.0]*7
        if battle.active_pokemon is not None and battle.active_pokemon.status is not None:
            for status in Status:
                if battle.active_pokemon.status == status:
                    active_status_effect[status.value - 1] = 1.0

        # opponent active status condition
        opponent_status_effect = [0.0]*7
        if battle.opponent_active_pokemon is not None and battle.opponent_active_pokemon.status is not None:
            for status in Status:
                if battle.opponent_active_pokemon.status == status:
                    opponent_status_effect[status.value - 1] = 1.0


        #########################################################################################################
        # Caluclate the length of the final_vector and make sure to update the value in _observation_size above #
        #########################################################################################################

        # Final vector - single array with health of both teams
        final_vector = np.concatenate(
            [
                health_team,  # N components for the health (bucket) of each pokemon
                health_opponent,  # N components for the health (bucket) of opponent pokemon
                move_effectiveness,  # 4 components for the effectiveness of each move
                move_categories,  # 12 components for the categories of each move (one hot encoded)
                active_stat_boosts,  # 5 components for the active stat boosts of the active pokemon
                my_types,  # 20 components for my types
                opponent_types,  # 20 components for opponent types
                available_switches, # 6 components for whether each pokemon can be switched to
                switch_types, # 120 components for the types of each pokemon that can be switched to
                active_status_effect, # 7 components for the status condition of the active pokemon (one hot encoded)
                opponent_status_effect, # 7 components for the status condition of the opponent active pokemon (one hot encoded)
            ]
        )

        return final_vector


########################################
# DO NOT EDIT THE CODE BELOW THIS LINE #
########################################


class SingleShowdownWrapper(SingleAgentWrapper):
    """
    A wrapper class for the PokeEnvironment that simplifies the setup of single-agent
    reinforcement learning tasks in a Pokémon battle environment.

    This class initializes the environment with a specified battle format, opponent type,
    and evaluation mode. It also handles the creation of opponent players and account names
    for the environment.

    Do NOT edit this class!

    Attributes:
        battle_format (str): The format of the Pokémon battle (e.g., "gen9randombattle").
        opponent_type (str): The type of opponent player to use ("simple", "max", "random").
        evaluation (bool): Whether the environment is in evaluation mode.
    Raises:
        ValueError: If an unknown opponent type is provided.
    """

    def __init__(
        self,
        team_type: str = "random",
        opponent_type: str = "random",
        evaluation: bool = False,
    ):
        opponent: Player
        unique_id = time.strftime("%H%M%S")

        opponent_account = "ot" if not evaluation else "oe"
        opponent_account = f"{opponent_account}_{unique_id}"

        opponent_configuration = AccountConfiguration(opponent_account, None)
        if opponent_type == "simple":
            opponent = SimpleHeuristicsPlayer(
                account_configuration=opponent_configuration
            )
        elif opponent_type == "max":
            opponent = MaxBasePowerPlayer(account_configuration=opponent_configuration)
        elif opponent_type == "random":
            opponent = RandomPlayer(account_configuration=opponent_configuration)
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

        account_name_one: str = "t1" if not evaluation else "e1"
        account_name_two: str = "t2" if not evaluation else "e2"

        account_name_one = f"{account_name_one}_{unique_id}"
        account_name_two = f"{account_name_two}_{unique_id}"

        team = self._load_team(team_type)

        battle_format = "gen9randombattle" if team is None else "gen9ubers"

        primary_env = ShowdownEnvironment(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        super().__init__(env=primary_env, opponent=opponent)

    def _load_team(self, team_type: str) -> str | None:
        bot_teams_folders = os.path.join(os.path.dirname(__file__), "teams")

        bot_teams = {}

        for team_file in os.listdir(bot_teams_folders):
            if team_file.endswith(".txt"):
                with open(
                    os.path.join(bot_teams_folders, team_file), "r", encoding="utf-8"
                ) as file:
                    bot_teams[team_file[:-4]] = file.read()

        if team_type in bot_teams:
            return bot_teams[team_type]

        return None
