import random
from functools import partial

import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point
from heuristics.search import a_star
from heuristics.heuristics import basic_attack_heursitic, distance_to_closest_food, distance_to_own_side, distance_to_enemy_in_own_side
from heuristics.goals import GetFoodGoal, GoBackGoal, ChaseAgentGoal, PatrolGoal





class BasicHeuristicAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None 

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)


class Take3AndGoOffensiveAgent(BasicHeuristicAgent):
    """Basic attack strategy: try to get 3 food items and go back to your basement"""
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.food_heuristic = distance_to_closest_food
        self.home_heuristic = distance_to_own_side
        self.search = a_star

    def choose_action(self, game_state):
        if game_state.get_agent_state(self.index).num_carrying < 3:
            goal = GetFoodGoal(game_state, self.index)
            h = partial(self.food_heuristic, distancer=self.distancer)   
        else:
            goal = GoBackGoal(self.index)
            h = partial(self.home_heuristic, distancer=self.distancer)

        actions = self.search(game_state, self.index, h, goal)
        return actions[0]

        

class Take3AndGoDefensiveAgent(BasicHeuristicAgent):
    """Basic attack strategy: chase the enemy agent in your basement"""
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.home_heuristic = distance_to_own_side
        self.chase_heuristic = distance_to_enemy_in_own_side
        self.patrol_heuristic = None
        self.search = a_star


    def choose_action(self, game_state):
        agent_state = game_state.get_agent_state(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]
        if invaders and game_state.get_agent_state(self.index).scared_timer == 0:
            goal = ChaseAgentGoal(game_state, self.index)
            h = partial(self.chase_heuristic, distancer=self.distancer)

        elif agent_state.num_carrying > 0:
            goal = GoBackGoal(self.index)
            h = partial(self.home_heuristic, distancer=self.distancer)

        else:
            goal = PatrolGoal(game_state, self.index)
            h = partial(goal.patrol_heuristic, distancer=self.distancer)
            # goal = PatrolGoal(self.index)
            # h = partial(self.patrol_heuristic, distancer=self.distancer)
        
        actions = self.search(game_state, self.index, h, goal)
        if actions:
            return actions[0]
        else:
            return 'Stop'
        # return actions[0]

        # ideas
        # - switch to defensive mode if wining by x, AND GO BACK TO ATTACK MODE if not winin by x anymore
        # - clever strategy using the capsules (grab as many points as possible?) (use costs?)
        # - avoid enemies when attacking?


        # defensive
        # - divide patrol zones
        # -
    