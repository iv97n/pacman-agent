import random
import sys
import os
from functools import partial


import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point
from .heuristics.search import a_star
from .heuristics.heuristics import (
    distance_to_closest_food, distance_to_own_side, distance_to_enemy_in_own_side,
    unary_cost, safe_cost, distance_to_closest_capsule, base_cost, distance_to_enemy
)
from .heuristics.goals import GetFoodGoal, GoBackGoal, ChaseAgentGoal, PatrolGoal, GetCapsuleGoal
from .reflex_agents import ReflexDefensiveAgent



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
        self.cost = unary_cost

    def choose_action(self, game_state):
        if game_state.get_agent_state(self.index).num_carrying < 3:
            goal = GetFoodGoal(game_state, self.index)
            h = partial(self.food_heuristic, distancer=self.distancer)   
            c = partial(self.cost, distancer=self.distancer)
        else:
            goal = GoBackGoal(self.index)
            h = partial(self.home_heuristic, distancer=self.distancer)
            c = partial(self.cost, distancer=self.distancer)

        actions = self.search(game_state, self.index, h, goal, c)
        return actions[0]

        

class Take3AndGoDefensiveAgent(BasicHeuristicAgent):
    """Basic attack strategy: chase the enemy agent in your basement"""
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.home_heuristic = distance_to_own_side
        self.chase_heuristic = distance_to_enemy_in_own_side
        self.patrol_heuristic = None
        self.search = a_star
        self.cost = unary_cost


    def choose_action(self, game_state):
        agent_state = game_state.get_agent_state(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]
        if invaders and game_state.get_agent_state(self.index).scared_timer == 0:
            goal = ChaseAgentGoal(game_state, self.index)
            h = partial(self.chase_heuristic, distancer=self.distancer)
            c = partial(self.cost, distancer=self.distancer)

        elif agent_state.num_carrying > 0:
            goal = GoBackGoal(self.index)
            h = partial(self.home_heuristic, distancer=self.distancer)
            c = partial(self.cost, distancer=self.distancer)

        else:
            goal = PatrolGoal(game_state, self.index)
            h = partial(goal.patrol_heuristic, distancer=self.distancer)
            c = partial(self.cost, distancer=self.distancer)
            # goal = PatrolGoal(self.index)
            # h = partial(self.patrol_heuristic, distancer=self.distancer)
        
        actions = self.search(game_state, self.index, h, goal, c)
        if actions:
            return actions[0]
        else:
            return 'Stop'
        

        # return actions[0]

        # ideas
        # - (DONE) switch to defensive mode if wining by x, AND GO BACK TO ATTACK MODE if not winin by x anymore
        # - (DONE) introduce custom costs in the search? (enemy positions)
        # - clever strategy using the capsules (grab as many points as possible?) (use costs?)
        # - (DONE w/ the safe cost) avoid enemies when attacking?
        # - avoid entering dead ends when an agent within 2 tiles?
        # - if scared move to offensive mode?
        # - improve the patroling
        # - both offensive and defensive agents have an offensive/defensive subagent?


        # defensive
        # - divide patrol zones
        # -
    
class CleverOffensiveAgent(BasicHeuristicAgent):
    """Improved attack strategy"""
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.food_heuristic = distance_to_closest_food
        self.home_heuristic = distance_to_own_side
        self.capsule_heuristic = distance_to_closest_capsule
        self.search = a_star
        self.cost = safe_cost
        self.capsule_cost = base_cost
        self.unary_cost = unary_cost
        
        self.defensive_threshold = 6
        self.defensive_subagent = Take3AndGoDefensiveAgent(index)


    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.defensive_subagent.register_initial_state(game_state)

        # --- PRE-COMPUTE DEAD ENDS ---
        # We identify all tiles on the map that are part of a dead-end corridor.
        # We do this by iteratively pruning nodes with only 1 neighbor (tips of dead ends).
        
        walls = game_state.get_walls()
        height = walls.height
        width = walls.width
        
        # Build a graph of the layout
        neighbors = {}
        for x in range(width):
            for y in range(height):
                if not walls[x][y]:
                    n_list = []
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height and not walls[nx][ny]:
                            n_list.append((nx, ny))
                    neighbors[(x,y)] = n_list
        
        # Iteratively remove leaves (nodes with degree 1)
        # We stop at the midline because crossing home is an "exit"
        active_nodes = set(neighbors.keys())
        
        # Define home boundary cols
        if self.red:
            home_cols = range(0, width // 2)
        else:
            home_cols = range(width // 2, width)
            
        while True:
            # Find leaves (only 1 neighbor in the active set)
            # IMPORTANT: Do not prune home tiles (they are safe exits)
            leaves = []
            for node in active_nodes:
                # Count active neighbors
                active_neighbors = [n for n in neighbors[node] if n in active_nodes]
                
                # If it's a dead end tip AND not in our home territory
                if len(active_neighbors) < 2 and int(node[0]) not in home_cols:
                    leaves.append(node)
            
            if not leaves:
                break
                
            for leaf in leaves:
                active_nodes.remove(leaf)
                
        # The nodes remaining in active_nodes are "Safe" (loops, junctions, or home).
        # The nodes removed are "Dead Ends".
        self.safe_positions = active_nodes
        self.num_capsules = 1
        self.capsules_timer = 0


    def choose_action(self, game_state):
        score = game_state.get_score() if self.red else -game_state.get_score()
        capsules = game_state.get_blue_capsules() if self.red else game_state.get_red_capsules()


        # check if some agent has eaten a capsule, if so go for the food ignoring the enemies
        if len(capsules) < self.num_capsules:
            print("hola")
            self.capsules_timer = 40
            self.num_capsules = len(capsules)
        
        # keep eating food while the capsule is active, ignore enemies. Decrease the capsules timer
        if self.capsules_timer > 0:
            goal = GetFoodGoal(game_state, self.index)
            h = partial(self.food_heuristic, distancer=self.distancer)   
            c = partial(self.unary_cost, distancer=self.distancer)
            self.capsules_timer -= 1

        # If there are capsules left go for the capsules
        # elif capsules:
            # goal = GetCapsuleGoal(game_state, self.index)
            # h = partial(self.capsule_heuristic, distancer=self.distancer) 
            # c = partial(self.cost, distancer=self.distancer, safe_positions=self.safe_positions)

        elif score >= self.defensive_threshold: 
            return self.defensive_subagent.choose_action(game_state)

        elif game_state.get_agent_state(self.index).num_carrying < 3:
            goal = GetFoodGoal(game_state, self.index)
            h = partial(self.food_heuristic, distancer=self.distancer)   
            c = partial(self.cost, distancer=self.distancer, safe_positions=self.safe_positions)

        else:
            goal = GoBackGoal(self.index)
            h = partial(self.home_heuristic, distancer=self.distancer)
            c = partial(self.cost, distancer=self.distancer, safe_positions=self.safe_positions)

        actions = self.search(game_state, self.index, h, goal, c)
        return actions[0]
        """
        score = game_state.get_score() if self.red else -game_state.get_score()


        if game_state.get_agent_state(self.index).scared_timer > 0:
            # offensive mode independently of everything else
            # avoid enemies?




            pass

        else:
            if score > self.defensive_threshold: 
                return self.defensive_subagent.choose_action(game_state)
        

        if game_state.get_agent_state(self.index).num_carrying < 3:
            goal = GetFoodGoal(game_state, self.index)
            h = partial(self.food_heuristic, distancer=self.distancer)   
        else:
            goal = GoBackGoal(self.index)
            h = partial(self.home_heuristic, distancer=self.distancer)

        actions = self.search(game_state, self.index, h, goal)
        return actions[0]
        """