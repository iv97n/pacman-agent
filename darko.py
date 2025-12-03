# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util
import os
import json
from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point
import time
import math

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]

# my_team.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  def register_initial_state(self, game_state):
    self.start = game_state.get_agent_position(self.index)
    CaptureAgent.register_initial_state(self, game_state)
    self.walls = game_state.get_walls()
    self.legal_positions = self.get_legal_positions(self.walls)
    
  def get_legal_positions(self, walls):
    # Returns a list of all legal (x,y) positions on the board
    width = walls.width
    height = walls.height
    positions = []
    for x in range(width):
        for y in range(height):
            if not walls[x][y]:
                positions.append((x, y))
    return positions

  def choose_action(self, game_state):
    actions = game_state.get_legal_actions(self.index)
    
    # Evaluate actions
    values = [self.evaluate(game_state, a) for a in actions]
    max_value = max(values)
    best_actions = [a for a, v in zip(actions, values) if v == max_value]

    return random.choice(best_actions)

  def get_successor(self, game_state, action):
    successor = game_state.generate_successor(self.index, action)
    pos = successor.get_agent_state(self.index).get_position()
    if pos != util.nearest_point(pos):
      return successor.generate_successor(self.index, action)
    return successor

  def evaluate(self, game_state, action):
    features = self.get_features(game_state, action)
    weights = self.get_weights(game_state, action)
    return features * weights

  def get_features(self, game_state, action):
    return util.Counter()

  def get_weights(self, game_state, action):
    return {}

###################
# Particle Filter #
###################

class ParticleFilter:
    """
    A particle filter for tracking a single agent.
    """
    def __init__(self, agent, start_pos, legal_positions, num_particles=300):
        self.agent = agent 
        self.num_particles = num_particles
        self.legal_positions = legal_positions
        self.particles = [start_pos] * num_particles

    def observe(self, my_pos, noisy_distance, enemy_pos_if_visible, game_state):
        """
        Update particle weights based on the noisy distance reading.
        """
        if enemy_pos_if_visible is not None:
            self.particles = [enemy_pos_if_visible] * self.num_particles
            return

        weights = util.Counter()
        
        for p in self.particles:
            true_dist = self.agent.get_maze_distance(my_pos, p)
            # We assume standard deviation of ~4 for the noise
            prob = game_state.get_distance_prob(true_dist, noisy_distance)
            weights[p] += prob

        # 3. Resample
        # FIX: Check for total_count (snake_case) or standard sum logic
        if sum(weights.values()) == 0:
            # If all weights zero, reset to uniform random
            self.particles = [random.choice(self.legal_positions) for _ in range(self.num_particles)]
        else:
            # FIX: Use random.choices instead of util.sample to avoid TypeError
            # Extract keys (positions) and values (probabilities)
            population = list(weights.keys())
            probs = list(weights.values())
            self.particles = random.choices(population, weights=probs, k=self.num_particles)

    def elapse_time(self, game_state):
        """
        Move particles to simulate enemy movement.
        """
        new_particles = []
        for old_pos in self.particles:
            x, y = int(old_pos[0]), int(old_pos[1])
            possible_moves = []
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_x, next_y = x + dx, y + dy
                if not game_state.has_wall(next_x, next_y):
                    possible_moves.append((next_x, next_y))
            
            if len(possible_moves) > 0:
                new_particles.append(random.choice(possible_moves))
            else:
                new_particles.append(old_pos)
        
        self.particles = new_particles

    def get_most_likely_position(self):
        """
        Return the position with the highest number of particles.
        """
        counts = util.Counter()
        for p in self.particles:
            counts[p] += 1
        # FIX: Changed argMax() to arg_max() just in case that is also renamed
        return counts.arg_max()

###################
# OFFENSIVE AGENT #
###################

class OffensiveAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food but avoids dead ends when ghosts are near.
  """
  
  def register_initial_state(self, game_state):
    ReflexCaptureAgent.register_initial_state(self, game_state)
    
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
                    if not walls[nx][ny]:
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

  def get_features(self, game_state, action):
    features = util.Counter()
    successor = self.get_successor(game_state, action)
    my_state = successor.get_agent_state(self.index)
    my_pos = my_state.get_position()
    
    # 1. Food Successor Score
    food_list = self.get_food(successor).as_list()
    features['successor_score'] = -len(food_list)

    # 2. Distance to nearest food
    if len(food_list) > 0:
      min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
      features['distance_to_food'] = min_distance

    # 3. Ghost Avoidance & Dead End Detection
    enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
    defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
    
    if len(defenders) > 0:
      dists = [self.get_maze_distance(my_pos, a.get_position()) for a in defenders]
      min_ghost_dist = min(dists)
      scared = min([a.scared_timer for a in defenders]) > 0
      
      if not scared:
        # --- NEW DEAD END LOGIC ---
        # If a ghost is somewhat near (threat range < 5), DO NOT enter/stay in a dead end.
        if min_ghost_dist <= 5:
            if my_pos not in self.safe_positions:
                # We are in a dead end (or entering one).
                # This is extremely dangerous.
                features['dead_end_trap'] = 1
        
        # Standard ghost avoidance
        if min_ghost_dist <= 1: 
            features['ghost_danger'] = 1000
        elif min_ghost_dist <= 5: 
            features['ghost_danger'] = (5 - min_ghost_dist) * 10
      else:
        # If scared, we can be aggressive
        features['eat_ghost'] = -min_ghost_dist

    # 4. Carrying Limit
    num_carrying = my_state.num_carrying
    if num_carrying > 0:
        dist_to_home = self.get_maze_distance(my_pos, self.start)
        features['distance_to_home'] = dist_to_home * num_carrying
    
    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def get_weights(self, game_state, action):
    return {
      'successor_score': 250,
      'distance_to_food': -2,
      'ghost_danger': -1000,
      'dead_end_trap': -10000,  # Huge penalty for being in a dead end near a ghost
      'eat_ghost': 100,
      'distance_to_home': -2,
      'stop': -100,
      'reverse': -2
    }
  
###################
# DEFENSIVE AGENT #
###################

class DefensiveAgent(ReflexCaptureAgent):
  
  def register_initial_state(self, game_state):
    ReflexCaptureAgent.register_initial_state(self, game_state)
    self.opponents = self.get_opponents(game_state)

    if not hasattr(self, 'legal_positions'):
        self.walls = game_state.get_walls()
        self.legal_positions = []
        for x in range(self.walls.width):
            for y in range(self.walls.height):
                if not self.walls[x][y]:
                    self.legal_positions.append((x, y))
    
    # Initialize Particle Filters for each opponent
    self.filters = {}
    for opp_index in self.opponents:
        # Enemy starts at their own start position (we need to look it up or guess)
        # Actually, in this contest, agents start at specific layout locations.
        # But for PF init, we can just initialize uniformly or at the "far corner".
        # Safe bet: Initialize at the furthest legal point from us (their spawn).
        enemy_start_pos = game_state.get_initial_agent_position(opp_index)
        self.filters[opp_index] = ParticleFilter(self, enemy_start_pos, self.legal_positions)

  def choose_action(self, game_state):
    # 1. Update Particle Filters BEFORE choosing action
    my_pos = game_state.get_agent_position(self.index)
    noisy_distances = game_state.get_agent_distances()
    
    for opp_index in self.opponents:
        # Get visible position (or None)
        enemy_obs = game_state.get_agent_position(opp_index)
        noisy_dist = noisy_distances[opp_index]
        
        # Elapse time (move particles)
        self.filters[opp_index].elapse_time(game_state)
        # Observe (weight particles)
        self.filters[opp_index].observe(my_pos, noisy_dist, enemy_obs, game_state)

    # 2. Proceed with normal action selection
    return ReflexCaptureAgent.choose_action(self, game_state)

  def get_features(self, game_state, action):
    features = util.Counter()
    successor = self.get_successor(game_state, action)
    my_state = successor.get_agent_state(self.index)
    my_pos = my_state.get_position()

    features['on_defense'] = 1
    if my_state.is_pacman: features['on_defense'] = 0

    # DETERMINE MOST LIKELY ENEMY POSITIONS
    enemy_positions = []
    for opp_index in self.opponents:
        # Ask the filter where it thinks the enemy is
        most_likely_pos = self.filters[opp_index].get_most_likely_position()
        enemy_positions.append((opp_index, most_likely_pos))

    # Check for invaders (enemies on our side)
    # We need to check if the *estimated* position is on our side
    invaders = []
    for idx, pos in enemy_positions:
        # Check if pos is on our side. 
        # Red is usually left (x < width/2), Blue is right.
        is_red = game_state.is_on_red_team(self.index)
        width = game_state.data.layout.width
        mid_x = width // 2
        
        # If I am red, invader is on left. If I am blue, invader is on right.
        is_invader = False
        if is_red and pos[0] < mid_x: is_invader = True
        if not is_red and pos[0] >= mid_x: is_invader = True
        
        if is_invader:
            invaders.append(pos)

    features['num_invaders'] = len(invaders)

    if len(invaders) > 0:
      # Intercept the closest estimated invader
      dists = [self.get_maze_distance(my_pos, pos) for pos in invaders]
      features['invader_distance'] = min(dists)
    else:
      # Patrol logic (if we think everyone is on their side)
      map_width = game_state.data.layout.width
      mid_x = map_width // 2
      if self.red: mid_x -= 1
      patrol_point = (mid_x, game_state.data.layout.height // 2)
      
      # Handle walls at patrol point
      while game_state.has_wall(int(patrol_point[0]), int(patrol_point[1])):
          patrol_point = (patrol_point[0], patrol_point[1] + 1)
      
      features['invader_distance'] = self.get_maze_distance(my_pos, patrol_point)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def get_weights(self, game_state, action):
    return {
      'num_invaders': -1000,
      'on_defense': 100,
      'invader_distance': -10,
      'stop': -100,
      'reverse': -2
    }
