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

# Shared team coordination data
team_patrol_assignments = {}

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

        # Resample with improved handling
        total_weight = sum(weights.values())
        if total_weight == 0:
            # If all weights zero, reset to uniform random
            self.particles = [random.choice(self.legal_positions) for _ in range(self.num_particles)]
        else:
            # Normalize weights and resample
            population = list(weights.keys())
            probs = [w / total_weight for w in weights.values()]
            self.particles = random.choices(population, weights=probs, k=self.num_particles)

    def elapse_time(self, game_state):
        """
        Move particles to simulate enemy movement with staying probability.
        """
        new_particles = []
        for old_pos in self.particles:
            x, y = int(old_pos[0]), int(old_pos[1])
            possible_moves = [(x, y)]  # Can stay in place
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_x, next_y = x + dx, y + dy
                if not game_state.has_wall(next_x, next_y):
                    possible_moves.append((next_x, next_y))
            
            # 20% chance to stay, 80% to move
            if len(possible_moves) > 1 and random.random() < 0.8:
                new_particles.append(random.choice(possible_moves[1:]))
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
    
    # Store team mate for coordination
    team_indices = [i for i in self.get_team(game_state) if i != self.index]
    self.team_mate_index = team_indices[0] if team_indices else None
    
    # Track capsules
    self.capsules = self.get_capsules(game_state)
    
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

  def get_features(self, game_state, action):
    features = util.Counter()
    successor = self.get_successor(game_state, action)
    my_state = successor.get_agent_state(self.index)
    my_pos = my_state.get_position()
    
    # Check if we're winning by a significant margin
    score = self.get_score(game_state)
    winning_threshold = 8
    
    # If winning by 5+ points, switch to defensive mode
    if score >= winning_threshold:
      return self._get_defensive_features(game_state, action, successor, my_state, my_pos)
    
    # 1. Food Successor Score
    food_list = self.get_food(successor).as_list()
    features['successor_score'] = -len(food_list)

    # Get ghost information first for other features
    enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
    defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
    min_ghost_dist = float('inf')
    scared = False
    min_scared_timer = 0
    # Belief-based rough risk using noisy distances if ghosts unseen
    noisy_dists = successor.get_agent_distances()
    opponent_indices = self.get_opponents(successor)
    min_noisy_def_dist = float('inf')
    
    if len(defenders) > 0:
      dists = [self.get_maze_distance(my_pos, a.get_position()) for a in defenders]
      min_ghost_dist = min(dists)
      scared_timers = [a.scared_timer for a in defenders]
      min_scared_timer = min(scared_timers)
      scared = min_scared_timer > 0
    # Estimate risk from noisy distances (for unseen defenders)
    for idx in opponent_indices:
      st = successor.get_agent_state(idx)
      if st.is_pacman:
        continue
      if st.get_position() is None:
        min_noisy_def_dist = min(min_noisy_def_dist, noisy_dists[idx])

    # 2. Capsule Prioritization
    capsules = self.get_capsules(successor)
    if len(capsules) > 0 and not scared:
      capsule_dists = [self.get_maze_distance(my_pos, cap) for cap in capsules]
      min_capsule_dist = min(capsule_dists)

      # If capsule is close and ghosts can't intercept next step, force grab
      ghost_can_intercept = False
      for d in defenders:
        if d.get_position() is not None:
          dist_ghost_to_cap = self.get_maze_distance(d.get_position(), capsules[capsule_dists.index(min_capsule_dist)])
          if dist_ghost_to_cap <= 1:
            ghost_can_intercept = True
            break
      if min_capsule_dist <= 3 and not ghost_can_intercept:
        features['capsule_force'] = 1
        features['distance_to_capsule'] = min_capsule_dist
        # So we don't hang around, reduce ghost penalty when going for it
        features['ghost_danger'] = 0

      # Opportunistic pickup when very close (even without nearby ghosts)
      if min_capsule_dist <= 2:
        features['capsule_opportunistic'] = 1
        features['distance_to_capsule'] = min_capsule_dist

      # Ghost-pressure-driven capsule seeking
      if min_ghost_dist < 10:
        # If capsule is closer than ghost and very near, prioritize it heavily
        if min_capsule_dist < min_ghost_dist and min_capsule_dist <= 3 and min_ghost_dist <= 5:
          features['distance_to_capsule'] = min_capsule_dist
          features['capsule_priority'] = 1
          # Override ghost danger when capsule is reachable
          features['ghost_danger'] = 0
        elif min_ghost_dist < 6:
          # Normal capsule seeking when ghost nearby
          features['distance_to_capsule'] = min_capsule_dist
          features['capsule_nearby'] = 1

    # 3. Smart Food Selection - prefer food closer to home when carrying
    num_carrying = my_state.num_carrying
    if len(food_list) > 0:
      if num_carrying > 3:
        # Prefer food closer to home
        food_scores = []
        for food in food_list:
          dist_to_food = self.get_maze_distance(my_pos, food)
          dist_food_to_home = self.get_maze_distance(food, self.start)
          score = dist_to_food + dist_food_to_home * 0.5
          food_scores.append(score)
        features['distance_to_food'] = min(food_scores)
      else:
        # Normal behavior - closest food
        min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
        features['distance_to_food'] = min_distance

    # 4. Ghost Avoidance & Dead End Detection
    if len(defenders) > 0 or min_noisy_def_dist < float('inf'):
      if not scared:
        # Dead end logic - stronger penalty
        if min_ghost_dist <= 5:
            if my_pos not in self.safe_positions:
                features['dead_end_trap'] = 1
        
        # Emergency escape - very close ghost
        if min_ghost_dist <= 2:
            features['ghost_danger'] = 1000
            # Boost return home urgency
            if num_carrying > 0:
                features['emergency_return'] = 1
        elif min_ghost_dist <= 5: 
          features['ghost_danger'] = (5 - min_ghost_dist) * 10
        
        # Additional buffer penalty for being within 3 steps of a ghost
        if min_ghost_dist <= 3:
          features['ghost_buffer'] = 4 - min_ghost_dist

        # Belief-based danger when ghosts unseen but noisy distance small
        if min_noisy_def_dist < float('inf') and min_noisy_def_dist <= 4:
          features['ghost_belief_danger'] = (5 - min_noisy_def_dist) * 5
      else:
        # If scared, focus on food but ensure we can return before timer expires
        if len(food_list) > 0:
          min_food_dist = min([self.get_maze_distance(my_pos, food) for food in food_list])
          features['scared_food_distance'] = min_food_dist
        
        # Make sure we can get home before scared timer runs out
        dist_to_home = self.get_maze_distance(my_pos, self.start)
        if dist_to_home + 4 >= min_scared_timer:  # larger buffer
          features['scared_return'] = dist_to_home * 2

    # 5. Dynamic Retreat Threshold - return earlier when ghosts near
    if num_carrying > 0:
        dist_to_home = self.get_maze_distance(my_pos, self.start)
        
        # Dynamic threshold based on ghost proximity
        if min_ghost_dist < 6:
          # Urgent return if ghost nearby and carrying
          features['distance_to_home'] = dist_to_home * num_carrying * 2
        elif num_carrying >= 5:
          # Normal return threshold
          features['distance_to_home'] = dist_to_home * num_carrying
        else:
          # Light penalty for being far when carrying few
          features['distance_to_home'] = dist_to_home * num_carrying * 0.5
    
    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def _get_defensive_features(self, game_state, action, successor, my_state, my_pos):
    """Defensive features when winning - protect the lead."""
    features = util.Counter()
    
    # Stay on defense
    features['on_defense'] = 1
    if my_state.is_pacman: features['on_defense'] = 0
    
    # Find invaders
    enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
    invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
    
    if len(invaders) > 0:
      dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
      features['invader_distance'] = min(dists)
      features['num_invaders'] = len(invaders)
    else:
      # Get coordinated patrol point
      patrol_point = self._get_patrol_zone(game_state)
      features['patrol_distance'] = self.get_maze_distance(my_pos, patrol_point)
    
    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1
    
    return features
  
  def _get_patrol_zone(self, game_state):
    """Get a patrol point that's coordinated with teammate."""
    food_defending = self.get_food_you_are_defending(game_state).as_list()
    
    if len(food_defending) == 0:
      # Don't patrol spawn - go to border
      width = game_state.data.layout.width
      height = game_state.data.layout.height
      mid_x = width // 2
      if self.red: mid_x -= 1
      return (mid_x, height // 2)
    
    # Divide patrol zones by y-coordinate (top/bottom)
    height = game_state.data.layout.height
    
    # Offensive agent (lower index) patrols top half
    if self.index < self.team_mate_index if self.team_mate_index else True:
      food_zone = [f for f in food_defending if f[1] >= height // 2]
    else:
      food_zone = [f for f in food_defending if f[1] < height // 2]
    
    # If our zone is empty, use all food
    if len(food_zone) == 0:
      food_zone = food_defending
    
    # Patrol near center of our zone (offensive agent when defending)
    avg_x = sum(f[0] for f in food_zone) / len(food_zone)
    avg_y = sum(f[1] for f in food_zone) / len(food_zone)
    patrol_point = (int(avg_x), int(avg_y))
    
    if game_state.has_wall(int(patrol_point[0]), int(patrol_point[1])):
      patrol_point = min(food_zone, key=lambda f: abs(f[0]-avg_x) + abs(f[1]-avg_y))

    # Make sure we're not too close to spawn
    spawn_dist = self.get_maze_distance(patrol_point, self.start)
    if spawn_dist < 3 and len(food_zone) > 1:
      patrol_point = max(food_zone, key=lambda f: self.get_maze_distance(f, self.start))

    # Avoid clustering with teammate: if too close, shift up/down if legal
    if self.team_mate_index is not None:
      mate_state = game_state.get_agent_state(self.team_mate_`index)
      if mate_state and mate_state.get_position() is not None:
        mate_pos = mate_state.get_position()
        if self.get_maze_distance(patrol_point, mate_pos) <= 2:
          # try shifting vertically
          for dy in [2, -2, 3, -3]:
            cand = (patrol_point[0], patrol_point[1] + dy)
            if 0 <= cand[1] < game_state.data.layout.height and not game_state.has_wall(int(cand[0]), int(cand[1])):
              patrol_point = cand
              break
    
    return patrol_point

  def get_weights(self, game_state, action):
    # Check if in defensive mode
    score = self.get_score(game_state)
    if score >= 8:
      return {
        'on_defense': 200,
        'num_invaders': -1000,
        'invader_distance': -10,
        'patrol_distance': -5,
        'stop': -100,
        'reverse': -2
      }
    
    return {
      'successor_score': 250,
      'distance_to_food': -2,
      'distance_to_capsule': -15,
      'capsule_nearby': 40,
      'capsule_priority': 2000,
      'capsule_opportunistic': 60,
      'capsule_force': 4000,
      'ghost_danger': -3000,
      'ghost_buffer': -800,
      'ghost_belief_danger': -1500,
      'dead_end_trap': -12000,
      'emergency_return': 5000,
      'scared_food_distance': -3,
      'scared_return': -120,
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
    
    # Store team mate for coordination
    team_indices = [i for i in self.get_team(game_state) if i != self.index]
    self.team_mate_index = team_indices[0] if team_indices else None
    
    # Track food state for detecting eaten food
    self.previous_food = self.get_food_you_are_defending(game_state).as_list()

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
        enemy_start_pos = game_state.get_initial_agent_position(opp_index)
        self.filters[opp_index] = ParticleFilter(self, enemy_start_pos, self.legal_positions)
    
    # Compute chokepoints - positions with limited paths
    self.chokepoints = self._compute_chokepoints(game_state)
    
    # Track initial food for protection
    self.initial_food_count = len(self.get_food_you_are_defending(game_state).as_list())
  
  def _compute_chokepoints(self, game_state):
    """Find narrow passages that are strategic defense points."""
    chokepoints = []
    walls = game_state.get_walls()
    
    for x, y in self.legal_positions:
      # Count open neighbors
      open_neighbors = 0
      for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
          open_neighbors += 1
      
      # Positions with only 2 neighbors are narrow passages
      if open_neighbors == 2:
        # Check if it's on our side
        is_red = game_state.is_on_red_team(self.index)
        mid_x = walls.width // 2
        if (is_red and x < mid_x) or (not is_red and x >= mid_x):
          chokepoints.append((x, y))
    
    # If no chokepoints found, use positions near midline as fallback
    if len(chokepoints) == 0:
      is_red = game_state.is_on_red_team(self.index)
      mid_x = walls.width // 2
      target_x = mid_x - 1 if is_red else mid_x
      for x, y in self.legal_positions:
        if x == target_x:
          chokepoints.append((x, y))
    
    return chokepoints

  def choose_action(self, game_state):
    # 1. Update Particle Filters BEFORE choosing action
    my_pos = game_state.get_agent_position(self.index)
    noisy_distances = game_state.get_agent_distances()
    
    # Detect eaten food
    current_food = self.get_food_you_are_defending(game_state).as_list()
    self.eaten_food = []
    if self.previous_food:
      self.eaten_food = [food for food in self.previous_food if food not in current_food]
    self.previous_food = current_food
    
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

    # Scared ghost behavior - flee from invaders
    if my_state.scared_timer > 0:
      features['scared'] = 1
      # Find invaders and flee
      enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
      invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
      if len(invaders) > 0:
        dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
        # Positive distance with positive weight = maximize distance (flee)
        features['flee_distance'] = min(dists)
      if action == Directions.STOP: features['stop'] = 1
      return features

    # DETERMINE MOST LIKELY ENEMY POSITIONS
    enemy_positions = []
    for opp_index in self.opponents:
        most_likely_pos = self.filters[opp_index].get_most_likely_position()
        enemy_positions.append((opp_index, most_likely_pos))

    # Check for invaders (enemies on our side)
    invaders = []
    for idx, pos in enemy_positions:
        is_red = game_state.is_on_red_team(self.index)
        width = game_state.data.layout.width
        mid_x = width // 2
        
        is_invader = False
        if is_red and pos[0] < mid_x: is_invader = True
        if not is_red and pos[0] >= mid_x: is_invader = True
        
        if is_invader:
            invaders.append(pos)

    features['num_invaders'] = len(invaders)

    if len(invaders) > 0:
      # If food was just eaten, prioritize going to that location
      if hasattr(self, 'eaten_food') and len(self.eaten_food) > 0:
        eaten_dists = [self.get_maze_distance(my_pos, food) for food in self.eaten_food]
        features['distance_to_eaten_food'] = min(eaten_dists)
        features['food_just_eaten'] = len(self.eaten_food)
      
      # Intercept the closest estimated invader
      dists = [self.get_maze_distance(my_pos, pos) for pos in invaders]
      features['invader_distance'] = min(dists)
      
      # Food protection - prioritize if food is being eaten
      food_defending = self.get_food_you_are_defending(successor).as_list()
      current_food_count = len(food_defending)
      if current_food_count < self.initial_food_count:
        features['food_being_eaten'] = self.initial_food_count - current_food_count
    else:
      # Dynamic patrol with zone coordination
      food_defending = self.get_food_you_are_defending(successor).as_list()
      
      if len(food_defending) > 0:
        # Check if teammate is also defending (offensive agent defending due to lead)
        teammate_state = game_state.get_agent_state(self.team_mate_index) if self.team_mate_index else None
        both_defending = teammate_state and not teammate_state.is_pacman
        
        if both_defending:
          # Coordinate zones - defensive agent takes bottom, offensive takes top
          height = game_state.data.layout.height
          food_zone = [f for f in food_defending if f[1] < height // 2]
          
          if len(food_zone) == 0:
            food_zone = food_defending
          
          avg_y = sum(f[1] for f in food_zone) / len(food_zone)
        else:
          # Solo defense - cover all food
          avg_y = sum(f[1] for f in food_defending) / len(food_defending)
        
        # Patrol near border on our side
        width = game_state.data.layout.width
        border_x = (width // 2) - 1 if self.red else (width // 2)
        
        # Use chokepoints if available in our zone
        if len(self.chokepoints) > 0:
          zone_chokes = [c for c in self.chokepoints if (not both_defending or c[1] < height // 2)]
          # Filter out chokepoints too close to spawn
          zone_chokes = [c for c in zone_chokes if self.get_maze_distance(c, self.start) >= 3]
          if len(zone_chokes) > 0:
            choke_dists = [abs(c[0] - border_x) + abs(c[1] - avg_y) for c in zone_chokes]
            best_choke = zone_chokes[choke_dists.index(min(choke_dists))]
            features['patrol_distance'] = self.get_maze_distance(my_pos, best_choke)
            features['guarding_chokepoint'] = 1
          else:
            patrol_point = (border_x, int(avg_y))
            if game_state.has_wall(patrol_point[0], patrol_point[1]):
              legal_near = min(self.legal_positions, 
                             key=lambda p: abs(p[0]-border_x) + abs(p[1]-avg_y))
              patrol_point = legal_near
            features['patrol_distance'] = self.get_maze_distance(my_pos, patrol_point)
        else:
          # Patrol near zone border center
          patrol_point = (border_x, int(avg_y))
          if game_state.has_wall(patrol_point[0], patrol_point[1]):
            legal_near = min(self.legal_positions, 
                           key=lambda p: abs(p[0]-border_x) + abs(p[1]-avg_y))
            patrol_point = legal_near
          features['patrol_distance'] = self.get_maze_distance(my_pos, patrol_point)
      else:
        # Fallback to midline patrol
        map_width = game_state.data.layout.width
        map_height = game_state.data.layout.height
        mid_x = map_width // 2
        if self.red: mid_x -= 1
        
        # Find a safe patrol point near midline
        patrol_point = None
        for offset in range(map_height):
          for y in [map_height // 2 + offset, map_height // 2 - offset]:
            if 0 <= y < map_height and not game_state.has_wall(mid_x, y):
              patrol_point = (mid_x, y)
              break
          if patrol_point:
            break
        
        # Ultimate fallback - just stay near our starting position
        if patrol_point is None:
          patrol_point = self.start
        
        features['patrol_distance'] = self.get_maze_distance(my_pos, patrol_point)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def get_weights(self, game_state, action):
    return {
      'num_invaders': -1000,
      'on_defense': 100,
      'invader_distance': -10,
      'distance_to_eaten_food': -50,
      'food_just_eaten': -300,
      'food_being_eaten': -500,
      'patrol_distance': -5,
      'guarding_chokepoint': 50,
      'scared': -5000,
      'flee_distance': 200,
      'stop': -500,
      'reverse': -2
    }
