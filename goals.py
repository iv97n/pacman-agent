# Definition of different types of goals depending on the objective of the agent
from abc import ABC, abstractmethod
import random

class Goal(ABC):
    @abstractmethod
    def goal_condition(self, game_state):
        """
        Check if a game state satisfies the goal conditions
        """
        pass

class GetFoodGoal(Goal):
    def __init__(self, initial_state, agent_index):
        self.index = agent_index
        self.food_positions = self.get_food_position(initial_state, agent_index)

    def get_food_position(self, game_state, agent_index):
        if game_state.is_on_red_team(agent_index):
            food = game_state.get_blue_food()
        else:
            food = game_state.get_red_food()

        food_positions = set()

        width = food.width
        height = food.height

        for x in range(width):
            for y in range(height):
                if food[x][y]:
                    food_positions.add((x, y))

        return food_positions


    def goal_condition(self, game_state):
        # Check if the agent is in a food position
        if game_state.get_agent_position(self.index) in self.food_positions:
            return True
        return False
    
class GoBackGoal(Goal):
    def __init__(self, agent_index):
        self.index = agent_index

    def goal_condition(self, game_state):
        position = game_state.get_agent_position(self.index)

        width = game_state.data.layout.width
        halfway = width // 2

        if game_state.is_on_red_team(self.index):
            return position[0] < halfway  
        else:
            return position[0] >= halfway 
        
class ChaseAgentGoal(Goal):
    def __init__(self, initial_state, agent_index):
        self.index = agent_index
        self.enemy_positions = self.get_enemy_positions(initial_state, agent_index)

    def get_enemy_positions(self, game_state, agent_index):
        width = game_state.data.layout.width
        halfway = width // 2

        is_red = game_state.is_on_red_team(agent_index)
        enemies = game_state.get_blue_team_indices() if is_red else game_state.get_red_team_indices()

        enemy_positions = []
        for e in enemies:
            position_enemy = game_state.get_agent_position(e)
            if position_enemy is None:
                continue

            if (is_red and position_enemy[0] < halfway) or (not is_red and position_enemy[0] >= halfway):
                enemy_positions.append(position_enemy)

        return enemy_positions

    def goal_condition(self, game_state):
        if game_state.get_agent_position(self.index) in self.enemy_positions:
            return True
        return False
    
class PatrolGoal(Goal):
    """
    Goal for a defensive agent to patrol its own side.
    The agent moves toward one of several predefined patrol points.
    """

    def __init__(self, initial_state, agent_index):
        self.index = agent_index
        self.patrol_point = self.get_patrol_point(initial_state)

    def get_patrol_point(self, game_state):
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        halfway = width // 2

        is_red = game_state.is_on_red_team(self.index)

        # Generate a grid of points on the agent's side
        if is_red:
            points = [(x, y) for x in range(halfway) for y in range(height)
                      if not game_state.has_wall(x, y)]
        else:
            points = [(x, y) for x in range(halfway, width) for y in range(height)
                      if not game_state.has_wall(x, y)]

        return random.choice(points)
    
    def patrol_heuristic(self, game_state, agent_index, distancer):
        position = game_state.get_agent_position(agent_index)
        return distancer.get_distance(position, self.patrol_point)

    def goal_condition(self, game_state):
        if game_state.get_agent_position(self.index) == self.patrol_point:
            return True
        return False


    


"LOOK AT THIS"
def get_agent_position(self, index):
    """
    Returns a location tuple if the agent with the given index is observable;
    if the agent is unobservable, returns None.
    """
    agent_state = self.data.agent_states[index]
    ret = agent_state.get_position()
    if ret:
        return tuple(int(x) for x in ret)
    return ret