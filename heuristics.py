# Definition of the different heuristic functions being implemented

import contest.util as util


def distance_to_enemy_in_own_side(game_state, agent_index, distancer):
    position = game_state.get_agent_position(agent_index)

    width = game_state.data.layout.width
    halfway = width // 2
    is_red = game_state.is_on_red_team(agent_index)

    # Get opponent indices
    enemies = game_state.get_blue_team_indices() if is_red else game_state.get_red_team_indices()

    # Keep only enemies in my side
    enemies_in_my_side = []
    for e in enemies:
        p = game_state.get_agent_position(e)
        if p is None:
            continue
        if (is_red and p[0] < halfway) or (not is_red and p[0] >= halfway):
            enemies_in_my_side.append(p)

    # Compute maze distances
    if not enemies_in_my_side:
        return 0

    distances = [distancer.get_distance(position, position_enemy) for position_enemy in enemies_in_my_side]
    return min(distances)



def distance_to_own_side(game_state, agent_index, distancer):
    width = game_state.data.layout.width
    halfway = width // 2
    walls = game_state.get_walls()
    
    position = game_state.get_agent_position(agent_index)
    is_red = game_state.is_on_red_team(agent_index)

    # Select the border x-coordinate for your home side
    border_x = halfway - 1 if is_red else halfway

    border_tiles = []
    height = walls.height

    # Collect all non-wall border tiles
    for y in range(height):
        if not walls[border_x][y]:
            border_tiles.append((border_x, y))

    distances = [distancer.get_distance(position, t) for t in border_tiles]

    return min(distances)


def distance_to_closest_food(game_state, agent_index, distancer):
    if game_state.is_on_red_team(agent_index):
        food_list = game_state.get_blue_food().as_list()
    else:
        food_list = game_state.get_red_food().as_list()

    # Compute distance to the nearest food
    if len(food_list) > 0:  # This should always be True,  but better safe than sorry
        my_pos = game_state.get_agent_state(agent_index).get_position()
        min_distance = min([distancer.get_distance(my_pos, food) for food in food_list])

        return min_distance
    return float('inf')


def basic_attack_heursitic(game_state, agent_index):
    # Compute the features used in the heuristic
    features = util.Counter()

    if game_state.is_on_red_team(agent_index):
        food_list = game_state.get_blue_food().as_list()
    else:
        food_list = game_state.get_red_food().as_list()

    features['successor_score'] = -len(food_list)  # self.getScore(successor)

    # Compute distance to the nearest food
    if len(food_list) > 0:  # This should always be True,  but better safe than sorry
        my_pos = game_state.get_agent_state(agent_index).get_position()

        # REVIEW (do we need to do this each time?)
        distancer = distance_calculator.Distancer(game_state.data.layout)
        distancer.get_maze_distances()
        min_distance = min([distancer.get_distance(my_pos, food) for food in food_list])

        features['distance_to_food'] = min_distance

    
    # Compute the weights
    weights = {'successor_score': 100, 'distance_to_food': -1}

    return features * weights

"""
def basic_defense_heuristic(game_state, agent_index):
    # Compute the features used in the heuristic
    features = util.Counter()
    successor = game_state

    my_state = game_state.get_agent_state(agent_index)
    my_pos = my_state.get_position()

    # Computes whether we're on defense (1) or offense (0)
    features['on_defense'] = 1
    if my_state.is_pacman: features['on_defense'] = 0

    # Computes distance to invaders we can see

    if game_state.is_on_red_team(agent_index):
        oponents = game_state.get_blue_team_indices()
    else:
        oponents = game_state.get_red_team_indices()

    enemies = [game_state.get_agent_state(i) for i in oponents]
    invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
    features['num_invaders'] = len(invaders)

    if len(invaders) > 0:
        distancer = distance_calculator.Distancer(game_state.data.layout)
        distancer.get_maze_distances()    
        dists = [distancer.get_distance(my_pos, a.get_position()) for a in invaders]
        features['invader_distance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

"""