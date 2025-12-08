# Definition of the different search functions being implemented
import heapq
import itertools



def backtrack(node, from_node):
    solution = []

    current = node
    while current:
        solution.append(current)
        current = from_node[current]

    solution.reverse()
    return solution

def backtrack_actions(node, from_node, action_map):
    actions = []

    current = node
    # Keep iterating while there is a parent node to the 'current' state
    while from_node[current] is not None:
        actions.append(action_map[current])
        current = from_node[current]

    actions.reverse()
    return actions

def a_star(game_state, agent_index, heuristic, goal, cost):
    """
    game_state: GameState instance representing the current state
    agent_index: index of the agent performing the search
    heuristic: callable that returns the heuristic value given a state

    returns a plan (array) of actions
    """

    # The frontier is a priority queue, implemented as a min-heap
    expanded_nodes = set()
    frontier = []
    counter = itertools.count()
    heapq.heappush(frontier, (0, next(counter), game_state))

    # Keep the cost from the initial state to the different nodes. Shape {GameState: int}
    costs = {game_state: 0}
    # Keep parents to perform backtracking once a solution is found
    from_node = {game_state: None}
    # Maps nodes to the actions that lead to it 
    action_map = {game_state: None}

    while frontier:

        # Choose the first value from the priority queue and remove it from the forntier
        _, _, node = heapq.heappop(frontier)

        # Skip if already expanded (to handle duplicates in heap)
        if node in expanded_nodes:
            continue

        # Add the selected node to the expanded nodes
        expanded_nodes.add(node)

        if goal.goal_condition(node):
            solution = backtrack_actions(node, from_node, action_map)
            return solution
        
        # Get all the possible actions starting from node
        actions = node.get_legal_actions(agent_index)

        # Add all possible successors states to the frontier
        for action in actions:
            successor = node.generate_successor(agent_index, action)                            

            # Only evaluate the successor if it has not already been expanded
            if successor not in expanded_nodes:
                c = costs[node] + cost(successor, agent_index)

                # Add the successor to the frontier
                if c < costs.get(successor, float('inf')):
                    costs[successor] = c
                    f = c + heuristic(successor, agent_index)
                    # Duplicated successors with different values are handled by the previous code
                    heapq.heappush(frontier, (f, next(counter), successor))
                    from_node[successor] = node
                    action_map[successor] = action

    # Return an empty list if no sequence of valid actions is found
    return []

