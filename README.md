# Pacman Agent

This project consists of the development of multiple multi-agent approaches to coordinate two Pac-Man players in a Capture The Flag (CTF) competition.
The strategies encompass multiple reflex and heuristic-search-based approaches, mostly divided into defensive and offensive strategies.


## Setting up the environment
1. Copy or clone the code from this framework to create your Pacman Agent, e.g., `git clone git@github.com:aig-upf/pacman-agent.git`
2. Go into pacman-agent folder, `cd pacman-agent/`
3. Run `git submodule update --init --remote` to pull the last pacman-contest
4. Create a virtual environment, e.g., `python3.8 -m venv venv`
5. Activate the virtual environment with `source venv/bin/activate`
6. Go to the pacman-contest folder and install the requirements:
    - `cd pacman-contest/`
    - `pip install -r requirements.txt`
    - `pip install -e .`

## Running a game
To run a game between the `baseline_team` and the current agent:
1. `cd pacman-contest/src/contest/`
2. `python capture.py -r baseline_team -b ../../../my_team.py`


## Coding a new agent
To create a new agent:
1. Create a new .py file
2. Following the structure of the `my_team.py` file, create a class in the new file with the name of your agent that inherits from `CaptureAgent`, e.g. `class ReflexCaptureAgent(CaptureAgent):`
2. In the new class, override the `def choose_action(self, game_state):` function to return the best next action (check the given source code example).
3. (Optional) Add any other functions to the class for reasoning / learning and improving your agents decision which could also use other code sources in the same folder.


