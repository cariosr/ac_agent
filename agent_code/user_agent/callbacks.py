import numpy as np
from random import shuffle
from time import time, sleep
from collections import deque
# from tempfile import TemporaryFile
import os.path
import pickle
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from decimal import Decimal

from settings import s, e

#Simple agent
def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x,y) for (x,y) in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if free_space[x,y]]
        #shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]

def get_free_cells(self, xa, ya, arena, bomb_dic, explosion_map, time):
    queue = deque([(xa, ya)])
    visited = {}
    visited[(xa, ya)] = 1
    free = 0
    while (len(queue) > 0):
        curr_x, curr_y = queue.popleft()
        curr = (curr_x, curr_y)
        directions = [(curr_x, curr_y - 1), (curr_x, curr_y + 1), (curr_x - 1, curr_y), (curr_x + 1, curr_y)]

        if (visited[curr]) == time:
            break

        for (xd, yd) in directions:
            d = (xd, yd)
            if ((arena[d] == 0) and
                    (explosion_map[curr] <= visited[curr] + 1) and
                    (not d in bomb_dic or not (visited[curr] + 1) in bomb_dic[d]) and
                    (not d in visited)):
                queue.append(d)
                visited[d] = visited[curr] + 1
                if not d in bomb_dic:
                    free += 1

    return free

def sel_action_target(self,targets):

    arena = self.game_state['arena']
    x, y, _, bombs_left, score = self.game_state['self']
    bombs = self.game_state['bombs']
    bomb_xys = [(x,y) for (x,y,t) in bombs]

    coins = self.game_state['coins']

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]
    # Take a step towards the most immediately interesting target
    free_space = arena == 0

    # if self.ignore_others_timer > 0:
    #     for o in others:
    #         free_space[o] = False
    #
    d = look_for_targets(free_space, (x,y), targets, self.logger)
    action = 'WAIT'
    if d == (x,y-1): action = 'UP'
    if d == (x,y+1): action ='DOWN'
    if d == (x-1,y): action = 'LEFT'
    if d == (x+1,y): action = 'RIGHT'

    return action

def mappping(self):
    # State definition
    #arena = np.zeros((s.rows,s.cols))
    # Gather information about the game state
    arena = self.game_state['arena']
    # aux_arena = np.zeros((s.rows, s.cols))
    # aux_arena[:,:] = arena
    x, y, _, bombs_left, score = self.game_state['self']
    bombs = self.game_state['bombs']
    bombs_xy = [(x, y) for (x, y,t) in bombs]
    bomb_dic = {}
    # others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
    coins = self.game_state['coins']
    explosion_map = self.game_state['explosions']

    bomb_map_timer = np.zeros(arena.shape)

    # dictionary of bombs
    for (xb, yb, t) in bombs:
        # when other agent mark arena as well
        vec_dir = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for v in vec_dir:
            hx, hy = v
            for i in range(4):
                xcoord = xb + hx * i
                ycoord = yb + hy * i
                if ((0 < xcoord < arena.shape[0]) and
                        (0 < ycoord < arena.shape[1]) and
                        (arena[(xcoord, ycoord)] == 1 or arena[(xcoord, ycoord)] == 0)):
                    bomb_map_timer[(xcoord, ycoord)] = t
                    if (xcoord, ycoord) in bomb_dic:
                        bomb_dic[(xcoord, ycoord)].append(t)
                    else:
                        bomb_dic[(xcoord, ycoord)] = [t]
                else:
                    break

    # map of bombs
    bomb_map = np.zeros(arena.shape)

    for (xb, yb, t) in bombs:
        vec_dir = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        if t==0:
            continue
        for v in vec_dir:
            hx, hy = v
            for i in range(0, 4 - t + 1):
                xcoord = xb + hx * i
                ycoord = yb + hy * i
                if ((0 < xcoord < arena.shape[0]) and
                        (0 < ycoord < arena.shape[1]) and
                        (arena[(xcoord, ycoord)] == 1 or arena[(xcoord, ycoord)] == 0)):
                    bomb_map[(xcoord, ycoord)] = 1
                else:
                    break

    self.dead_zone = bomb_map

    # General case
    # state = np.zeros(32, dtype = int)

    # Coins case
    state = np.zeros(self.state_size, dtype = int)

    # 0. UP ->    (x  , y-1)
    # 1. DOWN ->  (x  , y+1)
    # 2. LEFT ->  (x-1, y  )
    # 3. RIGHT -> (x+1, y  )

    # 4 bits for valid position
    valid = np.array([0, 0, 0, 0])

    directions = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
    for i in range(4):
        d = directions[i]
        if (arena[d] == 0 and explosion_map[d] <= 1 and d not in bombs_xy):
            valid[i] = 1

    state[:4] = valid

    # 1 bit  for flag bomb
    state[4] = bombs_left


    #Building the target state..
    # posibles targets
    # others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
    coins = self.game_state['coins']
#    targets = coins + dead_ends + crates
    bit_action = sel_action_target(self, coins)

    if (bit_action != 'WAIT'):
        idx_bit = self.map_actions[bit_action]
        state[5 + idx_bit] = 1
    else:
        crates = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 1)]
        bit_action = sel_action_target(self, crates)
        if (bit_action != 'WAIT'):
            idx_bit = self.map_actions[bit_action]
            state[5 + idx_bit] = 1


    # Compile a list of 'targets' the agent should head towards
    # dead_ends = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 0)
    #                 and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(0) == 1)]
    # crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]

    #def sel_action_target(self, targets):


    # Scape route
    if (bomb_map_timer[x, y] > 0):
        free_max = 0
        for i in range(4):
            x_curr, y_curr = directions[i]
            d = directions[i]

            if ((arena[d] == 0) and
                    (explosion_map[d] <= 1)):
                time = bomb_map_timer[x, y]
                free_curr = get_free_cells(self, x_curr, y_curr, arena, bomb_dic, explosion_map, time)
                if free_curr > free_max:
                    free_max = free_curr
                    idx_direction = i
        if free_max > 0:
            state[9 + idx_direction] = 1

    # Number of crates

    number_crates = 0
    vec_dir = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    for v in vec_dir:
        hx, hy = v
        for i in range(1, 4):
            xcoord = x + hx * i
            ycoord = y + hy * i
            if ((0 < xcoord < arena.shape[0]) and
                    (0 < ycoord < arena.shape[1]) and
                    (arena[(xcoord, ycoord)] == 1 or arena[(xcoord, ycoord)] == 0)):
                if arena[(xcoord, ycoord)] == 1:
                    number_crates += 1
            else:
                break


    if number_crates < 10:
        state[13] = number_crates
    else:
        state[13] = 9

    # when is next to the crate or opponent (State coins all zeros) activate
    if not state[5:13].any() and state[4] == 1:
        state[14] = 1
    #state[14] = flag_fit_to_bomb(self)

    #state[13] = number_crates

    # self.logger.debug(f'STATE VALID: {state[:4]}')
    # self.logger.debug(f'STATE BOMB: {state[4]}')
    # self.logger.debug(f'STATE COINS: {state[5:9]}')
    # self.logger.debug(f'STATE SCAPE: {state[9:13]}')
    # self.logger.debug(f'STATE CRATES: {state[13]}')

    print("STATE VALID: ",state[:4])
    print("STATE BOMB: ",state[4])
    print("STATE COINS: ",state[5:9])
    print("STATE SCAPE: ",state[9:13])
    print("STATE CRATES: ",state[13])
    print("STATE DROP BOMB: ", state[14], "\n")

    return state

def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('Successfully entered setup code')
    np.random.seed()

    # ACTIONS

    # Init the 6 possible actions
    # 0. UP ->    (x  , y-1)
    # 1. DOWN ->  (x  , y+1)
    # 2. LEFT ->  (x-1, y  )
    # 3. RIGHT -> (x+1, y  )
    # 4. WAIT ->  (x  , y  )
    # 5. BOMB ->  (x  , y  )

    self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

    # Init map of actions
    self.map_actions = {
        'UP': 0,
        'DOWN': 1,
        'LEFT': 2,
        'RIGHT': 3,
        'WAIT': 4,
        'BOMB': 5
    }

    # Number of possible actions
    self.num_actions = 6

    # action index (this is really the variable used as action)
    self.idx_action = 4

    # STATE

    # state size
    self.state_size = 15

    self.state = np.zeros(self.state_size)

    # next_state defined as state
    self.next_state = np.zeros(self.state_size)



    # NN for playing
    # self.model = load_model_NN(self)

    # REWARDS

    # Reward accumulated for every 4 frames
    self.total_reward = 0

    # Reward List

    number_of_free = (s.cols - 2) * (s.rows // 2) + (s.rows // 2 - 1) * ((s.cols - 2) // 2 + 1)
    number_of_crates = s.crate_density * (number_of_free - 12)
    reward_coin = 100
    if (number_of_crates > 0):
        ratio_coins_crates = number_of_crates / 9
        reward_crate = int(reward_coin / ratio_coins_crates)
    else:
        reward_crate = 0

    self.reward_list = {
        'OPPONENT_ELIMINATED': 500,
        'COIN_COLLECTED': reward_coin,
        'CRATE_DESTROYED': reward_crate,
        'INVALID_ACTION': -8,
        'VALID': -2,
        'DIE': -1500
    }


def act(self):
    # Gather information about the game state
    print("\n",mappping(self),"\n")

    self.logger.info('Pick action according to pressed key')
    self.next_action = self.game_state['user_input']

def reward_update(self):

    if e.CRATE_DESTROYED in self.events:
        NCrates =  list(self.events).count(9)
        print("Crates Destroyed: \n", NCrates)
    pass

def learn(self):
    if e.CRATE_DESTROYED in self.events:
        print("Events: \n",self.events)
    pass

    pass
