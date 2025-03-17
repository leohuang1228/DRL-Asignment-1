import gym
import numpy as np


class Addition_wall_env(gym.Wrapper):
    def __init__(self, fuel_limit=5000, grid_size: int = 5, obs_num: int = 5):
        self.grid_size = grid_size
        env = gym.make("Taxi-v3", render_mode="ansi")  # Taxi-v3 is **always 5x5**. If you want a different grid size, you must create a custom environment.
        super().__init__(env)

        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit

        self.stations = [(0, 0), (0, self.grid_size - 1), (self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)]
        self.passenger_loc = None
        self.passenger_picked_up = False
        self.obstacles = []
        self.obstacle_num = obs_num
        self.destination = None

    def set_obstacle(self):
        self.obstacles = [(np.random.randint(self.grid_size), np.random.randint(self.grid_size)) for _ in range(self.obstacle_num)]

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.current_fuel = self.fuel_limit

        taxi_row, taxi_col, pass_idx, dest_idx = self.env.unwrapped.decode(obs)

        taxi_row = min(taxi_row, self.grid_size - 1)
        taxi_col = min(taxi_col, self.grid_size - 1)
        self.passenger_loc = self.stations[pass_idx]
        self.destination = self.stations[dest_idx]
        self.set_obstacle()

        self.passenger_picked_up = False

        return self.get_state(), info

    def get_state(self):

        taxi_row, taxi_col, _, _ = self.env.unwrapped.decode(self.env.unwrapped.s)
        passenger_x, passenger_y = self.passenger_loc
        destination_x, destination_y = self.destination
        obstacle_north = int((taxi_row - 1, taxi_col) in self.obstacles or taxi_row == 0)
        obstacle_south = int((taxi_row + 1, taxi_col) in self.obstacles or taxi_row == self.grid_size - 1)
        obstacle_east = int((taxi_row, taxi_col + 1) in self.obstacles or taxi_col == self.grid_size - 1)
        obstacle_west = int((taxi_row, taxi_col - 1) in self.obstacles or taxi_col == 0)
        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle = int((taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle
        destination_loc_north = int((taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int((taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east = int((taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west = int((taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle = int((taxi_row, taxi_col) == self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle

        state = (taxi_row, taxi_col,
                 self.stations[0][0], self.stations[0][1],
                 self.stations[1][0], self.stations[1][1],
                 self.stations[2][0], self.stations[2][1],
                 self.stations[3][0], self.stations[3][1],
                 obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
        return state

    def step(self, action):
        """Perform an action and update the environment state."""
        taxi_row, taxi_col, pass_idx, dest_idx = self.env.unwrapped.decode(self.env.unwrapped.s)

        next_row, next_col = taxi_row, taxi_col
        if action == 0:  # Move South
            next_row += 1
        elif action == 1:  # Move North
            next_row -= 1
        elif action == 2:  # Move East
            next_col += 1
        elif action == 3:  # Move West
            next_col -= 1

        if action in [0, 1, 2, 3]:
            if not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward = -5
                self.current_fuel -= 1
                if self.current_fuel <= 0:
                    return self.get_state(), reward - 10, True, False, {}
                return self.get_state(), reward, False, False, {}

        taxi_row, taxi_col = next_row, next_col

        self.current_fuel -= 1
        obs, reward, terminated, truncated, info = super().step(action)

        if reward == 20:
            reward = 50
        elif reward == -1:
            reward = -0.1
        elif reward == -10:
            reward = -10

        if action == 4:
            if pass_idx == 4:
                self.passenger_picked_up = True
                self.passenger_loc = (taxi_row, taxi_col)
            else:
                self.passenger_picked_up = False

        elif action == 5:
            if self.passenger_picked_up:
                if (taxi_row, taxi_col) == self.destination:
                    reward += 50
                    return self.get_state(), reward - 0.1, True, {}, {}
                else:
                    reward -= 10
        if action in [4, 5]:
            reward -= 10

        if self.passenger_picked_up:
            self.passenger_loc = (taxi_row, taxi_col)
        if self.current_fuel <= 0:
            return self.get_state(), reward - 10, True, False, {}
        return self.get_state(), reward, False, truncated, info

    def get_action_name(self, action):
        """Returns a human-readable action name."""
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"
