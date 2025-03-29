import pygame
import sys
import random
import os
import numpy as np
from qlearning import QLearningAgent
import time

class MazeEnv:
    def __init__(self, board_number=1):
        pygame.init()
        # Constants for drawing and grid
        self.GRID_SIZE = 60
        self.LINE_WIDTH = 3
        self.WHITE = (255, 255, 255)
        self.GRAY = (200, 200, 200)
        self.BLACK = (0, 0, 0)

        # For simplicity, we fix the board dimensions.
        self.num_rows = 10
        self.num_cols = 10
        
        self.screen = pygame.display.set_mode((self.num_cols * self.GRID_SIZE, self.num_rows * self.GRID_SIZE))

        # Set the grass path before generating the board
        self.grass_path = os.path.join("assets", "background_images", "grass_patch_1.png")

        # Create the Pygame display sized to our board
        pygame.display.set_caption("Zombie Survival Game")

        # Generate the maze board (each cell holds a background and wall info)
        self.grid = self._generate_board()

        # Load background image and store it in background_images
        self.background_images = {}
        try:
            bg_image = pygame.image.load(self.grass_path)
            bg_image = pygame.transform.scale(bg_image, (self.GRID_SIZE, self.GRID_SIZE))
        except pygame.error as e:
            bg_image = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE))
            bg_image.fill((100, 200, 100))
        self.background_images[self.grass_path] = bg_image

        # Load human image 
        try:
            self.human_image = pygame.image.load(os.path.join("assets", "human_images", "human_1.png"))
            self.human_image = pygame.transform.scale(self.human_image, (self.GRID_SIZE, self.GRID_SIZE))
        except pygame.error as e:
            self.human_image = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
            self.human_image.fill((255, 0, 0))

        # Load zombie image
        try:
            self.zombie_image = pygame.image.load(os.path.join("assets", "zombie_images", "zombie_1.png"))
            self.zombie_image = pygame.transform.scale(self.zombie_image, (self.GRID_SIZE, self.GRID_SIZE))
        except pygame.error as e:
            self.zombie_image = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
            self.zombie_image.fill((0, 255, 0))

        # Load human dead image
        try:
            self.human_dead_image = pygame.image.load(os.path.join("assets", "human_images", "human_dead.png"))
            self.human_dead_image = pygame.transform.scale(self.human_dead_image, (self.GRID_SIZE, self.GRID_SIZE))
        except pygame.error as e:
            self.human_dead_image = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
            self.human_dead_image.fill((150, 0, 0))

        # Load a list of zombie images for animation (if available)
        self.zombie_images = [self.zombie_image]  # or load multiple images as needed
        self.frame_count = 0

        # Qlearning agent
        self.reset()
        state_size = len(self._get_obs())
        action_size = 4
        self.z_agent = QLearningAgent(state_size, action_size)

        # initialize the human and zombie positions
        self.reset()

    def reset(self):  
        self.human_pos = (self.num_rows - 1, 0)
        self.zombie_pos = (0, self.num_cols - 1)
        self.steps_taken = 0

    def _generate_board(self):
        """
        Generate a maze board as a grid of dictionaries.
        Each cell has:
            'background': path to background image,
            'walls': [top, right, bottom, left] (True means wall exists)
        We use a DFS recursive backtracker to remove walls between cells.
        """
        # Initialize grid with all walls intact.
        grid = [[{'background': self.grass_path, 'walls': [True, True, True, True]}
                 for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        visited = [[False for _ in range(self.num_cols)] for _ in range(self.num_rows)]

        def in_bounds(r, c):
            return 0 <= r < self.num_rows and 0 <= c < self.num_cols

        # Directions: (dr, dc, wall index in current, wall index in neighbor)
        directions = [(-1, 0, 0, 2),  # Up: remove top wall of current, bottom wall of neighbor
                      (0, 1, 1, 3),   # Right: remove right wall, left wall of neighbor
                      (1, 0, 2, 0),   # Down: remove bottom wall, top wall of neighbor
                      (0, -1, 3, 1)]  # Left: remove left wall, right wall of neighbor

        # Start DFS at a random cell
        start_r = random.randrange(self.num_rows)
        start_c = random.randrange(self.num_cols)
        stack = [(start_r, start_c)]
        visited[start_r][start_c] = True

        while stack:
            r, c = stack[-1]
            # Find unvisited neighbors
            neighbors = []
            for dr, dc, wall_idx, opp_wall_idx in directions:
                nr, nc = r + dr, c + dc
                if in_bounds(nr, nc) and not visited[nr][nc]:
                    neighbors.append((nr, nc, wall_idx, opp_wall_idx))
            if neighbors:
                nr, nc, wall_idx, opp_wall_idx = random.choice(neighbors)
                # Remove wall between current and neighbor
                grid[r][c]['walls'][wall_idx] = False
                grid[nr][nc]['walls'][opp_wall_idx] = False
                visited[nr][nc] = True
                stack.append((nr, nc))
            else:
                stack.pop()

        return grid

    def _get_obs(self):
        # example state: [zombie_row, zombie_col, human_row, human_col]
        hr, hc = self.human_pos
        zr, zc = self.zombie_pos
        return np.array([hr,hc,zr,zc], dtype=np.float32)
    
    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_new_position(self, position, action):
        """
        Given a grid position (row, col) and an action (0: up, 1: right, 2: down, 3: left),
        return the new position if no wall blocks the movement.
        """
        row, col = position
        walls = self.grid[row][col]['walls']

        if action == 0 and row > 0 and not walls[0]:
            row -= 1
        elif action == 1 and col < self.num_cols - 1 and not walls[1]:
            col += 1
        elif action == 2 and row < self.num_rows - 1 and not walls[2]:
            row += 1
        elif action == 3 and col > 0 and not walls[3]:
            col -= 1
        return (row, col)

    def step(self, zombie_action, human_action):
        """ 
        Move both the human and zombie. Then comute the reqard for the zombie
        Return next_stat, reqard, done, info
        """
        # move human
        if human_action is not None:
            self.human_pos = self._get_new_position(self.human_pos, human_action)

        # move zombie 
        prev_distance = self._manhattan_distance(self.zombie_pos, self.human_pos)
        self.zombie_pos = self._get_new_position(self.zombie_pos, zombie_action)

        self.steps_taken += 1
        new_distance = self._manhattan_distance(self.zombie_pos, self.human_pos)
        max_distance = self.num_rows + self.num_cols - 2
        distance_ratio = new_distance / max_distance
        time_penalty = -0.1 * (self.steps_taken / 100)

        # Reward system
        old_distance = prev_distance
        new_distance = self._manhattan_distance(self.zombie_pos, self.human_pos)
        
        if self.zombie_pos == self.human_pos:
            reward = +100.0
        else:
            if new_distance < old_distance:
                reward = +2.0
            elif new_distance > old_distance:
                reward = -2.0
            else:
                reward = -1.0

        done = (self.zombie_pos == self.human_pos)
        info = {
            'distance_to_human': new_distance,
            'human_pos': self.human_pos,
            'zombie_pos': self.zombie_pos,
        }

        # return next_state, reward. done, info, for Qlearning 
        return self._get_obs(), reward, done, info
    
    def single_update(self):
        """
        Called for each frame of run()
        1. Get user input for human
        2. Qlearning agent picks action for zombie
        3. Call self.step() then agent.update()
        """
        # pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
        keys = pygame.key.get_pressed()
        human_action = None
        if keys[pygame.K_UP]:
            human_action = 0
        elif keys[pygame.K_RIGHT]:
            human_action = 1
        elif keys[pygame.K_DOWN]:
            human_action = 2
        elif keys[pygame.K_LEFT]:
            human_action = 3

        current_state = self._get_obs()
        zombie_action = self.z_agent.get_action(current_state)
        next_state, reward, done, info = self.step(zombie_action, human_action)
        self.z_agent.update(current_state, zombie_action, reward, next_state, done)

        return done
    
    def render_frame(self):
        """Redraw the background and the characters"""
        self.screen.fill(self.WHITE)
        self._draw_maze()
        self._draw_characters()
        pygame.display.flip()

    def _draw_maze(self):
        font = pygame.font.Font(None, 24)
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                x = col * self.GRID_SIZE
                y = row * self.GRID_SIZE
                bg_path = self.grid[row][col]['background']
                bg_image = self.background_images[bg_path]
                self.screen.blit(bg_image, (x, y))

                # Draw walls as thin black lines if they exist
                walls = self.grid[row][col]['walls']
                if walls[0]:  # Top wall
                    pygame.draw.line(self.screen, self.BLACK, (x, y), (x + self.GRID_SIZE, y), self.LINE_WIDTH)
                if walls[1]:  # Right wall
                    pygame.draw.line(self.screen, self.BLACK, (x + self.GRID_SIZE, y), (x + self.GRID_SIZE, y + self.GRID_SIZE), self.LINE_WIDTH)
                if walls[2]:  # Bottom wall
                    pygame.draw.line(self.screen, self.BLACK, (x, y + self.GRID_SIZE), (x + self.GRID_SIZE, y + self.GRID_SIZE), self.LINE_WIDTH)
                if walls[3]:  # Left wall
                    pygame.draw.line(self.screen, self.BLACK, (x, y), (x, y + self.GRID_SIZE), self.LINE_WIDTH)
                # Uncomment the following lines for coordinate debugging:
                """
                text_surface = font.render(f"({row},{col})", True, self.GRAY)
                text_rect = text_surface.get_rect(center=(x + self.GRID_SIZE // 2, y + self.GRID_SIZE // 2))
                self.screen.blit(text_surface, text_rect)
                """

    def run_one_round(self, round_duration):
        """
        Run a single round for up to 'round_duration' seconds or until the zombie catches the human.
        Returns True if the zombie caught the human, else False if time ran out
        """
        self.grid = self._generate_board()
        self.reset()
        start_time = time.time()
        clock = pygame.time.Clock()

        caught_human = False
        while True:
            clock.tick(10)
            done = self.single_update()
            self.render_frame()
            
            elapsed = time.time() - start_time 
            if done:
                caught_human = True
                break
            if elapsed >= round_duration:
                break
        return caught_human

    def _get_neighbors(self, row, col):
        """
        Returns neighboring cell coordinates (row, col) that are accessible (i.e., no wall in between).
        """
        neighbors = []
        walls = self.grid[row][col]['walls']
        if row > 0 and not walls[0]:
            neighbors.append((row - 1, col))
        if col < self.num_cols - 1 and not walls[1]:
            neighbors.append((row, col + 1))
        if row < self.num_rows - 1 and not walls[2]:
            neighbors.append((row + 1, col))
        if col > 0 and not walls[3]:
            neighbors.append((row, col - 1))
        return neighbors

    def _draw_characters(self):
        # Draw human and zombie at their grid positions
        human_x = self.human_pos[1] * self.GRID_SIZE
        human_y = self.human_pos[0] * self.GRID_SIZE
        zombie_x = self.zombie_pos[1] * self.GRID_SIZE
        zombie_y = self.zombie_pos[0] * self.GRID_SIZE
        self.screen.blit(self.human_image, (human_x, human_y))
        self.screen.blit(self.zombie_image, (zombie_x, zombie_y))

    def update(self):
        """
        called each frame from run()
        This will do exactly ONE environment step here
        1. Get user input
        2. Let afent pick an action for zombie
        3. call self.step() to get (next_state, reqard, done, info)
        4. Then agent.update() to train the agent
        """

        # process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

        # Get Human Action via arrow keys
        human_action = None
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            human_action = 0
        elif keys[pygame.K_RIGHT]:
            human_action = 1
        elif keys[pygame.K_DOWN]:
            human_action = 2
        elif keys[pygame.K_LEFT]:
            human_action = 3

        # Get Zombie action from QLearning Agent
        current_state = self._get_obs()
        zombie_action = self.z_agent.get_action(current_state)

        # Environment step: move both agents then compute reward
        next_state, reward, done, info = self.step(zombie_action, human_action)

        # Update Qlearning agent
        self.z_agent.update(current_state, zombie_action, reward, next_state, done)

        if self.zombie_pos == self.human_pos:
            self._game_over_screen()
            return True

    def run(self):
        clock = pygame.time.Clock()
        while True:
            clock.tick(10)  # Set FPS (grid-based movement, so lower FPS is acceptable)
            self.screen.fill(self.WHITE)
            self._draw_maze()
            round_over = self.update()
            self._draw_characters()
            pygame.display.flip()

            if round_over:
                return
    
    def _find_zombie_escape_path(self, human_row, human_col):
        # Choose the corner farthest from the human as the escape target.
        corners = [
            (0, 0),
            (0, self.num_cols - 1),
            (self.num_rows - 1, 0),
            (self.num_rows - 1, self.num_cols - 1)
        ]
        escape_corner = max(corners, key=lambda corner: self._manhattan_distance(corner, (human_row, human_col)))
        # Use A* to compute a path from the zombie’s current position to the escape corner.
        path = self._a_star(self.zombie_pos, escape_corner)
        return path if path else [self.zombie_pos]

    def _a_star(self, start, goal):
        import heapq
        open_set = []
        heapq.heappush(open_set, (self._manhattan_distance(start, goal), 0, start))
        came_from = {}
        g_score = {start: 0}
        closed_set = set()

        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            if current == goal:
                # Reconstruct the path from start to goal.
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            if current in closed_set:
                continue
            closed_set.add(current)

            for neighbor in self._get_neighbors(current[0], current[1]):
                tentative_g = g_score[current] + 1
                if neighbor in closed_set:
                    continue
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._manhattan_distance(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        return None

    def _draw_character(self, row, col, image): 
        x = col * self.GRID_SIZE
        y = row * self.GRID_SIZE
        self.screen.blit(image, (x, y))
    
    def _game_over_screen(self):
        # Immediately render and display "Game Over" text.
        human_row, human_col = self.human_pos
        font = pygame.font.Font(None, 74)
        game_over_text = font.render("Game Over", True, self.BLACK)
        text_rect = game_over_text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2))
        
        # Draw the final state with the human dead and display the message.
        self.screen.fill(self.WHITE)
        self._draw_maze()
        self._draw_character(human_row, human_col, self.human_dead_image)
        # Draw the zombie at its current position
        self._draw_character(self.zombie_pos[0], self.zombie_pos[1], self.zombie_images[self.frame_count % len(self.zombie_images)])
        # Immediately display the Game Over message.
        self.screen.blit(game_over_text, text_rect)
        pygame.display.flip()
        
        # Animate the zombie moving away from the human until it reaches the escape corner.
        escape_path = self._find_zombie_escape_path(human_row, human_col)
        for (z_row, z_col) in escape_path:
            self.screen.fill(self.WHITE)
            self._draw_maze()
            # Keep the human dead image in place.
            self._draw_character(human_row, human_col, self.human_dead_image)
            # Draw the zombie at its new position.
            self._draw_character(z_row, z_col, self.zombie_images[self.frame_count % len(self.zombie_images)])
            # Keep the Game Over message visible.
            self.screen.blit(game_over_text, text_rect)
            pygame.display.flip()
            pygame.time.Clock().tick(5)

def loading_screen(screen, clock):
    screen.fill((0, 0, 0))
    font = pygame.font.Font(None, 74)
    loading_text = font.render("Loading...", True, (255, 255, 255))
    loading_rect = loading_text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
    screen.blit(loading_text, loading_rect)
    pygame.display.flip()
    pygame.time.delay(2000)

def show_about(screen,clock):
    about_lines = [
        "Zombie Survival Game",
        "Created by Jonah Pattison & Christopher Slogget",
        "use the arrow keys to move",
        "See if you can escape the zombie at all 10 levels",
        "Good Luck!",
        "Press ESC to return to the main menu"
    ]
    font = pygame.font.Font(None, 40)
    
    while True:
        screen.fill((0, 0, 0))
        for i, line in enumerate(about_lines):
            text = font.render(line, True, (255, 255, 255))
            text_rect = text.get_rect(center=(screen.get_width() // 2, 100 + i * 50))
            screen.blit(text, text_rect)
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return  # Return to the main menu
        clock.tick(30)

def start_session():
    """ 
    Run unlimited rounds of MazeEnv. Each round 90 seconds or until Zombie catches the human
    """
    env = MazeEnv()

    level = 5000
    model_path = f"training_models/zombie_agent_{level}.pkl"
    if os.path.exists(model_path):
        env.z_agent.load(model_path)
        print("Zombie Q-table loaded from zombie_agent.pkl.")

    round_num = 1

    while True:
        print(f"\nStarting Round {round_num} ...")
        caught_human = env.run_one_round(round_duration=90)

        if caught_human:
            print("Zombie caught the human!")
        else:
            print("Time is up! Human survived!")

        env.z_agent.save("zombie_agent.pkl")
        print("Zombie agent saved")

        prompt_clock = pygame.time.Clock()
        if continue_prompt(env.screen, prompt_clock):
            round_num += 1
        else:
            print("Returning to main menu")
            break 

def continue_prompt(screen, clock):
    """
    Displays a prompt asking if the user wants to continue,
    allowing selection of "Yes" or "No" via arrow keys.
    Returns True if "Yes" is selected, False otherwise.
    """
    options = ["Yes", "No"]
    selected = 0
    font = pygame.font.Font(None, 50)
    prompt_text = font.render("Continue?", True, (255, 255, 255))
    
    while True:
        screen.fill((0, 0, 0))
        # Draw the prompt title at the top
        prompt_rect = prompt_text.get_rect(center=(screen.get_width() // 2, 150))
        screen.blit(prompt_text, prompt_rect)
        
        # Draw the options and highlight the selected one
        for i, option in enumerate(options):
            color = (255, 0, 0) if i == selected else (255, 255, 255)
            option_text = font.render(option, True, color)
            option_rect = option_text.get_rect(center=(screen.get_width() // 2, 250 + i * 60))
            screen.blit(option_text, option_rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    return options[selected] == "Yes"
        clock.tick(30)

def main_menu():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Zombie Survival Game")
    clock = pygame.time.Clock()

    # Show loading screen first.
    loading_screen(screen, clock)

    menu_options = ["Start", "About", "Quit"]
    selected = 0  # Index of the currently selected option

    while True:
        screen.fill((50, 50, 50))
        font = pygame.font.Font(None, 50)
        
        # Draw each menu option, highlighting the selected one.
        for i, option in enumerate(menu_options):
            color = (255, 0, 0) if i == selected else (255, 255, 255)
            text = font.render(option, True, color)
            text_rect = text.get_rect(center=(screen.get_width() // 2, 200 + i * 70))
            screen.blit(text, text_rect)
        
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(menu_options)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(menu_options)
                elif event.key == pygame.K_RETURN:
                    if menu_options[selected] == "Start":
                        # Start the game.
                        start_session()
                    elif menu_options[selected] == "About":
                        # Show the about page.
                        show_about(screen, clock)
                    elif menu_options[selected] == "Quit":
                        pygame.quit()
                        sys.exit()
        clock.tick(30)

def main():
    main_menu()

if __name__ == "__main__":
    main()