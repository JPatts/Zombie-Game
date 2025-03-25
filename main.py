import pygame
import sys
import random
import os

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

        # Set the grass path before generating the board
        self.grass_path = os.path.join("assets", "background_images", "grass_patch_1.png")

        # Create the Pygame display sized to our board
        self.screen = pygame.display.set_mode((self.num_cols * self.GRID_SIZE, self.num_rows * self.GRID_SIZE))
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

        # Set initial positions (using grid coordinates: (row, col))
        self.human_pos = (self.num_rows - 1, 0)    # bottom-left
        self.zombie_pos = (0, self.num_cols - 1)     # top-right

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

    def _draw_maze(self):
        font = pygame.font.Font(None, 24)
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                x = col * self.GRID_SIZE
                y = row * self.GRID_SIZE

                # Draw background image
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

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

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
        Process user input for human movement and update the zombie's position
        using a simple chase strategy.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

        # Human movement via arrow keys (grid-based)
        keys = pygame.key.get_pressed()
        action = None
        if keys[pygame.K_UP]:
            action = 0
        elif keys[pygame.K_RIGHT]:
            action = 1
        elif keys[pygame.K_DOWN]:
            action = 2
        elif keys[pygame.K_LEFT]:
            action = 3
        if action is not None:
            new_human_pos = self._get_new_position(self.human_pos, action)
            # Only update if the position has changed
            self.human_pos = new_human_pos

        # Zombie simple chase: choose the move that minimizes Manhattan distance to human
        best_action = None
        current_distance = self._manhattan_distance(self.zombie_pos, self.human_pos)
        for a in range(4):
            new_pos = self._get_new_position(self.zombie_pos, a)
            d = self._manhattan_distance(new_pos, self.human_pos)
            if d < current_distance:
                current_distance = d
                best_action = a
        if best_action is not None:
            self.zombie_pos = self._get_new_position(self.zombie_pos, best_action)
        if self.zombie_pos == self.human_pos:
            self._game_over_screen()
            pygame.quit()
            sys.exit()

    def run(self):
        clock = pygame.time.Clock()
        while True:
            clock.tick(10)  # Set FPS (grid-based movement, so lower FPS is acceptable)
            self.screen.fill(self.WHITE)
            self._draw_maze()
            self.update()
            self._draw_characters()
            pygame.display.flip()
    
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


def main():
    env = MazeEnv()
    env.run()

if __name__ == "__main__":
    main()
