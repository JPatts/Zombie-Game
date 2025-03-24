import pygame
import sys

# constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
TILE_SIZE = 40
GRID_COLS = SCREEN_WIDTH // TILE_SIZE
GRID_ROWS = SCREEN_HEIGHT // TILE_SIZE

def load_images():
		# Load images
	try:
		tile_image = pygame.image.load("assets/background_images/grass_patch_1.png")
		tile_image = pygame.transform.scale(tile_image, (TILE_SIZE, TILE_SIZE))
	except pygame.error as e:
		print("tile.png not found, using placeholder.")
		tile_image = pygame.Surface((TILE_SIZE, TILE_SIZE))
		tile_image.fill((100, 100, 100))  # Gray placeholder

	try: 
		human_image = pygame.image.load("assets/human_images/human_1.png")
		human_image = pygame.transform.scale(human_image, (TILE_SIZE, TILE_SIZE))
	except pygame.error as e:
		print("human.png not found, using placeholder.")
		human_image = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
		human_image.fill((255, 0, 0))

	return tile_image, human_image

class Human(pygame.sprite.Sprite):
	def __init__(self, x, y, human_image):
		super().__init__()
		self.image = human_image
		self.rect = self.image.get_rect(topleft=(x, y))
		self.speed = 15
	
	def update(self, keys_pressed):
		# Update position based on arrow key input
		if keys_pressed[pygame.K_LEFT]: 
			self.rect.x -= self.speed
		if keys_pressed[pygame.K_RIGHT]:
			self.rect.x += self.speed
		if keys_pressed[pygame.K_UP]:
			self.rect.y -= self.speed
		if keys_pressed[pygame.K_DOWN]:
			self.rect.y += self.speed

		# Keep the player within the screen boundaries
		if self.rect.left < 0:
			self.rect.left = 0
		if self.rect.right > SCREEN_WIDTH:
			self.rect.right = SCREEN_WIDTH
		if self.rect.top < 0:
			self.rect.top = 0
		if self.rect.bottom > SCREEN_HEIGHT:
			self.rect.bottom = SCREEN_HEIGHT

def draw_grid(screen, tile_image):
    """Draw the background grid using the tile image."""
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            screen.blit(tile_image, (col * TILE_SIZE, row * TILE_SIZE))

def create_sprites(human_image):
	# Create instance of the human sprite at the top-left corner
	human_sprite = Human(0, 0, human_image)
	all_sprites = pygame.sprite.Group()
	all_sprites.add(human_sprite)
	return human_sprite, all_sprites

def game_loop(screen, tile_image, human_sprite, all_sprites):
    """The main game loop."""
    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(60)  # 60 FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Draw the grid background
        draw_grid(screen, tile_image)

        # Process key input and update human sprite
        keys_pressed = pygame.key.get_pressed()
        human_sprite.update(keys_pressed)

        # Draw all sprites on top of the background
        all_sprites.draw(screen)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

def main():
    # Initialize Pygame and set up the display
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Zombie Survival Game")

    # Load images and create sprites
    tile_image, human_image = load_images()
    human_sprite, all_sprites = create_sprites(human_image)

    # Run the game loop
    game_loop(screen, tile_image, human_sprite, all_sprites)

if __name__ == "__main__":
    main()