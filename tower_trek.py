import pygame
import random
import sys
import heapq
from enum import Enum
from typing import List, Tuple, Dict, Set

# Initialize pygame
pygame.init()

# Game constants
GRID_SIZE = 10  # 10x10 grid
CELL_SIZE = 60  # Size of each cell in pixels
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE + 100  # Extra space for UI
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
LIGHT_GRAY = (200, 200, 200)
BLUE = (0, 100, 255)
RED = (255, 50, 50)
GREEN = (50, 200, 50)
YELLOW = (255, 255, 0)
PURPLE = (150, 50, 200)
DARK_GREEN = (0, 100, 0)

# Game states
class GameState(Enum):
    MENU = 0
    PLAYING = 1
    GAME_OVER = 2
    LEVEL_COMPLETE = 3

# Difficulty levels
class Difficulty(Enum):
    EASY = 0
    MEDIUM = 1
    HARD = 2

class Cell:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.is_obstacle = False
        self.is_goal = False
        self.is_start = False
        
    def __eq__(self, other):
        if not isinstance(other, Cell):
            return False
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def get_pos(self) -> Tuple[int, int]:
        return (self.x, self.y)

class Grid:
    def __init__(self, width: int, height: int, difficulty: Difficulty):
        self.width = width
        self.height = height
        self.difficulty = difficulty
        self.cells = [[Cell(x, y) for y in range(height)] for x in range(width)]
        self.player_pos = (0, 0)
        self.ai_pos = (width - 1, height - 1)
        self.goal_pos = (width - 1, 0)
        self.generate_grid()
        
    def generate_grid(self):
        # Set start, goal, and initial positions
        self.cells[0][0].is_start = True
        self.cells[self.width - 1][0].is_goal = True
        self.player_pos = (0, self.height - 1)  # Player starts at bottom left
        self.ai_pos = (self.width - 1, self.height - 1)  # AI starts at bottom right
        self.goal_pos = (self.width - 1, 0)  # Goal is at top right
        
        # Generate obstacles based on difficulty
        obstacle_percentage = {
            Difficulty.EASY: 0.15,
            Difficulty.MEDIUM: 0.25,
            Difficulty.HARD: 0.35
        }
        
        num_obstacles = int(self.width * self.height * obstacle_percentage[self.difficulty])
        
        # Place obstacles randomly, but not on start, goal, player or AI positions
        obstacles_placed = 0
        while obstacles_placed < num_obstacles:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            
            # Skip if position is start, goal, player or AI
            if ((x, y) == (0, 0) or (x, y) == (self.width - 1, 0) or 
                (x, y) == self.player_pos or (x, y) == self.ai_pos):
                continue
                
            # Skip if already an obstacle
            if self.cells[x][y].is_obstacle:
                continue
                
            # Make sure the grid remains solvable by checking if there's a path
            # from player to goal before placing the obstacle
            self.cells[x][y].is_obstacle = True
            if not self.is_path_exists(self.player_pos, self.goal_pos):
                self.cells[x][y].is_obstacle = False
                continue
                
            obstacles_placed += 1
    
    def is_valid_move(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        if self.cells[x][y].is_obstacle:
            return False
        return True
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = pos
        neighbors = []
        
        # Check all four directions
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:  # Up, Right, Down, Left
            new_x, new_y = x + dx, y + dy
            if self.is_valid_move((new_x, new_y)):
                neighbors.append((new_x, new_y))
                
        return neighbors
    
    def is_path_exists(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Check if there's a path from start to end using BFS"""
        if start == end:
            return True
            
        visited = set()
        queue = [start]
        visited.add(start)
        
        while queue:
            current = queue.pop(0)
            
            if current == end:
                return True
                
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
        return False
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find a path from start to end using A* algorithm"""
        if start == end:  # Already at destination
            return [start]
            
        # A* algorithm
        open_set = []
        closed_set = set()
        
        # Priority queue with (f_score, position)
        heapq.heappush(open_set, (0, start))
        
        # For path reconstruction
        came_from = {}
        
        # g_score[n] is the cost of the cheapest path from start to n
        g_score = {start: 0}
        
        # f_score[n] = g_score[n] + heuristic(n)
        f_score = {start: self.heuristic(start, end)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
                
            closed_set.add(current)
            
            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                    
                tentative_g_score = g_score[current] + 1  # Cost is always 1 for adjacent cells
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, end)
                    
                    # Add to open set if not already there
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        
        return []  # No path found
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

class TowerTrek:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tower Trek: Escape the Pursuer")
        self.clock = pygame.time.Clock()
        
        # Load fonts
        try:
            self.font_small = pygame.font.SysFont("Arial", 16)
            self.font_medium = pygame.font.SysFont("Arial", 24)
            self.font_large = pygame.font.SysFont("Arial", 32)
            self.font_title = pygame.font.SysFont("Arial", 48, bold=True)
        except:
            self.font_small = pygame.font.Font(None, 16)
            self.font_medium = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 32)
            self.font_title = pygame.font.Font(None, 48)
        
        self.state = GameState.MENU
        self.difficulty = Difficulty.EASY
        self.level = 1
        self.score = 0
        self.moves = 0
        self.grid = None
        self.ai_won = False  # Flag to track if AI won
        self.initialize_game()
        
        # Debug flag
        self.debug = True  # Set to True by default to help diagnose issues
        
    def initialize_game(self):
        self.grid = Grid(GRID_SIZE, GRID_SIZE, self.difficulty)
        self.moves = 0
        self.ai_won = False
        
    def run(self):
        running = True
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                # Debug key - press D to print debug info
                if event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                    self.debug = not self.debug
                    print(f"Debug mode: {'ON' if self.debug else 'OFF'}")
                    print(f"Current state: {self.state}")
                    print(f"Player position: {self.grid.player_pos}")
                    print(f"AI position: {self.grid.ai_pos}")
                
                # Always check for ESC key to quit
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    if self.state == GameState.MENU:
                        running = False
                    else:
                        self.state = GameState.MENU
                
                # Process other events
                self.handle_event(event)
            
            # Update game state
            self.update()
            
            # Render
            self.render()
            
            # Debug info
            if self.debug:
                fps = self.clock.get_fps()
                debug_text = self.font_small.render(f"FPS: {fps:.1f} | State: {self.state.name}", True, GREEN)
                player_pos_text = self.font_small.render(f"Player: {self.grid.player_pos}", True, GREEN)
                ai_pos_text = self.font_small.render(f"AI: {self.grid.ai_pos}", True, GREEN)
                
                self.screen.blit(debug_text, (10, 10))
                self.screen.blit(player_pos_text, (10, 30))
                self.screen.blit(ai_pos_text, (10, 50))
            
            # Update display
            pygame.display.flip()
            
            # Cap the frame rate
            self.clock.tick(FPS)
            
        pygame.quit()
        sys.exit()
        
    def handle_event(self, event):
        # Print event info if in debug mode
        if self.debug and event.type == pygame.KEYDOWN:
            print(f"Key pressed: {pygame.key.name(event.key)}")
        
        if self.state == GameState.MENU:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print("Starting game from menu")
                    self.start_game()
                elif event.key in [pygame.K_1, pygame.K_KP1]:
                    self.difficulty = Difficulty.EASY
                    print("Difficulty set to EASY")
                elif event.key in [pygame.K_2, pygame.K_KP2]:
                    self.difficulty = Difficulty.MEDIUM
                    print("Difficulty set to MEDIUM")
                elif event.key in [pygame.K_3, pygame.K_KP3]:
                    self.difficulty = Difficulty.HARD
                    print("Difficulty set to HARD")
                    
        elif self.state == GameState.PLAYING:
            if event.type == pygame.KEYDOWN:
                # Handle player movement
                x, y = self.grid.player_pos
                new_pos = None
                
                # WASD or Arrow keys
                if event.key in [pygame.K_w, pygame.K_UP]:
                    new_pos = (x, y - 1)  # Up
                elif event.key in [pygame.K_s, pygame.K_DOWN]:
                    new_pos = (x, y + 1)  # Down
                elif event.key in [pygame.K_a, pygame.K_LEFT]:
                    new_pos = (x - 1, y)  # Left
                elif event.key in [pygame.K_d, pygame.K_RIGHT]:
                    new_pos = (x + 1, y)  # Right
                
                if new_pos and self.grid.is_valid_move(new_pos):
                    self.grid.player_pos = new_pos
                    self.moves += 1
                    print(f"Player moved to {new_pos}")
                    
                    # Check if player reached the goal
                    if new_pos == self.grid.goal_pos:
                        self.level_complete()
                    else:
                        # AI's turn to move
                        self.move_ai()
                        
                        # Check if AI caught the player (double-check)
                        self.check_collision()
                    
        elif self.state == GameState.GAME_OVER or self.state == GameState.LEVEL_COMPLETE:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if self.state == GameState.GAME_OVER:
                        print("Returning to menu from game over")
                        self.state = GameState.MENU
                    else:  # LEVEL_COMPLETE
                        print("Starting next level")
                        self.start_next_level()
    
    def update(self):
        # Check for collision between player and AI
        if self.state == GameState.PLAYING:
            self.check_collision()
    
    def check_collision(self):
        """Check if AI has caught the player"""
        if self.grid.player_pos == self.grid.ai_pos:
            print(f"COLLISION DETECTED! Player: {self.grid.player_pos}, AI: {self.grid.ai_pos}")
            self.game_over()
    
    def render(self):
        self.screen.fill(BLACK)
        
        if self.state == GameState.MENU:
            self.render_menu()
        elif self.state == GameState.PLAYING:
            self.render_game()
        elif self.state == GameState.GAME_OVER:
            self.render_game()  # Show the game board
            self.render_game_over()  # Overlay game over message
        elif self.state == GameState.LEVEL_COMPLETE:
            self.render_game()  # Show the game board
            self.render_level_complete()  # Overlay level complete message
    
    def render_menu(self):
        # Title
        title = self.font_title.render("TOWER TREK", True, WHITE)
        subtitle = self.font_large.render("Escape the Pursuer", True, LIGHT_GRAY)
        title_rect = title.get_rect(center=(SCREEN_WIDTH//2, 100))
        subtitle_rect = subtitle.get_rect(center=(SCREEN_WIDTH//2, 150))
        self.screen.blit(title, title_rect)
        self.screen.blit(subtitle, subtitle_rect)
        
        # Instructions
        instructions = [
            "Navigate to the top of the tower while avoiding the AI pursuer",
            "Use W,A,S,D or Arrow Keys to move",
            "",
            "Select Difficulty:",
            f"1: EASY {'[Selected]' if self.difficulty == Difficulty.EASY else ''}",
            f"2: MEDIUM {'[Selected]' if self.difficulty == Difficulty.MEDIUM else ''}",
            f"3: HARD {'[Selected]' if self.difficulty == Difficulty.HARD else ''}",
            "",
            "Press SPACE to start",
            "Press ESC to quit"
        ]
        
        for i, line in enumerate(instructions):
            text = self.font_medium.render(line, True, WHITE)
            text_rect = text.get_rect(center=(SCREEN_WIDTH//2, 220 + i * 30))
            self.screen.blit(text, text_rect)
    
    def render_game(self):
        # Draw grid
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                cell = self.grid.cells[x][y]
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                # Draw cell background
                if cell.is_obstacle:
                    pygame.draw.rect(self.screen, GRAY, rect)
                elif (x, y) == self.grid.goal_pos:
                    pygame.draw.rect(self.screen, GREEN, rect)
                elif (x, y) == (0, self.grid.height - 1):  # Start position
                    pygame.draw.rect(self.screen, BLUE, rect)
                else:
                    pygame.draw.rect(self.screen, WHITE, rect)
                
                # Draw cell border
                pygame.draw.rect(self.screen, BLACK, rect, 1)
        
        # Draw player
        player_x, player_y = self.grid.player_pos
        player_rect = pygame.Rect(player_x * CELL_SIZE + 5, player_y * CELL_SIZE + 5, 
                                 CELL_SIZE - 10, CELL_SIZE - 10)
        pygame.draw.rect(self.screen, BLUE, player_rect)
        
        # Draw AI
        ai_x, ai_y = self.grid.ai_pos
        ai_rect = pygame.Rect(ai_x * CELL_SIZE + 5, ai_y * CELL_SIZE + 5, 
                             CELL_SIZE - 10, CELL_SIZE - 10)
        pygame.draw.rect(self.screen, RED, ai_rect)
        
        # Draw UI
        ui_rect = pygame.Rect(0, GRID_SIZE * CELL_SIZE, SCREEN_WIDTH, 100)
        pygame.draw.rect(self.screen, LIGHT_GRAY, ui_rect)
        
        # Draw level and moves info
        level_text = self.font_medium.render(f"Level: {self.level}", True, BLACK)
        moves_text = self.font_medium.render(f"Moves: {self.moves}", True, BLACK)
        diff_text = self.font_medium.render(f"Difficulty: {self.difficulty.name}", True, BLACK)
        
        self.screen.blit(level_text, (20, GRID_SIZE * CELL_SIZE + 20))
        self.screen.blit(moves_text, (20, GRID_SIZE * CELL_SIZE + 50))
        self.screen.blit(diff_text, (SCREEN_WIDTH - 200, GRID_SIZE * CELL_SIZE + 20))
        
        # Draw controls reminder
        controls_text = self.font_small.render("Use W,A,S,D or Arrow Keys to move", True, BLACK)
        self.screen.blit(controls_text, (SCREEN_WIDTH // 2 - 120, GRID_SIZE * CELL_SIZE + 70))
    
    def render_game_over(self):
        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Game over message
        game_over = self.font_title.render("GAME OVER", True, RED)
        game_over_rect = game_over.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 80))
        self.screen.blit(game_over, game_over_rect)
        
        # AI won message
        ai_won_text = self.font_large.render("AI CAUGHT YOU!", True, RED)
        ai_won_rect = ai_won_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 30))
        self.screen.blit(ai_won_text, ai_won_rect)
        
        # Stats
        stats = self.font_medium.render(f"Level: {self.level}  Moves: {self.moves}", True, WHITE)
        stats_rect = stats.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 20))
        self.screen.blit(stats, stats_rect)
        
        # Instruction
        instruction = self.font_medium.render("Press SPACE to return to menu", True, WHITE)
        instruction_rect = instruction.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 60))
        self.screen.blit(instruction, instruction_rect)
    
    def render_level_complete(self):
        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Level complete message
        level_complete = self.font_title.render("LEVEL COMPLETE!", True, GREEN)
        level_complete_rect = level_complete.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 50))
        self.screen.blit(level_complete, level_complete_rect)
        
        # Stats
        stats = self.font_medium.render(f"Level: {self.level}  Moves: {self.moves}", True, WHITE)
        stats_rect = stats.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
        self.screen.blit(stats, stats_rect)
        
        # Instruction
        instruction = self.font_medium.render("Press SPACE for next level", True, WHITE)
        instruction_rect = instruction.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 50))
        self.screen.blit(instruction, instruction_rect)
    
    def start_game(self):
        self.state = GameState.PLAYING
        self.level = 1
        self.score = 0
        self.initialize_game()
    
    def game_over(self):
        print("Game over! AI caught the player.")
        self.ai_won = True
        self.state = GameState.GAME_OVER
    
    def level_complete(self):
        print(f"Level {self.level} complete!")
        self.score += 1000 - (self.moves * 10)  # Score based on moves
        self.state = GameState.LEVEL_COMPLETE
    
    def start_next_level(self):
        self.level += 1
        
        # Increase difficulty every 3 levels
        if self.level % 3 == 0:
            if self.difficulty == Difficulty.EASY:
                self.difficulty = Difficulty.MEDIUM
            elif self.difficulty == Difficulty.MEDIUM:
                self.difficulty = Difficulty.HARD
        
        self.initialize_game()
        self.state = GameState.PLAYING
    
    def move_ai(self):
        """Move the AI towards the player using pathfinding"""
        # First, check if AI is already on the player (should never happen, but just in case)
        if self.grid.ai_pos == self.grid.player_pos:
            print("AI already on player position!")
            self.game_over()
            return
            
        # Find path from AI to player
        path = self.grid.find_path(self.grid.ai_pos, self.grid.player_pos)
        
        if path and len(path) > 0:
            # Move one step along the path
            self.grid.ai_pos = path[0]
            print(f"AI moved to {self.grid.ai_pos} via pathfinding")
            
            # Check if AI caught the player after moving
            if self.grid.ai_pos == self.grid.player_pos:
                print("AI caught player after pathfinding move!")
                self.game_over()
                return
        else:
            # If no path found, try to move in a direction that gets closer to player
            ai_x, ai_y = self.grid.ai_pos
            player_x, player_y = self.grid.player_pos
            
            # Try to move in the direction of the player
            possible_moves = []
            
            # Check horizontal movement
            if ai_x < player_x and self.grid.is_valid_move((ai_x + 1, ai_y)):
                possible_moves.append((ai_x + 1, ai_y))
            elif ai_x > player_x and self.grid.is_valid_move((ai_x - 1, ai_y)):
                possible_moves.append((ai_x - 1, ai_y))
                
            # Check vertical movement
            if ai_y < player_y and self.grid.is_valid_move((ai_x, ai_y + 1)):
                possible_moves.append((ai_x, ai_y + 1))
            elif ai_y > player_y and self.grid.is_valid_move((ai_x, ai_y - 1)):
                possible_moves.append((ai_x, ai_y - 1))
                
            # If there are possible moves, choose one randomly
            if possible_moves:
                self.grid.ai_pos = random.choice(possible_moves)
                print(f"AI moved to {self.grid.ai_pos} (direct)")
                
                # Check if AI caught the player after moving
                if self.grid.ai_pos == self.grid.player_pos:
                    print("AI caught player after direct move!")
                    self.game_over()
                    return
            else:
                # If no direct moves, try any valid move
                neighbors = self.grid.get_neighbors(self.grid.ai_pos)
                if neighbors:
                    self.grid.ai_pos = random.choice(neighbors)
                    print(f"AI moved to {self.grid.ai_pos} (random)")
                    
                    # Check if AI caught the player after moving
                    if self.grid.ai_pos == self.grid.player_pos:
                        print("AI caught player after random move!")
                        self.game_over()
                        return
                else:
                    print("AI couldn't move")

# Start the game
if __name__ == "__main__":
    game = TowerTrek()
    game.run()