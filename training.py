import sys
import time
import pygame 
import os
import numpy as np
import subprocess
from main import MazeEnv

WATCH_GAME = True

def traing_agent(num_rounds=5, save_interval=500, commit_interval=2500, round_duration=250, load_file=None):
    """
    Training loop for the zombie agent
    Args:
        num_rounds (int): Number of training rounds
        save_interval (int): Interval for saving the agent
        commit_interval (int): Interval for committing changes
        round_duration (int): Duration of each round in seconds
    """
    # start timer
    overall_start_time = time.time()

    # create folder for all training data
    models_dir = "training_models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # initalize environment
    env = MazeEnv() 

    # disable rendering by overriding the render_frame method
    if not WATCH_GAME:
        env.render_frame = lambda: None
        pygame.display.set_mode((1,1))
    else:
        window_width = env.num_cols * env.GRID_SIZE
        window_height = env.num_rows * env.GRID_SIZE
        pygame.display.set_mode((window_width, window_height))
    
    # can make pygame winodw smaller to reduce overhead 

    # load agent stat from specified file if provided
    # otherwise use default file or start from scratch 
    if load_file is not None and os.path.exists(load_file):
        env.z_agent.load(load_file)
        print(f"Loaded agent state from {load_file}")
    else:
        print("No load file provided or file does not exist. Starting from scratch.")

    env.training_mode = True

    # override single_update function to simulate human movement during training
    def training_single_update():
        current_state = env._get_obs()
        zombie_action = env.z_agent.get_action(current_state)
        human_action = simulated_human_action(env)
        next_state, reward, done, info = env.step(zombie_action, human_action)
        env.z_agent.update(current_state, zombie_action, reward, next_state, done)
        return done 

    env.single_update = training_single_update

    def simulated_human_action(env):
        """
        Simulate player input by using rule-based logic
        - Human always moves to oppositte quadrant than zombie; attempts to flee to oppositte corner
        - If zombie is within 4 cells human will stop trying to move to oppositte quadrant and calculate moves
            to zombie and do exact oppsite; if zombie and human in hallway human moves other direction right away
        - no matter what every 15 steps human makes random move to simulate unpredictability
        """ 

        # init step counter for human
        if not hasattr(env, 'human_sim_step_count'):
            env.human_sim_step_count = 0
        env.human_sim_step_count += 1

        # every 15 steps make random move
        if env.human_sim_step_count % 15 == 0:
            return np.random.randint(4)
        
        # if zombie is within 4 cells; human moves directly away
        if env._manhattan_distance(env.human_pos, env.zombie_pos) <= 4:
            # Calculate the direction to move away from the zombie
            dx = env.zombie_pos[0] - env.human_pos[0]
            dy = env.human_pos[1] - env.zombie_pos[1]

            if abs(dx) >= abs(dy):
                return 0 if dx > 0 else 2  # Move left if zombie is to the left, right if to the right
            else:
                return 3 if dy > 0 else 1
        
        # otherwise rely on opp quadrant logic
        mid_row = env.num_rows // 2
        mid_col = env.num_cols // 2
        zr, zc = env.zombie_pos

        # determine zombie quadrant and set target for oppositte 
        # Quad 1: top right
        # Quad 2: top left
        # Quad 3: bottom left
        # Quad 4: bottom right
        if zr < mid_row and zc >= mid_col: # zombie in Q1
            target = (env.num_rows -1, 0)
        elif zr < mid_row and zc < mid_col:  # zombie in Q2
            target = (env.num_rows - 1, env.num_cols - 1)  # bottom right corner
        elif zr >= mid_row and zc < mid_col:  # zombie in Q3
            target = (0, env.num_cols - 1)  # top right corner
        else:  # zombie in Q4
            target = (0, 0)  # top left corner

        # Calculate the direction to move towards the target corner
        hr, hc = env.human_pos
        diff_r = target[0] - hr
        diff_c = target[1] - hc
        if abs(diff_r) >= abs(diff_c):
            return 2 if diff_r > 0 else 0
        else:
            return 1 if diff_c > 0 else 3

    # Main training loop
    for round_num in range(1, num_rounds + 1):
        env.reset()
        done = False
        round_start_time = pygame.time.get_ticks()

        if WATCH_GAME:
            clock = pygame.time.Clock()

        while not done:
            done = env.single_update()
            if WATCH_GAME:
                env.render_frame()
                clock.tick(30)
            if pygame.time.get_ticks() - round_start_time > round_duration * 1000:
                break
        
        if done:
            print(f"Round {round_num} complete. Zombie wins!")
        else:
            print(f"Round {round_num} complete. Human wins!")

        # save the agent every save_interval rounds
        if round_num % save_interval == 0:
            env.z_agent.save("zombie_agent.pkl")
            print(f"Saved agent state at round {round_num} to zombie_agent.pkl")

            model_filename = os.path.join(models_dir, f"zombie_agent_{round_num}.pkl")
            env.z_agent.save(model_filename)
            print(f"Saved agent state at round {round_num} to {model_filename}")

        # save the agent every save_interval rounds
        if round_num % commit_interval == 0:
            try:
                subprocess.call(["git", "add", "."])
                commit_message = f"Training round {round_num}"
                subprocess.call(["git", "commit", "-m", commit_message])
                subprocess.call(["git", "push"])
                print(f"Committed changes for round {round_num}")
            except Exception as e:
                print(f"Git Operation failed: {e}")
    
    final_model_filename = os.path.join(models_dir, f"zombie_agent_{round_num}.pkl")
    env.z_agent.save(final_model_filename)
    print(f"Final agent saved as {final_model_filename}")

    # end overall timer and output elapsed time
    overall_end_time = time.time()
    elapsed_time = overall_end_time - overall_start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    print(f"Time taken for training.py to complete with starting values of\n",
            f"number of rounds: {num_rounds}\n",
            f"duration of rounds: {round_duration}\n"
            "\n"
            f"Hours: {hours}\tminutes: {minutes}\tSeconds: {seconds}\n")

if __name__ == "__main__":
    load_file = sys.argv[1] if len(sys.argv) > 1 else None
    traing_agent(load_file=load_file) 