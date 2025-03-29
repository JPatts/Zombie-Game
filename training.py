import sys
import time
import pygame 
import os
import numpy as np
import subprocess
from main import MazeEnv

def traing_agent(num_rounds=5000, save_interval=500, commit_interval=2500, round_duration=5000, load_file=None):
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
    env.render_frame = lambda: None
    
    # can make pygame winodw smaller to reduce overhead 
    pygame.display.set_mode((1,1))

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
        human_action = np.random.randint(4)
        next_state, reward, done, info = env.step(zombie_action, human_action)
        env.z_agent.update(current_state, zombie_action, reward, next_state, done)
        return done 

    env.single_update = training_single_update

    # Main training loop
    for round_num in range(1, num_rounds + 1):
        env.reset()
        done = False
        round_start_time = pygame.time.get_ticks()
        while not done:
            done = env.single_update()
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