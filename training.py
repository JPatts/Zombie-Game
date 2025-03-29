import pygame 
import os
import numpy as np
import subprocess
from main import MazeEnv

def traing_agent(num_rounds=15, save_interval=5, commit_interval=15, round_duration=90):
    """
    Training loop for the zombie agent
    Args:
        num_rounds (int): Number of training rounds
        save_interval (int): Interval for saving the agent
        commit_interval (int): Interval for committing changes
        round_duration (int): Duration of each round in seconds
    """
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

    if os.path.exists("zombie_agent.pkl"):
        env.z_agent.load("zombie_agent.pkl")
        print("Loaded existing zombie agent from zombie_agent.pkl")
    
    env.training_mode = True

    # override single_update function to simulate human movement during training
    def training_single_update():
        current_state = env._get_obs()
        zombie_action = env.z_agent.get_action(current_state)
        human_action = np.random.randint(4)
        next_state, reward, done, info = env.step(zombie_action, human_action)
        env.z_agent.update(current_state, zombie_action, reward, next_state, done)
        return done 

    # Main training loop
    for round_num in range(1, num_rounds + 1):
        env.reset()
        done = False
        round_start_time = pygame.time.get_ticks()
        while not done:
            done = env.single_update()
            if pygame.time.get_ticks() - round_start_time > round_duration * 1000:
                break
        
        print(f"Round {round_num} complete")

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

if __name__ == "__main__":
    traing_agent()