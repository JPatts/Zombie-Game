import pygame 
import os
from main import MazeEnv

def traing_agent(num_rounds=1000, save_interval=50, round_duration=90):
    # initalize environment
    env = MazeEnv() 

    # disable rendering by overriding the render_frame method
    env.render_frame = lambda: None
    
    # can make pygame winodw smaller to reduce overhead 
    pygame.display.set_mode((1,1))

    if os.path.exists("zombie_agent.pkl"):
        env.z_agent.load("zombie_agent.pkl")
        print("Loaded existing zombie agent from zombie_agent.pkl")
    
    # Run training rounds
    for round_num in range(1, num_rounds + 1):
        caught_human = env.run_one_round(round_duration)
        print(f"Round {round_num} complete. Zombie caught human: {caught_human}")

        # save agent every save_interval rounds
        if round_num % save_interval == 0:
            filename = f"zombie_agent_{round_num}.pkl"
            env.z_agent.save(filename)
            print(f"Agent save to {filename}")

if __name__ == "__main__":
    traing_agent()