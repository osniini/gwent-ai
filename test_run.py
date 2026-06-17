import random
from src.engine.gwent_env import GwentEnv


env = GwentEnv()
state = env.reset()
done = False

print("Starting game...")
print("State:", state)

turns = 0

while not done and turns < 20:
    action = random.randint(0, 8)

    state, reward, done = env.step(action)
    turns += 1
    print(f"Turn {turns}: Action {action}, Reward {reward}, Done {done}")
    print("State:", state)

print("Game over!")
print("Final state:", state)
print("Final reward:", reward)
print("Final done:", done)