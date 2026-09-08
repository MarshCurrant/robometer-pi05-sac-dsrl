from droid.robot_env import RobotEnv

env = RobotEnv()

print("Connected, getting robot state")
state, _ = env.get_state()
print(f"state: {state}")
