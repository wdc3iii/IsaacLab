import os

with open("pycharm_env.txt", "w") as f:
    for key, value in os.environ.items():
        f.write(f"{key}: {value}\n")