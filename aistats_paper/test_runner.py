import subprocess
import time

if __name__ == "__main__":

    NUM_SIMULATIONS = 10
    ALGO = 'sgd'  # {sgd, sgd2, acrpn}

    t_ = time.time()

    for s in range(NUM_SIMULATIONS):
        print(f"SIMULATION NO : {s + 1}", end="\n")

        command = [
            'python',
            f'mujoco_experiments/{ALGO}.py',
            '--exp-name', f'{ALGO.upper()}{s + 1}_FINAL',
            '--env-seed', '-1',
            '--save', 'True',
            '--track', 'False',
            '--alpha', '100_000',
            '--normalize-returns', 'True',

            '--num-updates', '1000',
            '--max-timesteps', '1000',
            '--gym-id', 'Humanoid-v4',
            '--hidden-sizes', '(64, 64)',
        ]

        subprocess.run(command)

    print("Simulations complete. Time taken:", round(time.time() - t_, 3))
