import subprocess
import time

config = {
    "CartPole-v1": {
        'alpha': '1000',
        'num-updates': '1000',
        'max-timesteps': '500',  # horizon
        'batch-size': '50',
    }
}

if __name__ == "__main__":

    NUM_SIMULATIONS = 10
    ENV = "CartPole-v1"
    env_config = config[ENV]
    ALGO = 'reinforce'  # {reinforce, crpn}

    t_ = time.time()

    for s in range(NUM_SIMULATIONS):

        print(f"SIMULATION NO : {s + 1}", end="\n")

        command = [
            'python',
            f'linear/{ALGO}.py',
            '--exp-name', f'{ALGO.upper()}{s + 1}_LINEAR',
            '--env-seed', '-1',
            '--save', 'False',  # change to True if you want to save simulation data.
            '--track', 'False',
            '--alpha', env_config['alpha'],
            '--normalize-returns', 'True',

            '--num-updates', env_config['num-updates'],
            '--max-timesteps', env_config['max-timesteps'],
            '--gym-id', ENV,
            '--batch-size', env_config['batch-size']
        ]

        subprocess.run(command)

    print("Simulations complete. Time taken:", round(time.time() - t_, 3))
