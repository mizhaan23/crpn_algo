import subprocess
import time

config = {
    "CartPole-v1": {
        'alpha': '1000',
        'num-updates': '1000',
        'max-timesteps': '500',  # horizon
        'batch-size': '50',
        'hidden-sizes': '()',  # linear features ONLY
    },

    "Reacher-v4": {
        'alpha': '10000',
        'num-updates': '1000',
        'max-timesteps': '50',  # horizon
        'batch-size': '100',
        'hidden-sizes': '(32, 32)',
    },

    "Humanoid-v4": {
        'alpha': '10000',
        'num-updates': '1000',
        'max-timesteps': '1000',  # horizon
        'batch-size': '100',
        'hidden-sizes': '(64, 64)',
    }
}

if __name__ == "__main__":

    NUM_SIMULATIONS = 10
    ENV = "CartPole-v1"
    env_config = config[ENV]
    ALGO = 'acrpn'  # {sgd, acrpn}

    t_ = time.time()

    for s in range(NUM_SIMULATIONS):

        print(f"SIMULATION NO : {s + 1}", end="\n")

        command = [
            'python',
            f'aistats_paper/mujoco_experiments/{ALGO}.py',
            '--exp-name', f'{ALGO.upper()}{s + 1}_FINAL',
            '--env-seed', '-1',
            '--save', 'True',
            '--track', 'False',
            '--alpha', env_config['alpha'],
            '--normalize-returns', 'True',

            '--num-updates', env_config['num-updates'],
            '--max-timesteps', env_config['max-timesteps'],
            '--gym-id', ENV,
            '--hidden-sizes', env_config['hidden-sizes'],
            '--batch-size', env_config['batch-size']
        ]

        subprocess.run(command)

    print("Simulations complete. Time taken:", round(time.time() - t_, 3))
