import sys
import argparse

sys.path.append('./src')
sys.path.append('./src/models')
sys.path.append('./src/optimizers')

import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()

    cfg_path = args.config
    run_experiment.run(cfg_path)


if __name__ == '__main__':
    main()
