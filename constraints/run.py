"""
Run the experiment with the given config file.
"""
import argparse
from pathlib import Path
import shutil

from presp.evolution import Evolution
from presp.prescriptor import NNPrescriptor, NNPrescriptorFactory
import yaml

from constraints.constraint_prescriptor import ConstraintPrescriptor
from constraints.problems import BNH, Circle, Triangle


def main():
    """
    Main experiment runner. Loads the config file then runs the experiment accordingly.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "rb") as f:
        config = yaml.safe_load(f)

    save_path = Path(config["evolution_params"]["save_path"])
    if save_path.exists():
        inp = input(f"{save_path} already exists. Replace? (y/n)")
        if inp.lower() == "y":
            shutil.rmtree(save_path)
        else:
            print("Exiting...")
            return
    save_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, save_path / "config.yml")

    # Load problem
    eval_params = config["eval_params"]
    if config["problem"] == "bnh":
        problem = BNH(**eval_params)
        factory = NNPrescriptorFactory(ConstraintPrescriptor, **config["prescriptor_params"])
    elif config["problem"] == "circle":
        problem = Circle(**eval_params)
        factory = NNPrescriptorFactory(NNPrescriptor, **config["prescriptor_params"])
    elif config["problem"] == "triangle":
        problem = Triangle(**eval_params)
        factory = NNPrescriptorFactory(ConstraintPrescriptor, **config["prescriptor_params"])
    else:
        raise ValueError(f"Unknown problem {config['problem']}")

    factory = NNPrescriptorFactory(ConstraintPrescriptor, **config["prescriptor_params"])

    evolution = Evolution(**config["evolution_params"], prescriptor_factory=factory, evaluator=problem)
    evolution.run_evolution()


if __name__ == "__main__":
    main()
