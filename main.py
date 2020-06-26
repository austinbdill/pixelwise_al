import argparse
import yaml
from trainers.gumbel_trainer import GumbelTrainer
from trainers.target_trainer import TargetTrainer
from trainers.vanilla_trainer import VanillaTrainer
from trainers.baseline_trainer import BaselineTrainer
from trainers.reinforce_trainer import ReinforceTrainer
from trainers.a2c_trainer import A2CTrainer
from trainers.ddpg_trainer import DDPGTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='path to config file')

try:
    args = parser.parse_args()
except IOError as msg:
    parser.error(str(msg))

with open("configs/" + args.config, 'r') as f:
    params = yaml.load(f, yaml.FullLoader)

# Select training algorithm
if params["type"] == "vanilla":
    trainer = VanillaTrainer(params)
elif params["type"] == "baseline":
    trainer = BaselineTrainer(params)
elif params["type"] == "gumbel":
    trainer = GumbelTrainer(params)
elif params["type"] == "target":
    trainer = TargetTrainer(params)
elif params["type"] == "reinforce":
    trainer = ReinforceTrainer(params)
elif params["type"] == "a2c":
    trainer = A2CTrainer(params)
elif params["type"] == "ddpg":
    trainer = DDPGTrainer(params)


trainer.train()
