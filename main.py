import argparse
import yaml
from trainers.gumbel_trainer import GumbelTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='path to config file')

try:
    args = parser.parse_args()
except IOError as msg:
    parser.error(str(msg))

with open("configs/" + args.config, 'r') as f:
    params = yaml.load(f, yaml.FullLoader)

# Select training algorithm
if params["type"] == "gumbel":
    trainer = GumbelTrainer(params)

trainer.train()