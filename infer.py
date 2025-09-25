import os
from omegaconf import OmegaConf
import argparse
from pathlib import Path

import torch

from anytoedit.anytoedit_wrapper import AnytoEdit
from third_party.tokenflow.util import seed_everything

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, default='configs/infer.yaml')
  parser.add_argument("--outputs", type=str, default='results')
  parser.add_argument("--latents", type=str, default="latents")

  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--verbose", type=bool, default=True)
  args = parser.parse_args()
  return args

if __name__ == "__main__":
    #Prepare yamls
    args = parse_arguments()
    config = OmegaConf.load(args.config)
    seed_everything(args.seed)

    args.outputs = os.path.join(args.outputs, Path(args.config).stem)

    Path(args.latents).mkdir(parents=True, exist_ok=True)
    Path(args.outputs).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, os.path.join(args.outputs, 'config.yaml'))

    editor = AnytoEdit(args, config)
    editor.edit()

    