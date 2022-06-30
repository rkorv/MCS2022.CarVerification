import sys

import torch

sys.path.append(".")
import train


CHECKPOINTS_DIR = "./checkpoints"
LOGS_DIR = "./logs"

MODEL_RESUME = f"{LOGS_DIR}/epoch=7-step=81000.ckpt"

model = train.CarsReidentificationTrainEffnet.load_from_checkpoint(MODEL_RESUME)

model = model.eval().half()

torch.save(model.net.state_dict(), f"{CHECKPOINTS_DIR}/final.pt")
