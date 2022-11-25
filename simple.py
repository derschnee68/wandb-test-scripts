import wandb
import time


key = "386b13a786fcc130351f4b73f8db5d8ed7ede9fb"
wandb.login()


for ii in range(30):
  wandb.log({"acc": 1 - 2 ** -ii, "loss": 2 ** -ii})
  time.sleep(0.5)
 