import wandb
import pickle

wandb.login()

with wandb.init(project="test-project", job_type="load-data") as run:
    fake_dataset = {"labels": [0,1],"inputs":[False,True]}
    raw_data = wandb.Artifact("test-ds", type="dataset")

    with raw_data.new_file("test_ds.pt", mode="wb") as file:
        pickle.dump(fake_dataset,file)

    run.log_artifact(raw_data)

