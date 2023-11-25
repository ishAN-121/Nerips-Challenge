from datasets import load_dataset

data_name = "aqua_rat"
training_data = load_dataset(data_name, split="train",download_mode='force_redownload')
