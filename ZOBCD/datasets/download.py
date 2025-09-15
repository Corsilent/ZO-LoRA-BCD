from datasets import load_dataset
import os

save_path_train = 'glue_sst2/train/'
save_path_valid = 'glue_sst2/validation/'

os.makedirs(save_path_train, exist_ok=True)
os.makedirs(save_path_valid, exist_ok=True)

dataset = load_dataset('glue', 'sst2')

dataset['train'].save_to_disk(save_path_train)
dataset['validation'].save_to_disk(save_path_valid)

print("Successfully downloaded and saved the SST-2 dataset.")