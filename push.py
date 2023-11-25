
import torch
from datasets import load_from_disk,load_dataset
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer,RewardTrainer,PPOConfig,PPOTrainer

model_name = "neurips-model"
refined_model = "../../../scratch/tushar_s.iitr/models/new_model"
base_model = LlamaForCausalLM.from_pretrained(
    refined_model,
)
base_model.push_to_hub(model_name)
