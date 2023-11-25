import os
import torch
from datasets import load_from_disk,load_dataset
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer,RewardTrainer,PPOConfig,PPOTrainer

os.environ['TRANSFORMERS_CACHE'] = '../../../scratch/tushar_s.iitr/models'

dataset = load_dataset("piqa", split="train")

# Model and tokenizer names
base_model_name = "../../../scratch/tushar_s.iitr/models/model"
base_tokenizer_name = "../../../scratch/tushar_s.iitr/models/tokenizer"
refined_model = "../../../scratch/tushar_s.iitr/models/new_model_1"

# Tokenizer
llama_tokenizer = LlamaTokenizer.from_pretrained(base_tokenizer_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"  # Fix for fp16

#Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)
# Model
base_model = LlamaForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map={"": 0}
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1
print("base model done")

# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=32,
    bias="none",
    task_type="CAUSAL_LM"
)
# Training Params
train_params = TrainingArguments(
    output_dir="../../../scratch/tushar_s.iitr/models/enhanced_model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant"
)

# Trainer
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=peft_parameters,
    dataset_text_field="goal",
    tokenizer=llama_tokenizer,
    args=train_params,
    max_seq_length = 512
)

# Training
fine_tuning.train()

# Save Model
fine_tuning.model.save_pretrained(refined_model)
base_model = LlamaForCausalLM.from_pretrained(
    refined_model,
    device_map={"": 0}
)
base_model.push_to_hub("neurips-model")



