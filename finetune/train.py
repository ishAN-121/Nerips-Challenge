import torch
from datasets import load_dataset
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
from dotenv import dotenv_values

env_vars = dotenv_values('.env')
dataset = load_dataset("piqa", split="train")
llama_tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b_v2", trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right" 


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)
base_model = LlamaForCausalLM.from_pretrained(
    "openlm-research/open_llama_3b_v2",
    quantization_config=quant_config,
    device_map={"": 0}
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1
print("base model done")

peft_parameters = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=32,
    bias="none",
    task_type="CAUSAL_LM"
)
train_params = TrainingArguments(
    output_dir=env_vars.get("Output_Dir"),
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

fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=peft_parameters,
    dataset_text_field="goal",
    tokenizer=llama_tokenizer,
    args=train_params,
    max_seq_length = 512
)


fine_tuning.train()

fine_tuning.model.save_pretrained(env_vars.get("Refined_Model"))
base_model = LlamaForCausalLM.from_pretrained(
    env_vars.get("Refined_Model"),
    device_map={"": 0}
)
base_model.push_to_hub(env_vars.get("Refined_Model_Name"))



