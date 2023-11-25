from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    TrainingArguments
)
from trl import RewardTrainer
import os
from datasets import load_dataset

data_name = "Anthropic/hh-rlhf"
training_data = load_dataset(data_name, split="train")
os.environ['TRANSFORMERS_CACHE'] = '../../../scratch/tushar_s.iitr/models'

gpt_tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b")
gpt_model = AutoModelForSequenceClassification.from_pretrained("tiiuae/falcon-rw-1b")
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
gpt_tokenizer.padding_side = "right"
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

trainer = RewardTrainer(
    model=gpt_model,
    tokenizer=gpt_tokenizer,
    train_dataset=training_data,
    args = train_params
)
trainer.args.train_batch_size = 4
trainer.args.train_max_steps = 1000

trainer.train()