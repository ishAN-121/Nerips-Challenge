import os
import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
)
from dotenv import dotenv_values
import argparse

parser = argparse.ArgumentParser(description='Prompt for inference')
    
parser.add_argument('-p', '--prompt', dest='prompt', required=True,help='Input prompt')
args = parser.parse_args()
env_vars = dotenv_values('.env')
llama_tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b_v2", trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"  

base_model = LlamaForCausalLM.from_pretrained(
    env_vars.get("Refined_Model"),
    device_map={"": 0}
)
prompt = args.prompt
input_ids = llama_tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.to("cuda:0")

generation_output = base_model.generate(
    input_ids=input_ids, max_new_tokens=32
)
print(llama_tokenizer.decode(generation_output[0]))

