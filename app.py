from fastapi import FastAPI
import torch  # Import your LLM model's library
from typing import Optional
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging
)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)
model =  LlamaForCausalLM.from_pretrained(
    "IsHanGarg/neurips-model",
    device="cuda:0",
    quant_config=quant_config
)
llama_tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b_v2", trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

app = FastAPI()

@app.get("/generate/")
async def generate_text(prompt: str, max_length: Optional[int] = 100):
    # Process the 'prompt' and generate text using your LLM
    # Replace this with code to use your LLM model
    input_ids = llama_tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to("cuda:0")
    generation_output = model.generate(
    input_ids=input_ids, max_new_tokens=32
)
    return {"generated_text": generation_output[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
