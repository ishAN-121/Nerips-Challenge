from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForSequenceClassification, 
    AutoTokenizer
)
import os

os.environ['TRANSFORMERS_CACHE'] = '../../../scratch/tushar_s.iitr/models'

# Initialize the tokenizer
llama_tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b_v2", trust_remote_code=True)
llama_tokenizer.save_pretrained("../../../scratch/tushar_s.iitr/models/tokenizer")
print("=============================================================================")
print("Tokenizer downloaded")
print("=============================================================================")


# Initialize the base model
base_model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b_v2")
base_model.save_pretrained("../../../scratch/tushar_s.iitr/models/model")
print("=============================================================================")
print("Model downloaded")
print("=============================================================================")


       
     