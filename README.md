# Neurips LLM Efficiency Challenge

The repo contains our submission to the Neurips LLM efficiency challenge.

## Problem Statement 

 The problem statement was to fine tune a LLM model on a single gpu within 24 hours.

The repo has 3 directories : - 

## Finetune

The directory conatins the source code for fine tuning the LLM. We have used open llama-3b v2 for finetuning using LoRA(Low Rank Adaptation) technique and a SFT Trainer.
To finetune:
```
cd finetune
cp .env.example .env
```
Now fill the env file

```
pip install -r requirements.txt
huggingface-cli login
python train.py
```
For inference
```
python inference.py -p "Ask question"
```

## Submission

This contains the source code for a simple python server and a dockerfile as demanded in the submission.For running our submission first fill the dockerfile and the run
```
docker build -t app:tag
docker run -p 8000:8000 app:tag 
```
Now make a curl request to the docker container.
```
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Your prompt text here", "max_length": 100}' http://container_ip_address:8000/generate/
```

## Future Work

The directory conatins the source code for the future work we would have done if we had more time. We tried to use RLHF to fine tune the model. The directory contains the source code for training  a reward model and which can be used for RLHF in future.
