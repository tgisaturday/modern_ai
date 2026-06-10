# [프로그램 10-1] HuggingFace Transformers와 PEFT 라이브러리, GPT-2 모델 로드하기
#!pip install transformers datasets peft accelerate

import torch
import datasets
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 토크나이저 및 GPT-2 모델 불러오기
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model     = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

tokenizer.pad_token = tokenizer.eos_token

# [프로그램 10-2] 커스템 텍스트 데이터셋 준비하기
import datasets # Added import for datasets library

texts = [
    "Deep learning enables powerful representations.",
    "Large language models require careful training.",
    "Transformers changed the landscape of NLP.",
    "LoRA enables efficient fine-tuning with low-rank matrices."
]

# Create a Dataset object from the list of strings
# Then wrap it in a DatasetDict to match the typical output of load_dataset
dataset = datasets.DatasetDict({'train': datasets.Dataset.from_dict({'text': texts})})

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

tokenized = dataset.map(tokenize, batched=True)

# [프로그램 10-3]  LoRA 모듈을 구성하고 모델에 적용하기
config = LoraConfig(
    r=8,                     # low-rank 차원
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none"
)

lora_model = get_peft_model(model, config)
lora_model.print_trainable_parameters()

# [프로그램 10-4] 학습 준비하기
optimizer = torch.optim.AdamW(lora_model.parameters(), lr=2e-4)

# [프로그램 10-5] 학습 루프 실행하기
lora_model.train()

for epoch in range(3):
    total_loss = 0
    for batch in tokenized["train"]:
        inputs = torch.tensor(batch["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).unsqueeze(0).to(device)

        optimizer.zero_grad()
        labels = inputs.clone()
        labels[attention_mask == 0] = -100

        outputs = lora_model(
            input_ids=inputs,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# [프로그램 10-6] 파인튜닝된 모델로 텍스트 생성하기

lora_model.eval()

prompt = "Large language models"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = lora_model.generate(
        input_ids,
        max_length=50,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
