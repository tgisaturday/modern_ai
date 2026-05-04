# [프로그램 8-1] GPT-2 모델과 토크나이저 로드하기
#!pip install transformers

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 토크나이저와 모델 로드
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model     = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# 모델을 평가 모드로 설정
model.eval()

# [프로그램 8-2] 입력 문장을 토크나이저 및 텐서로 변환하기
prompt = "Artificial intelligence will"

input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# [프로그램 8-3] 텍스트 생성 수행하기

output_ids = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    top_p=0.9,
    temperature=0.8
)

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_text)

# [프로그램 8-4] 랜덤 시드 고정하기
torch.manual_seed(0)
output_ids = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    top_p=0.9,
    temperature=0.8
)
print(tokenizer.decode(output_ids[0]))

