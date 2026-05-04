# [프로그램 13-1] Qwen2.5 모델 불러오기
#!pip install transformers accelerate

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # 예시용 모델
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

# [프로그램 13-2] 기본 프롬프팅 적용하기 : 중간 추론이 없는 방식
def generate(prompt, max_new_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

problem = "한 상자에 사과가 8개 들어 있다. 상자 5개에는 사과가 모두 몇 개 있는가?"
print(generate(problem))

# [프로그램 13-3] CoT 프롬프팅 적용하기 : 단계적 추론 유도
cot_prompt = (
    "한 상자에 사과가 8개 들어 있다. 상자 5개에는 사과가 모두 몇 개 있는가?\n"
    "생각의 사슬을 따라 단계적으로 추론하되, 마지막 줄에만 정답을 다음 형식으로 출력하라.\n"
    "FINAL: <숫자>\n"
    "1. 상자 하나에 8개가 있다.\n"
    "2. 상자가 5개 있다.\n"
    "3. 따라서 총 사과 개수는 8*5 = 40\n"
    "FINAL: "
)


print(generate(cot_prompt))

# [프로그램 13-4] 자기 일관성 프롬프팅 적용 : 여러 개의 추론 경로를 샘플링
import re

def extract_last_int(text: str):
    nums = re.findall(r"\d+", text)
    return int(nums[-1]) if nums else None

def sample(
    prompt,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9
):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def sample_cot(prompt, n=5):
    answers = []
    for _ in range(n):
        out = sample(
                        prompt,
                        max_new_tokens=64,
                        temperature=0.7,
                        top_p=0.9
                    )
        answers.append(extract_last_int(out))
    return answers

answers = sample_cot(cot_prompt, n=8)
print("샘플링된 답:", answers)

# [프로그램 13-5] CoT 프롬프트 구조의 확장: 명시적 추론 템플릿 사용
template_prompt = (
    "문제를 단계별로 분석하자.\n"
    "문제: 두 수의 합은 17이고, 차는 3이다. 두 수를 구하라.\n"
    "해결 절차:\n"
    "1. 조건을 정리한다.\n"
    "2. 두 수 x,y에 대해 x+y=17, x-y=3이다.\n"
    "3. 두 식을 더하면 2x=20이다.\n"
    "4. 따라서 x=10이고 y=7이다.\n"
    "답:"
)
print(generate(template_prompt))
