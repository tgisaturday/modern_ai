# [프로그램 14-1] DPO 학습 도구 불러오기

#!pip install transformers datasets trl accelerate safetensors

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import DPOTrainer, DPOConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [프로그램 14-2] 토이 데이터셋으로 선호 데이터셋 구성하기
data = {
    "prompt": [
        "Explain what machine learning is.",
        "How should I study mathematics effectively?",
        "What is deep learning?",
        "How can I improve my programming skills?",
        "What is overfitting in machine learning?",
        "How should beginners start learning Python?",
        "What is the purpose of validation data?",
        "How do I prepare for technical interviews?",
        "What is the difference between AI and machine learning?",
        "How can I learn algorithms and data structures well?",
        "What is gradient descent?",
        "How should I read research papers effectively?",
        "What is a neural network?",
        "How can I manage my study time better?",
        "What is regularization and why is it important?",
        "How do I debug code efficiently?",
        "What is the role of data preprocessing?",
        "How can I stay motivated while studying?",
        "What is reinforcement learning?",
        "How should I practice problem solving in math?"
    ],
    "chosen": [
        "Machine learning is a field of study where systems learn patterns from data to make predictions or decisions without being explicitly programmed for every case.",
        "To study mathematics effectively, focus on understanding concepts first, then practice problems regularly and review your mistakes.",
        "Deep learning is a subset of machine learning that uses multi-layer neural networks to learn complex patterns from large amounts of data.",
        "You can improve programming skills by writing code consistently, working on small projects, and reviewing others' code to learn new techniques.",
        "Overfitting occurs when a model learns the training data too well, including noise, and performs poorly on new, unseen data.",
        "Beginners should start learning Python by understanding basic syntax, practicing simple programs, and gradually moving to small projects.",
        "Validation data is used to tune model hyperparameters and evaluate performance during training without biasing the final test results.",
        "Prepare for technical interviews by reviewing fundamentals, practicing coding problems, and explaining your reasoning clearly.",
        "Artificial intelligence is a broad field about building intelligent systems, while machine learning is a specific approach within AI that learns from data.",
        "Learn algorithms and data structures by understanding core ideas, implementing them yourself, and practicing problems of increasing difficulty.",
        "Gradient descent is an optimization method that iteratively updates parameters to minimize a loss function.",
        "Read research papers by first skimming the abstract and figures, then focusing on methods and experiments in detail.",
        "A neural network is a model composed of layers of connected units that transform inputs to outputs through learned weights.",
        "Manage study time by setting clear goals, breaking tasks into smaller steps, and scheduling regular review sessions.",
        "Regularization helps prevent overfitting by constraining model complexity, improving generalization to new data.",
        "Debug code efficiently by reproducing the bug, isolating the problem, and checking assumptions step by step.",
        "Data preprocessing cleans and transforms raw data so that models can learn effectively and reliably.",
        "Staying motivated is easier when you set achievable goals, track progress, and connect learning to long-term objectives.",
        "Reinforcement learning is a framework where an agent learns by interacting with an environment and receiving rewards.",
        "Practice math problem solving by attempting problems independently, analyzing solutions, and revisiting weak areas."
    ],
    "rejected": [
        "Machine learning is basically computer magic that makes machines smart automatically.",
        "Just memorize formulas; understanding the ideas behind them is unnecessary.",
        "Deep learning is just a fancy word for computers thinking like humans.",
        "You get better at programming just by reading code without writing any yourself.",
        "Overfitting means the model is too powerful and therefore always better.",
        "To learn Python, just copy code from the internet without understanding it.",
        "Validation data is not important and can be ignored most of the time.",
        "There is no need to prepare for interviews; talent alone is enough.",
        "AI and machine learning are exactly the same thing with different names.",
        "Memorizing algorithms without understanding them is the fastest way to learn.",
        "Gradient descent is when the computer randomly changes numbers until it works.",
        "Reading only the abstract is enough to fully understand a research paper.",
        "A neural network is a literal simulation of the human brain.",
        "Good time management means studying whenever you feel like it.",
        "Regularization just makes models worse by limiting their power.",
        "Debugging means randomly changing code until the error disappears.",
        "Data preprocessing is optional and usually unnecessary.",
        "Motivation is purely about willpower and cannot be influenced.",
        "Reinforcement learning is just supervised learning with rewards.",
        "The best way to practice math is to memorize solutions without trying."
    ]
}

dataset = Dataset.from_dict(data)

# [프로그램 14-3] 언어 모델 불러오기
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
#성능 향상을 위해 더 큰 모델 사용 가능
#model_name = "Qwen/Qwen2.5-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

policy_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
ref_model    = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# [프로그램 14-4] DPO 학습 구성하기
use_cuda = torch.cuda.is_available()

config = DPOConfig(
    beta=0.1,
    learning_rate=1e-5,
    max_length=256,
    use_cpu=not use_cuda,
    fp16=False,
    bf16=use_cuda,   # GPU면 bf16 사용
)
trainer = DPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    args=config,
    processing_class=tokenizer,
    train_dataset=dataset,
)

# [프로그램 14-5] DPO 학습 실행하기
trainer.train()

# [프로그램 14-6] 학습된 정책 평가하기
def generate_greedy(model, prompt):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = "How should I study mathematics effectively?"
print("=== Before (Reference, greedy) ===")
print(generate_greedy(ref_model, prompt))
print("\n=== After (Policy, greedy) ===")
print(generate_greedy(policy_model, prompt))
