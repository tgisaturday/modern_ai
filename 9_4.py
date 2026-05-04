# [프로그램 9-1] Stable Diffusion v1.5 모델 로드하기
#!pip install diffusers transformers accelerate safetensors sympy
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stable Diffusion v1.5 로드
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
).to(device)

pipe.safety_checker = lambda images, **kwargs: (images, False)  
# 실습 편의를 위한 safety checker 비활성화


# [프로그램 9-2] 기본 Text-to-Image 생성하기
prompt = "a watercolor painting of a small cottage in a quiet forest"
image = pipe(prompt).images
plt.imshow(image[0])

# [프로그램 9-3] 스타일 변경으로 이미지 생성 제어하기
prompt = "a small cottage in a quiet forest, cinematic lighting, 3d rendering"
image = pipe(prompt).images
plt.imshow(image[0])


# [프로그램 9-4] 부정 프롬프트로 이미지 생성 제어하기
prompt = "a cozy reading room with warm lighting"
negative = "blurry, distorted, low quality"

image = pipe(prompt, negative_prompt=negative).images
plt.imshow(image[0])


# [프로그램 9-6] 프롬프트의 민감도 관찰하기
generator = torch.Generator(device).manual_seed(0)

image = pipe(
    "a futuristic city skyline at sunset",
    height=512,
    width=512,
    generator=generator
).images

plt.imshow(image[0])

#(6) 프롬프트 민감도 관찰
prompts = [
    "a portrait of a young woman",
    "a portrait of a young woman, smiling",
    "a portrait of a young woman, dramatic lighting",
    "a portrait of a young woman, soft studio lighting"
]

images = [pipe(p).images[0] for p in prompts]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten() 
for i, img in enumerate(images):
    axes[i].imshow(img)
    axes[i].axis('off') 
    axes[i].set_title(f'Prompt {i+1}')
plt.tight_layout()
plt.show()