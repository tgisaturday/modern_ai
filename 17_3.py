# [프로그램 17-1] 서비스 배포를 위한 웹 서버 환경 구축하기
#!pip install fastapi uvicorn python-multipart gradio 
#!pip install llama-index llama-index-llms-openai \
#             llama-index-embeddings-huggingface \
#            transformers sentencepiece accelerate datasets \
#             sounddevice soundfile openai-whisper gtts


# [프로그램 17-2] 모델 초기화 및 API 엔드 포인트 구현하기
# 코드 실행시 "api_server.py" 파일이 생성된다.
api_code = """
# api_server.py
import os
import uuid
import tempfile

import whisper
import torch
from gtts import gTTS
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse

from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Whisper ASR
asr_model = whisper.load_model("small")

# RAG
personal_docs = [
    "사용자는 게임 오목을 좋아하며 주로 5x5 미니게임을 즐긴다.",
    "사용자는 커피보다 차(茶)를 선호한다.",
    "사용자는 매일 6시에 헬스장을 간다.",
    "사용자는 최근 AI 관련 대학원 프로젝트를 진행하고 있다.",
]
embed_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")
docs = [Document(text=d) for d in personal_docs]
rag_index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
rag_engine = rag_index.as_retriever(similarity_top_k=3)

# LLM
llm_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm = AutoModelForCausalLM.from_pretrained(llm_name).to(device)

OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def transcribe(path: str) -> str:
    result = asr_model.transcribe(path, fp16=False)
    return result["text"]

def rag_answer(query: str) -> str:
    nodes = rag_engine.retrieve(query)
    ctx = "\n".join([n.node.get_content() for n in nodes])
    prompt = (
        "문맥: " + ctx + "\n"
        "사용자 질문: " + query + "\n"
        "문맥과 질문의 관계를 해석하여 간결하고 정확하게 응답하라.\n"
        "답변:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("답변:")[-1]

def synthesize(text: str) -> str:
    out_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}.mp3")
    tts = gTTS(text=text, lang="ko")
    tts.save(out_path)
    return out_path

def run_voice_pipeline(audio_path: str):
    query_text = transcribe(audio_path)
    answer_text = rag_answer(query_text)
    out_path = synthesize(answer_text)
    return query_text, answer_text, out_path

@app.post("/voice")
async def voice_assistant(audio: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        query_text, answer_text, out_path = run_voice_pipeline(tmp_path)
        return {
            "query": query_text,
            "answer": answer_text,
            "audio_file": out_path
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

@app.get("/audio")
def get_audio(path: str):
    if not path.endswith(".mp3"):
        raise HTTPException(status_code=400, detail="mp3 only")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path, media_type="audio/mpeg", filename="response.mp3")
"""

with open("api_server.py", "w") as f:
    f.write(api_code)

# 실행:
# uvicorn api_server:app --host 0.0.0.0 --port 8000

# [프로그램 17-3] 웹 클라이언트 데모 구현하기
#코드 실행시 "demo_ui.py" 파일이 생성된다.
ui_code = """
# demo_ui.py
import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/voice"

def call_api(audio_path):
    if audio_path is None:
        return "", "", None

    with open(audio_path, "rb") as f:
        files = {"audio": ("audio.wav", f, "audio/wav")}
        r = requests.post(API_URL, files=files, timeout=600)

    r.raise_for_status()
    data = r.json()

    return (
        data.get("query", ""),
        data.get("answer", ""),
        data.get("audio_file", None)
    )

with gr.Blocks() as demo:
    gr.Markdown("## Voice Assistant Client (Gradio share)")
    audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="음성 입력")
    txt_query = gr.Textbox(label="전사된 텍스트")
    txt_answer = gr.Textbox(label="모델 응답")
    audio_out = gr.Audio(label="음성 응답", type="filepath")
    btn = gr.Button("실행")

    btn.click(
        fn=call_api,
        inputs=audio_in,
        outputs=[txt_query, txt_answer, audio_out]
    )
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
"""

with open("demo_ui.py", "w") as f:
    f.write(ui_code)

# [프로그램 17-4] 개인화 음성 어시스턴트 배포판 실행하기
# !uvicorn api_server:app --host 0.0.0.0 --port 8000 & python demo_ui.py

# API Server는 background에서 실행되므로, 
# INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
# 위 메시지가 뜬 후 gradio의 public URL에 접속해서 demo를 확인한다.

# [프로그램 17-5] 개인화 음성 어시스턴트 확장판 실행하기
#!pip install docsray
#!docsray setup
#!docsray download-models --model-type lite   # 4b model (~3GB)
#!docsray web
