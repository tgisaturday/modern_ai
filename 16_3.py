# [프로그램 16-1] 서비스 구현을 위한 라이브러리 설치하기
#!pip install llama-index llama-index-llms-openai \
#             llama-index-embeddings-huggingface \
#            transformers sentencepiece accelerate datasets \
#             sounddevice soundfile openai-whisper gtts

# [프로그램 16-2] 사용자 지식 기반 문서 구성하기
personal_docs = [
    "사용자는 게임 오목을 좋아하며 주로 5x5 미니게임을 즐긴다.",
    "사용자는 커피보다 차(茶)를 선호한다.",
    "사용자는 매일 6시에 헬스장을 간다.",
    "사용자는 최근 AI 관련 대학원 프로젝트를 진행하고 있다.",
]

# [프로그램 16-3] LlamaIndex 기반 RAG 구성하기
from llama_index.core import VectorStoreIndex, Document, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 임베딩 모델 선택
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 사용자 지식을 Document 형태로 변환
docs = [Document(text=d) for d in personal_docs]

# 인덱스 생성
index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)

query_engine = index.as_retriever(similarity_top_k=3)

# [프로그램 16-4] Whisper를 사용하여 음성 전사하기
import whisper

asr_model = whisper.load_model("small")  # 속도와 품질의 균형을 고려한 선택

# 음성 입력을 받아 텍스트로 변환하는 함수
def transcribe_audio(audio_path):
    result = asr_model.transcribe(audio_path, fp16=False)
    return result["text"]

# [프로그램 16-5] 언어 모델 응답 생성하고 RAG 결합하기
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

llm_name = "Qwen/Qwen2.5-1.5B-Instruct"  # 예시 소형 LLM
tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm = AutoModelForCausalLM.from_pretrained(llm_name).to(device)

def generate_llm_answer(query, context):
    prompt = (
        "다음은 사용자 개인 정보가 포함된 문서에서 추출한 문맥이다.\n"
        f"문맥: {context}\n\n"
        f"사용자 질문: {query}\n"
        "문맥을 고려하여 적절한 응답을 제공하라.\n"
        "답변:"
    )
    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    out = llm.generate(**tokens, max_new_tokens=128)
    return tokenizer.decode(out[0], skip_special_tokens=True).split("답변:")[-1]

def answer_with_rag(query):
    nodes = query_engine.retrieve(query)   
    context = "\n".join([n.node.get_content() for n in nodes])  # 검색된 문서 내용 합치기
    answer = generate_llm_answer(query, context)
    return answer


# [프로그램 16-6] gtts를 활용하여 음성 합성 구성하기
from gtts import gTTS
import os

def text_to_speech(text, out_path="response.mp3"):
    tts = gTTS(text=text, lang="ko")
    tts.save(out_path)
    return out_path

# [프로그램 16-7-1] 개인화 음성 어시스턴트 파이프라인 구성하기

#!apt-get -qq update
#!apt-get -qq install -y ffmpeg
#!pip -q install pydub

from google.colab import files
from pydub import AudioSegment
import os

def ensure_user_query_wav():
    print("user_query.wav 로 사용할 오디오 파일을 업로드하세요. (wav/mp3/m4a/webm/ogg 등 가능)")
    uploaded = files.upload()  # 업로드 UI 표시

    if not uploaded:
        raise RuntimeError("업로드된 파일이 없습니다.")

    # 업로드된 첫 파일 선택
    src_name = next(iter(uploaded.keys()))
    print("업로드된 파일:", src_name)

    # 이미 wav면 이름만 통일
    if src_name.lower().endswith(".wav"):
        if src_name != "user_query.wav":
            os.replace(src_name, "user_query.wav")
        print("저장 완료: user_query.wav")
        return "user_query.wav"

    # wav가 아니면 wav로 변환
    audio = AudioSegment.from_file(src_name)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export("user_query.wav", format="wav")

    print("변환 완료: user_query.wav (16kHz mono)")
    return "user_query.wav"

path = ensure_user_query_wav()
print("최종 파일:", path, "크기:", os.path.getsize(path), "bytes")

# [프로그램 16-7-2] 함수 호출하여 응답 확인하기
def personal_voice_assistant(audio_path):
    # 1. 음성 → 텍스트
    query = transcribe_audio(audio_path)

    # 2. 개인화 RAG 기반 LLM 응답 생성
    answer = answer_with_rag(query)

    # 3. 텍스트 → 음성 변환
    voice_path = text_to_speech(answer)

    return query, answer, voice_path

query_text, answer_text, audio_out = personal_voice_assistant("user_query.wav")
print("사용자 질문:", query_text)
print("모델 답변:", answer_text)
print("음성 파일:", audio_out)

