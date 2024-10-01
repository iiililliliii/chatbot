import streamlit as st
from utils import print_messages
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import requests
from streamlit_chat import message as st_message


st.set_page_config(page_title="ChatGPT", page_icon="★")
st.title("★ ChatGPT ★")

# Hugging Face Token 활용해서 M2M100 1.2B 모델 사용
API_URL = "https://api-inference.huggingface.co/models/facebook/m2m100_1.2B"
headers = {"Authorization": "Bearer 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"}
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
output = query({
    "inputs": "The answer to the universe is",
})


#대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Hugging Face API 호출 함수
def query_huggingface_api(input_text):
    data = {"inputs": input_text}
    response = requests.post(API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()[0]['generated_text']  # 모델이 생성한 텍스트 반환
    else:
        return f"Error: {response.status_code}"

# 이전 대화기록을 출력해 주는 코드
print_messages()

# 이전 대화 기록 출력
print_messages()

# 사용자 입력 처리
if user_input := st.chat_input("메시지를 입력해 주세요."):
    # 사용자의 메시지를 세션에 추가하고 표시
    st_message(user_input, is_user=True)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Hugging Face API 호출하여 AI의 답변 생성
    with st.spinner("AI가 답변을 생성하는 중입니다..."):
        ai_response = query_huggingface_api(user_input)

    # AI의 답변을 세션에 추가하고 표시
    st_message(ai_response)
    st.session_state["messages"].append({"role": "assistant", "content": ai_response})
