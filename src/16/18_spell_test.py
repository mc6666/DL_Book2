# pip install audio_recorder_streamlit
# 載入套件
import openai, random
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import time, os
from openai import OpenAI

# 建立 OpenAI 用戶端
client = OpenAI()

# 暫存檔名設定
tmp_audio_file = './whisper/tmp.wav'

@st.cache_data
def get_question_list():
    # 讀取檔案
    with open('./whisper/英文佳句座右銘.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()
    clean_lines = []
    for line in lines:
        if len(line)<=0 or line == '\n': continue
        clean_lines.append(line.split('.')[0]+'.')
    return clean_lines

def get_random_question(question_list):
    return random.choice(question_list)

question_list = get_question_list()

# 建立畫面
st.title('英語測驗')

# 測驗題目
placeholder = st.empty()
if 'key' not in st.session_state:
    question = get_random_question(question_list)
    st.session_state.key = question
    placeholder.text(question)
audio_bytes = audio_recorder(pause_threshold=2.0)
    
if audio_bytes:
    # 錄音存檔
    with open(tmp_audio_file, mode='wb') as f:
        f.write(audio_bytes)    

    # 呼叫 OpenAI API，取得辨識結果
    audio_file= open(tmp_audio_file, "rb")
    content = client.audio.translations.create(
                model="whisper-1", file=audio_file).text
    # time.sleep(3)
        
    # 辨識結果與題目比對
    st.text('題目：'+st.session_state.key)
    st.text('回答：'+content)
    st.markdown("## " + '答對了!!' 
        if content.lower()==st.session_state.key.lower() 
        else '答錯了!!')

    # 出下一題
    question = get_random_question(question_list)
    st.session_state.key = question
    placeholder.text(question)
