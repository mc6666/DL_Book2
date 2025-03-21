# insatll：pip install streamlit sqlalchemy -U
# run：streamlit run 02_SQL_generator_web.py
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text
from openai import OpenAI

client = OpenAI()
# 透過SQLAlchemy定義連線字串
connection_string = "sqlite:///Northwind_ORM.db"

# 定義存取所有資料表名稱的函數
@st.cache_data 
def get_table_list():
    engine = create_engine(connection_string, echo=False)
    sql = "SELECT tbl_name FROM sqlite_master where type = 'table'"
    df = pd.read_sql(sql, engine)
    return sorted(list(df.tbl_name))
    
# 定義存取資料表欄位的函數
def get_table_schema(engine, table_list):
    engine = create_engine(connection_string, echo=False)
    table_list_string = ','.join(["'"+table+"'" for table in table_list])
    sql = "SELECT sql FROM sqlite_master where type = 'table' and " + \
          f"tbl_name in ({table_list_string})"
    df = pd.read_sql(sql, engine)
    return '\n'.join(list(df.sql))

# 定義畫面
st.set_page_config(layout="wide")
st.title('SQL萬用產生器')

col1, col2 = st.columns(2)
with col1:
    table_list = st.multiselect('Tables:', get_table_list())
    prompt = st.text_input('SQL:', 
           value='find all orders for the customer name = "Alfreds Futterkiste"')
    
    button1 = st.button('執行')
    sql_generated = st.empty()
    
with col2:
    if button1:
        # SQL生成
        engine = create_engine(connection_string, echo=False)
        prompt_full = "return only SQL statement：\n" + get_table_schema(engine, table_list) \
                        + '\n-- ' + prompt
        print(prompt_full)
        messages=[
            {"role": "system", "content": "you are a Database expert."},
            {"role": "user", "content": prompt_full}
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=messages,
        )
        
        # 去除回答的標點符號
        try:
            if '```' in response.choices[0].message.content:
                sql = response.choices[0].message.content.split('```')[1]
            elif '\n\n' in response.choices[0].message.content:
                sql = response.choices[0].message.content.split('\n\n')[1]
            else:
                sql = response.choices[0].message.content
            if sql.startswith('sql'):
                sql = sql[4:]
        except:
            print(response.choices[0].message.content)
            
        # 執行 SQL，並顯示查詢結果
        print(f'sql:{sql}')
        sql = sql.replace('\n', ' ')
        with sql_generated.container():
            st.markdown(f"{sql}")
        try:
            df = pd.read_sql(sql, engine)
            df
        except Exception as e:
            st.markdown(f"###### {str(e)}")
