import os
import streamlit as st
# from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from streamlit_chat import message
from langchain.memory import (ConversationBufferMemory, ConversationSummaryMemory,
ConversationKGMemory, CombinedMemory, ConversationBufferWindowMemory)
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import time

from secrets_key import templateAgent

st.title('NBAgpt 🏀🧠')

user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-...",
    type="password")
if not user_api_key:
  st.warning('Please enter a valid API key', icon="⚠️")
if user_api_key:
  st.success('API key entered successfully', icon="✅")

with st.sidebar.expander(":link: Useful links"):
    st.write("**LLM implementation:** [LangChain documentation](https://python.langchain.com/docs/get_started/introduction.html)")
    st.write("**Dataset** [Kaggle NBA](https://www.kaggle.com/datasets/nathanlauga/nba-games/data)")

#### FILE CSV
games = pd.read_csv('C:/Users/Alessio/Desktop/TaskNBA/games.csv')
games_details_orig = pd.read_csv('C:/Users/Alessio/Desktop/TaskNBA/games_details.csv', low_memory=False)
games_details = pd.read_csv('C:/Users/Alessio/Desktop/TaskNBA/games_details_seas.csv', low_memory=False)
players = pd.read_csv('C:/Users/Alessio/Desktop/TaskNBA/players.csv')
ranking = pd.read_csv('C:/Users/Alessio/Desktop/TaskNBA/ranking.csv')
teams = pd.read_csv('C:/Users/Alessio/Desktop/TaskNBA/teams.csv')
games['GAME_DATE_EST']= pd.to_datetime(games['GAME_DATE_EST'])

os.environ['OPENAI_API_KEY'] = user_api_key

selected_mode = st.selectbox("What do you want to do?", ["", "Ask your data",
  'Ask to plot something'])

if selected_mode=='Ask your data':
    
    mname = 'gpt-4-1106-preview'
    llm=ChatOpenAI(model_name=mname,
            temperature=0,
            openai_api_key=user_api_key)

    chat_history_buffer = ConversationBufferMemory(memory_key="chat_history",input_key="input")
    chat_history_summary = ConversationSummaryMemory(llm=llm,memory_key="chat_history_summary",input_key="input")
    chat_history_KG = ConversationKGMemory(llm=llm,memory_key="chat_history_KG",input_key="input")
    memory = CombinedMemory(memories=[chat_history_buffer, chat_history_summary, chat_history_KG])
        
    ######## AGENT
    
    agent = create_pandas_dataframe_agent(llm, [games, games_details, players, ranking, teams], verbose=True)
    agent.agent.llm_chain.prompt.template = templateAgent
            
    if "chat_history" not in st.session_state:
      st.session_state.chat_history = []
    if 'something' not in st.session_state:
      st.session_state.something = ''
    def submit():
      st.session_state.something = st.session_state.widget
      st.session_state.widget = ''
    user_input = st.text_input("Enter your question:", key="input_field")
    st.button('Submit', type='primary')

    if user_input and st.button:
      answer = agent.run(user_input)
      st.session_state.chat_history.append(("user", user_input))
      st.session_state.chat_history.append(("agent", answer))

    for i, (sender, message_text) in enumerate(st.session_state.chat_history):
      if sender == "user":
        message(message_text, is_user=True, key=f"{i}_user")
      else:
        message(message_text, key=f"_{i}")

    fig = plt.gcf()
    if fig.get_axes():
      st.pyplot(fig)

      #st.write(answer)

    def clear_chat():
      st.session_state.chat_history=[]
      st.session_state.user_input = ""

    st.button('Clear conversation', on_click=clear_chat, type='primary')

if selected_mode=='Ask to plot something':
  from pandasai import SmartDataframe, SmartDatalake
  import pandas as pd
  # from pandasai.llm import OpenAI
  # from langchain.llms import OpenAI
  from langchain.chat_models import ChatOpenAI

  available_dataframes = {
        'Games': games,
        'Games Details Original': games_details_orig,
        'Games Details': games_details,
        'Players': players,
        'Ranking': ranking,
        'Teams': teams
    }

  
  selected_dataframe_name = st.selectbox("Select a DataFrame:", list(available_dataframes.keys()))

  df = available_dataframes[selected_dataframe_name]

  st.sidebar.write(f"### Selected DataFrame: {selected_dataframe_name}")

  df_explanations = {
        'Games': 'All games from 2004 season to 2020 season, teams, and some details like the number of points, etc.',
        'Games Details Original': 'Details of games dataset, all statistics of players for a given game.',
        'Games Details': 'Details of games dataset with season information, all statistics of players for a given game.',
        'Players': 'Players details (name).',
        'Ranking': 'Ranking of NBA given a day (split into west and east on CONFERENCE column).',
        'Teams': 'All teams of NBA with infos of each of them.'
    }
  
  st.sidebar.write(f"### DataFrame Explanation: {df_explanations[selected_dataframe_name]}")

  st.write("### Preview of the selected DataFrame:")
  
  with st.expander('Preview of the selected DataFrame:'):
      st.table(df.head())

  mname = 'gpt-4-1106-preview'
  llm=ChatOpenAI(model_name=mname,
          temperature=0,
          openai_api_key=user_api_key)
  
  # games = SmartDataframe(GGDPmerge_df, name="games", config={"llm":llm})
  # teams = SmartDataframe(teams_df, name="teams", config={"llm":llm})
  df_smart = SmartDataframe(df, name="df", config={"llm":llm})

  # lake = SmartDatalake(
  # [games, games_details], # players, ranking, teams],
  # config={"llm": llm}
  # )

  # sdf = SmartDataframe(df, config={"llm": llm})
  # sdf.chat("Plot a chart of the gdp by country")
  
  def submit():
    st.session_state.something = st.session_state.widget
    st.session_state.widget = ''
  user_input = st.text_input("Enter your question:", key="input_field")
  st.button('Submit', type='primary')

  if user_input and st.button:
    df_smart.chat(user_input)
    time.sleep(5)
    fig = Image.open('C:/Users/Alessio/Desktop/TaskNBA/temp_chart.png')
    st.image(fig) #, caption='Plot of points scored by Pelicans', use_column_width=True)
  
  def clear_chat():
    
    st.session_state.user_input = ""
    os.remove('C:/Users/Alessio/Desktop/TaskNBA/temp_chart.png')

  st.button('Clear conversation', on_click=clear_chat, type='primary')
