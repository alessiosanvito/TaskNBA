import os
import streamlit as st
# from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from streamlit_chat import message
from langchain.memory import (ConversationBufferMemory, ConversationSummaryMemory,
ConversationKGMemory, CombinedMemory, ConversationBufferWindowMemory)
from langchain.agents.agent_types import AgentType
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import time

from secrets_key import templateAgent

# Set the title for the Streamlit app
st.title('NBAgpt 🏀🧠')

# Get the OpenAI API key from the user through a password input field in the sidebar
user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-...",
    type="password")
# Display a warning if the API key is not provided
if not user_api_key:
  st.warning('Please enter a valid API key', icon="⚠️")
# Display a success message if the API key is provided
if user_api_key:
  st.success('API key entered successfully', icon="✅")

# Display useful links in the sidebar
with st.sidebar.expander(":link: Useful links"):
    st.write("**LLM implementation:** [LangChain documentation](https://python.langchain.com/docs/get_started/introduction.html)")
    st.write("**LLM for graphs:** [PandasAI documentation](https://docs.pandas-ai.com/en/latest/)")
    st.write("**Dataset** [Kaggle NBA](https://www.kaggle.com/datasets/nathanlauga/nba-games/data)")

#### FILE CSV
games = pd.read_csv('C:/Users/Alessio/Desktop/TaskNBA/games.csv')
games_details_orig = pd.read_csv('C:/Users/Alessio/Desktop/TaskNBA/games_details.csv', low_memory=False)
games_details = pd.read_csv('C:/Users/Alessio/Desktop/TaskNBA/games_details_seas.csv', low_memory=False)
players = pd.read_csv('C:/Users/Alessio/Desktop/TaskNBA/players.csv')
ranking = pd.read_csv('C:/Users/Alessio/Desktop/TaskNBA/ranking.csv')
teams = pd.read_csv('C:/Users/Alessio/Desktop/TaskNBA/teams.csv')
games['GAME_DATE_EST']= pd.to_datetime(games['GAME_DATE_EST'])

# Set the OpenAI API key in the environment variable
os.environ['OPENAI_API_KEY'] = user_api_key

# Ask the user what they want to do
selected_mode = st.selectbox("What do you want to do?", ["", "Ask your data",
  'Ask for some plot'])

# If the user chooses to ask about data
if selected_mode=='Ask your data':
    
    mname = 'gpt-4-1106-preview'
    # Set up the ChatOpenAI model with gpt4 model
    llm=ChatOpenAI(model_name=mname,
            temperature=0,
            openai_api_key=user_api_key)

    # ConversationBufferMemory - This memory allows for storing messages and then extracts the messages in a variable.
    chat_history_buffer = ConversationBufferMemory(memory_key="chat_history",input_key="input")

    # ConversationSummaryMemory - This type of memory creates a summary of the conversation over time. This can be useful for condensing information from the conversation over time.
    chat_history_summary = ConversationSummaryMemory(llm=llm,memory_key="chat_history_summary",input_key="input")

    # Conversation Knowledge Graph Memory - This type of memory uses a knowledge graph to recreate memory.
    chat_history_KG = ConversationKGMemory(llm=llm,memory_key="chat_history_KG",input_key="input")
    
    # All those memory classes are used in the same chain initializing and usingCombinedMemory class
    memory = CombinedMemory(memories=[chat_history_buffer, chat_history_summary, chat_history_KG])
        
    ######## AGENT
    
    # Create an agent that can answer questions about the pandas dataframe
    # passing the dataframes as a list, set verbose=True to print out more debugging info
    # use the AgentType.OPENAI_FUNCTIONS agent type which allows us to execute dataframe operations like .mean(), .shape etc inside the agent.
    agent = create_pandas_dataframe_agent(llm, [games, games_details, players, ranking, teams], verbose=True, memory=memory)
    # Define the prompt for the agent
    agent.agent.llm_chain.prompt.template = templateAgent

    # Initialize chat history in the session state        
    if "chat_history" not in st.session_state:
      st.session_state.chat_history = []
    if 'something' not in st.session_state:
      st.session_state.something = ''
    
    # Define a function to handle user submission
    def submit():
      st.session_state.something = st.session_state.widget
      st.session_state.widget = ''
    
    # Get user input through a text input field
    user_input = st.text_input("Enter your question:", key="input_field")
    st.button('Submit', type='primary')

    # Process user input and display chat history
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

    # Define a function to clear the chat history
    def clear_chat():
      st.session_state.chat_history=[]
      st.session_state.user_input = ""

    st.button('Clear conversation', on_click=clear_chat, type='primary')

# If the user chooses to ask for a plot
if selected_mode=='Ask for some plot':
  from pandasai import SmartDataframe, SmartDatalake
  import pandas as pd
  from langchain.chat_models import ChatOpenAI

  # List of available dataframe, choose one of them
  available_dataframes = {
        'Games': games,
        'Games Details Original': games_details_orig,
        'Games Details': games_details,
        'Players': players,
        'Ranking': ranking,
        'Teams': teams
    }

  # Box to select a dataframe
  selected_dataframe_name = st.selectbox("Select a DataFrame:", list(available_dataframes.keys()))

  df = available_dataframes[selected_dataframe_name]

  st.sidebar.write(f"### Selected DataFrame: {selected_dataframe_name}")

  # Define the explanations for each dataframe
  df_explanations = {
        'Games': 'All games from 2004 season to 2020 season, teams, and some details like the number of points, etc.',
        'Games Details Original': 'Details of games dataset, all statistics of players for a given game.',
        'Games Details': 'Details of games dataset with season information, all statistics of players for a given game.',
        'Players': 'Players details (name).',
        'Ranking': 'Ranking of NBA given a day (split into west and east on CONFERENCE column).',
        'Teams': 'All teams of NBA with infos of each of them.'
    }
  
  # Print it into the sidebar
  st.sidebar.write(f"### DataFrame Explanation: {df_explanations[selected_dataframe_name]}")

  st.write("### Preview of the selected DataFrame:")
  
  # Set up a preview for the selected dataframe
  with st.expander('Preview of the selected DataFrame:'):
      st.table(df.head())

  # Set up ChatOpenAI model for the SmartDataframe
  mname = 'gpt-4-1106-preview'
  llm=ChatOpenAI(model_name=mname,
          temperature=0,
          openai_api_key=user_api_key)
  
  # teams = SmartDataframe(teams_df, name="teams", config={"llm":llm})
  df_smart = SmartDataframe(df, name="df", config={"llm":llm})
  
  # Define a function to handle user submission
  def submit():
    st.session_state.something = st.session_state.widget
    st.session_state.widget = ''
  
  # Get user input through a text input field
  user_input = st.text_input("Enter your question:", key="input_field")
  st.button('Submit', type='primary')

  # Process user input and display the generated plot
  if user_input and st.button:
    df_smart.chat(user_input)
    # Wait for the full upload of the image
    time.sleep(5)
    # Show the image
    fig = Image.open('C:/Users/Alessio/Desktop/TaskNBA/temp_chart.png')
    st.image(fig)
  
  # Define a function to clear the chat history and delete the temporary chart file
  def clear_chat():
    st.session_state.user_input = ""
    os.remove('C:/Users/Alessio/Desktop/TaskNBA/temp_chart.png')

  # Display a button to clear the conversation
  st.button('Clear conversation', on_click=clear_chat, type='primary')