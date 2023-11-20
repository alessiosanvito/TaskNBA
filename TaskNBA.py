### IMPORT


from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
# from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import streamlit as st
from streamlit_chat import message
import tempfile
import os


#### KEY + PROMPT
from secrets_key import OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

from secrets_key import templateAgent, base_template_agent_pandas
# os.environ['single_template'] = single_template



#### FILE CSV
games = pd.read_csv('C:/Users/Alessio/Desktop/TaskNBA/games.csv')
games_details_orig = pd.read_csv('C:/Users/Alessio/Desktop/TaskNBA/games_details.csv', low_memory=False)
games_details = pd.read_csv('C:/Users/Alessio/Desktop/TaskNBA/games_details_seas.csv', low_memory=False)
players = pd.read_csv('C:/Users/Alessio/Desktop/TaskNBA/players.csv')
ranking = pd.read_csv('C:/Users/Alessio/Desktop/TaskNBA/ranking.csv')
teams = pd.read_csv('C:/Users/Alessio/Desktop/TaskNBA/teams.csv')
games['GAME_DATE_EST']= pd.to_datetime(games['GAME_DATE_EST'])


##### AGENT

agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0, model="gpt-4-1106-preview"), [games, games_details, players, ranking, teams], verbose=True)

### PROMPT GIUSTO ALL'AGENT
agent.agent.llm_chain.prompt.template = templateAgent
# agent.agent.llm_chain.prompt.template = base_template_agent_pandas



############# PROVE DOMANDE
# agent.run("how many points did the Pelicans score in the game of 22-12-2022 against the Spurs?") # 126
# agent.run("How many points did Devin Booker score against Boston in the 24-03-2017 match?") # 70
# agent.run("How many times did Bol Bol play 0 minutes?") # 2

# agent.run("Provide a summary of Bol Bol's career statistics")
# agent.run("What was the ranking of Memphis in the Western conference in 22022?")

# agent.run("Compare the shooting accuracy of free throws of Kobe bryant and Lebron James in the 2010 season")

agent.run("Tell me the top 5 players for shooting accuracy of free throws in season 2019 that have played at least 600 minutes")