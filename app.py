import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date
from fbprophet import Prophet
from fbprophet.plot import plot, plot_components
from plotly import graph_objs as go


DATA_INICIO = '2023-01-01'
DATA_FIM = date.today().strftime('%Y-%m-%d')

st.title('Análise de Ações')


#Creating side bar on app

st.sidebar.header('Escolha a ação')


n_dias = st.slider('Quantidade de dias de previsão', 30, 365)

#Create function to get action name, ticket. Concanete name + ticket

def pegar_dados_acoes():
    path = '/Users/juliodantas/PycharmProjects/testeprophet/acoes.csv'
    return pd.read_csv(path, delimiter=';')

df = pegar_dados_acoes()

acao = df['snome']
nome_acao_escolhida = st.sidebar.selectbox('Escolha uma ação:', acao)

df_acao = df[df['snome'] == nome_acao_escolhida]
acao_escolhida = df_acao.iloc[0]['sigla_acao']
acao_escolhida = acao_escolhida + '.SA'

@st.cache_data
def pegar_valores_online(sigla_acao):
    df = yf.download(sigla_acao, DATA_INICIO, DATA_FIM)
    df.reset_index(inplace=True)
    return df

df_valores = pegar_valores_online(acao_escolhida)

st.subheader('Tabela de valores - ' + nome_acao_escolhida)
st.write(df_valores.tail(10))

st.subheader('Gráfico de preços')
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_valores['Date'],
                         y=df_valores['Close'],
                         name='Preço Fechamento',
                         line_color='yellow'))

fig.add_trace(go.Scatter(x=df_valores['Date'],
                         y=df_valores['Open'],
                         name='Preço Abertura',
                         line_color='blue'))

st.plotly_chart(fig)

#Previsão

df_treino = df_valores[['Date', 'Close']]

#Renomear Colunas

df_treino = df_treino.rename(columns = {"Date": 'ds', 'Close': 'y'})

modelo = Prophet()
modelo.fit(df_treino)

futuro = modelo.make_future_dataframe(periods=n_dias, freq='B')
previsao = modelo.predict(futuro)

st.subheader('Previsao')
st.write(previsao[['ds','yhat','yhat_lower','yhat_upper']].tail(n_dias))
