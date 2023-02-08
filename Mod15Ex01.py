import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tkinter import *
from mttkinter import *

st.set_page_config(page_title='Projeto Final - Python - EBAC', page_icon='https://www.svgrepo.com/show/376344/python.svg', layout="wide", initial_sidebar_state="auto", menu_items=None)
with st.sidebar:
  
        st.title("Projeto Final - Curso Python - EBAC")
        st.subheader('Aluno: Matheus Feldberg')
        st.markdown('[LinkedIn](https://www.linkedin.com/in/matheus-feldberg-521a93259)')
        st.header('''
                Sumário:
                1. Compreensão do problema
                2. Exploração dos Dados
                3. Transformação e Limpeza dos dados
                4. Visualização de dados
                5. Conclusão
                ''')
        
        st.markdown('Acesse os dados completos no [Kaggle](https://www.kaggle.com/code/matheusfeldberg/ebac-projeto-final-python-01-09-23)')

st.header('1. Compreensão do problema:')
st.markdown('O Ojetivo do projeto é descobrir quais fatores estão relacionados com a inadinplência e a adimplência dos clientes de uma determinada empresa de cartão de crédito.')

st.header('2. Exploração dos Dados:')


st.markdown('Foi utilizada a base de dados de crédito presentes neste neste [link](https://raw.githubusercontent.com/andre-marcos-perez/ebac-course-utils/develop/dataset/credito.csv).')
st.markdown('Os dados estão no formato CSV e contém informações sobre clientes de uma instituição financeira. Em especial, estamos interessados em explicar a segunda coluna, chamada de default, que indica se um cliente é adimplente(default = 0), ou inadimplente (default = 1), ou seja, queremos entender o porque de um cliente deixar de honrar com suas dívidas baseado no comportamento  de outros atributos, como salário, escolaridade e movimentação financeira. Uma descrição completa dos atributos está abaixo:')

st.markdown(''' 
            
| Coluna  | Descrição |
| ------- | --------- |
| id      | Número da conta |
| default | Indica se o cliente é adimplente (0) ou inadimplente (1) |
| idade   | --- |
| sexo    | --- |
| depedentes | --- |
| escolaridade | --- |
| estado_civil | --- |
| salario_anual | Faixa do salario mensal multiplicado por 12 |
| tipo_cartao | Categoria do cartao: blue, silver, gold e platinium |
| meses_de_relacionamento | Quantidade de meses desde a abertura da conta |
| qtd_produtos | Quantidade de produtos contratados |
| iteracoes_12m | Quantidade de iteracoes com o cliente no último ano |
| meses_inatico_12m | Quantidade de meses que o cliente ficou inativo no último ano |
| limite_credito | Valor do limite do cartão de crédito |
| valor_transacoes_12m | Soma total do valor das transações no cartão de crédito no último ano |
| qtd_transacoes_12m | Quantidade total de transações no cartão de crédito no último ano |
                
 ''')   

st.markdown('Para iniciar, vamos fazer a leitura dos dados num dataframe pandas, já definindo como identificar os dados faltantes (na):')

df = pd.read_csv('https://raw.githubusercontent.com/andre-marcos-perez/ebac-course-utils/develop/dataset/credito.csv', na_values='na')

st.dataframe(df)

st.header('2.1. Estrutura:')

st.markdown('Para começar é interessante ver o quanto a base de dados está balanceda, ou seja, qual é a porporção de dados entre clientes adimplentes e inadimplentes:')

st.markdown('Usaremos o seguinte código:')
            
code = '''qtd_total, _ = df.shape
qtd_adimplentes, _ = df[df['default'] == 0].shape
qtd_inadimplentes, _ = df[df['default'] == 1].shape'''

st.code(code, language='python')


qtd_total, _ = df.shape

qtd_adimplentes, _ = df[df['default'] == 0].shape

qtd_inadimplentes, _ = df[df['default'] == 1].shape


st.write(f"A proporcão clientes adimplentes é de {round(100 * qtd_adimplentes / qtd_total, 2)}%.")
st.write(f"A proporcão clientes inadimplentes é de {round(100 * qtd_inadimplentes / qtd_total, 2)}.%")

st.subheader(':blue[Conclusão:]')
st.markdown('Com base nos valores de proporção apresentados acima, pode-se concluir que a base de dados é composta por mais dados referentes a clientes adimplentes do que inadimplentes.')

st.header('2.2. Schema:')

df.dtypes

st.markdown('Importante pontuar que os atributos :red[_limite_credito_ e _valor_transacoes_12m_] estão definidas com type objetc, no entanto não faz sentido, uma vez que são números decimais, ou seja, na etapa de limpeza de dados é preciso converter esses atributos para float.')

st.markdown('Aplicando uma visualização rápidas aos atributos categóricos:')

st.write(df.select_dtypes('object').describe().transpose())

st.markdown('Aqui nota-se que para os atributos :green[_escolaridade_, _estado_civil_ e _salario_anual_] a quantidade é inferior a 10127, ou seja, há dados faltantes para esses atributos, que deverão ser tratados. Outro aspecto interessante que corrobora para análise anterior é que os atributos :red[_limite_credito_ e _valor_transacoes_12m_] apresentaram quantidade de valores únicos muito perto da quantidade total de elementos, ou seja, a análise para atributos categóricos não é adequada para esses atributos pois eles estão no grupo incorreto.')

st.markdown('Aplicando uma visualização rápidas aos atributos numéricos:')

st.write(df.drop('id', axis=1).select_dtypes('number').describe().transpose())

st.markdown('A visualização acima nos mostra que não há dados faltantes para nenhum atributo numérico e que o atributo :blue[_qtd_transacoes_12m_] possui um desvio elevado, ou seja, os dados estão bem dispersos.')

st.header('2.3. Dados faltantes:')

st.markdown('Como já foi definido de que forma os dados faltantes estão especificados (na), o próximo passo é identificar em quais colunas esses dados estão, apesar de já termos uma desconfiança baseada nas análises acima: :green[_escolaridade_, _estado_civil_ e _salario_anual_].')

st.write(df.isna().any())

st.markdown('- A função abaixo levanta algumas estatísticas sobre as colunas dos dados faltantes:')

code = '''def stats_dados_faltantes(df: pd.DataFrame) -> None:

  stats_dados_faltantes = []
  for col in df.columns:
    if df[col].isna().any():
      qtd, _ = df[df[col].isna()].shape
      total, _ = df.shape
      dict_dados_faltantes = {col: {'quantidade': qtd, "porcentagem": round(100 * qtd/total, 2)}}
      stats_dados_faltantes.append(dict_dados_faltantes)

  for stat in stats_dados_faltantes:
    print(stat)'''

st.code(code, language='python')

st.markdown('O objetivo aqui é verificar se os dados faltantes alteram a proporção da quantidade de dados so três atributos: :green[_escolaridade_, _estado_civil_ e _salario_anual_].')

def stats_dados_faltantes(df: pd.DataFrame) -> None:

  stats_dados_faltantes = []
  for col in df.columns:
    if df[col].isna().any():
      qtd, _ = df[df[col].isna()].shape
      total, _ = df.shape
      dict_dados_faltantes = {col: {'quantidade': qtd, "porcentagem": round(100 * qtd/total, 2)}}
      stats_dados_faltantes.append(dict_dados_faltantes)

  for stat in stats_dados_faltantes:
    st.write(stat)
     
stats_dados_faltantes(df=df)
stats_dados_faltantes(df=df[df['default'] == 0])
stats_dados_faltantes(df=df[df['default'] == 1])


st.markdown('Pode-se notar que a proporção de dados faltantes praticamente se mantém para os três atributos nos três data frames analisados (df base completa original, df de adimplentes e df de inadimplentes). Ou seja, podemos eliminar as linhas com dados faltantes sem que coloquemos um viés na análise dos dados.')

st.header('3. Transformação e Limpeza dos dados:')

st.subheader('3.1. Correção de Schema:')

st.markdown('Na etapa de exploração, notamos que as colunas **limite_credito** e **valor_transacoes_12m** estavam sendo interpretadas como colunas categóricas (`dtype = object`). Dessa forma, nesse ponto vamos utilizar uuma função `lambda` para fazer essa correção.')

code = '''
df['valor_transacoes_12m'] = df['valor_transacoes_12m'].apply(lambda valor: float(valor.replace(".", "").replace(",", ".")))
df['limite_credito'] = df['limite_credito'].apply(lambda valor: float(valor.replace(".", "").replace(",", ".")))'''

st.code(code, language='python')

st.markdown('Checando o schema para verificar que a função `lamba` funcionou corretamente:')

df.dtypes

st.markdown('Vamos dar um olhada nos atributos numéricos para termos uma ideia de como os atributos **limite_credito** e **valor_transacoes_12m** se comportam.')

st.write(df.drop('id', axis=1).select_dtypes('number').describe().transpose())

st.markdown('Podemos observar que os atributos **limite_credito** e **valor_transacoes_12m** possuem um valor de desvio padrão elevado, ou seja, há uma dispersão grande dos valores, o que nos leva pensar que são atributos bem personalizados por cliente.')

st.subheader('3.2. Remoção de dados faltantes:')

df.dropna(inplace=True)

st.markdown('Analisando novamente a estrutura de dados após remover os itens faltantes:')

qtd_total_novo, _ = df.shape
qtd_adimplentes_novo, _ = df[df['default'] == 0].shape
qtd_inadimplentes_novo, _ = df[df['default'] == 1].shape

st.write(f"A proporcão adimplentes ativos é de {round(100 * qtd_adimplentes / qtd_total, 2)}%")
st.write(f"A nova proporcão de clientes adimplentes é de {round(100 * qtd_adimplentes_novo / qtd_total_novo, 2)}%")
st.write("")
st.write(f"A proporcão clientes inadimplentes é de {round(100 * qtd_inadimplentes / qtd_total, 2)}%")
st.write(f"A nova proporcão de clientes inadimplentes é de {round(100 * qtd_inadimplentes_novo / qtd_total_novo, 2)}%")

st.markdown('''Como era de se esperar, as proporcões de clientes adimplentes e iandimplentes após a remoção dos dados faltantes 
            ficou praticamente a mesma. Dessa forma as etapas de exploração e limpeza dos dados foi conluída com sucesso,
            podendo passar assim, para a etapa de visualização dos dados.''')

st.header('4. Visualização de dados:')

st.markdown('''Nesse ponto vamos criar diversas visualizações comparativas para podemos gerar insights que ajudem a responder ao 
            nosso objetivo: quais atributos estão relacionados com a adimplencia e quais estão relacionados com a inadimplencia 
            dos clientes. Começamos inportando pacotes de visualização e separando os clientes adimplentes e inadimplentes.''')


code='''import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

df_adimplente = df[df['default'] == 0]
df_inadimplente = df[df['default'] == 1]'''

st.code(code, language='python')

st.subheader('4.1. Visualizações categóricas:')

st.markdown(' - Analisando o atributo Escolaridade:')

st.set_option('deprecation.showPyplotGlobalUse', False)
            
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_style("whitegrid")


df_adimplente = df[df['default'] == 0]
df_inadimplente = df[df['default'] == 1]

coluna = 'escolaridade'
titulos = ['Escolaridade dos Clientes', 'Escolaridade dos Clientes Adimplentes', 'Escolaridade dos Clientes Inadimplentes']

eixo = 0
max_y = 0
max = df.select_dtypes('object').describe()[coluna]['freq'] * 1.1

figura, eixos = plt.subplots(1,3, figsize=(20, 5), sharex=True)

for dataframe in [df, df_adimplente, df_inadimplente]:

    df_to_plot = dataframe[coluna].value_counts().to_frame()
    df_to_plot.rename(columns={coluna: 'frequencia_absoluta'}, inplace=True)
    df_to_plot[coluna] = df_to_plot.index
    df_to_plot.sort_values(by=[coluna], inplace=True)
    df_to_plot.sort_values(by=[coluna])

    f = sns.barplot(x=df_to_plot[coluna], y=df_to_plot['frequencia_absoluta'], ax=eixos[eixo])
    f.set(title=titulos[eixo], xlabel=coluna.capitalize(), ylabel='Frequência Absoluta')
    f.set_xticklabels(labels=f.get_xticklabels(), rotation=90)

    _, max_y_f = f.get_ylim()
    max_y = max_y_f if max_y_f > max_y else max_y
    f.set(ylim=(0, max_y))

    eixo += 1

st.pyplot(figura)

st.markdown('''Podemos notar que o atributo escolaridade não é relevante para o objetivo do projeto, pois a distribuição 
            se mantém semelhante nos três gráficos.''')

st.markdown('- Analisando o atributo tipo_cartao:')

coluna = 'tipo_cartao'
titulos = ['Tipo Cartão dos Clientes', 'Tipo Cartão dos Clientes Adimplentes', 'Tipo Cartão dos Clientes Inadimplentes']

eixo = 0
max_y = 0
max = df.select_dtypes('object').describe()[coluna]['freq'] * 1.1

figura, eixos = plt.subplots(1,3, figsize=(20, 5), sharex=True)

for dataframe in [df, df_adimplente, df_inadimplente]:

  df_to_plot = dataframe[coluna].value_counts().to_frame()
  df_to_plot.rename(columns={coluna: 'frequencia_absoluta'}, inplace=True)
  df_to_plot[coluna] = df_to_plot.index
  df_to_plot.sort_values(by=[coluna], inplace=True)
  df_to_plot.sort_values(by=[coluna])

  f = sns.barplot(x=df_to_plot[coluna], y=df_to_plot['frequencia_absoluta'], ax=eixos[eixo])
  f.set(title=titulos[eixo], xlabel=coluna.capitalize(), ylabel='Frequência Absoluta')
  f.set_xticklabels(labels=f.get_xticklabels(), rotation=90)

  _, max_y_f = f.get_ylim()
  max_y = max_y_f if max_y_f > max_y else max_y
  f.set(ylim=(0, max_y))

  eixo += 1

st.pyplot(figura)

st.markdown('''Para esse atributo podemos verificar algo interessante, apesar de poucos clientes possuírem o cartão gold, 
            todos são adimplentes. Não há clientes com cartão platinum na base de dados. ''')

st.subheader('4.2. Visualizações numéricas:')

st.markdown('- Analisando o atributo Quantidade de Transações nos Últimos 12 Meses:')

coluna = 'qtd_transacoes_12m'
titulos = ['Qtd. de Transações no Último Ano', 'Qtd. de Transações no Último Ano de Adimplentes', 'Qtd. de Transações no Último Ano de Inadimplentes']

eixo = 0
max_y = 0
figura, eixos = plt.subplots(1,3, figsize=(20, 5), sharex=True)

for dataframe in [df, df_adimplente, df_inadimplente]:

  f = sns.histplot(x=coluna, data=dataframe, stat='count', ax=eixos[eixo])
  f.set(title=titulos[eixo], xlabel=coluna.capitalize(), ylabel='Frequência Absoluta')

  _, max_y_f = f.get_ylim()
  max_y = max_y_f if max_y_f > max_y else max_y
  f.set(ylim=(0, max_y))

  eixo += 1

st.pyplot(figura)

st.markdown('''No gráfico que representa todos os clientes, podemos notar dois picos: um em torno de 40 transações e outro em torno de 80 transações.
            No entanto qdo avaliamos o gráfico de adimplentes, apesar de termos os dois piscos semelhantes, o pico que ocorre em torno de 40 transações 
            parece mais suave. E, ao avaliarmos o gráfico dos inadimplentes vemos um pico em torno de 40 transações, completanto o que falta no gráfico 
            dos clientes adimplentes. Dessa forma podemos concluir que o atributo **qtd_transacoes_12m** tem relação com o obejtivo do projeto.''')

st.markdown('- Verificando o atributo Valor das transações nos últimos 12 Meses:')

coluna = 'valor_transacoes_12m'
titulos = ['Valor das Transações no Último Ano', 'Valor das Transações no Último Ano de Adimplentes', 'Valor das Transações no Último Ano de Inadimplentes']

eixo = 0
max_y = 0
figura, eixos = plt.subplots(1,3, figsize=(20, 5), sharex=True)

for dataframe in [df, df_adimplente, df_inadimplente]:

  f = sns.histplot(x=coluna, data=dataframe, stat='count', ax=eixos[eixo])
  f.set(title=titulos[eixo], xlabel=coluna.capitalize(), ylabel='Frequência Absoluta')

  _, max_y_f = f.get_ylim()
  max_y = max_y_f if max_y_f > max_y else max_y
  f.set(ylim=(0, max_y))

  eixo += 1

st.pyplot(figura)

st.markdown('''Algo semelhante ocorre com o atributo **valor_transacoes_12m** como ocorreu com o atributo **qtd_transacoes_12m**: 
            Em torno de transações com Valor de 2.500 o que falta no gráfico dos clientes adimplentes é complementado pelo que está 
            no gráfico dos clientes inadimplentes. Algo semelhante ocorre com o atributo **valor_transacoes_12m** como ocorreu com 
            o atributo **qtd_transacoes_12m**: Em torno de transações com Valor de 2.500 o que falta no gráfico dos clientes adimplentes 
            é complementado pelo que está no gráfico dos clientes inadimplentes.''')

st.markdown(' - Valor de Transações nos Últimos 12 Meses x Quantidade de Transações nos Últimos 12 Meses:')

f = sns.relplot(x='valor_transacoes_12m', y='qtd_transacoes_12m', data=df, hue='default')
_ = f.set(
    title='Relação entre Valor e Quantidade de Transações no Último Ano', 
    xlabel='Valor das Transações no Último Ano', 
    ylabel='Quantidade das Transações no Último Ano'
  )


st.markdown('''Ao analisarmos o gráfico acima podemos concluir que há dois grupos distintos que contém adimplentes e inadimplentes e um grupo que só contém adimplentes.
Analisando os grupos que contém somente adimplentes podemos conluir que cliente com muitas transações (acima de 80) com valores altos (acima de 12 Mil) 
tendem a ser adimplentes.''')

st.markdown('''Analisando os grupos que possuem inadimplentes podemos chegar as seguintes combinações entre valor de transação e qte de transação:

1.   Qtde transações até 60 com valores póximos a 3000: tendência a inadimplência.
2.   Qtde transações de 40 até 90 com valores de 3000 até  10000: tendência a inadimplência.
3.   Qtde transações de 60 até 100 com valores de 2500 até  5000: tendência a adimplência.
4.   Qtde transações de 90 até 120 com valores de 7500 até  9000: tendência a adimplência.''')

st.markdown('- Analisando o atributo Idade')
 
coluna = 'idade'
titulos = ['Idade Todos Clientes', 'Idade Clientes Adimplentes', 'Idade Clientes Inadimplentes']

eixo = 0
max_y = 0
figura, eixos = plt.subplots(1,3, figsize=(20, 5), sharex=True)

for dataframe in [df, df_adimplente, df_inadimplente]:

  f = sns.histplot(x=coluna, data=dataframe, stat='count', ax=eixos[eixo])
  f.set(title=titulos[eixo], xlabel=coluna.capitalize(), ylabel='Frequência Absoluta')

  _, max_y_f = f.get_ylim()
  max_y = max_y_f if max_y_f > max_y else max_y
  f.set(ylim=(0, max_y))

  eixo += 1

st.pyplot(figura)


st.markdown('''Analisando o atributo da Idade podemos notar que clientes entre 40 e  50 anos possuem maior tendência a 
            serem inadimplentes.''')

coluna = 'iteracoes_12m'
titulos = ['Iterações 12m Todos Clientes', 'Iterações 12m Clientes Adimplentes', 'Iterações 12m Clientes Inadimplentes']

eixo = 0
max_y = 0
figura, eixos = plt.subplots(1,3, figsize=(20, 5), sharex=True)

for dataframe in [df, df_adimplente, df_inadimplente]:

  f = sns.histplot(x=coluna, data=dataframe, stat='count', ax=eixos[eixo])
  f.set(title=titulos[eixo], xlabel=coluna.capitalize(), ylabel='Frequência Absoluta')

  _, max_y_f = f.get_ylim()
  max_y = max_y_f if max_y_f > max_y else max_y
  f.set(ylim=(0, max_y))

  eixo += 1

st.pyplot(figura)

st.markdown('''Ao analisarmos os gráficos acima, podemos verificar que quando não há iteração o cliente tende a ser adinmplente e conforme a qtde 
de iterações vai aumentando os clientes tedem a se tornar inandimplentes. Vale ressaltar que quando há a mudança de 2 pra 3 iterações
é possível que seja um "ponto de virada", ou seja, seria interessante olhar esse cliente mais de perto pois é onde há maior tendência
que ele se torne inadimplente.''')

st.markdown('- Verificando se há relação entre os atirbutos **idade** e **iteracao_12m**:')
 
f = sns.relplot(x='idade', y='iteracoes_12m', data=df, hue='default')
_ = f.set(
    title='Relação entre Idade e Quantidade de Iterações no Último Ano', 
    xlabel='Idade', 
    ylabel='Quantidade de iterações no Último Ano'
  )

st.markdown('''Ao avaliarmos os adois atributos **idade** e **iteracoes_12m** juntos com a coluna default podemos verificar que independente da idade, 
conforme maior o número de iterações maior a qte de clientes inandimplentes. Verificamos tb que clientes entre 40 e 50 anos que possuem 
4 iterações possuem uma forte tendência a ser inadimplente.''')

st.header('5. Conclusão:')

st.markdown('''Pode-se concluir que os atributos categóricos não conseguem mapear o comportamento de adimplência e inadimplência dos clientes da base de dados analisada.
No entanto, quando avaliados os atributos numéricos pode-se destacar os seguintes pontos:

- Relação entre **qtd_transacoes_12m** e **valor_transacoes_12m**:

1.   Qtde transações até 60 com valores póximos a 3000: tendência a inadimplência.
2.   Qtde transações de 40 até 90 com valores de 3000 até  10000: tendência a inadimplência.
3.   Qtde transações de 60 até 100 com valores de 2500 até  5000: tendência a adimplência.
4.   Qtde transações de 90 até 120 com valores de 7500 até  9000: tendência a adimplência.
5.   Cliente com muitas transações (acima de 80) com valores altos (acima de 12 Mil) tendem a ser adimplentes.


- Atributo **iteracao_12m**: Vale atencão ao cliente no momento em que a quantidade de iteração muda de de 2 pra 3 pois há indicativo de ser um  "ponto de virada", ou seja, é um indicativo de que que ele possa se tornar inadimplente.


- Atributo **idade** e **iteracao_12m** juntos: Clientes entre 40 e 50 anos que possuem 4 iterações possuem uma forte tendência a ser inadimplente.''')
