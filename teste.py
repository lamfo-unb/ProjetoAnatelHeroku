from typing import TextIO
from gensim.corpora.dictionary import Dictionary
import numpy as np
from gensim import models
import random
import nltk
import spacy
import re
import unidecode
import streamlit as st
import pandas as pd
import plotly.express as px
import time


st.title('Classificador de tópicos de reclamação LDA')


stopwords = nltk.corpus.stopwords.words('portuguese')
stopwords.extend(
["pois", "pra", "ate", "porem", "ha", "desde", "porque", "tambem", "entao", "dia", "apos","nao",
    "consumidor", "reclama", "",
    'eramos', 'estao', 'estavamos', 'estiveramos', 'estivessemos', 'foramos', 'fossemos', 'hao', 'houveramos',
    'houverao',
    'houveriamos', 'houvessemos', 'ja', 'sao', 'sera', 'serao', 'seriamos', 'so', 'tera', 'terao', 'teriamos',
    'tinhamos',
    'tiveramos', 'tivessemos', 'voce', 'voces', "após", "quero", "havia", "sky", "janeiro",
    "fevereiro", "marco", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro",
    "dezembro",'claro' ,'tim' ,'vivo' ,'nextel', 'net' ,'oi' ,'sky' ,'outros' ,'sercomtel',
    'algar' , 'embratel', 'anatel', 'cabo',"mes","dias","ms","vezes","hoje", "vcs","jamais","deu","to","ira","eletronica","netcombo","faz",
    "sendo","assim","vo","vez", "ter","falou", "vou", "sendo", "conforme", "fiz", "fazer", "ainda liguei","nada"," "])



get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF), range(n)))
mycolors = get_colors(40)

# import streamlit as st

texto = st.text_area("Insira texto para classificação abaixo:",height = 200,max_chars = 1000)

if not st.button('Processar'):
    st.stop()

if len(texto) < 100:   
    st.error('Reclamação fornecida é muito curta, por favor digite reclamação com 100 caracteres ou mais.')
    st.stop()

texto_bk = texto 

def classificalda(texto,texto_bk):
    #texto = "gostaria de reclamar sobre sinal de telefonia móvel"
    
    def correctforsite(coluna,x):
        coluna = [i for i in coluna.split() if i not in stopwords]
        x = x.lower().split()
        y = [i[:-1] for i in x]
        x.extend(y)
        result = all(elem in x for elem in coluna)
        return  result


    

    texto = texto.lower()
    texto = re.sub('(\W|\d)', ' ', texto)
    texto = unidecode.unidecode(texto)
    texto =  re.sub(' [A-z] '," ", texto)
    texto =  re.sub('rede ',"rede_", texto)
    texto =  re.sub('nao ',"nao_", texto)
    texto =  re.sub('pre ',"pre_", texto)
    texto= re.sub('pos ',"pos_", texto)
    st.progress(20)

    

    nlp = spacy.load("pt_core_news_sm",disable=['parser', 'ner'])
    st.progress(30)
    texto = nlp(texto)
    st.progress(40)
    texto = [word.lemma_ for word in texto ]
    st.progress(50)

    texto = [word for word in texto if word not in stopwords]
    

    common_dictionary = Dictionary.load("LDA1/model_40_reclamacao.id2word")#
    common_corpus = np.array([common_dictionary.doc2bow(texto)])
    st.progress(60)

    lda_model = models.LdaModel.load("LDA1/model_40_reclamacao")

    corp_cur = common_corpus[0]

    st.progress(65)

    print(len(corp_cur))
    if len(corp_cur) < 5:
        st.warning('Essa reclamação parece ter muitas palavras fora do vocabulário, classificação pode ser imprecisa')
        time.sleep(5)

    topic_percs, wordid_topics, wordid_phivalues = lda_model[corp_cur]
    word_dominanttopic = []
    for wd, topic in wordid_topics:
        try:
            word_dominanttopic.append((lda_model.id2word[wd], topic[0]))
        except:
            #palavra está no dic, porém não tem tópico (filtrada no modelo)
            continue 

    topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True) 

    st.progress(70)

    TOPS1 = str(int(topic_percs_sorted[0][1] * 100))
    TOPS1_N = str(int(topic_percs_sorted[0][0]) + 1)
    TOPS2 = str(int(topic_percs_sorted[1][1] * 100))
    TOPS2_N = str(int(topic_percs_sorted[1][0]) + 1)
    TOPS3 = str(int(topic_percs_sorted[2][1] * 100))
    TOPS3_N = str(int(topic_percs_sorted[2][0]) + 1)
    COLOR1 = mycolors[int(TOPS1_N)-1]
    COLOR2 = mycolors[int(TOPS2_N)-1]
    COLOR3 = mycolors[int(TOPS3_N)-1]

    if int(TOPS1) < 0.1:
        st.error('Resultado com baixa acurácia, por favor reescreva a reclamação.')
        st.stop()

    topicos = [i[0] for i in topic_percs_sorted]
    topicos_percentual = [i[1] for i in topic_percs_sorted]
    df = pd.DataFrame(list(zip(topicos, topicos_percentual)), 
                columns =['Tópicos', 'Percentual'])

    def converteldas(x,rodada):
        x = x + 1
        if rodada == 1:
            if x in [13,14,17,32,34]:
                return "cobrança indevida"
            elif x == 1:
                return "contatos inoportunos"
            elif x in [8,19]:
                return "desacordo entre o contratado x entrega"
            elif x == 23:
                return "dificuldades com cancelamento"
            elif x in [11,33]:
                return "divergência crédito x serviço pré-pago"
            elif x == 9:
                return "insatisfação com o atendimento da operadora em geral"
            elif x == 12:
                return "problemas com ajustes em dados cadastrais"
            elif x == 7: 
                return "problemas com pacotes empresariais"
            elif x == 37:
                return "problemas com pagamentos"
            elif x in [20,38]:
                return "problemas com planos"
            elif x == 35:
                return "problemas com portabilidade"
            elif x in [2,40]:
                return "problemas com vendas"
            elif x in [4,10,18,30,31]:
                return "problemas com instalações e reparo"
            elif x == 27:
                return "problemas de acesso a serviços digitais via rede da prestadora"
            elif x == 5:
                return "reclamação recorrente"
            else:
                return "outros assuntos"


    st.progress(75)

    df.Tópicos = df.Tópicos.apply(lambda x: converteldas(x,1))


    common_dictionary = Dictionary.load("LDA2/model_40_reclamacao.id2word")#
    common_corpus = np.array([common_dictionary.doc2bow(texto)])
    lda_model = models.LdaModel.load("LDA2/model_40_reclamacao")

    corp_cur = common_corpus[0]

    topic_percs, wordid_topics, wordid_phivalues = lda_model[corp_cur]
    word_dominanttopic = []
    for wd, topic in wordid_topics:
        try:
            word_dominanttopic.append((lda_model.id2word[wd], topic[0]))
        except:
            #palavra está no dic, porém não tem tópico (filtrada no modelo)
            continue 

    topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True) 

    TOPS1 = str(int(topic_percs_sorted[0][1] * 100))
    TOPS1_N = str(int(topic_percs_sorted[0][0]) + 1)
    TOPS2 = str(int(topic_percs_sorted[1][1] * 100))
    TOPS2_N = str(int(topic_percs_sorted[1][0]) + 1)
    TOPS3 = str(int(topic_percs_sorted[2][1] * 100))
    TOPS3_N = str(int(topic_percs_sorted[2][0]) + 1)
    COLOR1 = mycolors[int(TOPS1_N)-1]
    COLOR2 = mycolors[int(TOPS2_N)-1]
    COLOR3 = mycolors[int(TOPS3_N)-1]

    if int(TOPS1) < 0.1:
        st.error('Resultado com baixa acurácia, por favor reescreva a reclamação.')
        st.stop()

    topicos = [i[0] for i in topic_percs_sorted]
    topicos_percentual = [i[1] for i in topic_percs_sorted]
    df2 = pd.DataFrame(list(zip(topicos, topicos_percentual)), 
                columns =['Tópicos', 'Percentual'])

    st.progress(80)
    def converteldas(x,rodada):
        x = x + 1
        if rodada == 2:
            if x == 40:
                return "bloqueio indevido"
            elif x in [11, 15, 20, 27, 31, 36, 39]:
                return "cobrança indevida"
            elif x in [6, 16, 19]:
                return "desacordo entre o contratado x entrega"
            elif x == 18:
                return "dificuldades com cancelamento"
            elif x in [4, 12, 21, 29]:
                return "falhas no funcionamento geral"
            elif x == 28: 
                return "problemas com pacotes empresariais"
            elif x == 7:
                return "problemas com pagamentos"
            elif x == 1:
                return "problemas com portabilidade"
            elif x in [10, 2, 32]:
                return "problemas com instalações e reparo"
            elif x in [35, 37]:
                return "problemas em geral com a internet"
            elif x == 34:
                return "troca ou aquisição de chip ou aparelho"
            else:
                return "outros assuntos"



    df2.Tópicos = df2.Tópicos.apply(lambda x: converteldas(x,2))

    df.sort_values(by=['Percentual'], inplace=True, ascending=False)
    df = df.drop_duplicates(subset=["Tópicos"], keep='first')

    df2.sort_values(by=['Percentual'], inplace=True, ascending=False)
    df2 = df2.drop_duplicates(subset=["Tópicos"], keep='first')


    df3 = pd.merge(df,df2,on = "Tópicos",how = "outer",validate="one_to_one")

    print(df3.Tópicos.apply(lambda x: correctforsite(x,texto_bk)))
    df3['Percentual'] = np.where((df3.Percentual_x>df3.Percentual_y),df3.Percentual_x,df3.Percentual_y)

    index_names = df3[ (df3['Tópicos'] == "outros assuntos")].index 
    df3.drop(index_names, inplace = True)
    df3['Percentual'].fillna(0, inplace=True)
    print(df3.Percentual)
    st.progress(90)
    df3['Percentual'] = np.where((df3.Tópicos.apply(lambda x: correctforsite(x,texto_bk))),df3.Percentual + 0.5,df3.Percentual)
    df3 = df3.sort_values(by='Percentual')
    principaltopico = str(df3.Tópicos[0])

    return df3, principaltopico

with st.progress(0):
     df3, principaltopico = classificalda(texto,texto_bk)
     st.progress(100)
     st.success("Pronto! Esta reclamação parece conter principalmente o tópico de AAA. Confira abaixo os demais tópicos.".replace("AAA",principaltopico))




# html = ""
# frase_col =[]
# for word in texto_bk.split():
#     match = False
#     for (palavra, topics) in word_dominanttopic:
#         if palavra[:-2] in unidecode.unidecode(word.lower()):
#             palavra = ("<mark style='color: COLOR; background-color: white'>" + word + "</mark>").replace(
#                                 "COLOR", mycolors[topics])
#             frase_col.append(palavra)
#             match = True
#     if match != True:
#         frase_col.append(word)
# frase_col = " ".join(frase_col)
# st.components.v1.html(frase_col)

fig = px.bar(df3, x='Tópicos', y='Percentual' , height=600)#x='topico', y='percentual'

st.plotly_chart(fig)

with st.beta_expander('Detalhes'):
    st.write('O modelo apresentado neste portal é baseado nos resultados da pesquisa desenvolvida na frente de reclamações. Contudo, os valores apresentados podem ser difentes dos reais utilizados na pesquisa para adequação no visualizador e processador em tempo real, e deve-se portanto utilizar o painel apenas para consulta exploratória informal do modelo.')