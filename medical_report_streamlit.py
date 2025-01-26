import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from utils import DropFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from joblib import load
import xgboost as xgb
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline


# Configuração inicial do Streamlit
st.set_page_config(page_title="Análise de Doença Vascular", layout="wide")
dados_clean = pd.read_csv("https://raw.githubusercontent.com/Tamireees/Desenvolvimento-de-um-Modelo-Preditivo-para-Identifica-o-de-Risco-Cardiovascular-com-Deploy-no-Stre/refs/heads/main/dados_clean", sep=',')

# Definir as páginas
pages = {
    "Questões Análise": "main",
    "Apresentação": "page_2",
    "Etapas do Desenvolvimento": "page_3"}

# Barra de navegação
selected_page = st.sidebar.radio("Selecione a Página", list(pages.keys()))

if selected_page == "Questões Análise":
    
    # Conteúdo da terceira página
    st.title("Questões Análise")
    
    #carregando os dados
    dados_clean = pd.read_csv("https://raw.githubusercontent.com/Tamireees/Desenvolvimento-de-um-Modelo-Preditivo-para-Identifica-o-de-Risco-Cardiovascular-com-Deploy-no-Stre/refs/heads/main/dados_clean", sep=',')

    
    
    
    st.markdown('<style>div[role="listbox"] ul{background-color: #6e42ad}; </style>', unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; '> Formulário para Identificação de Risco Cardiovascular em pacientes:</h1>", unsafe_allow_html=True)
    st.warning('Preencha o formulário com todos os seus dados pessoais e clique no botão **ENVIAR** no final da página.')

    class DropFeatures(BaseEstimator, TransformerMixin):
        def __init__(self, feature_to_drop=['id']): 
            self.feature_to_drop = feature_to_drop

        def fit(self, df, y=None):
            return self

        def transform(self, df):
            missing_features = set(self.feature_to_drop) - set(df.columns)
            if missing_features:
                print(f"Uma ou mais features não estão no DataFrame: {', '.join(missing_features)}")
            return df.drop(columns=self.feature_to_drop, errors='ignore')


    class CustomMinMaxScaler(BaseEstimator, TransformerMixin):
        def __init__(self, min_max_scaler=['Idade', 'Genero', 'Altura', 'Peso', 'PressaoArterialSistolica',
                                           'PressaoArterialDiastolica']):
            self.min_max_scaler = min_max_scaler
            self.min_max_enc = MinMaxScaler()

        def fit(self, df, y=None):
            if set(self.min_max_scaler).issubset(df.columns):
                self.min_max_enc.fit(df[self.min_max_scaler])
            return self

        def transform(self, df):
            if set(self.min_max_scaler).issubset(df.columns):
                df_copy = df.copy()
                scaled_values = self.min_max_enc.transform(df_copy[self.min_max_scaler])
                df_copy[self.min_max_scaler] = scaled_values
                return df_copy
            else:
                print('Uma ou mais features não estão no DataFrame.')
                return df

    class CustomOneHotEncoder:
        def __init__(self, OneHotEncoding, handle_unknown='ignore'):
            self.OneHotEncoding = OneHotEncoding
            self.handle_unknown = handle_unknown
            self.ohe = OneHotEncoder(sparse=False, handle_unknown=self.handle_unknown)  # sparse=False para retornar uma matriz densa
    
        def fit(self, X, y=None):
            # Ajusta o OneHotEncoder nas colunas selecionadas
            self.ohe.fit(X[self.OneHotEncoding])
            return self
    
        def transform(self, X):
            # Transforma as colunas selecionadas
            X_copy = X.copy()
            
            # Realiza a transformação
            transformed = self.ohe.transform(X_copy[self.OneHotEncoding])
            
            # Converte para um DataFrame e adiciona ao DataFrame original
            transformed_df = pd.DataFrame(transformed, columns=self.ohe.get_feature_names_out(self.OneHotEncoding))
            
            # Remove as colunas originais e adiciona as transformadas
            X_copy = X_copy.drop(self.OneHotEncoding, axis=1)
            X_copy = pd.concat([X_copy, transformed_df], axis=1)
            
            return X_copy

    class CustomOrdinalEncoder:
        def __init__(self, ordinal_feature, handle_unknown='use_encoded_value', unknown_value=-1):
            self.ordinal_feature = ordinal_feature
            self.handle_unknown = handle_unknown
            self.unknown_value = unknown_value
            self.oenc = OrdinalEncoder(handle_unknown=self.handle_unknown, unknown_value=self.unknown_value)

        def fit(self, X, y=None):
            # Ajusta o codificador nas colunas selecionadas
            self.oenc.fit(X[self.ordinal_feature])
            return self

        def transform(self, X):
            # Transforma as colunas selecionadas
            X_copy = X.copy()

            # Verifica se a coluna a ser transformada existe em X
            if self.ordinal_feature in X_copy.columns:
                X_copy[self.ordinal_feature] = self.oenc.transform(X_copy[[self.ordinal_feature]])
            else:
                raise ValueError(f"A coluna '{self.ordinal_feature}' não existe no DataFrame fornecido.")

            return X_copy



    # Idade
    st.write('### Idade')
    input_idade = int(st.slider('Selecione a sua idade', 18, 100))

    # Gênero
    st.write('### Gênero do participante')
    input_genero = st.selectbox('Qual o seu gênero (0: Masculino / 1: Feminino?', dados_clean['Genero'].unique())

    # Altura
    st.write('### Altura')
    input_altura = int(st.slider('Selecione a sua altura em cm', 140, 240))

    # Peso
    st.write('### Peso')
    input_Peso = int(st.slider('Selecione o seu peso em Kg', 30, 240))

    # Pressão Arterial Sistólica
    st.write('### Pressão Arterial Sistólica')
    input_PressaoArterialSistolica = int(st.text_input('Digite a leitura da pressão arterial Sistólica feita pelo paciente e pressione ENTER para confirmar', 0))

    # Pressão Arterial Diastólica
    st.write('### Pressão Arterial Diastólica')
    input_PressaoArterialDiastolica = int(st.text_input('Digite a leitura da pressão arterial Diastólica feita no paciente e pressione ENTER para confirmar', 0))

    # Colesterol
    st.write('### Colesterol')
    input_Colesterol = int(st.slider('Digite o seu nível de colesterol total lido como mg/dll', 0, 10))

    # Glicose
    st.write('### Glicose')
    input_Glicose = int(st.slider('Digite o seu nível de glicose lido como mmol/l', 0, 20))

    # Fumante
    st.write('### Fumante')
    input_Fumante = st.radio('Você fuma?', ['Sim', 'Não'], index=0)
    input_Fumante_dict = {'Sim': 1, 'Não': 0}
    input_Fumante = input_Fumante_dict.get(input_Fumante)

    # Usa Álcool
    st.write('### Usa Álcool')
    input_UsaAlcool = st.radio('Você faz uso de bebida alcoólica?', ['Sim', 'Não'], index=0)
    input_UsaAlcool_dict = {'Sim': 1, 'Não': 0}
    input_UsaAlcool = input_UsaAlcool_dict.get(input_UsaAlcool, 0)  # Default para 0 caso o valor seja inválido

    # Ativo Fisicamente
    st.write('### Ativo Fisicamente')
    input_AtivoFisicamente = st.radio('Você faz atividade física?', ['Sim', 'Não'], index=0)
    input_AtivoFisicamente_dict = {'Sim': 1, 'Não': 0}
    input_AtivoFisicamente = input_AtivoFisicamente_dict.get(input_AtivoFisicamente)

    # Definindo o novo cliente
    novo_cliente = [0,  # ID
                    input_idade, 
                    input_genero, 
                    input_altura, 
                    input_Peso, 
                    input_PressaoArterialSistolica,  
                    input_PressaoArterialDiastolica,  
                    input_Colesterol, 	
                    input_Glicose, 
                    input_Fumante, 
                    input_UsaAlcool, 
                    input_AtivoFisicamente, 
                    0  # target 
    ]

    def data_split(df, test_size):
        SEED = 1561651
        df_treino, df_teste = train_test_split(df, test_size=test_size, random_state=SEED)
        return df_treino.reset_index(drop=True), df_teste.reset_index(drop=True)

    df_treino, df_teste = data_split(dados_clean, 0.2)

    # Novo cliente para previsão
    cliente_predict_df = pd.DataFrame([novo_cliente], columns=df_teste.columns)

    # Concatenando novo cliente ao dataframe de teste
    teste_novo_cliente = pd.concat([df_teste, cliente_predict_df], ignore_index=True)

    def criar_pipeline(df):
        pipeline = Pipeline([
    ('drop_features', DropFeatures(feature_to_drop=['id'])),
    ('minmax_scaler', CustomMinMaxScaler(min_max_scaler=['Idade', 'Altura', 'Peso'])),
    ('onehot_encoder', CustomOneHotEncoder(OneHotEncoding=['Fumante', 'UsaAlcool'], handle_unknown='ignore')),
    ('ordinal_encoder', CustomOrdinalEncoder(ordinal_feature=['Colesterol', 'Glicose'], handle_unknown='use_encoded_value', unknown_value=-1))
])
        return pipeline.fit(df)

    # Ajuste: fit no df_treino, transform no df_teste e novo cliente
    pipeline = criar_pipeline(df_treino)
    df_transformado_treino = pipeline.transform(df_treino)
    df_transformado_teste = pipeline.transform(df_teste)
    cliente_pred_transformado = pipeline.transform(cliente_predict_df)

    # Agora, para prever o novo cliente, basta passar os dados transformados
    cliente_pred = cliente_pred_transformado.drop(['DoencaVascular'], axis=1, errors='ignore')

    # Predições
    if st.button('Enviar'):
        model = joblib.load('modelo/xgb.vascular')
        final_pred = model.predict(cliente_pred)
        if final_pred[-1] == 0:
            st.success('### Parabéns! Você possui baixo risco cardiovascular.')
            st.balloons()
        else:
            st.error('### Atenção! Seu risco cardiovascular é alto. Consulte um médico.')










    # Exibir conteúdo de acordo com a página selecionada
if selected_page == "Apresentação":
    # Título
        st.markdown('# <div style="text-align: center;">**Desenvolvimento de um Modelo Preditivo para Identificação de Risco Cardiovascular com Deploy no Streamlit**', unsafe_allow_html=True)
    # Subtítulo
        st.markdown('## Introdução', unsafe_allow_html=True)
    # Texto
        st.markdown('<div style="text-align: left;">    O presente projeto visa a elaboração de um modelo preditivo para identificar o risco de desenvolvimento de doenças cardiovasculares em pacientes, utilizando a base de dados "DoençaVascular.xlsx". A partir desse modelo, pretende-se criar uma aplicação interativa com o Streamlit, com o objetivo de auxiliar médicos na compreensão de como determinados fatores demográficos, comportamentos de saúde e marcadores biológicos impactam o desenvolvimento dessas enfermidades. Abaixo está descrito o plano para análise e desenvolvimento do projeto.</div>', unsafe_allow_html=True)
        # Subtítulo
        st.markdown('## Dados do DataFrame', unsafe_allow_html=True)
    # Descrição das variáveis
        st.markdown('''
                    *O dataset contém as seguintes variáveis:*
    * **Idade:** Idade do(a) participante.
    * **Sexo:** Sexo do participante (masculino/feminino).
    * **Altura:** Altura medida em centímetros.
    * **Peso:** Peso medido em quilogramas.
    * **PressãoArterialSistólica:** Leitura da pressão arterial sistólica feita pelo paciente.
    * **PressãoArterialDiastólica:** Leitura da pressão arterial diastólica feita no paciente.
    * **Colesterol:** Nível de colesterol total lido como mg/dl em uma escala de 0 a 5+ unidades (inteiro). Cada unidade denota aumento/diminuição de 20 mg/dL, respectivamente.
    * **Glicose:** Nível de glicose lido como mmol/l em uma escala de 0 a 16+ unidades (número inteiro). Cada unidade denota aumento/diminuição em 1 mmol/L, respectivamente.
    * **Fumante:** Indica se a pessoa fuma ou não (binário: 0 = Não, 1 = Sim).
    * **UsaAlcool:** Indica se a pessoa bebe álcool ou não (binário: 0 = Não, 1 = Sim).
    * **AtivoFisicamente:** Indica se a pessoa é fisicamente ativa ou não (binário: 0 = Não, 1 = Sim).
    * **DoencaVascular:** Indica se a pessoa sofre de doenças cardiovasculares ou não (binário: 0 = Não, 1 = Sim).''', 
    unsafe_allow_html=True)
        # Subtítulo
        st.markdown('## Objetivo',  unsafe_allow_html=True)
        st.markdown('Criar uma ferramenta preditiva e interpretável que auxilie médicos na tomada de decisão, promovendo um maior entendimento dos fatores de risco associados às doenças cardiovasculares e, consequentemente, contribuindo para a prevenção e tratamento eficazes.')
        
        
        
        
        
        
        
if selected_page == "Etapas do Desenvolvimento":
    st.title("Etapas do Desenvolvimento")
    st.markdown("## 1. Exploração dos Dados:", unsafe_allow_html=True)
    st.markdown("* ### Carregar os Dados:")
    st.code('''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    dados = pd.read_csv('path_to_file.csv')
    dados_clean = dados''')
    caminho_arquivo = "https://raw.githubusercontent.com/Tamireees/modelo_preditivo/refs/heads/main/Doen%C3%A7aVascular.csv"
    dados_clean = pd.read_csv(caminho_arquivo, sep=';')
    st.write("### Visualização dos Dados:")
    st.dataframe(dados_clean.head())
    st.code("dados_clean.shape")
    dados_clean.shape
    st.divider()
    st.code("dados_clean.info()")
    dados_clean.info()
    # Limpeza dos Dados
    st.markdown("* ### Limpeza e Transformação dos Dados")
    st.markdown("Nesta seção, removemos valores nulos e ajustamos os tipos das colunas para garantir consistência nos dados.""", unsafe_allow_html=True)
    st.code("dados_clean = dados_clean.drop('index', axis=1)")
    st.code("dados_clean.isnull().sum()")
    missing_values = dados_clean.isnull().sum()
    st.write(missing_values)
    # Remover valores nulos
    dados_clean = dados_clean.dropna()
    st.write("**Após a limpeza (remoção de valores nulos):**")
    st.code('''dados_clean = dados_clean.dropna()
    # neste caso a idade é uma descrição importante para fazer a predição, portanto, será necessário deletar esses dados nulos.''')
    st.dataframe(dados_clean.head())
    # Ajustar Tipos
    st.write("**Dados após ajuste de tipos:**")
    st.code('''
    # Ajustar tipos
    dados_clean['Idade'] = dados_clean['Idade'].round().astype(int)  # Arredonda Idade para inteiro
    dados_clean['Genero'] = dados['Genero'].replace({1: 0, 2: 1})  # Genero: 0 (Masculino), 1 (Feminino
    dados_clean['id'] = dados_clean['id'].astype('object')
    dados_clean['Genero'] = dados_clean['Genero'].astype('int')''')
    dados_clean["Idade"] = dados_clean["Idade"].round().astype(int)
    dados_clean["Genero"] = dados_clean["Genero"].replace({1: 0, 2: 1}).astype("object")
    dados_clean["id"] = dados_clean["id"].astype("object")
    dados_clean['Genero'] = dados_clean['Genero'].astype('int')
    st.dataframe(dados_clean.head())
    # Separador visual
    st.divider()
    # Análise de Correlação
    st.markdown("* ### Análise de Correlação e Visualizações")
    st.markdown("""Aqui exploramos a correlação entre variáveis numéricas e visualizamos os dados por meio de gráficos.""", unsafe_allow_html=True)
    st.code('''corr = dados_clean.select_dtypes(include=[np.number]).corr()
                 plt.figure(figsize = (20,10))
                 sns.heatmap(corr, cmap="Blues", annot=True)
    # Maior correlação são: Altura x Peso // colesterol x glicose''')
    corr = dados_clean.select_dtypes(include=[np.number]).corr()
    st.markdown("<h3 style='color:white;'>Matriz de Correlação:</h3>", unsafe_allow_html=True)
    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")
    sns.heatmap(corr, cmap="magma", annot=True, cbar=True, annot_kws={"size": 12, "color": "white"})
    plt.xticks(color="white")
    plt.yticks(color="white")
    st.pyplot(plt, bbox_inches='tight', transparent=True)
    plt.clf()
    # Gráfico de Dispersão
    st.subheader("Gráfico de Dispersão: Altura x Peso:")
    if "Altura" in dados_clean.columns and "Peso" in dados_clean.columns:
        fig, ax = plt.subplots(figsize=(10, 5)) 
        sns.set_style("darkgrid")
        sns.scatterplot(data=dados_clean, x="Altura", y="Peso", hue="Genero", ax=ax)
        plt.xlabel("Altura", color="white")
        plt.ylabel("Peso", color="white")
        plt.title("Altura vs Peso por Gênero", color="white")
        plt.xticks(color="white")
        plt.yticks(color="white")
        plt.xticks(rotation=45)
        st.pyplot(fig, bbox_inches='tight', transparent=True)
    else:
        st.warning("As colunas 'Altura' ou 'Peso' não foram encontradas.")
    st.markdown('''## 2. Preparação dos Dados:''', unsafe_allow_html=True)
    st.markdown("* ### *Explorando as variáveis quantitativas: Glicose, Colesterol, PressaoArterialDiastolica, PressaoArterialSistolica, Altura, Idade.*")
    st.code("colunas_quantitativas = ['PressaoArterialDiastolica', 'PressaoArterialSistolica', 'Altura', 'Idade']")
    colunas_quantitativas = ['PressaoArterialDiastolica', 'PressaoArterialSistolica', 'Altura', 'Idade']
    st.code("dados_clean[colunas_quantitativas].describe().round(2).T")
    dados_clean[colunas_quantitativas].describe().round(2).T
    st.divider()
    st.markdown('''
    **PressaoArterialDiastolica**
    * Média: 96.63
    * Desvio padrão: 188.48
    * Valores atípicos: -70 a 11.000
    * Os valores negativos e extremamente altos são errôneos. A pressão arterial diastólica deveria estar em uma faixa normal entre 40 e 120 mmHg. Esses valores precisam ser corrigidos ou removidos
    **PressaoArterialSistolica**
    * Média: 128.82
    * Desvio padrão: 154.02
    * Valores atípicos: -150 a 16.020
    * Assim como na pressão diastólica, os valores negativos e acima de 160 são anormais e precisam ser corrigidos
    **Altura**
    * Média: 164.36 cm
    * Desvio padrão: 8.21 cm
    * Intervalo: 55 a 250 cm
    * Embora a altura média e o desvio padrão estejam razoáveis, um valor mínimo de 55 cm e máximo de 250 cm podem ser plausíveis para o contexto de um estudo de saúde, mas valores muito baixos ou altos podem indicar erros
    **Idade**
    * Média: 53.34 anos
    * Desvio padrão: 6.77 anos
    * Intervalo: 30 a 65 anos
    * A idade parece estar bem distribuída.''', unsafe_allow_html=True)
    st.code('''
        def plotar_boxplot_geral(dataset, y):
            ax = sns.boxplot(data=dataset, y=y)
            ax.figure.set_size_inches(4,4)
            plt.xticks(rotation=45)
            ''')
    def plotar_boxplot_geral(dataset, y):
        sns.set_style("darkgrid")
        plt.figure(figsize=(5, 3))
        ax = sns.boxplot(data=dataset, y=y, palette="Set2")
        plt.title("Boxplot Geral", color="white")  # Adicionar título com fonte branca
        plt.xticks(color="white", rotation=45)  # Rotação e cor branca
        plt.yticks(color="white")  # Cor dos ticks do eixo Y
        st.pyplot(plt, bbox_inches='tight', transparent=True)
        plt.clf()
    st.code("# Deletando dados que contem 'outlier', como não há nenhuma informação para considerar a altura dos participantes, iremos deixas apenas participantes acima de 140cm e menores que 240cm de altura.")
    st.code("dados_clean = dados_clean[(dados_clean['Altura'] >= 140) & (dados_clean['Altura'] <= 240)]")
    dados_clean = dados_clean[(dados_clean['Altura'] >= 140) & (dados_clean['Altura'] <= 240)]
    plotar_boxplot_geral(dados_clean, 'Altura')
    sns.set_style("darkgrid")
    plt.figure(figsize=(6, 4))
    sns.histplot(data=dados_clean, x='Altura', bins=20)
    plt.xlabel("Altura", color="white")
    plt.ylabel("Frequência", color="white")
    plt.title("Histograma de Altura", color="white")
    plt.xticks(color="white")
    plt.yticks(color="white")
    plt.xticks(rotation=45)
    st.pyplot(plt, bbox_inches='tight', transparent=True)
    plt.clf()
    baixo_40 = dados_clean[dados_clean['PressaoArterialDiastolica'] < 40].shape[0]
    maior_120 = dados_clean[dados_clean['PressaoArterialDiastolica'] > 120].shape[0]
    st.write(f'Valores de Pressão Arterial Diastólica abaixo de 40: {baixo_40}')
    st.write(f'Valores de Pressão Arterial Diastólica acima de 120: {maior_120}')
    st.code("# A pressão arterial diastólica deveria estar em uma faixa normal entre 40 e 120 mmHg. Esses valores serão removidos.")
    st.code("dados_clean = dados_clean[(dados_clean['PressaoArterialDiastolica'] >= 40) & (dados_clean['PressaoArterialDiastolica'] <= 120)])")
    dados_clean = dados_clean[(dados_clean['PressaoArterialDiastolica'] >= 40) & (dados_clean['PressaoArterialDiastolica'] <= 120)]
    plotar_boxplot_geral(dados_clean, 'PressaoArterialDiastolica')
    sns.set_style("darkgrid")
    plt.figure(figsize=(6, 4))
    sns.histplot(data=dados_clean, x='PressaoArterialDiastolica', bins=20)
    plt.xlabel("PressaoArterialDiastolica", color="white")
    plt.ylabel("Frequência", color="white")
    plt.title("Histograma de PressaoArterialDiastolica", color="white")
    plt.xticks(color="white")
    plt.yticks(color="white")
    plt.xticks(rotation=45)
    st.pyplot(plt, bbox_inches='tight', transparent=True)
    plt.clf()
    st.code("# Os valores negativos e acima de 160 são anormais e precisam ser corrigidos")
    st.code("dados_clean = dados_clean[(dados_clean['PressaoArterialSistolica'] >= 90) & (dados_clean['PressaoArterialSistolica'] <= 160)]")
    dados_clean = dados_clean[(dados_clean['PressaoArterialSistolica'] >= 90) & (dados_clean['PressaoArterialSistolica'] <= 160)]
    plotar_boxplot_geral(dados_clean, 'PressaoArterialSistolica')
    sns.set_style("darkgrid")
    plt.figure(figsize=(6, 4))
    sns.histplot(data=dados_clean, x='PressaoArterialSistolica', bins=20)
    plt.xlabel("PressaoArterialSistolica", color="white")
    plt.ylabel("Frequência", color="white")
    plt.title("Histograma de PressaoArterialSistolica", color="white")
    plt.xticks(color="white")
    plt.yticks(color="white")
    plt.xticks(rotation=45)
    st.pyplot(plt, bbox_inches='tight', transparent=True)
    plt.clf()
    st.markdown("* ### *Explorando as variáveis qualitativas: DoencaVascular, AtivoFisicamente, UsaAlcool, Fumante, Genero.*")
    st.code('''dados_clean['DoencaVascular'].value_counts(normalize=True)*100
    # Cerca de 50% das pessoas não têm doença vascular, enquanto 49% é uma porcentagem significativa entre os que têm a condição.''')
    dados_clean['DoencaVascular'].value_counts(normalize=True)*100
    st.divider()
    st.code('''dados_clean['AtivoFisicamente'].value_counts(normalize=True)*100
    # A maior parte da amostra, 80,37%, pratica atividade física, enquanto 19,63% não adotam um estilo de vida ativo.''')
    dados_clean['AtivoFisicamente'].value_counts(normalize=True)*100
    st.divider()
    st.code('''dados_clean['UsaAlcool'].value_counts(normalize=True)*100
    # A grande maioria das pessoas não consome álcool, representando 94,62% da amostra, enquanto apenas 5,38% indicam que fazem uso de álcool.''')
    dados_clean['UsaAlcool'].value_counts(normalize=True)*100
    st.divider()
    st.code('''dados_clean['Fumante'].value_counts(normalize=True)*100
    # A grande maioria da amostra, 91,19%, não é fumante, enquanto 8,81% são fumantes.''')
    dados_clean['Fumante'].value_counts(normalize=True)*100
    st.divider()
    st.code('''dados_clean['Genero'].value_counts(normalize=True)*100
    # Na amostra, 65,05% são do gênero masculino e 34,95% são do gênero feminino.''')
    dados_clean['Genero'].value_counts(normalize=True)*100
    st.divider()
    st.markdown("* ### *Explorando as variáveis qualitativas ordinais: Glicose e Colesterol.*")
    st.code("# Substituir os valores para 0, 1, 2 (mantendo a ordem)")
    dados_clean['Glicose'] = dados_clean['Glicose'].replace({1: 0, 2: 1, 3: 2})
    dados_clean['Colesterol'] = dados_clean['Colesterol'].replace({1: 0, 2: 1, 3: 2})
    st.code("dados_clean['Glicose'].value_counts(normalize=True)*100")
    st.write("Distribuição de Glicose:")
    st.write(dados_clean['Glicose'].value_counts(normalize=True)*100)
    st.code("dados_clean['Colesterol'].value_counts(normalize=True)*100")
    st.write("Distribuição de Colesterol:")
    st.write(dados_clean['Colesterol'].value_counts(normalize=True)*100)
    st.divider()
    st.markdown("* ### *Análise das variáveis em relação ao Target*")
    descricao = dados_clean.groupby('DoencaVascular').mean()
    st.write("Média das variáveis por Doença Vascular:")
    st.write(descricao)
    st.divider()
    st.markdown("* ### *Dividindo os dados em conjunto de treino e teste.*")
    st.code('''SEED = 1561651
    df_treino, df_teste = train_test_split(dados_clean, test_size=0.2, random_state=SEED)''')
    SEED = 1561651
    df_treino, df_teste = train_test_split(dados_clean, test_size=0.2, random_state=SEED) 
    st.code("df_treino.shape")
    df_treino.shape
    st.divider()
    st.code("df_teste.shape")   
    df_teste.shape
    st.divider()
    st.markdown('''
                ### 3. Pipeline de Dados:
    Para garantir um fluxo organizado e eficiente dos dados para o modelo preditivo, será criada uma pipeline com as seguintes etapas:
    * **Leitura e cade dados: Normalizar variáveis contínuas (altura, peso, colesterol, glicose) e codificar variáveis categóricas (sexo).**
    * **Divisão do dataset: Separar os dados em conjuntos de treino (80%) e teste (20%).**
    * **Balanceamento de classes: Aplicar técnicas como oversampling ou undersampling, se necessário, para lidar com possíveis desbalanceamentos nos dados.**
    * **Feature engineering: Criar novas variáveis relevantes, se aplicável, para melhorar o desempenho do modelo.**
    * **Pré-processamento final: Garantir que todos os dados estejam padronizados antes de serem enviados para o modelo.rregamento dos dados: Importar os dados do arquivo "DoençaVascular.xlsx".**
    * **Tratamento de valores ausentes: Identificar e corrigir dados ausentes ou inconsistentes.**
    * **Transformação.**''', 
    unsafe_allow_html=True)
    st.code('''   
    class DropFeatures(BaseEstimator, TransformerMixin):
        def __init__(self, feature_to_drop=['id']): 
            self.feature_to_drop = feature_to_drop
        def fit(self,df):
            return self
        def transform(self,df):
            if (set(self.feature_to_drop).issubset(df.columns)):
                df.drop(self.feature_to_drop,axis=1,inplace=True)
                return df
            else:
            print('Uma ou mais features não estão no DataFrame')
            return df
    class MinMAx(BaseEstimator, TransformerMixin):
        def __init__(self, min_max_scaler=['Idade', 'Genero', 'Altura', 'Peso', 'PressaoArterialSistolica',
            'PressaoArterialDiastolica']):
            self.min_max_scaler = min_max_scaler
            self.min_max_enc = MinMaxScaler()
        def fit(self, df):
            if set(self.min_max_scaler).issubset(df.columns):
                self.min_max_enc.fit(df[self.min_max_scaler])
            return self
        def transform(self, df):
            if set(self.min_max_scaler).issubset(df.columns):
                scaled_values = self.min_max_enc.transform(df[self.min_max_scaler])
                df[self.min_max_scaler] = scaled_values
                return df
            else:
                print('Uma ou mais features não estão no DataFrame.')
                return df
    class OneHotEncodingNames(BaseEstimator, TransformerMixin):
        def __init__(self, OneHotEncoding=['Fumante', 'UsaAlcool', 'AtivoFisicamente']):
            self.OneHotEncoding = OneHotEncoding
            self.one_hot_enc = OneHotEncoder(sparse_output=False)  # Retorna um array denso para facilitar
        def fit(self, df):
            if set(self.OneHotEncoding).issubset(df.columns):
                self.one_hot_enc.fit(df[self.OneHotEncoding])
            return self
        def transform(self, df):
            if set(self.OneHotEncoding).issubset(df.columns):
                # Obter as colunas codificadas
                encoded_array = self.one_hot_enc.transform(df[self.OneHotEncoding])
                encoded_df = pd.DataFrame(encoded_array, 
                                          columns=self.one_hot_enc.get_feature_names_out(self.OneHotEncoding), 
                                          index=df.index)
            # Concatenar as colunas codificadas com o restante do DataFrame
                outras_features = df.drop(columns=self.OneHotEncoding)
                df_full = pd.concat([outras_features, encoded_df], axis=1)
                return df_full
            else:
                print('Uma ou mais features não estão no DataFrame.')
                return df
    class OrdinalFeature(BaseEstimator, TransformerMixin):
        def __init__(self, ordinal_feature=['Colesterol', 'Glicose']):
                self.ordinal_feature = ordinal_feature
        def fit(self, df):
            return self
        def transform(self, df):
            missing_columns = [col for col in self.ordinal_feature if col not in df.columns]
            if missing_columns:
                print(f"As colunas seguintes não estão no DataFrame: {', '.join(missing_columns)}")
            else:
                ordinal_encoder = OrdinalEncoder()
                df[self.ordinal_feature] = ordinal_encoder.fit_transform(df[self.ordinal_feature])
            return df
    class Oversample(BaseEstimator, TransformerMixin):
        def __init__(self, target_column='DoencaVascular'):
            self.target_column = target_column
            self.oversample = SMOTE(sampling_strategy='minority')
        def fit(self, df):
        # Não é necessário treinar ou ajustar nada, apenas retorna self
            return self
        def transform(self, df):
            if self.target_column in df.columns:
                X = df.drop(columns=[self.target_column])
                y = df[self.target_column]
                X_bal, y_bal = self.oversample.fit_resample(X, y)
                return pd.concat([pd.DataFrame(X_bal, columns=X.columns),
                                  pd.DataFrame(y_bal, columns=[self.target_column])], axis=1)
            else:
                print(f"A coluna target '{self.target_column}' não está no DataFrame.")
                return df
    def pipeline(df):
        pipeline = Pipeline([
            ('feature_dropper', DropFeatures()),
            ('min_max_scaler', MinMAx()),
            ('OneHotEncoding', OneHotEncodingNames()),
            ('ordinal_feature', OrdinalFeature()),
            ('oversample', Oversample())
    ])
        return pipeline.fit_transform(df)
                ''') 
    class DropFeatures(BaseEstimator, TransformerMixin):
        def __init__(self, feature_to_drop=['id']): 
            self.feature_to_drop = feature_to_drop

        def fit(self, df, y=None):
            return self

        def transform(self, df):
            missing_features = set(self.feature_to_drop) - set(df.columns)
            if missing_features:
                print(f"Uma ou mais features não estão no DataFrame: {', '.join(missing_features)}")
            return df.drop(columns=self.feature_to_drop, errors='ignore')


    class CustomMinMaxScaler(BaseEstimator, TransformerMixin):
        def __init__(self, min_max_scaler=['Idade', 'Genero', 'Altura', 'Peso', 'PressaoArterialSistolica',
                                           'PressaoArterialDiastolica']):
            self.min_max_scaler = min_max_scaler
            self.min_max_enc = MinMaxScaler()

        def fit(self, df, y=None):
            if set(self.min_max_scaler).issubset(df.columns):
                self.min_max_enc.fit(df[self.min_max_scaler])
            return self

        def transform(self, df):
            if set(self.min_max_scaler).issubset(df.columns):
                df_copy = df.copy()
                scaled_values = self.min_max_enc.transform(df_copy[self.min_max_scaler])
                df_copy[self.min_max_scaler] = scaled_values
                return df_copy
            else:
                print('Uma ou mais features não estão no DataFrame.')
                return df


    class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
        def __init__(self, OneHotEncoding=['Fumante', 'UsaAlcool', 'AtivoFisicamente']):
            self.OneHotEncoding = OneHotEncoding
            self.one_hot_enc = OneHotEncoder(sparse_output=False)  # Retorna um array denso

        def fit(self, df, y=None):
            if set(self.OneHotEncoding).issubset(df.columns):
                self.one_hot_enc.fit(df[self.OneHotEncoding])
            return self

        def transform(self, df):
            if set(self.OneHotEncoding).issubset(df.columns):
                encoded_array = self.one_hot_enc.transform(df[self.OneHotEncoding])
                encoded_df = pd.DataFrame(encoded_array, 
                                          columns=self.one_hot_enc.get_feature_names_out(self.OneHotEncoding), 
                                          index=df.index)
                df_copy = df.drop(columns=self.OneHotEncoding)
                return pd.concat([df_copy, encoded_df], axis=1)
            else:
                print('Uma ou mais features não estão no DataFrame.')
                return df


    class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
        def __init__(self, ordinal_feature=['Colesterol', 'Glicose']):
            self.ordinal_feature = ordinal_feature
            self.ordinal_enc = OrdinalEncoder()

        def fit(self, df, y=None):
            if set(self.ordinal_feature).issubset(df.columns):
                self.ordinal_enc.fit(df[self.ordinal_feature])
            return self

        def transform(self, df):
            if set(self.ordinal_feature).issubset(df.columns):
                df_copy = df.copy()
                df_copy[self.ordinal_feature] = self.ordinal_enc.transform(df_copy[self.ordinal_feature])
                return df_copy
            else:
                print(f"Uma ou mais features não estão no DataFrame.")
                return df
    def criar_pipeline(df):
        pipeline = Pipeline([
            ('drop_features', DropFeatures(feature_to_drop=['id'])),
            ('minmax_scaler', CustomMinMaxScaler(min_max_scaler=['Idade', 'Altura', 'Peso'])),
            ('onehot_encoder', CustomOneHotEncoder(OneHotEncoding=['Fumante', 'UsaAlcool'])),
            ('ordinal_encoder', CustomOrdinalEncoder(ordinal_feature=['Colesterol', 'Glicose']))
    ])

        return pipeline.fit_transform(df)


    st.markdown('''
                ### 4. Seleção e Treinamento do Modelo:
    * **Testar diferentes algoritmos de machine learning, como Regressão Logística, Random Forest, e XGBoost.**
    * **Avaliar o desempenho dos modelos com métricas como acurácia, precisão, recall e F1-score.**''', 
    unsafe_allow_html=True)
    st.code('''
    X_teste = df_teste.drop('DoencaVascular', axis=1)
    y_teste = df_teste['DoencaVascular']
    X_treino = df_treino.drop('DoencaVascular', axis=1)
    y_treino = df_treino['DoencaVascular']
    SEED = 1561651
    def roda_modelo(modelo):
        modelo.fit(X_treino, y_treino)
        prob_predict = modelo.predict_proba(X_teste)
        print(f'\n----------------Resultados {modelo} ------------------\n')
        data_sem_chances = np.sort(modelo.predict_proba(X_teste)[:, 0])
        data_com_chances = np.sort(modelo.predict_proba(X_teste)[:, 1])
        kstest = stats.ks_2samp(data_sem_chances, data_com_chances)
        print(f'Métrica KS: {kstest}')
        print('\n Confusion Matrix \n')
        predicao = modelo.predict(X_teste)
        cm = confusion_matrix(y_teste, predicao, normalize='true')
        fig, ax = plt.subplots(figsize=(7,7))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                    xticklabels=['sem_chances', 'com_chances'],
                    yticklabels=['sem_chances', 'com_chances'])
        ax.set_title('Matriz de Confusão Normalizada', fontsize=16, fontweight='bold')
        ax.set_xlabel('Label predita', fontsize=18)
        ax.set_ylabel('Label verdadeira', fontsize=18)
        plt.show()
        print('\nClassification Report\n')
        print(classification_report(y_teste, predicao, zero_division=0))
        print('\nROC curve\n')
        y_proba = prob_predict[:, 1]  # Probabilidades da classe positiva
        fpr, tpr, _ = roc_curve(y_teste, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # Linha de referência
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.show()
    from sklearn.tree import DecisionTreeClassifier
    modelo_tree = DecisionTreeClassifier()
    roda_modelo(modelo_tree)
    from sklearn.ensemble import GradientBoostingClassifier
    modelo_xgb = GradientBoostingClassifier()
    roda_modelo(modelo_xgb)
    # Salvando modelo com melhor performace:
    joblib.dump(modelo_xgb, 'xgb.vascular')
    ''')
    
    def roda_modelo(modelo, X_treino, y_treino, X_teste, y_teste):
        # Treinamento do modelo
        modelo.fit(X_treino, y_treino)
        prob_predict = modelo.predict_proba(X_teste)
        # Resultados do modelo
        st.write(f'\n----------------Resultados {modelo} ------------------\n')
        # Separar as probabilidades para a classe positiva e negativa
        data_sem_chances = np.sort(modelo.predict_proba(X_teste)[:, 0])
        data_com_chances = np.sort(modelo.predict_proba(X_teste)[:, 1])
        # Teste KS
        kstest = stats.ks_2samp(data_sem_chances, data_com_chances)
        st.write(f'Métrica KS: {kstest}')
        # Matriz de Confusão
        st.write('\nConfusion Matrix\n')
        predicao = modelo.predict(X_teste)
        cm = confusion_matrix(y_teste, predicao, normalize='true')
        # Exibir matriz de confusão
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                    xticklabels=['sem_chances', 'com_chances'],
                    yticklabels=['sem_chances', 'com_chances'])
        ax.set_title('Matriz de Confusão Normalizada', fontsize=16, fontweight='bold', color='white')
        ax.set_xlabel('Label predita', fontsize=18, color='white')
        ax.set_ylabel('Label verdadeira', fontsize=18, color='white')
        plt.xticks(color='white')
        plt.yticks(color='white')
        st.pyplot(fig, bbox_inches='tight', transparent=True)
        # Relatório de classificação
        st.write('\nClassification Report\n')
        st.text(classification_report(y_teste, predicao, zero_division=0))
        # Curva ROC
        st.write('\nROC curve\n')
        y_proba = prob_predict[:, 1]  # Probabilidades da classe positiva
        fpr, tpr, _ = roc_curve(y_teste, y_proba)
        roc_auc = auc(fpr, tpr)
        # Exibir curva ROC
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='yellow', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color='white', lw=2, linestyle='--')  # Linha de referência
        plt.xlabel("False Positive Rate", color='white')
        plt.ylabel("True Positive Rate", color='white')
        plt.title("Receiver Operating Characteristic (ROC) Curve", color='white')
        plt.legend(loc="lower right", fontsize=12, labelcolor='white')
        plt.xticks(color='white')
        plt.yticks(color='white')
        st.pyplot(plt, bbox_inches='tight', transparent=True)
        # Limpar a figura para evitar sobreposição em gráficos futuros
        plt.clf()
    # Preparação dos dados
    df_transformado = criar_pipeline(df_treino)
    X = df_transformado.drop(columns=['DoencaVascular'])  # Variáveis independentes
    y = df_transformado['DoencaVascular']  # Variável dependente

    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    # Treinando e avaliando os modelos
    modelo_logistico = LogisticRegression(random_state=SEED)
    roda_modelo(modelo_logistico, X_treino, y_treino, X_teste, y_teste)
    modelo_tree = DecisionTreeClassifier(random_state=SEED)
    roda_modelo(modelo_tree, X_treino, y_treino, X_teste, y_teste)
    modelo_xgb = GradientBoostingClassifier(random_state=SEED)
    roda_modelo(modelo_xgb, X_treino, y_treino, X_teste, y_teste)
    # Salvando o modelo com melhor desempenho
    joblib.dump(modelo_xgb, 'xgb.vascular')
    st.markdown('''
        ### 5. Implementação da Aplicação:
        * **Desenvolver uma interface intuitiva no Streamlit para entrada de dados e exibição dos resultados.**
        * **Exibir explicações interpretáveis sobre como cada fator contribui para o risco predito.**''', unsafe_allow_html=True)
    st.markdown('''
        ### 6. Implantação do Modelo:
        * **Realizar o deploy da aplicação no Streamlit, garantindo que esteja acessível para médicos e profissionais de saúde.**''', unsafe_allow_html=True)
# streamlit run medical_report_streamlit.py