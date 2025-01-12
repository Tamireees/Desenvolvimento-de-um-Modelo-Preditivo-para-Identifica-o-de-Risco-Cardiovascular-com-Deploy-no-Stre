from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder  # Importação correta do sklearn
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np


# Classes para pipeline

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
        
        
class MinMaxScaler(BaseEstimator, TransformerMixin):
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
        
        
class OneHotEncoder(BaseEstimator, TransformerMixin):
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
     

        
class OrdinalEncoder(BaseEstimator, TransformerMixin):
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