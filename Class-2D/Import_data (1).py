import pandas as pd

excel = '6npl_map_100x100_10mN_filtered.xlsx'


def DataToDF(excel):
    sheet_name = 'Test 1'
    df = pd.read_excel(excel, sheet_name=sheet_name)
    df.dropna(subset=['HARDNESS'], inplace=True)
    #remove inf
    df = df.iloc[2:]
    df = df.astype(float)
    return df

def DFtoMatrix(df):
    # Transform dataframe into a matrix X with dimensions: num of data x 2
    matrix = df[['HARDNESS', 'MODULUS']].values.reshape(-1, 2)
    return matrix
