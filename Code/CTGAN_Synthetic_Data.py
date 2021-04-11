import ctgan as ctgan
import pandas as pd
from ctgan import CTGANSynthesizer
data = pd.read_csv('cleaned_data.csv')  #tabularised dataset

discrete_columns = range(100, 120)  #20 bins

ctgan = CTGANSynthesizer(epochs=10)
ctgan.fit(df, discrete_columns)  

 #create synthetic data for 1000000 number of rows 
 #1000000 prodcuts
samples = ctgan.sample(1000000)