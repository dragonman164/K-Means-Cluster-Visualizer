from pycaret.clustering import plot_model, setup, create_model
import pandas as pd
from pycaret.internal.tabular import pull

df = pd.read_csv('scrapped_Data.csv')

def nopreprocessing():
    pre = setup(data = df,silent = True,verbose=False)
    model = create_model('kmeans',verbose=False)
    results = pull()
    print(results['Silhouette'][0])
    plot_model(model, plot = 'elbow',save=True)

def zscorenormalize():
    pre = setup(data = df, silent = True, verbose = False,normalize=True,normalize_method='zscore')
    model = create_model('kmeans',verbose= False)
    results = pull()
    print(results['Silhouette'][0])
    plot_model(model, plot = 'elbow',save = True)

zscorenormalize()