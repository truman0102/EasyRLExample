from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def halfheatmap(data: pd.DataFrame, method: str = 'spearman'):
    corr = data.corr(method=method)
    mask = np.zeros_like(corr, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, annot=True, cmap="YlGnBu", mask=mask)
    plt.show()
