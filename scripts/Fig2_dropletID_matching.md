```python
import numpy as np
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from scipy.spatial import distance
from scipy import sparse
```


```python
adata_proc = ad.read_h5ad('PATH_TO_h5ad')
```


```python
input_prefix = '~/PATH/fk209_2022-05-24-10X_1e-3_th50_ee_'
```


```python
output_prefix = '~/PATH/fk209_stat_'
```

# Identifying cells with dBB type combinations


```python
def dist_12(tab, i1, i2):  # TODO also use color distance?
    hamdist = len(list(tab.loc[i1, 'idx'])) * distance.hamming(list(tab.loc[i1, 'idx']), list(tab.loc[i2, 'idx'])) # hamming dist
    return (hamdist < dist) 

def create_adj_matrix(tab):
    return np.asarray([[dist_12(tab, i1, i2) for i1 in tab.index] for i2 in tab.index])

def group_idx(tab):
    mat = create_adj_matrix(tab)
    n, components = sparse.csgraph.connected_components(mat, directed=True)
    return n, components
```


```python
def matching(adata, cond, dist):

    idx = pd.read_csv(input_prefix + cond + '_pattern.txt', sep='\t')
    idx.columns = ["sc_id","idx"]

    n, clusters = group_idx(idx)
    print("Num_dropletIDs, " + str(n))
    idx = idx.assign(cluster=clusters)
    idx = idx.sort_values('cluster')
    idx = idx.loc[:,["sc_id","cluster"]]
    
    labels = []

    for i in range(len(adata.obs.index)):
        if idx['sc_id'].isin([adata.obs.index[i]]).any():
            label = idx[idx['sc_id']==adata.obs.index[i]]['cluster']
            labels.append(label.values[0])
        else:
            labels.append('no_idx')

    print("Num_cells_with_dropletIDs:" + str(len(labels) - labels.count('no_idx')))
    
    adata.obs['idx'] = labels

    new = pd.concat([adata.obs['leiden'], adata.obs['K562_score'], adata.obs['THP1_score'], adata.obs['idx']], axis=1)
    new = new[new.duplicated(subset='idx', keep = False)]
    new = new[new.idx != 'no_idx']
    new = new.sort_values('idx')
    new.leiden = new.leiden.astype(int)

    grouped = new.groupby('idx')

    stat = pd.concat([grouped.size(), grouped.mean()], axis=1)
    stat.columns = (['num_cells', 'cell_types', 'K562_score ', 'THP1_score '])

    plt.hist(stat.cell_types)
    plt.title(cond + "_dist" + str(dist-1))
    plt.savefig(output_prefix + cond + '_dist' + str(dist-1) + '.png')
    plt.close()

    stat.to_csv(output_prefix + cond + '_dist' + str(dist-1) + '.txt', sep='\t', header=True)
```


```python
def summary(cond_list, dist_list):
    
    colnames = ['cond','dist','num_units','max_num_cells/unit','min_num_cells/unit','error_rate']
    stats_summary = pd.DataFrame(columns = colnames)

    for i in range(len(cond_list)):

        cond = cond_list[i]
        dist = dist_list[i]

        stat = pd.read_csv(output_prefix + cond + '_dist' + str(dist-1) + '.txt', sep='\t')
        
        identity = []

        for l in range(len(stat.cell_types)):
            if ((stat.cell_types.iloc[l] == 0.0)|(stat.cell_types.iloc[l] == 1.0)):
                identity.append(1)
            else:
                identity.append(0)

        stats_summary = pd.concat([stats_summary,pd.DataFrame([[cond, \
                                                                dist-1, \
                                                                len(stat.cell_types), \
                                                                max(stat.num_cells.tolist()), \
                                                                min(stat.num_cells.tolist()), \
                                                                1-statistics.mean(identity)]],\
                                                               columns = colnames)], axis=0)
        
    return stats_summary
```


```python
cond_list = ["2-9", "3-9", "4-9", '5-9']
dist_list =  [1,1,1,1]

for i in range(len(cond_list)):
    
    cond = cond_list[i]
    print(cond_list[i])
    
    dist = dist_list[i]

    matching(adata=adata_proc, cond=cond, dist=dist)

stats_summary = summary(cond_list, dist_list)
stats_summary
```

    2-9
    Num_dropletIDs, 1442
    Num_cells_with_dropletIDs:1155
    3-9
    Num_dropletIDs, 851
    Num_cells_with_dropletIDs:584
    4-9
    Num_dropletIDs, 439
    Num_cells_with_dropletIDs:289
    5-9
    Num_dropletIDs, 211
    Num_cells_with_dropletIDs:134





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cond</th>
      <th>dist</th>
      <th>num_units</th>
      <th>max_num_cells/unit</th>
      <th>min_num_cells/unit</th>
      <th>error_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2-9</td>
      <td>0</td>
      <td>144</td>
      <td>5</td>
      <td>2</td>
      <td>0.375</td>
    </tr>
    <tr>
      <th>0</th>
      <td>3-9</td>
      <td>0</td>
      <td>40</td>
      <td>3</td>
      <td>2</td>
      <td>0.125</td>
    </tr>
    <tr>
      <th>0</th>
      <td>4-9</td>
      <td>0</td>
      <td>15</td>
      <td>3</td>
      <td>2</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>5-9</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
stats_summary.to_csv(output_prefix + 'error_rates.txt', sep='\t', header=True)
```


```python

```
