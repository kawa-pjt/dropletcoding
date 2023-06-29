# dBB type detection with downsized DNA barcode libraries, to estimate the required sequencing depth (relevant to Figure S7). 


```python
import numpy as np
import pandas as pd
import sklearn.covariance
import scipy.stats
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
import statistics
from scipy.stats import gmean
from scipy.spatial import distance
from scipy import sparse
```


```python
class IBBSeqLabeler:
    def __init__(self, counts, min_sig_count, alpha=1e-3):  
        self.mapped= counts.iloc[:-1]
        psuedo_count = 1
        self.min_log_std = 1.
        self.log_mapped = np.log(self.mapped + psuedo_count)
        self.alpha = alpha
        self.min_sig_count = min_sig_count

    def _fit(self, log_mapped):
        cov = sklearn.covariance.EllipticEnvelope(
                contamination=0.1,
                random_state=0)
        cov.fit(log_mapped.values[:, None])
        return cov

    def run(self):
        try:
            self._cov = self._fit(self.log_mapped)
            self.bg_loc = np.ravel(self._cov.location_)
            self.bg_prec = np.ravel(self._cov.precision_)
        except ValueError as e:
            self.bg_loc = self.log_mapped.mean()
            self.bg_prec = 1./ min(self.log_mapped.std(ddof=1), self.min_log_std)
        self.bg_z_values = (self.log_mapped - self.bg_loc) * self.bg_prec
        self.bg_probs = np.where(self.mapped >= self.min_sig_count, 1 - scipy.stats.norm.cdf(self.bg_z_values), 1)
        self.signals = self.bg_probs <= self.alpha

    def get_signals(self):
        return self.signals

    def get_bg_probs(self):
        return self.bg_probs
```


```python
class IBBSeqTableLabeler:
    def __init__(self, table, min_sig_count, min_total_count):
        
        if min_total_count:
            table = table.loc[:, table[:-1].sum(axis=0) >= min_total_count]
        self.table = table  
        self.labelers = [IBBSeqLabeler(self.table[cell], min_sig_count=min_sig_count) for cell in self.table.columns]
   
    def run(self):

        for labeler in tqdm.tqdm(self.labelers):
            labeler.run()
        self.signals = np.asarray([l.get_signals() for l in self.labelers], dtype=int).T  
        self.bg_probs = np.asarray([l.get_bg_probs() for l in self.labelers]).T  

    def save(self, out_prefix):
        output = out_prefix + '.signals.txt.gz'
        tab = pd.DataFrame(self.signals, index=self.table[:-1].index, columns=self.table.columns)
        tab.to_csv(output, sep='\t', compression='gzip')

        pos_sel = self.signals.sum(axis=0) > 0
        pos_columns = self.table.columns.values[pos_sel]
        pos_signals = self.signals[:, pos_sel]
        output = out_prefix + '.pos_signals.txt.gz'
        tab = pd.DataFrame(pos_signals, index=self.table[:-1].index, columns=pos_columns)
        tab.to_csv(output, sep='\t', compression='gzip')

        output = out_prefix + '.bg_probs.txt.gz'
        tab = pd.DataFrame(self.bg_probs, index=self.table[:-1].index, columns=self.table[:-1].columns)
        tab.to_csv(output, float_format='%.6g', sep='\t', compression='gzip')

```


```python
Dir = 'PATH_TO_FOLDER'    
```

# Reading 4 downsized samples + the original sample


```python
Data = ['Rep2_dBB_0p2M', 'Rep2_dBB_1M', 'Rep2_dBB_5M', 'Rep2_dBB_10M', 'Rep2_dBB']
```


```python
# Adjusted by library sizes
Min_total_count = [1 ,2 ,10 ,20, 25]
```


```python
# Adjusted by library sizes
Min_sig_count =  [1 ,2 ,10 ,20, 25]
```


```python
alpha = "1e-3" 
```


```python
for i in range(len(Data)):
    dense_umis = Dir + Data[i] + "/out/dense_umis.tsv"
    tab = pd.read_csv(dense_umis, sep='\t', index_col=0)
    print(tab.shape)
```

    (59, 7370)
    (59, 7370)
    (59, 7367)
    (59, 7364)
    (59, 7358)


# Positive signal detection


```python
for i in range(len(Data)):
    dense_umis = Dir + Data[i] + "/out/dense_umis.tsv"
    tab = pd.read_csv(dense_umis, sep='\t', index_col=0)
    out_prefix = Dir + Data[i] + '/' + Data[i]
    labeler = IBBSeqTableLabeler(tab, min_total_count=Min_total_count[i], min_sig_count=Min_sig_count[i])
    labeler.run()
    labeler.save(out_prefix)
    test = pd.read_csv(out_prefix + '.pos_signals.txt.gz', sep='\t', index_col=0) 
    print(Data[i] + ": The number of data points with at least one dBB type, " , str(len(test.T)))
```

    100%|█████████████████████████████████████| 6592/6592 [00:02<00:00, 2562.28it/s]


    Rep2_dBB_0p2M: The number of data points with at least one dBB type,  6324


    100%|█████████████████████████████████████| 7256/7256 [00:02<00:00, 2488.47it/s]


    Rep2_dBB_1M: The number of data points with at least one dBB type,  4592


    100%|█████████████████████████████████████| 7286/7286 [00:06<00:00, 1080.88it/s]


    Rep2_dBB_5M: The number of data points with at least one dBB type,  3416


    100%|██████████████████████████████████████| 7284/7284 [00:08<00:00, 907.81it/s]


    Rep2_dBB_10M: The number of data points with at least one dBB type,  3300


    100%|██████████████████████████████████████| 7304/7304 [00:08<00:00, 869.74it/s]


    Rep2_dBB: The number of data points with at least one dBB type,  3452



```python
for i in range(len(Data)):
    dense_umis = Dir + Data[i] + "/out/dense_umis.tsv"
    tab = pd.read_csv(dense_umis, sep='\t', index_col=0)
    out_prefix = Dir + Data[i] + '/' + Data[i]
    
    test = pd.read_csv(out_prefix + '.pos_signals.txt.gz', sep='\t', index_col=0)
    print(Data[i] + ': Mean_num_indexes_per_cell, ' + str(np.mean(test.apply(lambda x: x[0:58].sum()))))
```

    Rep2_dBB_0p2M: Mean_num_indexes_per_cell, 2.012333965844402
    Rep2_dBB_1M: Mean_num_indexes_per_cell, 1.8656358885017421
    Rep2_dBB_5M: Mean_num_indexes_per_cell, 2.144906323185012
    Rep2_dBB_10M: Mean_num_indexes_per_cell, 2.1224242424242425
    Rep2_dBB: Mean_num_indexes_per_cell, 2.1891657010428736


# Generating combinatorial codes


```python
num_beads = 3 # minumum positive iBB types per droplet
```


```python
for i in range(len(Data)):
    
    cond = '3-9_'
    dense_umis = Dir + Data[i] + "/out/dense_umis.tsv"
    tab = pd.read_csv(dense_umis, sep='\t', index_col=0)
    out_prefix = Dir + Data[i] + '/' + Data[i]
    
    test = pd.read_csv(out_prefix + '.pos_signals.txt.gz', sep='\t', index_col=0)
    test = test.loc[:, test.apply(lambda x: x[0:58].sum()) > num_beads-1]
    test = test.loc[:, test.apply(lambda x: x[0:58].sum()) < 10] 
    pos_counts = test.apply(np.sum, axis=0)
    unique_ibb_nunits = pos_counts.value_counts().sort_index()
    
    temp = pd.DataFrame(test.T)
    temp['idx'] = 0

    print(Data[i] + ": The number of data points after filtering, " + str(len(temp)))
    
    for k in range(0, len(temp)):
        temp.iloc[k,58] = ''.join(str(s) for s in temp.iloc[k,0:58].tolist())
    temp=temp.sort_values(by=['idx'])
    temp.iloc[:,58].to_csv(out_prefix  + '_' + cond + 'pattern.txt', sep='\t', header=True)
```

    Rep2_dBB_0p2M: The number of data points after filtering, 1719
    Rep2_dBB_1M: The number of data points after filtering, 542
    Rep2_dBB_5M: The number of data points after filtering, 888
    Rep2_dBB_10M: The number of data points after filtering, 891
    Rep2_dBB: The number of data points after filtering, 987


# Matching


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

    idx = pd.read_csv(input_prefix + '_' + cond + 'pattern.txt', sep='\t')
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
    stat.to_csv(output_prefix + cond + '_dist' + str(dist-1) + '.txt', sep='\t', header=True)
```


```python
def summary(data, cond, dist):
    
    colnames = ['cond','dist','num_units','max_num_cells_per_unit','min_num_cells_per_unit','error_rate']
    stats = pd.DataFrame(columns = colnames)  
    
    data = data
    input_prefix = Dir + Data[i] + '/' + Data[i]
    
    stat = pd.read_csv(input_prefix + cond + '_dist' + str(dist-1) + '.txt', sep='\t')
    
    identity = []
    
    for l in range(len(stat.cell_types)):
            if ((stat.cell_types.iloc[l] == 0.0)|(stat.cell_types.iloc[l] == 1.0)):
                identity.append(1)
            else:
                identity.append(0)
    
    stats = pd.concat([stats, pd.DataFrame([[cond, \
                                            dist-1, \
                                            len(stat.cell_types), \
                                            max(stat.num_cells.tolist()), \
                                            min(stat.num_cells.tolist()), \
                                            1-statistics.mean(identity)]],\
                                           columns = colnames)], axis=0)
    
    stats.to_csv(input_prefix + '_stats.txt', sep='\t', header=True)
    return stats
```


```python
adata_proc = ad.read_h5ad('PATH_TO_h5ad') 
```


```python
cond = '3-9_'
dist = 1
```


```python
for i in range(len(Data)):
    input_prefix = Dir + Data[i] + '/' + Data[i]
    output_prefix = Dir + Data[i] + '/' + Data[i]
    print(Data[i])
    matching(adata=adata_proc, cond=cond, dist=dist)
```

    Rep2_dBB_0p2M
    Num_dropletIDs, 1689
    Num_cells_with_dropletIDs:1230
    Rep2_dBB_1M
    Num_dropletIDs, 510
    Num_cells_with_dropletIDs:345
    Rep2_dBB_5M
    Num_dropletIDs, 814
    Num_cells_with_dropletIDs:551
    Rep2_dBB_10M
    Num_dropletIDs, 800
    Num_cells_with_dropletIDs:563
    Rep2_dBB
    Num_dropletIDs, 891
    Num_cells_with_dropletIDs:612



```python
Colnames = ['cond','dist','num_units','max_num_cells/unit','min_num_cells/unit','error_rate', 'data']
stats_summary = pd.DataFrame(columns = Colnames)

for i in range(len(Data)):
    
    data = Data[i]
    summary(data=data, cond=cond, dist=dist)
    stats = pd.read_csv(Dir + Data[i] + '/' + Data[i] + '_stats.txt', sep='\t',index_col=0)
    stats['data'] = Data[i]
    stats.columns = Colnames
    stats_summary = pd.concat([stats_summary, stats], axis=0)
```


```python
stats_summary
```




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
      <th>data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3-9_</td>
      <td>0</td>
      <td>16</td>
      <td>2</td>
      <td>2</td>
      <td>0.500000</td>
      <td>Rep2_dBB_0p2M</td>
    </tr>
    <tr>
      <th>0</th>
      <td>3-9_</td>
      <td>0</td>
      <td>16</td>
      <td>3</td>
      <td>2</td>
      <td>0.062500</td>
      <td>Rep2_dBB_1M</td>
    </tr>
    <tr>
      <th>0</th>
      <td>3-9_</td>
      <td>0</td>
      <td>36</td>
      <td>3</td>
      <td>2</td>
      <td>0.111111</td>
      <td>Rep2_dBB_5M</td>
    </tr>
    <tr>
      <th>0</th>
      <td>3-9_</td>
      <td>0</td>
      <td>39</td>
      <td>3</td>
      <td>2</td>
      <td>0.102564</td>
      <td>Rep2_dBB_10M</td>
    </tr>
    <tr>
      <th>0</th>
      <td>3-9_</td>
      <td>0</td>
      <td>38</td>
      <td>3</td>
      <td>2</td>
      <td>0.078947</td>
      <td>Rep2_dBB</td>
    </tr>
  </tbody>
</table>
</div>




```python
stats_summary.to_csv(Dir + 'Suppl_Rep2_3-9_dist0_error_rates.txt', sep='\t', header=True)
```


```python

```
