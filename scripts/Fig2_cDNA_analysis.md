```python
import numpy as np
import anndata as ad
import pandas as pd
from pandas import api
from matplotlib import interactive
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import rc_context
import scanpy as sc
sc.settings.verbosity = 3
```


```python
def load_gene_table(gene_table):

    if isinstance(gene_table, pd.DataFrame):
        return gene_table
    else:
        return pd.read_csv(gene_table, sep='\t').set_index('gene_id')
```


```python
Cellline_genes = pd.read_csv('~/PATH_TO/E-MTAB-2770-query-results.tpms.tsv', sep='\t',skiprows=4)
#https://www.ebi.ac.uk/gxa/experiments/E-MTAB-2770/
```


```python
Cellline_genes= Cellline_genes[['Gene ID','Gene Name','K-562, blast phase chronic myelogenous leukemia, BCR-ABL1 positive', 'THP-1, childhood acute monocytic leukemia']]
```


```python
K562_genes= Cellline_genes[Cellline_genes['K-562, blast phase chronic myelogenous leukemia, BCR-ABL1 positive'] > 100]
K562_genes= K562_genes[K562_genes['THP-1, childhood acute monocytic leukemia'] < 3]
K562_markerID = K562_genes['Gene Name'].tolist()

THP1_genes= Cellline_genes[Cellline_genes['THP-1, childhood acute monocytic leukemia'] > 200]
THP1_genes=  THP1_genes[THP1_genes['K-562, blast phase chronic myelogenous leukemia, BCR-ABL1 positive'] < 5]
THP1_markerID = THP1_genes['Gene Name'].tolist()

print(len(THP1_markerID), len(K562_markerID))
```


```python
ref_ID= K562_markerID + THP1_markerID
```


```python
adata = sc.read_umi_tools('~/PATH_TO/processed/cDNA/cDNA_fk209_2022-05-24-10X_re.star.gene_cell_counts.txt.gz')
```


```python
var_names = adata.var_names.intersection(adata.var_names)
adata = adata[:, var_names]
```


```python
adata
```


```python
gene_table = load_gene_table('~/PATH_TO/gencode.v38.annotation.txt')
for col in gene_table.columns:
   adata.var.loc[:, col] = gene_table[col]
```


```python
for col in gene_table.columns:
   adata.var.loc[:, col] = gene_table[col]
```


```python
def K562(adata):
    #adata = self.adata
    K562_markers = K562_genes['Gene Name'].tolist()
    K562_genes_mm_ens = adata.var_names[np.in1d(adata.var.gene_name, K562_markers)]
    sc.tl.score_genes(adata, K562_genes_mm_ens, score_name='K562_score')
```


```python
def THP1(adata):
    #adata = self.adata
    THP1_markers = THP1_genes['Gene Name'].tolist()
    THP1_genes_mm_ens = adata.var_names[np.in1d(adata.var.gene_name, THP1_markers)]
    sc.tl.score_genes(adata, THP1_genes_mm_ens, score_name='THP1_score')
```


```python
#gene全体を使うときのみ
sc.pp.filter_cells(adata, min_genes=250)
sc.pp.filter_genes(adata, min_cells=25)
```


```python
adata
```


```python
sc.pl.highest_expr_genes(adata, n_top=20)
```


```python
adata.var['mt'] = adata.var.seqname == 'chrM'  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
```


```python
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.1, multi_panel=True)
```


```python
adata = adata[adata.obs.pct_counts_mt < 8, :] 
adata = adata[adata.obs.pct_counts_mt > 2, :] 
adata = adata[adata.obs.total_counts < 8000, :]
adata = adata[adata.obs.total_counts > 1000, :]
adata = adata[adata.obs.n_genes_by_counts < 3500,:]
```


```python
K562(adata)
THP1(adata)
```


```python
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.1, multi_panel=True)
```


```python
adata
```


```python
sc.pp.normalize_total(adata, target_sum=8e3)
sc.pp.log1p(adata)
adata_raw = adata
```


```python
#adata_raw = adata
adata = adata_raw
```


```python
sc.pp.highly_variable_genes(adata, min_mean=0.025, max_mean=2, min_disp=0.5)
#sc.pp.highly_variable_genes(adata)
```


```python
sc.pl.highly_variable_genes(adata)
```


```python
adata.var['ref_ID'] = np.in1d(adata.var.gene_name, ref_ID)
adata
```


```python
adata = adata[:, adata.var.ref_ID]
adata
```


```python
sc.pp.scale(adata, max_value=10)
```


```python
sc.tl.pca(adata, svd_solver='arpack')
```


```python
sc.pl.pca(adata, color=adata.obs, projection='2d', size=100)
```


```python
sc.pl.pca_variance_ratio(adata, log=True, n_pcs=7)
```


```python
sc.pp.neighbors(adata, n_neighbors=200, n_pcs=5)
sc.tl.umap(adata)
sc.pl.umap(adata)
```


```python
sc.tl.leiden(adata, resolution = 0.1)
```


```python
with rc_context({'figure.figsize': (5, 5)}):
    sc.pl.umap(adata, color=adata.obs)
```


```python
plt.set_cmap('viridis')
plt.cm.get_cmap()
```


```python
for col in adata.obs:
    with rc_context({'figure.figsize': (5, 5)}):
        sc.pl.umap(adata, color=col, size=10, save="_fk209_clusteing"+str(col)+".pdf")
```


```python
adata.write_h5ad('~/PATH_TO/processed/others/2022-05-24-10X_final/2022-05-24-10X/fk209_proc_cDNA.h5ad')
```


```python

```
