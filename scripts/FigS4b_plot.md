```python
import FlowCal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import math
import seaborn as sns
import os
```


```python
# Indicate a folder containing cell data
# folder_c = CELL_DATA_FOLDER('~/20220426)
```


```python
# Indicate a folder containing beads data
# folder_b = BEADS_DATA_FOLDER('~/PCdBB check 20220426)
```


```python
# out_folder = OUTPUT_FOLDER ('~/processed/20220426_out')
```


```python
filenames_c = ['K562-aCD71_MACS-Pre.fcs', \
             'K562-aCD71_MACS-Post.fcs', \
             'K562-aCD71_MACS-Pass through.fcs']
```


```python
filenames_b = ['PCdBB-bcd-biotin_NC.fcs', \
             'PCdBB-bcd-biotin_UV- T24V-Cy5.fcs', \
             'PCdBB-bcd-biotin_UV+ T24V-Cy5.fcs']
```


```python
Gate_FITC = 1e3
Gate_AF647 = 1e3
```

# Plotting FACS analysis data for cells (with a AF647 probe)


```python
for filename in filenames_c:
    # read a fcs file
    fcs = FlowCal.io.FCSData(folder_c + filename)
    
    # clean up
    fcs = FlowCal.gate.density2d(fcs, channels=['FSC-A', 'SSC-A'], gate_fraction=0.8)
    fcs_df = pd.DataFrame(fcs, columns = fcs.channels)

    # define pct_FITC and pct_AF647
    pct_FITC = 100*((fcs_df['FITC-A'] > Gate_FITC) & (fcs_df['Alexa Fluor 647-A'] <= Gate_AF647)).sum() / len(fcs_df)
    pct_AF647 = 100*((fcs_df['FITC-A'] <= Gate_FITC) & (fcs_df['Alexa Fluor 647-A'] > Gate_AF647)).sum() / len(fcs_df)

    pct_FITC = math.ceil(pct_FITC * 10) / 10
    pct_AF647 = math.ceil(pct_AF647 * 10) / 10

    # plot cleaned data with gating conditions & results
    
    fig, ax = plt.subplots(1,1,figsize=[4,4], dpi=300)

    ax.set_aspect('equal')
    ax.set_title(filename)
    ax.text(10**4.5, 10, pct_FITC)
    ax.text(10, 10**4.5, pct_AF647)
    ax.axvline(x=Gate_FITC, color = 'k', linestyle='dashed', linewidth=1)
    ax.axhline(y=Gate_AF647, color = 'k', linestyle='dashed', linewidth=1)
    ax.axhline(y=Gate_AF647, color = 'k', linestyle='dashed', linewidth=1)
    ax = FlowCal.plot.density2d(fcs, channels=['FITC-A', 'Alexa Fluor 647-A'], mode='scatter', xscale='log',yscale='log', xlim=(10, 125000), ylim=(10, 125000))
    plt.savefig(out_folder + filename[:-4] + '_processed_plot.pdf')
    plt.show()
    plt.close()
```


    
![png](output_8_0.png)
    



    
![png](output_8_1.png)
    



    
![png](output_8_2.png)
    



```python
# histplot of collected beads

for filename in filenames_b:
    # read a fcs file
    fcs = FlowCal.io.FCSData(folder_b + filename)
    
    
    fig, ax = plt.subplots(1,1,figsize=[4,4], dpi=300)
    ax.set_title(filename)
    ax = FlowCal.plot.hist1d(fcs, channel='Alexa Fluor 647-A', facecolor = 'gray', edgecolor = 'gray')
    plt.savefig(out_folder + filename[:-4] + '_processed_plot_postcheck.pdf')
    plt.show()
    plt.close()
```


    
![png](output_9_0.png)
    



    
![png](output_9_1.png)
    



    
![png](output_9_2.png)
    



```python

```
