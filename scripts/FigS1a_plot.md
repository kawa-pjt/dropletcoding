```python
import sys
import os
import numpy as np
import glob
import re
from skimage import io 
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
```

###  move to the data folder


```python
os.mkdir("~/PATH_TO/Imagedata/processed/2022-02-16_out")
root = "~/PATH_TO/Imagedata/2022-02-16/2022-02-16_images/"
OUTPUTroot = "~/PATH_TO/Imagedata/processed/2022-02-16_out/"
```


```python
Akey = "*d3.tif"
A =  [os.path.abspath(p) for p in glob.glob(root + Akey)]
A = sorted(A, key = lambda x : (re.search(r'(\d*)d3.tif', x).groups()[0]))
```


```python
Bkey = "*d4.tif"
B =  [os.path.abspath(p) for p in glob.glob(root + Bkey)]
B = sorted(B, key = lambda x : (re.search(r'(\d*)d4.tif', x).groups()[0]))
```

## Data 
### Before light exposure: A[0:3]
### After light exposure: A[3:6]


```python
fig = plt.figure(figsize=(10,30), dpi=300)
for i in range(len(A)):    
    ax1 = fig.add_subplot(len(A),2, 2*i+1)
    ax1.imshow(np.rot90(io.imread(A[i])), cmap ='gray')
    
    scalebar = ScaleBar(0.309, "um", length_fraction=0.2)
    plt.gca().add_artist(scalebar)
    
    ax2 = fig.add_subplot(len(A),2, 2*(i+1))
    ax2.imshow(np.rot90(io.imread(B[i])), cmap ='gray')
    
    scalebar = ScaleBar(0.309, "um", length_fraction=0.2)
    plt.gca().add_artist(scalebar)
plt.savefig(OUTPUTroot + '2022-02-16_proc.svg', format="svg")
```


    
![png](output_6_0.png)
    



```python

```
