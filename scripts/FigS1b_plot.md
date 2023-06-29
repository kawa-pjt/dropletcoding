```python
import sys
import os
import math
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
from skimage import io 
from matplotlib_scalebar.scalebar import ScaleBar
```


```python
#os.mkdir("~/PATH_TO/Imagedata/processed/2022-03-11_out")
root = "~/PATH_TO/Imagedata/2022-03-11/2022-03-11_images/"
OUTPUTroot = "~/PATH_TO/Imagedata/processed/2022-03-11_out/"
```


```python
Akey = "*d1.tif"
A =  [os.path.abspath(p) for p in glob.glob(root + Akey)]
```


```python
Bkey = "*d4.tif"
B =  [os.path.abspath(p) for p in glob.glob(root + Bkey)]
```


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
plt.savefig(OUTPUTroot + '2022-03-11_proc.svg', format="svg")
```


    
![png](output_4_0.png)
    



```python

```
