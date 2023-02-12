# 03 - Stats Review: The Most Dangerous Equation

In his famous article of 2007, Howard Wainer writes about very dangerous equations:

"Some equations  are  dangerous  if you  know them, and others are dangerous if you do not. The first category may  pose  danger  because the secrets  within its bounds open  doors  behind which lies terrible peril. The obvious winner in this is Einstein’s iconic equation $E = MC^2$, for it  provides  a  measure of  the  enormous energy hidden  within  ordinary  matter. \[...\] Instead I am interested in equations that unleash their danger not when we know about them, but rather when we do not. Kept close at hand, these equations allow us to understand things clearly, but their absence leaves us dangerously ignorant."

The equation he talks about is Moivre’s equation:

$
SE = \dfrac{\sigma}{\sqrt{n}} 
$

where $SE$ is the standard error of the mean, $\sigma$ is the standard deviation, and $n$ is the sample size. Sounds like a piece of math the brave and true should master, so let's get to it.

To see why not knowing this equation is very dangerous, let's look at some education data. I've compiled data on ENEM scores (Brazilian standardised high school scores, similar to SAT) from different schools for 3 years. I also cleaned the data to keep only the information relevant to us. The original data can be downloaded on the [Inep website](http://portal.inep.gov.br/web/guest/microdados#).

If we look at the top-performing school, something catches the eye: those schools have a reasonably small number of students. 
```
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style
style.use("fivethirtyeight")
```
