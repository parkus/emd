emd
===
Carries out empirical mode decomposition of the provided function. Written following [Huang et al. (1998; RSPA 454:903)](http://adsabs.harvard.edu/abs/1998RSPSA.454..903E).

I wrote this to be used with the Astrophysically Robust Correction of [Roberts et al. (2013 MNRAS 435:3639)](http://adsabs.harvard.edu/abs/2013MNRAS.tmp.2261R), see
https://github.com/parkus/arc.

Please acknowledge R. O. Parke Loyd if you use this code in your research.

Example Use
-----------
(also in test_script.py)

```python
import numpy as np
import emd
import matplotlib.pyplot as plt

# GENERATE A POLYNOMIAL + SINE TEST FUNCTION
N = 200
t = np.arange(200, dtype=float) - N / 2
amp = 10.0

# polynomial to start
y = t**2
fac = amp / np.max(y) #but let's make the numbers easier to read
y *= fac

# but let's give it an offset just to make sure that doesn't screw with things
y += amp / 5.0

# and now add in a sine
period = N / 10.0
phase = np.random.uniform(0.0, 2*np.pi)
y += (amp / 10.0) * np.sin(2*np.pi * t / period + phase)

# ADD NOISE, IF DESIRED
y += np.random.normal(0.0, amp/50.0, N)

# DECOMPOSE
c, r = emd.emd(t, y)

# PLOT
pf, = plt.plot(t, y)
pr, = plt.plot(t, r)
pcs = plt.plot(t, c, 'k-')

plt.legend((pf, pcs[0], pr), ('original function', 'modes', 'residual'))
```