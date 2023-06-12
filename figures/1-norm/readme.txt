-- CDL tests --
Old version of the code, where the input signal and the atoms were normalized.
This led to an explosion of the parameters (although invisible here) because the algorithm desperetaly tried to increase E_0 and E_max values to reach the signal's amplitue without success, because the atoms were always normalized.