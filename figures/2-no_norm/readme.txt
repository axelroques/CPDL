-- CDL tests --
New version of the code, where the norms were removed.
We achieve good convergence results (with step size=0.0001 et regularization=1).
Depends on the initial estimates of the parameters. I found that one of the atoms can be stuck in an impossible state where t_50 exceeds the length of the atom L
	-> a supplementary step could be added after updating the parameters so that this does not happen.