
import numpy as np

retour = np.searchsorted([0.1, 0.11, 0.11, 0.12, 0.12, 4,5], 0.13, side="left")


print(retour)
