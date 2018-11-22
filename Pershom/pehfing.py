import gudhi
import Yosdat as yd
import numpy as np
from skimage import io
from mapper import filters
global Patches
def main():
	yos = io.imread("yosemitehikes.jpg", as_grey=True)
	global Patches
	Patches = yd.buildPatches(yos, 0.8)
	Patches = np.array([Patches[i] for i in np.random.choice(len(Patches),10000, replace=False)])
	Patches = yd.smoothKNN(Patches, 15)
	Patches = yd.smoothKNN(Patches, 15)
def dpmg(n, m):
	deninds = np.argpartition(filters.kNN_distance(Patches, 30),n)
	denspat = [Patches[i] for i in deninds[0:n]]
	return yd.buildMeshFromPatches(denspat,m)
def rstg(n, A):
	rp = gudhi.RipsComplex(A)
	st = rp.create_simplex_tree(max_dimension=n)
	diag = st.persistence()
	gudhi.plot_persistence_barcode(diag)
