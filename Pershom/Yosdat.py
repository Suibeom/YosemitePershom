import time
from tqdm import *
from numpy import linalg
import numpy as np
import matplotlib.pyplot as plt
import csv
from skimage import io
D = np.array([[2, -1, 0, -1, 0, 0, 0, 0, 0],
		[-1, 3, -1, 0, -1, 0, 0, 0, 0],
		[0, -1, 2, 0, 0, -1, 0, 0, 0],
		[-1, 0, 0, 3, -1, 0, -1, 0, 0],
		[0, -1, 0, -1, 4, -1, 0, -1, 0],
		[0, 0, -1, 0, -1, 3, 0, 0, -1],
		[0, 0, 0, -1, 0, 0, 2, -1, 0],
		[0, 0, 0, 0, -1, 0, -1, 3, -1],
		[0, 0, 0, 0, 0, -1, 0, -1, 2]])
Pro = np.array([[1,0,0,0,0,0,0,0,-1],
		[0,1,0,0,0,0,0,0,-1],
		[0,0,1,0,0,0,0,0,-1],
		[0,0,0,1,0,0,0,0,-1],
		[0,0,0,0,1,0,0,0,-1],
		[0,0,0,0,0,1,0,0,-1],
		[0,0,0,0,0,0,1,0,-1],
		[0,0,0,0,0,0,0,1,-1]])
def cnorm(y):
	return np.dot(np.dot(y,D),y)

def samp(B, A):
	return np.log(A[B[0]:B[0]+3, B[1]:B[1]+3].reshape(9))
def L1D(B, C, A):
	return np.sum(np.abs(nsamp(B, A) - nsamp(C,A)))
def L1(y1,y2):
	return np.sum(np.abs(y1-y2))
def L1SD(B,C, A):
	return np.min(np.fromiter([L1D(D, C, A) for D in B], float))
def L1S(S, y):
	return np.min(np.fromiter([L1(y, y2) for y2 in S],float))
def nsamp(B, A):
	#y = np.dot(Pro,samp(B, A))
	y = samp(B, A)
	return y /np.sqrt(np.dot(y,y)) 
def picknm(A):
	return np.random.choice(A[0]-3), np.random.choice(A[1]-3)
def KNN(A, v, n):
	D = np.fromiter([np.linalg.norm(v-A[i]) for i in range(len(A))],float)
	k = np.argpartition(D, n)[n]
	return np.linalg.norm(v-D[k])
def smoothKNN(A, n):
	Ap = np.zeros_like(A)
	for i in tqdm(range(len(A))):
		D = np.fromiter([np.linalg.norm(A[i]-A[j]) for j in range(len(A))],float)
		Ap[i] = np.mean([A[j] for j in  np.argpartition(D, n)[0:n-1]],axis=0)
	return Ap
def augmentKNN(A, n):
	D = np.fromiter([KNN(A, A[i],n) for i in range(len(A))])
	return np.hstack([D, A])
def density(A, v, sig):
	n = A.shape[1]
	N = len(A)
	nf = 1/(np.sqrt(2*np.pi)*sig)**n
	pd = np.mean(np.exp([np.linalg.norm(v-A[i])/(2*sig) for i in range(N)]))
	return pd*nf
def augmentDensity(A, sig):
	D = np.fromiter([density(A,A[i],sig) for i in range(len(A))])
	return np.hstack([D,A])
def contrastThreshold(yos, k):
	Dat = [cnorm(samp(picknm(yos.shape), yos)) for i in range(10000)]
	k = np.sort(Dat)[int(k*10000)]
	return k
def hCPatches(yos, t):
	A = []
	for i in range(yos.shape[0] - 3):
		for j in range(yos.shape[1] - 3):
			if cnorm(samp((i,j), yos)) > t:
				A.append((i,j))
	return A
	
def buildPatches(yos, t, s="", n=0):
	A = hCPatches(yos, t)	
	if n>0:
		B = np.random.choice(len(A), n)
		As = [A[I] for I in B]
		A = As
	Samps = np.array([nsamp(I, yos) for I in A])
	if len(s)>4:
		fop = open(s, "write")
		f = csv.writer(fop)
		f.writerows(Samps)
		fop.close()
	return Samps

def weirdMesh(yos, A, m):
	fop = open("DatMebh"+str(m+1)+".csv","write")
	f = csv.writer(fop)
	B = []
	B.append(A[np.random.choice(len(A))])
	mesh=0.1
	l = 40
	n = 1000
	idx = int(0.2*l + 0.5)
#pick 'l' points and add the point furthest from all points in B!
	for i in range(n):
		C = [A[j] for j in np.random.choice(len(A), l)]
		Cp = np.fromiter([L1SD(B, P, yos) for P in C], float)
		B.append(C[np.argpartition(Cp, idx)[idx]])
		q = float(i)/n		
		idx = int(l*(.2 + .7*(1-q)**.25))
		print(m)
		print(i)

	Samps = np.array([nsamp(I, yos) for I in B])
	f.writerows(Samps)
	fop.close()
#Start at a random point and sample as many points as you can get.
def buildMesh(yos, A, mesh, outfile=""):
	L = np.random.choice(len(A),len(A), replace=False)

	B = [A[L[0]]]
	for i in tqdm(range(len(A))):
		yd = L1SD(B, A[L[i]],yos)
		if yd > mesh:
			B.append(A[L[i]])
			print(i)
	Samps = np.array([nsamp(I, yos) for I in B])
	if len(outfile)>4:	
		f.writerows(Samps)
		fop = open(s,"write")
		f = csv.writer(fop)
		fop.close()
	return Samps
def buildMeshFromPatches(A, mesh, outfile=""):
	L = np.random.choice(len(A),len(A), replace=False)

	B = [A[L[0]]]
	for i in range(len(A)):
		yd = L1S(B, A[L[i]])
		if yd > mesh:
			B.append(A[L[i]])
			#print(i)
	if len(outfile)>4:	
		f.writerows(B)
		fop = open(s,"write")
		f = csv.writer(fop)
		fop.close()
	return B
def main():
	yos = io.imread("yosemitehikes.jpg", as_grey=True)
	buildPatches(yos,0.8, "Dat9.csv")
	np.sort(Dat)[9800]
