#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from multiprocessing import Pool
from functools import partial
from copy import deepcopy

def collision(a,b):
	xa1		= a[0]
	xb1		= b[0]
	xa2		= a[0]+a[2]
	xb2		= b[0]+b[2]
	ya1		= a[1]
	yb1		= b[1]
	ya2		= a[1]+a[3]
	yb2		= b[1]+b[3]
	iLeft	= max(xa1,xb1)
	iRight	= min(xa2,xb2)
	iTop	= max(ya1,yb1)
	iBottom	= min(ya2,yb2)
	return max(0, iRight-iLeft) * max(0, iBottom-iTop)

def score(lab_w,lab_l,fixed_params,objs):
	s = 0
	for o0 in objs:
		s += 5*collision(o0,[0.0,0.0,lab_w,lab_l])
		for o1 in fixed_params:
			s -= 2*collision(o0,o1)
		for o1 in objs:
			s -= collision(o0,o1)
	return s

# def mutate(params,sigma,i):
# 	# for i in range(len(params)):
# 	params[i][0] += np.random.normal(0.0,sigma)
# 	params[i][1] += np.random.normal(0.0,sigma)
# 	return params

def mutate(params,sigma):
	for i in range(len(params)):
		params[i][0] += np.random.normal(0.0,sigma)
		params[i][1] += np.random.normal(0.0,sigma)
	return params


def movable_to_params(movable,lab_w,lab_l):
	params = np.zeros((len(movable),5))
	for i,o in enumerate(movable):
		params[i,0] = np.random.randint(lab_w-o['size'][0])
		params[i,1] = np.random.randint(lab_l-o['size'][0])
		params[i,2] = o['size'][0]
		params[i,3] = o['size'][1]
	return params

def params_to_movable(movable,params):
	for i,p in enumerate(params):
		movable[i]['loc'] = (p[0],p[1])
		movable[i]['size'] = (p[2],p[3])


def main():

	POP = 64
	EPOCHS = 2000
	sigma = 100

	# Object dataset
	df = pd.read_csv('lab.csv')
	df = df.set_index(df['Name'])

	# Lab dimensions
	lab_w = df.loc['Lab']['w']
	lab_l = df.loc['Lab']['l']

	# Fixed Objects 
	fixed = [
		{'name':'Pillar','loc': (0,610)},
		{'name':'Pillar','loc': (630,610)},
		{'name':'Conference','loc': (0,0)},
		{'name':'Hatch','loc': (630,120)},
	]
	fixed = [dict([(k,v) for k,v in o.items()]+[('Color',df.loc[o['name']]['Color']),('size',(df.loc[o['name']]['w'],df.loc[o['name']]['l']))]) for o in fixed]



	# Movable Objects
	movable = []
	for o in df[df['Color']=='b'].values:
		for n in range(o[1]):
			movable.append({'name':o[0],'loc': (0.0,0.0),'size': (o[2],o[3]),'Color': 'b'})

	# Init
	pop = [None for p in range(POP)]
	for p in range(POP):
		pop[p] = movable_to_params(movable,lab_w,lab_l)
	fixed_params = movable_to_params(fixed,lab_w,lab_l)

	evaluator = partial(score,lab_w,lab_l,fixed_params)
	pool = Pool(processes=POP)
	for e in range(EPOCHS):

		# Evaluation
		scores = pool.map(evaluator,pop)
		series = pd.Series(scores)
		series = series.sort_values(ascending=False)
		if e%10==0:
			print '%3d %3d %9.3f' % (e,series.index[0],series.iloc[0])
		best = deepcopy(pop[series.index[0]])

		# Selection
		new_pop = [None for p in range(POP)]
		for p in range(POP):
			idx = np.random.randint(p+1)
			new_pop[p] = deepcopy(pop[series.index[idx]])
		pop = new_pop

		# Mutation
		for p in range(POP):
			mutate(pop[p],sigma)
			# mutate(pop[p],sigma,np.random.randint(len(pop[p])-1))

		# Elitism
		pop[-1] = deepcopy(best)


	# Params to movable
	params_to_movable(movable,best)
		

	# Plot Size
	fig,ax = plt.subplots(1)
	ax.set_aspect('equal')
	ax.set_xlim((0.0,lab_w))
	ax.set_ylim((0.0,lab_l))

	# Plotting Objects 
	for o in fixed+movable:
		w = o['size'][0]
		l = o['size'][1]
		c = o['Color']
		ax.add_patch(Rectangle((o['loc']),w,l,linewidth=1,edgecolor=c,facecolor='none'))
		ax.annotate(o['name'], (o['loc'][0]+0.5*w,o['loc'][1]+0.5*l), color=c, fontsize=6, ha='center', va='center')


	# Figure
	plt.xlabel('Position X (centimeters)')
	plt.ylabel('Position Y (centimeters)')
	plt.grid(which='major',axis='both')
	plt.savefig('out.svg')



if __name__ == "__main__":
	main()
