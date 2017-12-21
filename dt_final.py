import numpy as np


import scipy.io
from final_fv import *

def divide(db):
	labels = list(set(data[-1] for data in db))
	a, b, c, split_index = 999, 999, 999, None
	for i in range(len(db[0])-1):
		for data in db:
			splits = split_d(i, data[i], db)
			score_temp = calculate_score(splits, labels)
			if score_temp < c:
				a, b, c, split_index = i, data[i], score_temp, splits
	return {'index':a, 'value':b, 'splits':split_index}


def calculate_score(splits, classes):
	a = float(sum([len(group) for group in splits]))
	# print(a)
	score = 0.0
	for group in splits:
		size = float(len(group))
		if size == 0:
			continue
		b = 0.0
		for j in classes:
			p = [temp[-1] for temp in group].count(j) / size
			# p = [r[-1] for r in range / size
			# p = r[0]
			 
			b += p * p
		score += (1.0 - b) * (size / a)
	return score

def split_d(key, v, db):
	l, r = list(), list()
	for i in db:
		if i[key] < v:
			l.append(i)
		else:
			r.append(i)
	# print(l)
	# print(r)
	return l, r
 
def leaf(group):
	# 	values = [i[-2] for i in group]
	# values = [i[-1] for i in group, j = 0]

	values = [i[-1] for i in group]
	# print(values)
	return max(set(values), key=values.count)

def predict(node, data):
	# print(node.shape)
	# print(row.shape)
	if data[node['index']] < node['value']:
		if isinstance(node['l'], dict):
			return predict(node['l'], data)
		else:
			return node['l']
	else:
		if isinstance(node['r'], dict):
			return predict(node['r'], data)
		else:
			return node['r']
 
def divide_data(split_node,depth):
	d = 5
	size = 10
	l, r = split_node['splits']
	del(split_node['splits'])
	if not l or not r:
		split_node['l'] = split_node['r'] = leaf(l + r)
		return
	if depth >= d:
		split_node['l'], split_node['r'] = leaf(l), leaf(r)
		return
	if len(l) <= size:
		split_node['l'] = leaf(l)
	else:
		split_node['l'] = divide(l)
		divide_data(split_node['l'], d, size, depth+1)
	if len(r) <= size:
		split_node['r'] = leaf(r)
	else:
		split_node['r'] = divide(r)
		divide_data(split_node['r'], d, size, depth+1)
 
# db = scipy.io.loadmat('knn_data.mat')


def create_model(train):
	r = divide(train)
	divide_data(r, 1)
	return r

def dt(train, test):
	t = create_model(train)
	output = list()
	for r in test:
		temp = predict(t, r)
		output.append(temp)
	return(output)
 



x_train,y_train,x_test = loaddataTrain()
y_train = np.reshape(y_train,(y_train.shape[0],1))
x_train_y_train = np.hstack([x_train,y_train])

predictions_temp = dt(x_train_y_train[:,1:], x_test[:,1:])
for i in range(x_test.shape[0]):
    print (" {} : {}".format(x_test[i][0],predictions_temp[i]))
# print(predictions_temp)
