# Adapted from Kevin Keraudren's code at https://github.com/kevin-keraudren/randomforest-python

import numpy as np
from tree import *
import os
from glob import glob
import itertools
import shutil
import multiprocessing as mp
from pprint import pprint

from weakLearner import WeakLearner, AxisAligned


def _grow_trees(params):
    thread_id, points, responses, labels, tree = params

    tree.grow(points, responses, labels)
    return tree


class Forest:
    def __init__( self,
                  ntrees=20,
                  tree_params={ 'max_depth' : 10,
                           'min_sample_count' : 5,
                           'test_count' : 100,
                           'test_class' : AxisAligned() },
                  nprocs=1):
        self.ntrees = ntrees
        self.tree_params = tree_params
        self.trees=[]
        self.labels = []
        self.nprocs = nprocs

    def __len__(self):
        return self.ntrees
        
    def grow(self,points,responses):
        for r in responses:
            if r not in self.labels:
                self.labels.append(r)

        if self.nprocs == 1:
            for i in range(self.ntrees):
                self.trees.append( Tree( self.tree_params ) )
                self.trees[i].grow( points, responses, self.labels )
        else:
            thread_ids = np.arange(self.ntrees)
            grow_input = itertools.izip(
                thread_ids,
                itertools.repeat(points),
                itertools.repeat(responses),
                itertools.repeat(self.labels),
                itertools.repeat(Tree(self.tree_params))
                )

            pool = mp.Pool(processes=self.nprocs)
            results = pool.map(func=_grow_trees, iterable=grow_input)
            self.trees = list(results)

    def predict(self, point, soft=False):
        r = {}

        for c in self.labels:
            r[c] = 0.0

        for i in range(self.ntrees):
            response = int(self.trees[i].predict(point))
            r[response] += 1

        if soft:
            for c in self.labels:
                r[c] /= self.ntrees

            return r
        else:
            response = None
            max_count = -1

            for c in self.labels:
                if r[c] > max_count:
                    response = c
                    max_count = r[c]

            return response

    def save(self,folder):
        if os.path.exists(folder):
            shutil.rmtree(folder)

        os.makedirs(folder)
        template = '%0'+str(int(np.log10(self.ntrees))) + 'd.data'

        for i in range(self.ntrees):
            filename = template % i
            self.trees[i].save(folder + '/' + filename)

        return

    def load(self,folder,test=WeakLearner()):
        self.trees = []

        for f in glob(folder+'/*'):
            self.trees.append( Tree() )
            self.trees[-1].load( f, test )

        self.ntrees = len(self.trees)
        self.labels = self.trees[0].labels

        return

