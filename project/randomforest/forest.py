#!/usr/bin/python

import numpy as np
from tree import *
import os
from glob import glob
import itertools
import shutil
import multiprocessing as mp

from weakLearner import WeakLearner, AxisAligned

class Forest:
    def __init__( self,
                  ntrees=20,
                  tree_params={ 'max_depth' : 10,
                           'min_sample_count' : 5,
                           'test_count' : 100,
                           'test_class' : AxisAligned() } ):
        self.ntrees = ntrees
        self.tree_params = tree_params
        self.trees=[]
        self.labels = []

    def __len__(self):
        return self.ntrees
        
    def grow(self,points,responses,nprocs=1):
        for r in responses:
            if r not in self.labels:
                self.labels.append(r)

        if nprocs == 1:
            for i in range(self.ntrees):
                self.trees.append( Tree( self.tree_params ) )
                self.trees[i].grow( points, responses, self.labels )
        else:
            raise NotImplementedError, "The parallel version of the forest.grow() function has yet to be implemented."
            
            # self.points = points
            # self.responses = responses
            # grow_input = np.arange(self.ntrees)
            thread_ids = np.arange(self.ntrees)

            grow_input = itertools.izip(
                itertools.repeat(thread_ids),
                itertools.repeat(points),
                itertools.repeat(responses),
                itertools.repeat(self.labels)
                # thread_ids,
                # self.points,
                # self.responses,
                # self.labels
                # itertools.repeat(trading_algo),
                # itertools.repeat(commission),
                # itertools.repeat(self.stop_loss_percent),
                # itertools.repeat(tickers_spreads),
                # itertools.repeat(data)
                )

            for i in range(self.ntrees):
                self.trees.append( Tree( self.tree_params ) )

            pool = mp.Pool(processes=nprocs)
            pool.map(self.grow_trees, grow_input)

    def grow_trees(self, params):
        thread_id, points, responses, labels = params

        # self.trees.append( Tree( self.tree_params ) )
        # self.trees[thread_id].grow( self.points, self.responses, self.labels )
        self.trees[thread_id].grow( points, responses, labels )

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

