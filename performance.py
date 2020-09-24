#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 20:16:37 2020

@author: juan
"""

import motmetrics as mm
import numpy as np

from absl import app, flags
from absl.flags import FLAGS
import logging
import glob 

flags.DEFINE_string('annotations',None,'Output detector labels')
flags.DEFINE_string('outputDet',None,'Annotations of video')

logger = logging.getLogger()
logger.disabled = True

def parse(labels, coma = True):
    file = open(labels,'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x.rstrip().lstrip() for x in lines]
    if coma:
        lines = [x.split(',') for x in lines]
    else: 
        lines = [x.split(' ') for x in lines]
    all_ = np.zeros((len(lines),6))
    for i,j in enumerate(lines):
        temp =  [float(x) for x in j]
        all_[i,:] =temp[:6]

    frame = {}
    detect = []
    for i in range(1,int(max(all_[:,0]))+1):
        frame[i] = all_[all_[:,0] == i,1:]

    return frame
  
def calcular(pathresult,pathlabels):
    acc = mm.MOTAccumulator(auto_id=True)
    mh = mm.metrics.create()
    result = parse(pathresult)
    labels = parse(pathlabels)

    for key in result.keys():
    
        ids_verdaderos = labels[key][:,0]
        ids_hypotesis = result[key][:,0]

        a = labels[key][:,1:]
        b = result[key][:,1:]
    
        dist = mm.distances.iou_matrix(a, b, max_iou=0.5)
    
        frameid = acc.update(ids_verdaderos,ids_hypotesis,dist)

    summary = mh.compute_many(
        [acc],
        names=['Resultado'],
        metrics=mm.metrics.motchallenge_metrics,
        )

    strsummary = mm.io.render_summary(
        summary.iloc[:,:-3],
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
        )
    print("Archivos: ",pathlabels,pathresult)
    print(strsummary)
    
    
def main(_arg):
    results = sorted(glob.glob(FLAGS.outputDet))
    labels = sorted(glob.glob(FLAGS.annotations))
    for result, label in zip(results,labels):
        
        calcular(result,label)
       

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

