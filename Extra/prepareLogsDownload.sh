#!/bin/bash

cd $HOME
mkdir logs
cd logs
mkdir CNN
mkdir RNN

cd $SHARED_SCRATCH/pao3/Parkinsons-Disease-Digital-Biomarker/

cd CNN
cp -a tf_logs $HOME/logs/CNN
cp -a checkpoints $HOME/logs/CNN

cd ../RNN
cp -a tf_logs $HOME/logs/RNN
cp -a checkpoints $HOME/logs/RNN

cd $HOME
tar -cvf logs.tar logs
