#!/bin/bash

cd $HOME
mkdir Results
cd Results
mkdir CSVtables
mkdir logs
cd logs
mkdir CNN
mkdir RNN

cd $SHARED_SCRATCH/pao3/Parkinsons-Disease-Digital-Biomarker/

cd CNN
ls | grep -P "^slurm-.*.out" | xargs -d"\n" rm
rm output.txt
mv tf_logs $HOME/Results/logs/CNN
mkdir tf_logs
mv checkpoints $HOME/Results/logs/CNN
mkdir checkpoints
mv Folds $HOME/Results/logs/CNN
mkdir Folds

cd ../RNN
ls | grep -P "^slurm-.*.out" | xargs -d"\n" rm
rm output.txt
mv tf_logs $HOME/Results/logs/RNN
mkdir tf_logs
mv checkpoints $HOME/Results/logs/RNN
mkdir checkpoints

cd ../data
mv features.csv $HOME/Results/CSVtables
mv features_extra_columns.csv $HOME/Results/CSVtables
mv features_noOutliers_extra_columns.csv $HOME/Results/CSVtables
mv train.csv $HOME/Results/CSVtables
mv train_extra_columns.csv  $HOME/Results/CSVtables
mv train_noOutliers_extra_columns.csv $HOME/Results/CSVtables
mv train_augmented_extra_columns.csv  $HOME/Results/CSVtables
mv train_augmented_noOutliers_extra_columns.csv $HOME/Results/CSVtables
mv val.csv $HOME/Results/CSVtables
mv val_extra_columns.csv $HOME/Results/CSVtables
mv val_noOutliers_extra_columns.csv $HOME/Results/CSVtables
mv test.csv $HOME/Results/CSVtables
mv test_extra_columns.csv $HOME/Results/CSVtables
mv test_noOutliers_extra_columns.csv $HOME/Results/CSVtables
for (( i=0; i<10; i++))
	do
		mv "fold"$i".csv" $HOME/Results/CSVtables
		mv "fold"$i"_extra_columns.csv" $HOME/Results/CSVtables
		mv "fold"$i"_augmented_extra_columns.csv" $HOME/Results/CSVtables
		mv "fold"$i"_noOutliers_extra_columns.csv" $HOME/Results/CSVtables
		mv "fold"$i"_noOutliers.csv" $HOME/Results/CSVtables
		mv "fold"$i"_noOutliers_augmented_extra_columns.csv" $HOME/Results/CSVtables
	done

cd ../Features
ls | grep -P "^slurm-.*.out" | xargs -d"\n" rm
rm output.txt

cd $HOME
tar -cvf results.tar Results