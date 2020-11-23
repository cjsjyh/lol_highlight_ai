#!/bin/bash
#test_and_val.sh

# RAN=$(seq 0 4)

# echo -n "set name of model to run : "
# read MODEL

# echo -n "give name of data except .h5 : "
# read H5

echo -n "set name of experiment's log : "
read LOGNAME

echo -n "set name of result_file : "
read RESULTDIR

mkdir result
mkdir result/result_$RESULTDIR
python3 hpo_test.py >& result/result_$RESULTDIR/$LOGNAME.log
# python3 hpo_test.py -m $MODEL > result/result_$RESULTDIR/evaluation.$MODEL.$LOGNAME.log
# mv data/models/$H5* result/result_$RESULTDIR
# #mv data/results.txt result/result_$RESULTDIR
# mv data/splits/$H5* result/result_$RESULTDIR
# cp datasets/$H5* result/result_$RESULTDIR
# cp machine_sum.py result/result_$RESULTDIR
# cp precision.py result/result_$RESULTDIR

# cd result/result_$RESULTDIR
# for i in $RAN
# do
#     python3 precision.py -i $i -d $H5  > evaluation$i.txt
# done

# rm $H5.h5