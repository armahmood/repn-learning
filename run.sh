#!/bin/bash

while getopts r:f:s:x: option
    do
        case "${option}" in
            r) N=${OPTARG};;
            f) features=${OPTARG};;
            s) seeds=${OPTARG};;
            x) search=${OPTARG};;
        esac
done

for feature in ${features[@]}; do
    for (( i = 1; i <= N; i++ )); do
        for seed in ${seeds[@]}; do
            if [ $search -eq 1 ]
            then
                nohup python learner.py --search -f $feature --seeds $seed --save_losses=True &> nohup${feature}_${i}_${search}.out &
            else
                nohup python learner.py -f $feature --seeds $seed --save_losses=True &> nohup${feature}_${i}_${search}.out &
            fi
        done
    done
done

