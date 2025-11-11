#!/bin/bash

########################
##### qsub options #####
########################
#
# Any line starting with #$ is interpreted as command line option for qsub

for id in 54 56 57 58 59; do
    for method in 5; do
        touch /home/akul/sddp_ac/experiments/gep/shfiles/gep_${id}_${method}.sh
        f=/home/akul/sddp_ac/experiments/gep/shfiles/gep_${id}_${method}.sh
        echo "#$ -N gep_${id}_m${method}">$f
        echo "#$ -j y">>$f
        echo "#$ -p -0">>$f
        echo "#$ -S /bin/bash">>$f
        echo "#$ -pe smp 2">>$f
        echo "#$ -o /home/akul/sddp_ac/experiments/gep/logfiles/gep_${id}_${method}.log">>$f
        echo "#$ -l h_rt=24:00:00">>$f
        echo "#$ -l h_vmem=10g">>$f
        echo "julia --project=/home/akul/sddp_ac /home/akul/sddp_ac/experiments/gep/gep_experiment.jl $id $method">>$f
        qsub /home/akul/sddp_ac/experiments/gep/shfiles/gep_${id}_${method}.sh
        sleep 5
    done
done
