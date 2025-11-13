#!/bin/bash

########################
##### qsub options #####
########################
#
# Any line starting with #$ is interpreted as command line option for qsub

for seed in 1 2 3; do
    # Running one instance from each (stages, scenarios) group:
    # 62: (10, 10) - representative of 60, 61, 62
    # 65: (10, 20) - representative of 63, 64, 65
    # 68: (10, 50) - representative of 66, 67, 68
    # 71: (15, 3)  - representative of 69, 70, 71
    # 74: (20, 3)  - representative of 72, 73, 74
    # 77: (25, 3)  - representative of 75, 76, 77
    for id in 62 65 68 71 74 77; do
        for method in 0 1 2 3 4 5; do
            touch /home/akul/sddp_ac/experiments/gep/shfiles/gep_${id}_${method}_s${seed}.sh
            f=/home/akul/sddp_ac/experiments/gep/shfiles/gep_${id}_${method}_s${seed}.sh
            echo "#$ -N gep_${id}_m${method}_s${seed}">$f
            echo "#$ -j y">>$f
            echo "#$ -p -0">>$f
            echo "#$ -S /bin/bash">>$f
            echo "#$ -pe smp 2">>$f
            echo "#$ -o /home/akul/sddp_ac/experiments/gep/logfiles/gep_${id}_${method}_s${seed}.log">>$f
            echo "#$ -l h_rt=24:00:00">>$f
            echo "#$ -l h_vmem=20g">>$f
            echo "julia --project=/home/akul/sddp_ac /home/akul/sddp_ac/experiments/gep/gep_experiment.jl $id $method $seed">>$f
            qsub /home/akul/sddp_ac/experiments/gep/shfiles/gep_${id}_${method}_s${seed}.sh
            sleep 5
        done
    done
done
