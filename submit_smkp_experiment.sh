#!/bin/bash

########################
##### qsub options #####
########################
#
# Any line starting with #$ is interpreted as command line option for qsub

for seed in 1 2 3; do
    # Running all 18 selected instance IDs:
    # 256-258: (3, 10, 30, 3)
    # 316-318: (3, 10, 30, 5)
    # 346-348: (4, 10, 30, 3)
    # 234, 240, 246: (4, 10, 30, 10)
    # 456-458: (5, 15, 40, 3)
    # 236, 242, 248: (5, 15, 40, 10)
    for id in 234 236 240 242 246 248 256 257 258 316 317 318 346 347 348 456 457 458; do
        for method in 0 1 2 3 4 5; do
            touch /home/akul/sddp_ac/experiments/smkp/shfiles/smkp_${id}_${method}_s${seed}.sh
            f=/home/akul/sddp_ac/experiments/smkp/shfiles/smkp_${id}_${method}_s${seed}.sh
            echo "#$ -N smkp_${id}_m${method}_s${seed}">$f
            echo "#$ -j y">>$f
            echo "#$ -p -0">>$f
            echo "#$ -S /bin/bash">>$f
            echo "#$ -pe smp 2">>$f
            echo "#$ -o /home/akul/sddp_ac/experiments/smkp/logfiles/smkp_${id}_${method}_s${seed}.log">>$f
            echo "#$ -l h_rt=24:00:00">>$f
            echo "#$ -l h_vmem=20g">>$f
            echo "julia --project=/home/akul/sddp_ac /home/akul/sddp_ac/experiments/smkp/smkp_experiment.jl $id $method $seed">>$f
            qsub /home/akul/sddp_ac/experiments/smkp/shfiles/smkp_${id}_${method}_s${seed}.sh
            sleep 5
        done
    done
done

