#$ -N gep_73_m1_s3
#$ -j y
#$ -p -0
#$ -S /bin/bash
#$ -pe smp 2
#$ -o /home/akul/sddp_ac/experiments/gep/logfiles/gep_73_1_s3.log
#$ -l h_rt=24:00:00
#$ -l h_vmem=20g
julia --project=/home/akul/sddp_ac /home/akul/sddp_ac/experiments/gep/gep_experiment.jl 73 1 3
