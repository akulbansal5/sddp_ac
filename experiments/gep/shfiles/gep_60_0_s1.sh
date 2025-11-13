#$ -N gep_60_m0_s1
#$ -j y
#$ -p -0
#$ -S /bin/bash
#$ -pe smp 2
#$ -o /home/akul/sddp_ac/experiments/gep/logfiles/gep_60_0_s1.log
#$ -l h_rt=24:00:00
#$ -l h_vmem=20g
julia --project=/home/akul/sddp_ac /home/akul/sddp_ac/experiments/gep/gep_experiment.jl 60 0 1
