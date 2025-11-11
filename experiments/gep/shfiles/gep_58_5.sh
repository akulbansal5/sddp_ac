#$ -N gep_58_m5
#$ -j y
#$ -p -0
#$ -S /bin/bash
#$ -pe smp 2
#$ -o /home/akul/sddp_ac/experiments/gep/logfiles/gep_58_5.log
#$ -l h_rt=24:00:00
#$ -l h_vmem=10g
julia --project=/home/akul/sddp_ac /home/akul/sddp_ac/experiments/gep/gep_experiment.jl 58 5
