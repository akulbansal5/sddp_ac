#$ -N gep_65_m3_s3
#$ -j y
#$ -p -0
#$ -S /bin/bash
#$ -pe smp 2
#$ -o /home/akul/sddp_ac/experiments/gep/logfiles/gep_65_3_s3.log
#$ -l h_rt=24:00:00
#$ -l h_vmem=20g
julia --project=/home/akul/sddp_ac /home/akul/sddp_ac/experiments/gep/gep_experiment.jl 65 3 3
