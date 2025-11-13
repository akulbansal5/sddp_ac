#$ -N gep_76_m2_s2
#$ -j y
#$ -p -0
#$ -S /bin/bash
#$ -pe smp 2
#$ -o /home/akul/sddp_ac/experiments/gep/logfiles/gep_76_2_s2.log
#$ -l h_rt=24:00:00
#$ -l h_vmem=20g
julia --project=/home/akul/sddp_ac /home/akul/sddp_ac/experiments/gep/gep_experiment.jl 76 2 2
