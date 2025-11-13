#$ -N smkp_248_m5_s2
#$ -j y
#$ -p -0
#$ -S /bin/bash
#$ -pe smp 2
#$ -o /home/akul/sddp_ac/experiments/smkp/logfiles/smkp_248_5_s2.log
#$ -l h_rt=24:00:00
#$ -l h_vmem=20g
julia --project=/home/akul/sddp_ac /home/akul/sddp_ac/experiments/smkp/smkp_experiment.jl 248 5 2
