#$ -N smkp_457_m4_s1
#$ -j y
#$ -p -0
#$ -S /bin/bash
#$ -pe smp 2
#$ -o /home/akul/sddp_ac/experiments/smkp/logfiles/smkp_457_4_s1.log
#$ -l h_rt=24:00:00
#$ -l h_vmem=20g
julia --project=/home/akul/sddp_ac /home/akul/sddp_ac/experiments/smkp/smkp_experiment.jl 457 4 1
