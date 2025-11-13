#$ -N smkp_348_m0_s3
#$ -j y
#$ -p -0
#$ -S /bin/bash
#$ -pe smp 2
#$ -o /home/akul/sddp_ac/experiments/smkp/logfiles/smkp_348_0_s3.log
#$ -l h_rt=24:00:00
#$ -l h_vmem=20g
julia --project=/home/akul/sddp_ac /home/akul/sddp_ac/experiments/smkp/smkp_experiment.jl 348 0 3
