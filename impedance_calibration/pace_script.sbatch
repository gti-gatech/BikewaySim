#!/bin/bash
#SBATCH -testCalibrate
#SBATCH --account=gts-rg49
#SBATCH –N2 --ntasks-per-node=2
#SBATCH --mem-per-cpu=6G
#SBATCH -t1:00:00
#SBATCH -qembers
#SBATCH -oReport-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL #SBATCH --mail-user=tpassmore6@gatech.edu

cd $SLURM_SUBMIT_DIR
module load anaconda3
conda activate bikewaysim
python step_2_run_calibration.py