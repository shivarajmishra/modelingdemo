#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition snowy 
#SBATCH -A punim1497
#SBATCH --time 1:30:00 
#SBATCH --cpus-per-task=8
#SBATCH --job-name="TobNZ"
#SBATCH --mail-user=tim.wilson@unimelb.edu.au
#SBATCH --mail-type=END
#SBATCH --array=1-1

date '+%A %W %Y %X'

module load python/3.8.6
module load web_proxy
pip install -e .

#module load netlogo/6.2.0-64

BASE_FOLDER='/data/gpfs/projects/punim1497/pmslt/TobaccoPMSLT'
cd $BASE_FOLDER

date '+%A %W %Y %X'

run_uncertainty_analysis model_specs/denicotine.yaml -d 0


date '+%A %W %Y %X'
