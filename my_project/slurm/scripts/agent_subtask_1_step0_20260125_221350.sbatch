#!/bin/bash
#SBATCH -J agent_subtask_1_step0
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o /work/sdz852/WORKING/AGI/my_project/slurm/logs/agent_subtask_1_step0_20260125_221350_%j.out
#SBATCH -e /work/sdz852/WORKING/AGI/my_project/slurm/logs/agent_subtask_1_step0_20260125_221350_%j.err

# Cluster: zeus
# Partition: normal
# Cores: 4, Memory: 16G, Time: 04:00:00

# Environment setup
set -e  # Exit on error
cd /work/sdz852/WORKING/AGI/my_project

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate agi_project_20260125

# Main script
echo 'Could not parse plan, listing files' && ls -la

# Mark completion
echo ""
echo "=== Job Completed ==="
echo "End time: $(date)"
echo "Exit code: $?"
touch /work/sdz852/WORKING/AGI/my_project/slurm/logs/agent_subtask_1_step0_20260125_221350_$SLURM_JOB_ID.complete