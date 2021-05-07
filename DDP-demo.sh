#!/bin/bash
#SBATCH -A bigbayes
#SBATCH --time=17:00:00
#SBATCH --mail-user=muhammad.taufiq@stats.ox.ac.uk
#SBATCH --partition=ziz-medium
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=3
#SBATCH --mem=5G
#SBATCH --output="/tmp/ddp-std-output"
#SBATCH --error="/tmp/ddp-err-output"
echo Starting on `hostname`

ip1=`hostname -I | awk '{print $2}'`
echo $ip1

# Store the master node’s IP address in the MASTER_ADDR environment variable.
export MASTER_ADDR=$(hostname)

echo "r$SLURM_NODEID master: $MASTER_ADDR"

echo "r$SLURM_NODEID Launching python script"

source ./.venv/bin/activate
srun python /data/ziz/taufiq/DDP-demo/example_ddp.py --nodes=3 --ip_address $(hostname)