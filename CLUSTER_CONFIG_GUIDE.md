# AGI Pipeline v3.2 - Cluster Configuration Guide

## Overview

The cluster configuration flows from `RUN_AGI_PIPELINE.sh` through `main.py` to the sub-agent:

```
RUN_AGI_PIPELINE.sh
    │
    ├── Sets: AGI_CLUSTER=zeus_cpu (or gpu_v100, etc.)
    ├── Sets: AGI_CLUSTER_CONFIG=/path/to/cluster_config.yaml
    │
    └── Calls: python main.py --cluster zeus_cpu
                    │
                    └── Passes to workflow → sub-agent
                                               │
                                               └── Reads AGI_CLUSTER env var
                                                   Loads cluster_config.yaml
                                                   Generates appropriate sbatch
```

## File Locations

Place these files in your AGI repository:

```
AGI/
├── config/
│   └── cluster_config.yaml    # Cluster definitions
├── agents/
│   └── sub_agent_v3.py        # Updated sub-agent
└── main.py                    # Add --cluster argument
```

## Cluster Config Example

```yaml
# config/cluster_config.yaml
default_cluster: zeus_cpu

clusters:
  zeus_cpu:
    name: "Zeus CPU Cluster"
    slurm:
      partition: "normal"
      account: "jlehle"
      cpus_per_task: 40
      memory: "64G"
      time: "08:00:00"
    gpu:
      available: false
  
  gpu_v100:
    name: "GPU V100"
    slurm:
      partition: "gpu1v100"
      account: "sdz852"
      cpus_per_task: 10
      time: "12:00:00"
      # NOTE: No memory - this cluster doesn't use it
    gpu:
      available: true
      default_count: 1
      directive_format: "--gpus {count}"
```

## main.py Updates

Add the `--cluster` argument:

```python
# In main.py argument parser
parser.add_argument(
    '--cluster',
    type=str,
    default=os.environ.get('AGI_CLUSTER', 'zeus_cpu'),
    help='Cluster profile for subtask SLURM settings'
)

# In main(), pass to workflow:
workflow = MultiAgentWorkflow(
    ...,
    cluster=args.cluster,
    slurm_config={
        'cluster': args.cluster,
        'config_path': os.environ.get('AGI_CLUSTER_CONFIG'),
    }
)
```

## RUN_AGI_PIPELINE.sh Usage

### For CPU Cluster (default)

```bash
# Edit SBATCH headers for the MASTER job (not subtasks)
#SBATCH --partition=normal
#SBATCH --account=jlehle
#SBATCH --cpus-per-task=192
#SBATCH --mem=900GB

# Set cluster for SUBTASKS
AGI_CLUSTER="zeus_cpu"
```

### For GPU Cluster

Create a separate `RUN_AGI_PIPELINE_GPU.sh`:

```bash
#!/bin/bash
#SBATCH --partition=gpu1v100
#SBATCH --gpus=1
#SBATCH --account=sdz852
#SBATCH --time=3-00:00:00
# ... rest of GPU-appropriate settings

# Set cluster for SUBTASKS
AGI_CLUSTER="gpu_v100"
```

Or use `--export`:

```bash
sbatch --export=AGI_CLUSTER=gpu_v100 RUN_AGI_PIPELINE_GPU.sh
```

## How the Sub-Agent Uses Cluster Config

The sub-agent reads `AGI_CLUSTER` environment variable and loads settings:

```python
class ClusterConfig:
    def __init__(self):
        self.cluster_name = os.environ.get('AGI_CLUSTER', 'zeus_cpu')
        self.config = self._load_config()
        self.cluster = self.config['clusters'][self.cluster_name]
```

When generating sbatch:

```python
def _generate_sbatch_script(...):
    cfg = self.cluster_config
    
    # Get cluster-specific settings
    partition = cfg.get_slurm_value('partition')
    account = cfg.get_slurm_value('account')
    memory = cfg.get_slurm_value('memory')  # None for GPU clusters
    
    lines = [
        f"#SBATCH --partition={partition}",
    ]
    
    # Only add account if specified
    if account:
        lines.append(f"#SBATCH --account={account}")
    
    # Only add memory if specified (GPU clusters may not use it)
    if memory:
        lines.append(f"#SBATCH --mem={memory}")
    
    # Add GPU directive if needed
    if cfg.is_gpu_cluster():
        gpu_directive = cfg.get_gpu_directive(gpu_count)
        lines.append(f"#SBATCH {gpu_directive}")
```

## Generated Sbatch Comparison

### Zeus CPU Cluster

```bash
#!/bin/bash
#SBATCH --job-name=agi_subtask_1_dryrun
#SBATCH --partition=normal
#SBATCH --account=jlehle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/agi_subtask_1_dryrun_%j.out
#SBATCH --error=slurm/logs/agi_subtask_1_dryrun_%j.err

#======================================================================
# Cluster: zeus_cpu (Zeus CPU Cluster)
```

### GPU V100 Cluster

```bash
#!/bin/bash
#SBATCH --job-name=agi_subtask_1_dryrun
#SBATCH --partition=gpu1v100
#SBATCH --account=sdz852
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --output=slurm/logs/agi_subtask_1_dryrun_%j.out
#SBATCH --error=slurm/logs/agi_subtask_1_dryrun_%j.err

#======================================================================
# Cluster: gpu_v100 (GPU V100)
```

Note: No `--mem` line for GPU cluster (as specified in config).

## Environment Variable Overrides

You can override any cluster setting via environment:

```bash
# In RUN_AGI_PIPELINE.sh
export AGI_SUBTASK_PARTITION="special_queue"
export AGI_SUBTASK_CPUS="64"
export AGI_SUBTASK_TIME="48:00:00"
```

These override the cluster config for ALL subtasks.

## Quick Start

1. Copy files:
```bash
cp cluster_config.yaml $AGI_ROOT/config/
cp sub_agent_v3_cluster.py $AGI_ROOT/agents/sub_agent.py
```

2. Update main.py to accept `--cluster` argument

3. Update workflow to pass cluster config to sub-agent

4. Run:
```bash
# CPU cluster
sbatch RUN_AGI_PIPELINE.sh

# GPU cluster  
sbatch --export=AGI_CLUSTER=gpu_v100 RUN_AGI_PIPELINE_GPU.sh
```

5. Verify generated sbatch files match your cluster:
```bash
cat project/slurm/scripts/*.sbatch | head -20
```
