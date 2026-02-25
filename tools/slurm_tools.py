"""
SLURM job submission and management tools v1.2.7.
Supports multiple clusters with different configurations (CPU and GPU).

Features:
- Multiple cluster configurations (zeus, gpu_cluster)
- GPU job submission (V100, A100, DGX)
- Specific node selection (--nodelist)
- Job submission via sbatch
- Job monitoring via squeue/sacct
- Parallel job execution
- Job dependencies
- Output collection
"""

import subprocess
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import yaml

from tools.sandbox import Sandbox


class ClusterConfig:
    """Configuration for a specific SLURM cluster"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize cluster configuration from dictionary.
        
        Args:
            config_dict: Cluster configuration dictionary from config.yaml
        """
        self.name = config_dict.get("name", "default")
        self.description = config_dict.get("description", "")
        
        # Node specifications
        self.nodes_total = config_dict.get("nodes_total", 10)
        self.cores_per_node = config_dict.get("cores_per_node", 80)
        self.memory_per_node = config_dict.get("memory_per_node", "256G")
        self.has_gpu = config_dict.get("has_gpu", False)
        
        # Defaults
        self.default_partition = config_dict.get("default_partition", "normal")
        self.default_cpus = config_dict.get("default_cpus", 4)
        self.default_memory = config_dict.get("default_memory", "16G")
        self.default_time = config_dict.get("default_time", "04:00:00")
        self.default_gpus = config_dict.get("default_gpus", 0)
        
        # Partitions
        self.partitions = config_dict.get("partitions", {})
        
        # Extra SBATCH directives
        self.sbatch_extras = config_dict.get("sbatch_extras", [])
    
    def get_partition_config(self, partition_name: str) -> Dict[str, Any]:
        """Get configuration for a specific partition"""
        return self.partitions.get(partition_name, {})
    
    def list_partitions(self) -> List[str]:
        """List all available partitions"""
        return list(self.partitions.keys())
    
    def list_gpu_partitions(self) -> List[str]:
        """List partitions that have GPUs"""
        return [name for name, config in self.partitions.items() 
                if config.get("has_gpu", False)]
    
    def list_cpu_partitions(self) -> List[str]:
        """List partitions that are CPU-only"""
        return [name for name, config in self.partitions.items() 
                if not config.get("has_gpu", False)]
    
    def get_max_gpus(self, partition: str) -> int:
        """Get maximum GPUs for a partition"""
        partition_config = self.get_partition_config(partition)
        return partition_config.get("max_gpus", 0)
    
    def get_gpu_type(self, partition: str) -> Optional[str]:
        """Get GPU type for a partition"""
        partition_config = self.get_partition_config(partition)
        return partition_config.get("gpu_type")
    
    def get_nodes_for_partition(self, partition: str) -> str:
        """Get node list string for a partition"""
        partition_config = self.get_partition_config(partition)
        return partition_config.get("nodes", "")


class SlurmConfig:
    """Global SLURM configuration with multi-cluster support"""
    
    def __init__(self, config_path: str = None, config_dict: Dict = None):
        """
        Initialize SLURM configuration.
        
        Args:
            config_path: Path to config.yaml file
            config_dict: Configuration dictionary (alternative to file)
        """
        self.clusters: Dict[str, ClusterConfig] = {}
        self.default_cluster = "zeus"
        self.poll_interval = 10
        self.max_poll_attempts = 720
        self.job_prefix = "agi"
        
        # Load configuration
        if config_dict:
            self._load_from_dict(config_dict)
        elif config_path:
            self._load_from_file(config_path)
        else:
            # Try default locations
            for path in ["config/config.yaml", "../config/config.yaml", "config.yaml"]:
                if Path(path).exists():
                    self._load_from_file(path)
                    break
            else:
                # No config found, use defaults
                self._set_defaults()
    
    def _load_from_file(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self._load_from_dict(config)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            self._set_defaults()
    
    def _load_from_dict(self, config: Dict):
        """Load configuration from dictionary"""
        # SLURM settings
        slurm_config = config.get("slurm", {})
        self.default_cluster = slurm_config.get("default_cluster", "zeus")
        self.poll_interval = slurm_config.get("poll_interval", 10)
        self.max_poll_attempts = slurm_config.get("max_poll_attempts", 720)
        self.job_prefix = slurm_config.get("job_prefix", "agi")
        
        # Load cluster configurations
        clusters_config = config.get("clusters", {})
        for cluster_name, cluster_dict in clusters_config.items():
            self.clusters[cluster_name] = ClusterConfig(cluster_dict)
        
        # If no clusters defined, set defaults
        if not self.clusters:
            self._set_defaults()
    
    def _set_defaults(self):
        """Set default cluster configuration (zeus)"""
        self.clusters["zeus"] = ClusterConfig({
            "name": "zeus",
            "cores_per_node": 192,
            "memory_per_node": "1000G",
            "default_partition": "normal",
            "default_cpus": 4,
            "default_memory": "16G",
            "default_time": "04:00:00",
            "has_gpu": False,
            "partitions": {
                "normal": {
                    "max_time": "7-00:00:00",
                    "max_cpus": 192,
                    "max_memory": "1000G",
                    "has_gpu": False
                }
            }
        })
    
    def get_cluster(self, cluster_name: str = None) -> ClusterConfig:
        """Get cluster configuration by name"""
        name = cluster_name or self.default_cluster
        if name not in self.clusters:
            print(f"Warning: Cluster '{name}' not found, using default '{self.default_cluster}'")
            name = self.default_cluster
        return self.clusters.get(name, list(self.clusters.values())[0])
    
    def list_clusters(self) -> List[str]:
        """List all configured clusters"""
        return list(self.clusters.keys())


class SlurmJob:
    """Represents a SLURM job"""
    
    def __init__(
        self,
        job_id: str,
        name: str,
        script_path: str,
        cluster: str = None,
        partition: str = None,
        gpus: int = 0,
        nodes: str = None,
        status: str = "PENDING"
    ):
        self.job_id = job_id
        self.name = name
        self.script_path = script_path
        self.cluster = cluster
        self.partition = partition
        self.gpus = gpus
        self.nodes = nodes
        self.status = status
        self.submit_time = datetime.now()
        self.start_time = None
        self.end_time = None
        self.exit_code = None
        self.output_file = None
        self.error_file = None
    
    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "script_path": self.script_path,
            "cluster": self.cluster,
            "partition": self.partition,
            "gpus": self.gpus,
            "nodes": self.nodes,
            "status": self.status,
            "submit_time": self.submit_time.isoformat(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "exit_code": self.exit_code,
            "output_file": self.output_file,
            "error_file": self.error_file
        }


class SlurmTools:
    """
    SLURM job management for multi-agent system.
    Supports multiple clusters, GPU jobs, and node specification.
    """
    
    def __init__(
        self, 
        sandbox: Sandbox, 
        config: SlurmConfig = None,
        config_path: str = None,
        cluster_name: str = None
    ):
        """
        Initialize SLURM tools.
        
        Args:
            sandbox: Sandbox instance for file operations
            config: SlurmConfig instance (loads from file if not provided)
            config_path: Path to config.yaml (used if config not provided)
            cluster_name: Specific cluster to use (uses default if not provided)
        """
        self.sandbox = sandbox
        
        # Load configuration
        if config:
            self.config = config
        else:
            self.config = SlurmConfig(config_path=config_path)
        
        # Set active cluster
        self.cluster_name = cluster_name or self.config.default_cluster
        self.cluster = self.config.get_cluster(self.cluster_name)
        
        # Job tracking
        self.active_jobs: Dict[str, SlurmJob] = {}
        self.completed_jobs: List[SlurmJob] = []
        
        # Directories
        self.slurm_dir = sandbox.project_dir / "slurm"
        self.slurm_scripts_dir = self.slurm_dir / "scripts"
        self.slurm_logs_dir = self.slurm_dir / "logs"
        
        # Create directories
        self.slurm_scripts_dir.mkdir(parents=True, exist_ok=True)
        self.slurm_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Check SLURM availability
        self.slurm_available = self._check_slurm()
    
    def set_cluster(self, cluster_name: str):
        """
        Switch to a different cluster configuration.
        
        Args:
            cluster_name: Name of cluster (e.g., 'zeus', 'gpu_cluster')
        """
        if cluster_name not in self.config.clusters:
            raise ValueError(f"Unknown cluster: {cluster_name}. Available: {self.config.list_clusters()}")
        self.cluster_name = cluster_name
        self.cluster = self.config.get_cluster(cluster_name)
    
    def _check_slurm(self) -> bool:
        """Check if SLURM commands are available"""
        try:
            result = subprocess.run(
                ["sinfo", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_cluster_status(self, partition: str = None) -> Dict[str, Any]:
        """
        Get current cluster status via sinfo.
        
        Args:
            partition: Optional specific partition to query
            
        Returns:
            Dict with cluster status information
        """
        if not self.slurm_available:
            return {"success": False, "error": "SLURM not available"}
        
        try:
            cmd = ["sinfo", "-N", "-l"]
            if partition:
                cmd.extend(["-p", partition])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return {"success": False, "error": result.stderr}
            
            # Parse sinfo output
            lines = result.stdout.strip().split('\n')
            nodes = []
            idle_nodes = []
            gpu_nodes = []
            
            for line in lines[2:]:  # Skip header lines
                parts = line.split()
                if len(parts) >= 4:
                    node_name = parts[0]
                    node_partition = parts[2] if len(parts) > 2 else ""
                    state = parts[3] if len(parts) > 3 else "unknown"
                    
                    node_info = {
                        "name": node_name,
                        "partition": node_partition,
                        "state": state
                    }
                    nodes.append(node_info)
                    
                    if "idle" in state.lower():
                        idle_nodes.append(node_name)
                    
                    # Check if GPU node
                    if "gpu" in node_name.lower() or "dgx" in node_name.lower():
                        gpu_nodes.append(node_info)
            
            return {
                "success": True,
                "cluster": self.cluster_name,
                "partition": partition,
                "total_nodes": len(nodes),
                "idle_nodes": idle_nodes,
                "idle_count": len(idle_nodes),
                "gpu_nodes": gpu_nodes,
                "gpu_count": len(gpu_nodes),
                "nodes": nodes,
                "raw_output": result.stdout
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "sinfo timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_partition_info(self, partition: str = None) -> Dict[str, Any]:
        """
        Get detailed info about a partition including config and live status.
        
        Args:
            partition: Partition name (uses default if not specified)
            
        Returns:
            Dict with partition configuration and status
        """
        partition = partition or self.cluster.default_partition
        
        # Get config info
        partition_config = self.cluster.get_partition_config(partition)
        
        # Get live status
        live_status = self.get_cluster_status(partition=partition)
        
        return {
            "success": True,
            "partition": partition,
            "cluster": self.cluster_name,
            "config": {
                "max_time": partition_config.get("max_time"),
                "max_cpus": partition_config.get("max_cpus", self.cluster.cores_per_node),
                "max_memory": partition_config.get("max_memory"),
                "has_gpu": partition_config.get("has_gpu", False),
                "gpu_type": partition_config.get("gpu_type"),
                "max_gpus": partition_config.get("max_gpus", 0),
                "vram_per_gpu": partition_config.get("vram_per_gpu"),
                "nodes": partition_config.get("nodes"),
                "description": partition_config.get("description", "")
            },
            "status": {
                "total_nodes": live_status.get("total_nodes", 0),
                "idle_count": live_status.get("idle_count", 0),
                "idle_nodes": live_status.get("idle_nodes", [])
            } if live_status.get("success") else None
        }
    
    def list_partitions(self, gpu_only: bool = False, cpu_only: bool = False) -> List[Dict[str, Any]]:
        """
        List available partitions with their properties.
        
        Args:
            gpu_only: Only show GPU partitions
            cpu_only: Only show CPU-only partitions
            
        Returns:
            List of partition info dictionaries
        """
        partitions = []
        for name, config in self.cluster.partitions.items():
            has_gpu = config.get("has_gpu", False)
            
            # Filter based on GPU requirement
            if gpu_only and not has_gpu:
                continue
            if cpu_only and has_gpu:
                continue
            
            partitions.append({
                "name": name,
                "has_gpu": has_gpu,
                "gpu_type": config.get("gpu_type"),
                "max_gpus": config.get("max_gpus", 0),
                "vram_per_gpu": config.get("vram_per_gpu"),
                "max_cpus": config.get("max_cpus", self.cluster.cores_per_node),
                "max_memory": config.get("max_memory"),
                "max_time": config.get("max_time"),
                "nodes": config.get("nodes"),
                "description": config.get("description", "")
            })
        return partitions
    
    def generate_sbatch_script(
        self,
        script_content: str,
        job_name: str,
        language: str = "bash",
        cpus: int = None,
        memory: str = None,
        time_limit: str = None,
        partition: str = None,
        gpus: int = None,
        gpu_type: str = None,
        nodelist: str = None,
        exclude_nodes: str = None,
        conda_env: str = None,
        dependencies: List[str] = None,
        array: str = None,
        extra_directives: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an sbatch submission script.
        
        Args:
            script_content: The actual script code to run
            job_name: Name for the job
            language: Script language (bash, python, r, perl)
            cpus: Number of CPUs (default: from cluster config)
            memory: Memory allocation (default: from cluster config)
            time_limit: Time limit (default: from cluster config)
            partition: SLURM partition (default: from cluster config)
            gpus: Number of GPUs to request (0 for CPU-only)
            gpu_type: GPU type (v100, a100, etc.) - auto-detected from partition if not specified
            nodelist: Specific node(s) to run on (e.g., "gpu001" or "gpu[001-003]")
            exclude_nodes: Node(s) to exclude (e.g., "gpu002")
            conda_env: Conda environment to activate
            dependencies: List of job IDs this job depends on
            array: Job array specification (e.g., "1-10", "1-100%5")
            extra_directives: Additional SBATCH directives
            
        Returns:
            Dict with script path and content
        """
        # Use cluster defaults
        cpus = cpus if cpus is not None else self.cluster.default_cpus
        memory = memory or self.cluster.default_memory
        time_limit = time_limit or self.cluster.default_time
        partition = partition or self.cluster.default_partition
        gpus = gpus if gpus is not None else self.cluster.default_gpus
        
        # Get partition config for GPU info and validation
        partition_config = self.cluster.get_partition_config(partition)
        partition_has_gpu = partition_config.get("has_gpu", False)
        
        # Auto-detect GPU type from partition if not specified
        if gpus and gpus > 0 and partition_has_gpu and not gpu_type:
            gpu_type = partition_config.get("gpu_type")
        
        # Validate GPU request
        if gpus and gpus > 0 and not partition_has_gpu:
            return {
                "success": False,
                "error": f"Partition '{partition}' does not have GPUs. Use one of: {self.cluster.list_gpu_partitions()}"
            }
        
        max_gpus = partition_config.get("max_gpus", 0)
        if gpus and gpus > max_gpus:
            return {
                "success": False,
                "error": f"Requested {gpus} GPUs but partition '{partition}' only has {max_gpus} per node"
            }
        
        # Sanitize job name
        safe_name = re.sub(r'[^\w\-]', '_', job_name)[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Output and error files
        output_file = self.slurm_logs_dir / f"{safe_name}_{timestamp}_%j.out"
        error_file = self.slurm_logs_dir / f"{safe_name}_{timestamp}_%j.err"
        
        # Build SBATCH header
        # -N 1: Single node (required for serial/threaded jobs)
        # -n 1: Single task (not using MPI)
        # -c X: Number of cores available for threading
        sbatch_lines = [
            "#!/bin/bash",
            f"#SBATCH -J {safe_name}",           # Job name
            f"#SBATCH -p {partition}",            # Partition (queue) name
            f"#SBATCH -N 1",                      # Total # of nodes (must be 1 for serial/threaded)
            f"#SBATCH -n 1",                      # Total # of MPI tasks (should be 1 for serial/threaded)
            f"#SBATCH -c {cpus}",                 # Total # of cores for threading
            f"#SBATCH --mem={memory}",            # Memory per node
            f"#SBATCH -t {time_limit}",           # Time limit
            f"#SBATCH -o {output_file}",          # Standard output file
            f"#SBATCH -e {error_file}",           # Standard error file
        ]
        
        # Add GPU request if needed
        if gpus and gpus > 0 and partition_has_gpu:
            if gpu_type:
                sbatch_lines.append(f"#SBATCH --gres=gpu:{gpu_type}:{gpus}")
            else:
                sbatch_lines.append(f"#SBATCH --gres=gpu:{gpus}")
        
        # Add node specification if provided
        if nodelist:
            sbatch_lines.append(f"#SBATCH -w {nodelist}")         # Specific nodes to use
        
        if exclude_nodes:
            sbatch_lines.append(f"#SBATCH -x {exclude_nodes}")    # Nodes to exclude
        
        # Add job array if specified
        if array:
            sbatch_lines.append(f"#SBATCH --array={array}")
        
        # Add dependencies if specified
        if dependencies:
            dep_str = ":".join(str(d) for d in dependencies)
            sbatch_lines.append(f"#SBATCH --dependency=afterok:{dep_str}")
        
        # Add cluster-specific extra directives
        for directive in self.cluster.sbatch_extras:
            sbatch_lines.append(f"#SBATCH {directive}")
        
        # Add user-specified extra directives
        if extra_directives:
            for directive in extra_directives:
                if not directive.startswith("#SBATCH"):
                    directive = f"#SBATCH {directive}"
                sbatch_lines.append(directive)
        
        # Add blank line after directives
        sbatch_lines.append("")
        
        # Add cluster/partition info as comments
        sbatch_lines.extend([
            f"# Cluster: {self.cluster_name}",
            f"# Partition: {partition}",
            f"# Cores: {cpus}, Memory: {memory}, Time: {time_limit}",
        ])
        if gpus and gpus > 0:
            sbatch_lines.append(f"# GPUs: {gpus} x {gpu_type or 'default'}")
        if nodelist:
            sbatch_lines.append(f"# Node(s): {nodelist}")
        sbatch_lines.append("")
        
        # Add environment setup
        sbatch_lines.extend([
            "# Environment setup",
            "set -e  # Exit on error",
            f"cd {self.sandbox.project_dir}",
            "",
        ])
        
        # Add GPU/CUDA setup for GPU jobs
        if gpus and gpus > 0 and partition_has_gpu:
            sbatch_lines.extend([
                "# GPU environment setup",
                "echo \"=== GPU Information ===\"",
                "echo \"SLURM Job ID: $SLURM_JOB_ID\"",
                "echo \"Running on node: $(hostname)\"",
                "echo \"CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES\"",
                "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true",
                "echo \"=========================\"",
                "",
            ])
        
        # Add conda activation if specified
        if conda_env:
            sbatch_lines.extend([
                "# Activate conda environment",
                "source $(conda info --base)/etc/profile.d/conda.sh",
                f"conda activate {conda_env}",
                "",
            ])
        
        # Add the actual script content
        sbatch_lines.append("# Main script")
        
        if language == "python":
            py_script_path = self.sandbox.get_scripts_dir() / f"{safe_name}_{timestamp}.py"
            py_script_path.write_text(script_content)
            sbatch_lines.append(f"python {py_script_path}")
        elif language == "r":
            r_script_path = self.sandbox.get_scripts_dir() / f"{safe_name}_{timestamp}.R"
            r_script_path.write_text(script_content)
            sbatch_lines.append(f"Rscript {r_script_path}")
        elif language == "perl":
            pl_script_path = self.sandbox.get_scripts_dir() / f"{safe_name}_{timestamp}.pl"
            pl_script_path.write_text(script_content)
            sbatch_lines.append(f"perl {pl_script_path}")
        else:
            # Bash - inline the content
            sbatch_lines.append(script_content)
        
        # Add completion marker
        sbatch_lines.extend([
            "",
            "# Mark completion",
            "echo \"\"",
            "echo \"=== Job Completed ===\"",
            "echo \"End time: $(date)\"",
            "echo \"Exit code: $?\"",
            f"touch {self.slurm_logs_dir}/{safe_name}_{timestamp}_$SLURM_JOB_ID.complete",
        ])
        
        # Write sbatch script
        sbatch_content = "\n".join(sbatch_lines)
        sbatch_path = self.slurm_scripts_dir / f"{safe_name}_{timestamp}.sbatch"
        sbatch_path.write_text(sbatch_content)
        
        return {
            "success": True,
            "script_path": str(sbatch_path),
            "script_content": sbatch_content,
            "output_file": str(output_file),
            "error_file": str(error_file),
            "job_name": safe_name,
            "cluster": self.cluster_name,
            "partition": partition,
            "cpus": cpus,
            "memory": memory,
            "gpus": gpus if partition_has_gpu else 0,
            "gpu_type": gpu_type,
            "nodelist": nodelist
        }
    
    def submit_job(
        self,
        sbatch_script: str,
        job_name: str = None
    ) -> Dict[str, Any]:
        """
        Submit a job via sbatch.
        
        Args:
            sbatch_script: Path to sbatch script
            job_name: Optional job name for tracking
            
        Returns:
            Dict with job_id and submission status
        """
        if not self.slurm_available:
            return {"success": False, "error": "SLURM not available"}
        
        # Validate script exists
        script_path = Path(sbatch_script)
        if not script_path.exists():
            return {"success": False, "error": f"Script not found: {sbatch_script}"}
        
        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.sandbox.project_dir)
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr,
                    "stdout": result.stdout
                }
            
            # Parse job ID from output (format: "Submitted batch job 12345")
            match = re.search(r'Submitted batch job (\d+)', result.stdout)
            if not match:
                return {
                    "success": False,
                    "error": f"Could not parse job ID from: {result.stdout}"
                }
            
            job_id = match.group(1)
            
            # Track the job
            job = SlurmJob(
                job_id=job_id,
                name=job_name or script_path.stem,
                script_path=str(script_path),
                cluster=self.cluster_name
            )
            self.active_jobs[job_id] = job
            
            return {
                "success": True,
                "job_id": job_id,
                "cluster": self.cluster_name,
                "message": f"Job {job_id} submitted to {self.cluster_name}"
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "sbatch timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def submit_script(
        self,
        script_content: str,
        job_name: str,
        language: str = "bash",
        conda_env: str = None,
        cpus: int = None,
        memory: str = None,
        time_limit: str = None,
        partition: str = None,
        gpus: int = None,
        gpu_type: str = None,
        nodelist: str = None,
        exclude_nodes: str = None,
        dependencies: List[str] = None
    ) -> Dict[str, Any]:
        """
        Convenience method to generate and submit in one call.
        
        Args:
            script_content: Script code to run
            job_name: Name for the job
            language: Script language
            conda_env: Conda environment to use
            cpus: CPU count
            memory: Memory allocation
            time_limit: Time limit
            partition: SLURM partition
            gpus: Number of GPUs
            gpu_type: GPU type (v100, a100, etc.)
            nodelist: Specific node(s) to run on
            exclude_nodes: Node(s) to exclude
            dependencies: Job IDs this depends on
            
        Returns:
            Dict with job_id and submission details
        """
        # Generate sbatch script
        gen_result = self.generate_sbatch_script(
            script_content=script_content,
            job_name=job_name,
            language=language,
            conda_env=conda_env,
            cpus=cpus,
            memory=memory,
            time_limit=time_limit,
            partition=partition,
            gpus=gpus,
            gpu_type=gpu_type,
            nodelist=nodelist,
            exclude_nodes=exclude_nodes,
            dependencies=dependencies
        )
        
        if not gen_result["success"]:
            return gen_result
        
        # Submit the job
        submit_result = self.submit_job(
            sbatch_script=gen_result["script_path"],
            job_name=job_name
        )
        
        # Combine results
        if submit_result["success"]:
            return {
                **submit_result,
                "script_path": gen_result["script_path"],
                "output_file": gen_result["output_file"],
                "error_file": gen_result["error_file"],
                "partition": gen_result["partition"],
                "cpus": gen_result["cpus"],
                "memory": gen_result["memory"],
                "gpus": gen_result["gpus"],
                "gpu_type": gen_result["gpu_type"],
                "nodelist": gen_result["nodelist"]
            }
        else:
            return submit_result
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of a specific job.
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            Dict with job status information
        """
        if not self.slurm_available:
            return {"success": False, "error": "SLURM not available"}
        
        try:
            # First try squeue for running/pending jobs
            result = subprocess.run(
                ["squeue", "-j", str(job_id), "--format=%i|%j|%T|%M|%l|%N", "--noheader"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split('|')
                status = parts[2] if len(parts) > 2 else "UNKNOWN"
                node = parts[5] if len(parts) > 5 else None
                
                return {
                    "success": True,
                    "job_id": job_id,
                    "status": status,
                    "node": node,
                    "running": status == "RUNNING",
                    "pending": status == "PENDING",
                    "completed": False
                }
            
            # Job not in queue - check sacct for completed jobs
            result = subprocess.run(
                ["sacct", "-j", str(job_id), "--format=JobID,State,ExitCode,Start,End,NodeList", "--noheader", "--parsable2"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                # Get the main job line (not .batch or .extern)
                for line in lines:
                    parts = line.split('|')
                    if len(parts) >= 3 and '.' not in parts[0]:
                        status = parts[1]
                        exit_code = parts[2].split(':')[0] if ':' in parts[2] else parts[2]
                        node = parts[5] if len(parts) > 5 else None
                        
                        return {
                            "success": True,
                            "job_id": job_id,
                            "status": status,
                            "exit_code": exit_code,
                            "node": node,
                            "running": False,
                            "pending": False,
                            "completed": status in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"]
                        }
            
            return {
                "success": True,
                "job_id": job_id,
                "status": "UNKNOWN",
                "completed": True  # Assume completed if not found
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Status check timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def wait_for_job(
        self,
        job_id: str,
        poll_interval: int = None,
        callback: callable = None
    ) -> Dict[str, Any]:
        """
        Wait for a job to complete.
        
        v1.2.7: No artificial timeout. Polls indefinitely until SLURM
        reports a terminal state. The job's own wall time is the only
        exit condition.
        
        Args:
            job_id: Job ID to wait for
            poll_interval: Seconds between status checks
            callback: Optional callback function called each poll
            
        Returns:
            Final job status
        """
        poll_interval = poll_interval or self.config.poll_interval
        attempt = 0
        
        while True:
            status = self.get_job_status(job_id)
            
            if not status["success"]:
                return status
            
            if callback:
                callback(status, attempt)
            
            if status.get("completed"):
                # Update tracked job
                if job_id in self.active_jobs:
                    job = self.active_jobs[job_id]
                    job.status = status.get("status", "COMPLETED")
                    job.exit_code = status.get("exit_code")
                    job.end_time = datetime.now()
                    self.completed_jobs.append(job)
                    del self.active_jobs[job_id]
                
                return status
            
            time.sleep(poll_interval)
            attempt += 1
    
    def wait_for_jobs(
        self,
        job_ids: List[str],
        poll_interval: int = None
    ) -> Dict[str, Any]:
        """
        Wait for multiple jobs to complete.
        
        v1.2.7: No artificial timeout. Polls indefinitely until all
        jobs reach a terminal state. Each job's own wall time is the
        only exit condition.
        
        Args:
            job_ids: List of job IDs to wait for
            poll_interval: Seconds between status checks
            
        Returns:
            Dict with status of all jobs
        """
        poll_interval = poll_interval or self.config.poll_interval
        
        remaining = set(str(j) for j in job_ids)
        results = {}
        
        while remaining:
            newly_completed = []
            
            for job_id in remaining:
                status = self.get_job_status(job_id)
                
                if status.get("completed"):
                    results[job_id] = status
                    newly_completed.append(job_id)
            
            for job_id in newly_completed:
                remaining.remove(job_id)
            
            if not remaining:
                return {
                    "success": True,
                    "all_completed": True,
                    "jobs": results
                }
            
            time.sleep(poll_interval)
    
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a running or pending job"""
        if not self.slurm_available:
            return {"success": False, "error": "SLURM not available"}
        
        try:
            result = subprocess.run(
                ["scancel", str(job_id)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                if job_id in self.active_jobs:
                    job = self.active_jobs[job_id]
                    job.status = "CANCELLED"
                    job.end_time = datetime.now()
                    self.completed_jobs.append(job)
                    del self.active_jobs[job_id]
                
                return {"success": True, "job_id": job_id, "message": "Job cancelled"}
            else:
                return {"success": False, "error": result.stderr}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def cancel_all_jobs(self) -> Dict[str, Any]:
        """Cancel all active jobs for this project"""
        results = []
        for job_id in list(self.active_jobs.keys()):
            result = self.cancel_job(job_id)
            results.append(result)
        
        return {
            "success": all(r["success"] for r in results) if results else True,
            "cancelled": len([r for r in results if r.get("success")]),
            "failed": len([r for r in results if not r.get("success")]),
            "results": results
        }
    
    def get_job_output(self, job_id: str) -> Dict[str, Any]:
        """
        Get output and error content from a completed job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Dict with stdout, stderr content
        """
        # Find output files (try multiple patterns)
        output_files = list(self.slurm_logs_dir.glob(f"*_{job_id}.out"))
        if not output_files:
            output_files = list(self.slurm_logs_dir.glob(f"*{job_id}*.out"))
        
        error_files = list(self.slurm_logs_dir.glob(f"*_{job_id}.err"))
        if not error_files:
            error_files = list(self.slurm_logs_dir.glob(f"*{job_id}*.err"))
        
        result = {
            "success": True,
            "job_id": job_id,
            "stdout": "",
            "stderr": "",
            "output_file": None,
            "error_file": None
        }
        
        if output_files:
            result["output_file"] = str(output_files[0])
            try:
                result["stdout"] = output_files[0].read_text()
            except Exception as e:
                result["stdout"] = f"Error reading output: {e}"
        
        if error_files:
            result["error_file"] = str(error_files[0])
            try:
                result["stderr"] = error_files[0].read_text()
            except Exception as e:
                result["stderr"] = f"Error reading errors: {e}"
        
        return result
    
    def get_active_jobs(self) -> List[Dict]:
        """Get list of all active jobs"""
        return [job.to_dict() for job in self.active_jobs.values()]
    
    def get_completed_jobs(self) -> List[Dict]:
        """Get list of all completed jobs"""
        return [job.to_dict() for job in self.completed_jobs]
    
    def estimate_resources(
        self,
        tasks: List[Dict],
        cpus_per_task: int = None,
        memory_per_task: str = None,
        gpus_per_task: int = 0,
        partition: str = None
    ) -> Dict[str, Any]:
        """
        Estimate resources needed for a list of tasks.
        
        Args:
            tasks: List of task dictionaries
            cpus_per_task: CPUs per task
            memory_per_task: Memory per task
            gpus_per_task: GPUs per task
            partition: Target partition
            
        Returns:
            Resource estimation
        """
        cpus_per_task = cpus_per_task or self.cluster.default_cpus
        partition = partition or self.cluster.default_partition
        
        # Parse memory
        mem_gb = 16
        if memory_per_task:
            match = re.match(r'(\d+)([GM])', memory_per_task.upper())
            if match:
                value = int(match.group(1))
                unit = match.group(2)
                mem_gb = value if unit == 'G' else value / 1024
        
        task_count = len(tasks)
        total_cpus = task_count * cpus_per_task
        total_memory_gb = task_count * mem_gb
        total_gpus = task_count * gpus_per_task
        
        # Calculate nodes needed
        nodes_by_cpu = (total_cpus + self.cluster.cores_per_node - 1) // self.cluster.cores_per_node
        
        # Get current availability
        status = self.get_cluster_status(partition=partition)
        idle_count = status.get("idle_count", 0) if status["success"] else 0
        
        # GPU partition recommendation
        gpu_recommendation = None
        if gpus_per_task > 0:
            gpu_partitions = self.list_partitions(gpu_only=True)
            if gpu_partitions:
                gpu_recommendation = {
                    "available_partitions": [p["name"] for p in gpu_partitions],
                    "total_gpus_needed": total_gpus,
                    "suggested": gpu_partitions[0]["name"] if gpu_partitions else None
                }
        
        return {
            "cluster": self.cluster_name,
            "partition": partition,
            "task_count": task_count,
            "cpus_per_task": cpus_per_task,
            "memory_per_task": f"{mem_gb}G",
            "gpus_per_task": gpus_per_task,
            "total_cpus_needed": total_cpus,
            "total_memory_gb": total_memory_gb,
            "total_gpus_needed": total_gpus,
            "nodes_needed": nodes_by_cpu,
            "idle_nodes_available": idle_count,
            "can_run_parallel": idle_count >= nodes_by_cpu,
            "gpu_recommendation": gpu_recommendation
        }
    
    def print_cluster_summary(self) -> str:
        """Print a summary of the current cluster configuration"""
        lines = [
            f"=== Cluster: {self.cluster_name} ===",
            f"Description: {self.cluster.description}",
            f"Cores/Node: {self.cluster.cores_per_node}",
            f"Memory/Node: {self.cluster.memory_per_node}",
            f"Has GPU: {self.cluster.has_gpu}",
            f"Default Partition: {self.cluster.default_partition}",
            "",
            "Available Partitions:"
        ]
        
        for p in self.list_partitions():
            gpu_info = f" (GPU: {p['max_gpus']}x {p['gpu_type']})" if p['has_gpu'] else ""
            lines.append(f"  - {p['name']}: {p['description']}{gpu_info}")
        
        summary = "\n".join(lines)
        print(summary)
        return summary
