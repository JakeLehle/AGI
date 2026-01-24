"""
SLURM job submission and management tools.
Designed for zeus cluster with 10 nodes (192 cores, 1TB RAM each).

Supports:
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

from tools.sandbox import Sandbox


class SlurmConfig:
    """Default SLURM configuration for zeus cluster"""
    
    # Partition settings
    DEFAULT_PARTITION = "normal"
    INTERACTIVE_PARTITION = "interactive"
    
    # Resource defaults (conservative to allow parallel jobs)
    DEFAULT_CPUS = 4
    DEFAULT_MEMORY = "16G"  # Per job
    DEFAULT_TIME = "04:00:00"  # 4 hours
    MAX_TIME = "7-00:00:00"  # 7 days (normal partition limit)
    
    # Node info
    NODES_TOTAL = 10
    CORES_PER_NODE = 192
    MEMORY_PER_NODE = "1000G"  # ~1TB
    
    # Job settings
    MAX_CONCURRENT_JOBS = 20  # Limit concurrent submissions
    POLL_INTERVAL = 10  # Seconds between status checks
    MAX_POLL_ATTEMPTS = 360  # Max attempts (1 hour at 10s intervals)


class SlurmJob:
    """Represents a SLURM job"""
    
    def __init__(
        self,
        job_id: str,
        name: str,
        script_path: str,
        status: str = "PENDING"
    ):
        self.job_id = job_id
        self.name = name
        self.script_path = script_path
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
    Handles job submission, monitoring, and output collection.
    """
    
    def __init__(self, sandbox: Sandbox, config: SlurmConfig = None):
        """
        Initialize SLURM tools.
        
        Args:
            sandbox: Sandbox instance for file operations
            config: SLURM configuration (uses defaults if not provided)
        """
        self.sandbox = sandbox
        self.config = config or SlurmConfig()
        
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
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status via sinfo"""
        if not self.slurm_available:
            return {"success": False, "error": "SLURM not available"}
        
        try:
            result = subprocess.run(
                ["sinfo", "-N", "-l"],
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
            
            for line in lines[2:]:  # Skip header lines
                parts = line.split()
                if len(parts) >= 4:
                    node_name = parts[0]
                    state = parts[3] if len(parts) > 3 else "unknown"
                    
                    nodes.append({
                        "name": node_name,
                        "state": state
                    })
                    
                    if "idle" in state.lower():
                        idle_nodes.append(node_name)
            
            return {
                "success": True,
                "total_nodes": len(nodes),
                "idle_nodes": idle_nodes,
                "idle_count": len(idle_nodes),
                "nodes": nodes,
                "raw_output": result.stdout
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "sinfo timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def generate_sbatch_script(
        self,
        script_content: str,
        job_name: str,
        language: str = "bash",
        cpus: int = None,
        memory: str = None,
        time_limit: str = None,
        partition: str = None,
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
            cpus: Number of CPUs (default: 4)
            memory: Memory allocation (default: 16G)
            time_limit: Time limit (default: 4 hours)
            partition: SLURM partition (default: normal)
            conda_env: Conda environment to activate
            dependencies: List of job IDs this job depends on
            array: Job array specification (e.g., "1-10", "1-100%5")
            extra_directives: Additional SBATCH directives
            
        Returns:
            Dict with script path and content
        """
        # Use defaults
        cpus = cpus or self.config.DEFAULT_CPUS
        memory = memory or self.config.DEFAULT_MEMORY
        time_limit = time_limit or self.config.DEFAULT_TIME
        partition = partition or self.config.DEFAULT_PARTITION
        
        # Sanitize job name
        safe_name = re.sub(r'[^\w\-]', '_', job_name)[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Output and error files
        output_file = self.slurm_logs_dir / f"{safe_name}_{timestamp}_%j.out"
        error_file = self.slurm_logs_dir / f"{safe_name}_{timestamp}_%j.err"
        
        # Build SBATCH header
        # Note: -N 1 -n 1 -c X is required for proper multithreading on zeus cluster
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
        
        # Add job array if specified
        if array:
            sbatch_lines.append(f"#SBATCH --array={array}")
        
        # Add dependencies if specified
        if dependencies:
            dep_str = ":".join(dependencies)
            sbatch_lines.append(f"#SBATCH --dependency=afterok:{dep_str}")
        
        # Add extra directives
        if extra_directives:
            for directive in extra_directives:
                sbatch_lines.append(f"#SBATCH {directive}")
        
        # Add blank line after directives
        sbatch_lines.append("")
        
        # Add environment setup
        sbatch_lines.extend([
            "# Environment setup",
            "set -e  # Exit on error",
            f"cd {self.sandbox.project_dir}",
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
            # Write Python script to temp file and execute
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
            f"echo 'Job completed at' $(date) >> {self.slurm_logs_dir}/{safe_name}_{timestamp}.complete",
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
            "job_name": safe_name
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
                script_path=str(script_path)
            )
            self.active_jobs[job_id] = job
            
            return {
                "success": True,
                "job_id": job_id,
                "message": f"Job {job_id} submitted successfully"
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
                "error_file": gen_result["error_file"]
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
                ["squeue", "-j", job_id, "--format=%i|%j|%T|%M|%l|%S", "--noheader"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split('|')
                status = parts[2] if len(parts) > 2 else "UNKNOWN"
                
                return {
                    "success": True,
                    "job_id": job_id,
                    "status": status,
                    "running": status == "RUNNING",
                    "pending": status == "PENDING",
                    "completed": False
                }
            
            # Job not in queue - check sacct for completed jobs
            result = subprocess.run(
                ["sacct", "-j", job_id, "--format=JobID,State,ExitCode,Start,End", "--noheader", "--parsable2"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                # Get the main job line (not .batch or .extern)
                for line in lines:
                    parts = line.split('|')
                    if len(parts) >= 3 and not '.' in parts[0]:
                        status = parts[1]
                        exit_code = parts[2].split(':')[0] if ':' in parts[2] else parts[2]
                        
                        return {
                            "success": True,
                            "job_id": job_id,
                            "status": status,
                            "exit_code": exit_code,
                            "running": False,
                            "pending": False,
                            "completed": status in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]
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
        max_attempts: int = None,
        callback: callable = None
    ) -> Dict[str, Any]:
        """
        Wait for a job to complete.
        
        Args:
            job_id: Job ID to wait for
            poll_interval: Seconds between status checks
            max_attempts: Maximum polling attempts
            callback: Optional callback function called each poll
            
        Returns:
            Final job status
        """
        poll_interval = poll_interval or self.config.POLL_INTERVAL
        max_attempts = max_attempts or self.config.MAX_POLL_ATTEMPTS
        
        for attempt in range(max_attempts):
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
        
        return {
            "success": False,
            "job_id": job_id,
            "error": f"Timeout waiting for job after {max_attempts * poll_interval} seconds"
        }
    
    def wait_for_jobs(
        self,
        job_ids: List[str],
        poll_interval: int = None,
        max_attempts: int = None
    ) -> Dict[str, Any]:
        """
        Wait for multiple jobs to complete.
        
        Args:
            job_ids: List of job IDs to wait for
            poll_interval: Seconds between status checks
            max_attempts: Maximum polling attempts
            
        Returns:
            Dict with status of all jobs
        """
        poll_interval = poll_interval or self.config.POLL_INTERVAL
        max_attempts = max_attempts or self.config.MAX_POLL_ATTEMPTS
        
        remaining = set(job_ids)
        results = {}
        
        for attempt in range(max_attempts):
            newly_completed = []
            
            for job_id in remaining:
                status = self.get_job_status(job_id)
                
                if status.get("completed"):
                    results[job_id] = status
                    newly_completed.append(job_id)
            
            for job_id in newly_completed:
                remaining.remove(job_id)
            
            if not remaining:
                # All jobs completed
                return {
                    "success": True,
                    "all_completed": True,
                    "jobs": results
                }
            
            time.sleep(poll_interval)
        
        # Timeout - return what we have
        for job_id in remaining:
            results[job_id] = {
                "success": False,
                "job_id": job_id,
                "error": "Timeout"
            }
        
        return {
            "success": False,
            "all_completed": False,
            "jobs": results,
            "timed_out": list(remaining)
        }
    
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a running or pending job"""
        if not self.slurm_available:
            return {"success": False, "error": "SLURM not available"}
        
        try:
            result = subprocess.run(
                ["scancel", job_id],
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
            "success": all(r["success"] for r in results),
            "cancelled": len([r for r in results if r["success"]]),
            "failed": len([r for r in results if not r["success"]]),
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
        # Find output files
        output_files = list(self.slurm_logs_dir.glob(f"*_{job_id}.out"))
        error_files = list(self.slurm_logs_dir.glob(f"*_{job_id}.err"))
        
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
        memory_per_task: str = None
    ) -> Dict[str, Any]:
        """
        Estimate resources needed for a list of tasks.
        
        Args:
            tasks: List of task dictionaries
            cpus_per_task: CPUs per task
            memory_per_task: Memory per task
            
        Returns:
            Resource estimation
        """
        cpus_per_task = cpus_per_task or self.config.DEFAULT_CPUS
        
        # Parse memory
        mem_gb = 16  # default
        if memory_per_task:
            match = re.match(r'(\d+)([GM])', memory_per_task.upper())
            if match:
                value = int(match.group(1))
                unit = match.group(2)
                mem_gb = value if unit == 'G' else value / 1024
        
        task_count = len(tasks)
        total_cpus = task_count * cpus_per_task
        total_memory_gb = task_count * mem_gb
        
        # Calculate nodes needed
        nodes_by_cpu = (total_cpus + self.config.CORES_PER_NODE - 1) // self.config.CORES_PER_NODE
        nodes_by_mem = (total_memory_gb + 1000 - 1) // 1000  # 1TB per node
        nodes_needed = max(nodes_by_cpu, nodes_by_mem)
        
        # Get current availability
        cluster_status = self.get_cluster_status()
        idle_nodes = cluster_status.get("idle_count", 0) if cluster_status["success"] else 0
        
        return {
            "task_count": task_count,
            "cpus_per_task": cpus_per_task,
            "memory_per_task": f"{mem_gb}G",
            "total_cpus_needed": total_cpus,
            "total_memory_gb": total_memory_gb,
            "nodes_needed": nodes_needed,
            "idle_nodes_available": idle_nodes,
            "can_run_parallel": idle_nodes >= nodes_needed,
            "recommended_concurrent": min(task_count, idle_nodes * (self.config.CORES_PER_NODE // cpus_per_task))
        }
