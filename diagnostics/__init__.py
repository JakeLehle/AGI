"""
Diagnostic Runner: Execute diagnostic scripts to gather system information.

The diagnostic runner executes small Python scripts that check for specific
conditions (file existence, module availability, memory, GPU, etc.) and
returns structured results that can be used for failure classification.

Usage:
    from agi.diagnostics import DiagnosticRunner
    
    runner = DiagnosticRunner()
    
    # Run a specific diagnostic
    result = runner.run("check_file_exists", path="/path/to/file.py")
    
    # Run multiple diagnostics
    results = runner.run_batch(["check_memory", "check_gpu", "check_disk_space"])
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticResult:
    """Result from running a diagnostic script."""
    name: str
    success: bool
    output: Dict[str, Any]
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class DiagnosticRunner:
    """
    Runner for diagnostic scripts.
    
    Diagnostic scripts are small Python files that check system conditions
    and return JSON results. They should:
    - Accept arguments via command line or environment
    - Print JSON to stdout
    - Exit with 0 on success, non-zero on failure
    - Complete quickly (< 30 seconds)
    """
    
    # Default timeout for diagnostic scripts
    DEFAULT_TIMEOUT = 30
    
    def __init__(
        self,
        scripts_dir: Optional[str] = None,
        python_path: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """
        Initialize the diagnostic runner.
        
        Args:
            scripts_dir: Directory containing diagnostic scripts
            python_path: Path to Python interpreter (uses current if None)
            timeout: Timeout in seconds for each script
        """
        if scripts_dir:
            self.scripts_dir = Path(scripts_dir)
        else:
            # Default: look for scripts relative to this file
            self.scripts_dir = Path(__file__).parent / "scripts"
        
        self.python_path = python_path or sys.executable
        self.timeout = timeout
        
        # Create scripts directory if it doesn't exist
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DiagnosticRunner initialized: scripts_dir={self.scripts_dir}")
    
    def run(
        self,
        diagnostic_name: str,
        **kwargs
    ) -> DiagnosticResult:
        """
        Run a single diagnostic script.
        
        Args:
            diagnostic_name: Name of the diagnostic (e.g., "check_file_exists")
            **kwargs: Arguments to pass to the script
            
        Returns:
            DiagnosticResult with output
        """
        script_path = self.scripts_dir / f"{diagnostic_name}.py"
        
        if not script_path.exists():
            return DiagnosticResult(
                name=diagnostic_name,
                success=False,
                output={},
                error=f"Script not found: {script_path}",
            )
        
        # Build command
        cmd = [self.python_path, str(script_path)]
        
        # Pass kwargs as JSON via environment
        env = os.environ.copy()
        env["DIAGNOSTIC_ARGS"] = json.dumps(kwargs)
        
        # Also pass common args directly
        for key, value in kwargs.items():
            env[f"DIAG_{key.upper()}"] = str(value)
        
        # Run the script
        import time
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Parse JSON output
            try:
                output = json.loads(result.stdout) if result.stdout.strip() else {}
            except json.JSONDecodeError:
                output = {"raw_output": result.stdout}
            
            return DiagnosticResult(
                name=diagnostic_name,
                success=result.returncode == 0,
                output=output,
                error=result.stderr if result.returncode != 0 else None,
                duration_ms=duration_ms,
            )
            
        except subprocess.TimeoutExpired:
            return DiagnosticResult(
                name=diagnostic_name,
                success=False,
                output={},
                error=f"Timeout after {self.timeout}s",
                duration_ms=self.timeout * 1000,
            )
        except Exception as e:
            return DiagnosticResult(
                name=diagnostic_name,
                success=False,
                output={},
                error=str(e),
            )
    
    def run_batch(
        self,
        diagnostics: List[str],
        **kwargs
    ) -> Dict[str, DiagnosticResult]:
        """
        Run multiple diagnostic scripts.
        
        Args:
            diagnostics: List of diagnostic names
            **kwargs: Arguments passed to all scripts
            
        Returns:
            Dict mapping diagnostic name to result
        """
        results = {}
        for diag in diagnostics:
            results[diag] = self.run(diag, **kwargs)
        return results
    
    def run_all(self, **kwargs) -> Dict[str, DiagnosticResult]:
        """
        Run all available diagnostic scripts.
        
        Args:
            **kwargs: Arguments passed to all scripts
            
        Returns:
            Dict mapping diagnostic name to result
        """
        diagnostics = self.list_available()
        return self.run_batch(diagnostics, **kwargs)
    
    def list_available(self) -> List[str]:
        """List all available diagnostic scripts."""
        scripts = []
        if self.scripts_dir.exists():
            for f in self.scripts_dir.glob("check_*.py"):
                scripts.append(f.stem)
        return sorted(scripts)
    
    def to_dict(self, results: Dict[str, DiagnosticResult]) -> Dict[str, Any]:
        """Convert results to a JSON-serializable dict."""
        return {
            name: {
                "success": r.success,
                "output": r.output,
                "error": r.error,
                "duration_ms": r.duration_ms,
                "timestamp": r.timestamp,
            }
            for name, r in results.items()
        }


# =============================================================================
# Quick diagnostic functions (no subprocess needed)
# =============================================================================

def quick_check_file(path: str) -> Dict[str, Any]:
    """Quick file existence check without subprocess."""
    p = Path(path)
    return {
        "path": str(p),
        "exists": p.exists(),
        "is_file": p.is_file() if p.exists() else False,
        "is_dir": p.is_dir() if p.exists() else False,
        "size_bytes": p.stat().st_size if p.exists() and p.is_file() else 0,
    }


def quick_check_module(module_name: str) -> Dict[str, Any]:
    """Quick module import check without subprocess."""
    try:
        import importlib
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown")
        return {
            "module": module_name,
            "installed": True,
            "version": version,
        }
    except ImportError as e:
        return {
            "module": module_name,
            "installed": False,
            "error": str(e),
        }


def quick_check_memory() -> Dict[str, Any]:
    """Quick memory check without subprocess."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "percent_used": mem.percent,
            "ok": mem.percent < 90,
        }
    except ImportError:
        # Fallback to /proc/meminfo on Linux
        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()
            info = {}
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
            
            total = info.get("MemTotal", 0) / (1024**2)
            available = info.get("MemAvailable", info.get("MemFree", 0)) / (1024**2)
            
            return {
                "total_gb": round(total, 2),
                "available_gb": round(available, 2),
                "percent_used": round((1 - available/total) * 100, 1) if total > 0 else 0,
                "ok": available > 1,  # At least 1GB available
            }
        except:
            return {"error": "Could not check memory"}


def quick_check_gpu() -> Dict[str, Any]:
    """Quick GPU availability check."""
    result = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpus": [],
    }
    
    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            result["gpu_count"] = torch.cuda.device_count()
            for i in range(result["gpu_count"]):
                props = torch.cuda.get_device_properties(i)
                result["gpus"].append({
                    "index": i,
                    "name": props.name,
                    "memory_gb": round(props.total_memory / (1024**3), 2),
                })
    except ImportError:
        # Try nvidia-smi
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                text=True,
                timeout=5,
            )
            lines = output.strip().split("\n")
            result["cuda_available"] = len(lines) > 0
            result["gpu_count"] = len(lines)
            for i, line in enumerate(lines):
                parts = line.split(", ")
                result["gpus"].append({
                    "index": i,
                    "name": parts[0] if parts else "unknown",
                    "memory": parts[1] if len(parts) > 1 else "unknown",
                })
        except:
            pass
    
    return result


def quick_check_disk(path: str = ".") -> Dict[str, Any]:
    """Quick disk space check."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        return {
            "path": path,
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "percent_used": round(used / total * 100, 1),
            "ok": free > 1 * (1024**3),  # At least 1GB free
        }
    except Exception as e:
        return {"path": path, "error": str(e)}
