"""
Dependency Parser — Script-informed dependency extraction for AGI Pipeline v1.2.0

Provides the tooling for Phase 2 of the sub-agent lifecycle: analyzing a
complete analysis script (in any supported language) to determine every
dependency it needs, then generating a correct conda environment YAML.

The key design decision is that the **LLM reviews the entire script**, not a
regex-extracted import list.  This catches:
  - subprocess/system() calls to external binaries (samtools, bedtools, ...)
  - Implicit dependencies (scanpy needs anndata, scvi-tools needs torch, ...)
  - R library() / require() calls including Bioconductor packages
  - Perl CPAN modules
  - Java library imports
  - Bash tool invocations

Workflow (called from sub_agent.py Phase 2):
    1. detect_language_from_extension(script_path)
    2. prompt = build_dependency_review_prompt(script_content, language)
    3. llm_response = invoke_resilient(llm, prompt)
    4. dep_list = parse_dependency_response(llm_response)
    5. yaml_str = generate_env_yaml(dep_list, env_name)

Usage:
    from utils.dependency_parser import (
        detect_language_from_extension,
        build_dependency_review_prompt,
        parse_dependency_response,
        generate_env_yaml,
        LANGUAGE_DISPATCH_TABLE,
    )
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DependencyList:
    """
    Structured representation of all dependencies a script requires.

    Populated by parsing the LLM's dependency review response.  Used as
    input to ``generate_env_yaml()`` for YAML generation.
    """

    conda_packages: List[str] = field(default_factory=list)
    """Packages available on conda-forge or bioconda (e.g. 'scanpy', 'samtools')."""

    pip_packages: List[str] = field(default_factory=list)
    """Packages only available via pip (e.g. 'celltypist', 'popv')."""

    system_binaries: List[str] = field(default_factory=list)
    """CLI tools that should be installed via conda if possible (e.g. 'bedtools')."""

    r_packages: List[str] = field(default_factory=list)
    """R packages — will be mapped to r-{pkg} or bioconductor-{pkg} on conda."""

    language: str = "python"
    """Primary script language: python, r, bash, perl, java."""

    python_version: str = "3.10"
    """Python version to pin in the environment."""

    r_version: str = ""
    """R version if an R environment is needed (e.g. '4.3')."""

    notes: List[str] = field(default_factory=list)
    """LLM notes about special requirements, conflicts, or version pins."""

    version_pins: Dict[str, str] = field(default_factory=dict)
    """Explicit version pins, e.g. {'numpy': '<2.0', 'torch': '>=2.0'}."""

    def all_packages(self) -> List[str]:
        """Return a flat list of all package names (no version pins)."""
        all_pkgs = list(self.conda_packages)
        all_pkgs.extend(self.pip_packages)
        all_pkgs.extend(self.r_packages)
        all_pkgs.extend(self.system_binaries)
        return all_pkgs

    def is_empty(self) -> bool:
        return (
            not self.conda_packages
            and not self.pip_packages
            and not self.system_binaries
            and not self.r_packages
        )


# =============================================================================
# LANGUAGE DETECTION & DISPATCH
# =============================================================================

# Maps file extension → language identifier
EXTENSION_MAP: Dict[str, str] = {
    ".py": "python",
    ".R": "r",
    ".r": "r",
    ".Rmd": "r",
    ".rmd": "r",
    ".sh": "bash",
    ".bash": "bash",
    ".pl": "perl",
    ".pm": "perl",
    ".java": "java",
    ".jar": "java",
}

# Dispatch table for sbatch script generation and diagnostics.
# Used by Phase 3 (sbatch generation) and the diagnostic agent.
LANGUAGE_DISPATCH_TABLE: Dict[str, Dict[str, Any]] = {
    "python": {
        "command": "python {script}",
        "shebang": "#!/usr/bin/env python3",
        "comment_prefix": "#",
        "import_patterns": [
            r"^\s*import\s+(\S+)",
            r"^\s*from\s+(\S+)\s+import",
        ],
        "error_traceback_pattern": r'File "(.+?)", line (\d+)',
        "package_check_cmd": 'python -c "import {package}"',
        "permissions": None,
    },
    "r": {
        "command": "Rscript {script}",
        "shebang": "#!/usr/bin/env Rscript",
        "comment_prefix": "#",
        "import_patterns": [
            r"^\s*library\s*\(\s*['\"]?(\w+)['\"]?\s*\)",
            r"^\s*require\s*\(\s*['\"]?(\w+)['\"]?\s*\)",
            r"^\s*(?:if\s*\(!requireNamespace\s*\(\s*['\"]?)(\w+)",
        ],
        "error_traceback_pattern": r"Error in .+? : |Execution halted",
        "package_check_cmd": 'Rscript -e "library({package})"',
        "permissions": None,
    },
    "bash": {
        "command": "bash {script}",
        "shebang": "#!/bin/bash",
        "comment_prefix": "#",
        "import_patterns": [],  # Bash doesn't import — binaries are detected differently
        "error_traceback_pattern": r"line (\d+):",
        "package_check_cmd": "which {package}",
        "permissions": "+x",
    },
    "perl": {
        "command": "perl {script}",
        "shebang": "#!/usr/bin/env perl",
        "comment_prefix": "#",
        "import_patterns": [
            r"^\s*use\s+(\S+?)[\s;]",
            r"^\s*require\s+(\S+?)[\s;]",
        ],
        "error_traceback_pattern": r"at (.+?) line (\d+)",
        "package_check_cmd": 'perl -e "use {package}"',
        "permissions": None,
    },
    "java": {
        "command": "java {script}",
        "shebang": None,
        "comment_prefix": "//",
        "import_patterns": [
            r"^\s*import\s+([\w.]+);",
        ],
        "error_traceback_pattern": r"at (\S+)\((\S+):(\d+)\)",
        "package_check_cmd": None,  # Java dependency checking is more complex
        "compile_first": "javac {script}",
        "permissions": None,
    },
}


def detect_language_from_extension(filepath: str) -> str:
    """
    Determine script language from file extension.

    Args:
        filepath: Path to the script file.

    Returns:
        Language string (python, r, bash, perl, java).
        Defaults to 'python' if extension is unrecognized.
    """
    ext = Path(filepath).suffix
    lang = EXTENSION_MAP.get(ext, None)
    if lang is None:
        logger.warning(
            f"Unrecognized extension '{ext}' for {filepath}, "
            f"defaulting to python"
        )
        return "python"
    return lang


def get_dispatch_entry(language: str) -> Dict[str, Any]:
    """
    Return the dispatch table entry for a language.

    Args:
        language: Language identifier.

    Returns:
        Dict with command, shebang, comment_prefix, etc.
        Falls back to python entry if language is unknown.
    """
    entry = LANGUAGE_DISPATCH_TABLE.get(language)
    if entry is None:
        logger.warning(
            f"No dispatch entry for language '{language}', "
            f"falling back to python"
        )
        return LANGUAGE_DISPATCH_TABLE["python"]
    return entry


def get_execution_command(language: str, script_path: str) -> str:
    """
    Build the execution command string for a script.

    Args:
        language: Language identifier.
        script_path: Path to the script file.

    Returns:
        Command string ready for use in an sbatch script.
    """
    entry = get_dispatch_entry(language)
    return entry["command"].format(script=script_path)


# =============================================================================
# LLM PROMPT CONSTRUCTION
# =============================================================================

def build_dependency_review_prompt(
    script_content: str,
    language: str,
    subtask_description: str = "",
) -> str:
    """
    Construct the LLM prompt for full dependency analysis of a script.

    The prompt instructs the LLM to examine the **entire** script and
    identify all dependencies, including implicit ones, external binaries,
    and packages that are only available via pip vs conda.

    Args:
        script_content: The complete script source code.
        language: Script language (python, r, bash, perl, java).
        subtask_description: Optional task context for better inference.

    Returns:
        A prompt string to send to the LLM.
    """
    lang_specific_guidance = _get_language_guidance(language)

    prompt = f"""You are a dependency analysis expert for computational biology and bioinformatics.

TASK: Analyze the following {language.upper()} script and identify ALL dependencies required to execute it successfully in a fresh conda environment.

{f"CONTEXT: This script is part of a larger analysis pipeline. The subtask description is: {subtask_description}" if subtask_description else ""}

SCRIPT TO ANALYZE:
```{language}
{script_content}
```

{lang_specific_guidance}

INSTRUCTIONS:
1. Identify EVERY package, library, module, and external tool this script needs.
2. For each dependency, determine the correct installation source:
   - conda-forge or bioconda (preferred for most scientific packages)
   - pip only (for packages NOT on conda channels — e.g. celltypist, popv, scvi-tools, decoupler)
   - system binary via conda (for CLI tools like samtools, bedtools, bowtie2, STAR, etc.)
   - R package via conda (r-{{name}} for CRAN, bioconductor-{{name}} for Bioconductor)
3. Include IMPLICIT dependencies that are not directly imported but required:
   - e.g. scanpy needs anndata, leidenalg; scvi-tools needs torch, jax
   - e.g. Seurat needs Matrix, ggplot2; SingleR needs BiocParallel
4. Note any version constraints or known conflicts.
5. Identify the primary language and recommended Python/R version.

RESPOND WITH ONLY a JSON object in this exact format (no other text, no markdown fences):
{{
    "conda_packages": ["package1", "package2"],
    "pip_packages": ["package3"],
    "system_binaries": ["samtools", "bedtools"],
    "r_packages": ["Seurat", "SingleR"],
    "language": "{language}",
    "python_version": "3.10",
    "r_version": "",
    "version_pins": {{"numpy": "<2.0"}},
    "notes": ["any special requirements or known issues"]
}}

RULES:
- Use ONLY conda-forge and bioconda channels (no defaults/main — commercial license).
- Put packages in pip_packages ONLY if they are genuinely not available on conda-forge or bioconda.
- For R packages: use the R package name (e.g. "Seurat" not "r-seurat") — the YAML generator handles the prefix.
- For system binaries: use the conda package name (e.g. "samtools" not "/usr/bin/samtools").
- Include the base language runtime (python, r-base, perl, etc.) in conda_packages.
- Be thorough — a missing dependency causes a pipeline failure and retry cycle.
"""
    return prompt


def _get_language_guidance(language: str) -> str:
    """Return language-specific analysis guidance for the LLM prompt."""
    guidance = {
        "python": """PYTHON-SPECIFIC GUIDANCE:
- Check all `import` and `from ... import` statements
- Check `subprocess.run()`, `os.system()`, `shutil.which()` for external binary calls
- Check `importlib.import_module()` for dynamic imports
- Common bioinformatics packages that are pip-only: popv, celltypist, scvi-tools, decoupler, episcanpy, cell2location, squidpy (check latest — some may have moved to conda)
- If torch/tensorflow is imported, note GPU requirements
- Check for `pip install` commands embedded in the script""",

        "r": """R-SPECIFIC GUIDANCE:
- Check all `library()` and `require()` calls
- Check `BiocManager::install()` for Bioconductor packages
- Check `system()` and `system2()` for external binary calls
- Distinguish CRAN packages (r-{name} on conda-forge) from Bioconductor (bioconductor-{name})
- Include r-base in conda_packages
- If the script uses reticulate, also include python dependencies""",

        "bash": """BASH-SPECIFIC GUIDANCE:
- Every command invoked is a potential dependency (samtools, bedtools, awk, etc.)
- Check for piped commands, loops, and conditionally invoked tools
- Check for `python`, `Rscript`, `perl`, `java` invocations within the script
- Common bioinformatics binaries: samtools, bcftools, bedtools, bowtie2, STAR, hisat2, minimap2, fastp, trimmomatic, fastqc, multiqc
- Most of these are available via bioconda
- Standard Unix tools (grep, sed, awk, sort, cut) do NOT need to be listed""",

        "perl": """PERL-SPECIFIC GUIDANCE:
- Check all `use` and `require` statements
- Check for CPAN modules (many available via conda-forge as perl-{name})
- Check for `system()` and backtick calls for external binaries
- BioPerl modules are available via bioconda""",

        "java": """JAVA-SPECIFIC GUIDANCE:
- Check `import` statements for library dependencies
- Check if the script is a JAR file or needs compilation
- Java bioinformatics tools (GATK, Picard, Trimmomatic) are often available via bioconda
- Include appropriate JDK version in conda_packages (e.g. openjdk)""",
    }
    return guidance.get(language, "")


# =============================================================================
# LLM RESPONSE PARSING
# =============================================================================

def parse_dependency_response(llm_response: str) -> DependencyList:
    """
    Parse the LLM's dependency analysis response into a DependencyList.

    Handles common LLM response quirks:
      - JSON wrapped in markdown code fences
      - Trailing commas
      - Explanatory text before/after the JSON
      - Single quotes instead of double quotes

    Args:
        llm_response: Raw LLM response string.

    Returns:
        Populated DependencyList. Returns a minimal default if parsing fails.
    """
    # Strip markdown fences
    cleaned = llm_response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Try to extract JSON object if there's surrounding text
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if json_match:
        cleaned = json_match.group(0)

    # Fix trailing commas (common LLM mistake)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    # Fix single quotes → double quotes (crude but handles simple cases)
    # Only apply if there are no double quotes at all
    if '"' not in cleaned and "'" in cleaned:
        cleaned = cleaned.replace("'", '"')

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse dependency JSON: {e}")
        logger.debug(f"Raw response:\n{llm_response[:500]}")
        return DependencyList(notes=[f"Parse error: {e}"])

    # Build DependencyList from parsed data
    dep_list = DependencyList(
        conda_packages=_to_str_list(data.get("conda_packages", [])),
        pip_packages=_to_str_list(data.get("pip_packages", [])),
        system_binaries=_to_str_list(data.get("system_binaries", [])),
        r_packages=_to_str_list(data.get("r_packages", [])),
        language=data.get("language", "python"),
        python_version=str(data.get("python_version", "3.10")),
        r_version=str(data.get("r_version", "")),
        notes=_to_str_list(data.get("notes", [])),
        version_pins=data.get("version_pins", {}),
    )

    # Deduplicate
    dep_list.conda_packages = _dedupe(dep_list.conda_packages)
    dep_list.pip_packages = _dedupe(dep_list.pip_packages)
    dep_list.system_binaries = _dedupe(dep_list.system_binaries)
    dep_list.r_packages = _dedupe(dep_list.r_packages)

    # Cross-check: remove anything in pip that's also in conda
    conda_lower = {p.lower().split("=")[0].split(">")[0].split("<")[0]
                    for p in dep_list.conda_packages}
    dep_list.pip_packages = [
        p for p in dep_list.pip_packages
        if p.lower().split("=")[0].split(">")[0].split("<")[0] not in conda_lower
    ]

    logger.info(
        f"Parsed dependencies: "
        f"{len(dep_list.conda_packages)} conda, "
        f"{len(dep_list.pip_packages)} pip, "
        f"{len(dep_list.system_binaries)} binaries, "
        f"{len(dep_list.r_packages)} R packages"
    )

    return dep_list


# =============================================================================
# CONDA ENV YAML GENERATION
# =============================================================================

# Packages that are known to require pip installation.
# This is a safety net — the LLM should also identify these, but having
# a hard-coded fallback prevents build failures.
KNOWN_PIP_ONLY: set = {
    "popv",
    "celltypist",
    "scvi-tools",
    "decoupler",
    "episcanpy",
    "cell2location",
    "moscot",
    "pertpy",
    "mudata",
    "scirpy",
    "schist",
    "palantir",
    "cellrank",
    "cellcharter",
    "cornucopia",
    "novosparc",
    "tangram-sc",
    "stlearn",
}

# Bioconductor R packages — mapped to bioconductor-{name} on conda
KNOWN_BIOCONDUCTOR: set = {
    "SingleR",
    "SingleCellExperiment",
    "BiocParallel",
    "DESeq2",
    "edgeR",
    "limma",
    "GenomicRanges",
    "SummarizedExperiment",
    "clusterProfiler",
    "org.Hs.eg.db",
    "org.Mm.eg.db",
    "scran",
    "scater",
    "batchelor",
    "DropletUtils",
    "AUCell",
    "GSVA",
    "fgsea",
    "ComplexHeatmap",
    "monocle",
    "slingshot",
    "destiny",
    "infercnv",
}


def generate_env_yaml(
    dep_list: DependencyList,
    env_name: str,
    channels: Optional[List[str]] = None,
) -> str:
    """
    Generate a conda environment YAML string from a DependencyList.

    Follows these rules:
      - Channels: conda-forge and bioconda ONLY (no defaults/main)
      - Python/R version pinned from dep_list
      - System binaries → conda dependencies (most are on bioconda)
      - R packages → r-{name} (CRAN) or bioconductor-{name} (Bioconductor)
      - Known pip-only packages → pip section
      - Version pins applied where specified

    Args:
        dep_list: Parsed dependency information.
        env_name: Name for the conda environment.
        channels: Override channel list (default: conda-forge, bioconda).

    Returns:
        YAML string ready to write to a .yml file.
    """
    ch = channels or ["conda-forge", "bioconda"]

    # Start building the conda deps list
    conda_deps: List[str] = []
    pip_deps: List[str] = []

    # --- Language runtime ---------------------------------------------------
    if dep_list.language in ("python", "bash"):
        py_version = dep_list.python_version or "3.10"
        conda_deps.append(f"python={py_version}")
        conda_deps.append("pip")  # Always include pip for fallback installs

    if dep_list.language == "r" or dep_list.r_packages:
        if dep_list.r_version:
            conda_deps.append(f"r-base={dep_list.r_version}")
        else:
            conda_deps.append("r-base")

    if dep_list.language == "perl":
        conda_deps.append("perl")

    if dep_list.language == "java":
        conda_deps.append("openjdk")

    # --- Conda packages -----------------------------------------------------
    for pkg in dep_list.conda_packages:
        base_name = _package_base_name(pkg)

        # Safety net: redirect known pip-only packages
        if base_name.lower() in KNOWN_PIP_ONLY:
            pip_deps.append(_apply_version_pin(pkg, dep_list.version_pins))
            continue

        # Skip if it's the language runtime (already added above)
        if base_name.lower() in ("python", "r-base", "perl", "openjdk", "pip"):
            continue

        conda_deps.append(_apply_version_pin(pkg, dep_list.version_pins))

    # --- System binaries (most available on bioconda) -----------------------
    for binary in dep_list.system_binaries:
        base_name = _package_base_name(binary)
        if base_name.lower() not in {_package_base_name(d).lower() for d in conda_deps}:
            conda_deps.append(binary)

    # --- R packages ---------------------------------------------------------
    for r_pkg in dep_list.r_packages:
        conda_name = _r_package_to_conda(r_pkg)
        if conda_name.lower() not in {_package_base_name(d).lower() for d in conda_deps}:
            conda_deps.append(conda_name)

    # --- Pip packages -------------------------------------------------------
    for pkg in dep_list.pip_packages:
        pinned = _apply_version_pin(pkg, dep_list.version_pins)
        if pinned not in pip_deps:
            pip_deps.append(pinned)

    # --- Build YAML ---------------------------------------------------------
    lines = [
        f"name: {env_name}",
        "channels:",
    ]
    for c in ch:
        lines.append(f"  - {c}")

    lines.append("dependencies:")
    for dep in conda_deps:
        lines.append(f"  - {dep}")

    if pip_deps:
        lines.append("  - pip:")
        for dep in pip_deps:
            lines.append(f"    - {dep}")

    yaml_str = "\n".join(lines) + "\n"
    return yaml_str


def add_package_to_yaml(
    yaml_content: str,
    package: str,
    section: str = "conda",
) -> str:
    """
    Add a single package to an existing env YAML string.

    Used by the diagnostic agent when it discovers a missing package and
    needs to update the YAML before a rebuild.

    Args:
        yaml_content: Existing YAML string.
        package: Package name (optionally with version pin).
        section: "conda" to add to dependencies, "pip" to add to pip section.

    Returns:
        Updated YAML string.
    """
    lines = yaml_content.splitlines()
    result = []

    if section == "pip":
        # Find the pip: section or create one
        pip_section_idx = None
        deps_end_idx = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "- pip:":
                pip_section_idx = i
            if stripped.startswith("- ") and not stripped.startswith("- pip:"):
                deps_end_idx = i

        if pip_section_idx is not None:
            # Insert after the last pip entry
            insert_idx = pip_section_idx + 1
            while insert_idx < len(lines):
                stripped = lines[insert_idx].strip()
                if stripped.startswith("- ") and lines[insert_idx].startswith("    "):
                    insert_idx += 1
                else:
                    break
            lines.insert(insert_idx, f"    - {package}")
        else:
            # No pip section exists — add one at the end of dependencies
            # Find the last dependency line
            last_dep = len(lines) - 1
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip().startswith("- "):
                    last_dep = i
                    break
            lines.insert(last_dep + 1, f"  - pip:")
            lines.insert(last_dep + 2, f"    - {package}")
    else:
        # Add to conda dependencies — insert before pip: section or at end
        insert_idx = len(lines)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "- pip:":
                insert_idx = i
                break
            # Track last conda dep line
            if stripped.startswith("- ") and line.startswith("  - ") and not line.startswith("    "):
                insert_idx = i + 1

        lines.insert(insert_idx, f"  - {package}")

    return "\n".join(lines) + "\n"


def remove_package_from_yaml(
    yaml_content: str,
    package: str,
) -> str:
    """
    Remove a package from an existing env YAML string.

    Searches both conda and pip sections. Matches on base package name
    (ignoring version pins).

    Args:
        yaml_content: Existing YAML string.
        package: Package name to remove.

    Returns:
        Updated YAML string.
    """
    base = _package_base_name(package).lower()
    lines = yaml_content.splitlines()
    result = []

    for line in lines:
        stripped = line.strip()
        # Check if this line is a package entry matching the target
        if stripped.startswith("- "):
            pkg_in_line = stripped.lstrip("- ").strip()
            if _package_base_name(pkg_in_line).lower() == base:
                continue  # Skip this line
        result.append(line)

    return "\n".join(result) + "\n"


# =============================================================================
# QUICK REGEX EXTRACTORS (used as sanity checks, not primary source)
# =============================================================================

def extract_imports_regex(script_content: str, language: str) -> List[str]:
    """
    Quick regex-based import extraction from a script.

    This is NOT the primary dependency detection method (the LLM is).
    It's used for sanity checking and quick pre-scans.

    Args:
        script_content: Script source code.
        language: Script language.

    Returns:
        List of imported module/package names.
    """
    dispatch = LANGUAGE_DISPATCH_TABLE.get(language, {})
    patterns = dispatch.get("import_patterns", [])
    imports = set()

    for pattern in patterns:
        for match in re.finditer(pattern, script_content, re.MULTILINE):
            module = match.group(1)
            # Take only the top-level package (e.g. 'os.path' → 'os')
            top_level = module.split(".")[0]
            imports.add(top_level)

    return sorted(imports)


def extract_subprocess_binaries(script_content: str) -> List[str]:
    """
    Extract external binary names from subprocess/system calls in a script.

    Catches patterns like:
      - subprocess.run(['samtools', ...])
      - subprocess.run("samtools view ...")
      - os.system("bedtools ...")
      - system("bowtie2 ...")

    Args:
        script_content: Script source code.

    Returns:
        List of binary names found.
    """
    binaries = set()

    # subprocess.run/call/Popen with list argument
    for match in re.finditer(
        r"subprocess\.(?:run|call|Popen|check_call|check_output)"
        r"\s*\(\s*\[?\s*['\"](\w[\w\-]*)['\"]",
        script_content,
    ):
        binaries.add(match.group(1))

    # os.system("command ...")
    for match in re.finditer(
        r"os\.system\s*\(\s*['\"](\w[\w\-]*)",
        script_content,
    ):
        binaries.add(match.group(1))

    # R: system("command ...")  and  system2("command", ...)
    for match in re.finditer(
        r"system2?\s*\(\s*['\"](\w[\w\-]*)",
        script_content,
    ):
        binaries.add(match.group(1))

    # Perl: system("command ...")  and  `command ...`
    for match in re.finditer(
        r"system\s*\(\s*['\"](\w[\w\-]*)",
        script_content,
    ):
        binaries.add(match.group(1))
    for match in re.finditer(
        r"`(\w[\w\-]*)\s",
        script_content,
    ):
        binaries.add(match.group(1))

    # Filter out common non-binary false positives
    false_positives = {
        "echo", "print", "cat", "cd", "export", "source",
        "set", "if", "for", "while", "do", "done", "then",
        "fi", "else", "elif", "true", "false", "test",
    }
    binaries -= false_positives

    return sorted(binaries)


# =============================================================================
# HELPERS
# =============================================================================

def _to_str_list(value: Any) -> List[str]:
    """Coerce a value to a list of strings."""
    if isinstance(value, list):
        return [str(v).strip() for v in value if v]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []


def _dedupe(items: List[str]) -> List[str]:
    """Remove duplicates preserving order."""
    seen = set()
    result = []
    for item in items:
        key = item.lower().split("=")[0].split(">")[0].split("<")[0].strip()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _package_base_name(pkg: str) -> str:
    """Extract base package name, stripping version pins."""
    return re.split(r"[=<>!~\[]", pkg)[0].strip()


def _apply_version_pin(
    pkg: str, version_pins: Dict[str, str]
) -> str:
    """Apply a version pin from the pins dict if the package has one."""
    base = _package_base_name(pkg)

    # If the package already has a pin, keep it
    if any(c in pkg for c in "=<>!~"):
        return pkg

    pin = version_pins.get(base) or version_pins.get(base.lower())
    if pin:
        return f"{base}{pin}"
    return pkg


def _r_package_to_conda(r_package: str) -> str:
    """
    Convert an R package name to its conda equivalent.

    CRAN packages → r-{lowercase_name}
    Bioconductor packages → bioconductor-{lowercase_name}
    """
    if r_package in KNOWN_BIOCONDUCTOR:
        return f"bioconductor-{r_package.lower()}"
    return f"r-{r_package.lower()}"
