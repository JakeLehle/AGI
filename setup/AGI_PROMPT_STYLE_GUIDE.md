# AGI Pipeline — Master Prompt Style Guide v1.0

## Purpose

This document defines the canonical format for master prompts consumed by the AGI
Multi-Agent Pipeline. It serves two audiences:

1. **Prompt authors (humans):** How to structure your project prompts so the pipeline
   extracts exactly the steps you intend, with no over-decomposition or missed steps.

2. **Parser developers:** The formal specification that `_extract_steps_from_prompt()`
   must implement. Every parsing rule derives from this document.

### Design Philosophy

The master prompt is a **contract** between the human and the agent system. The human
does the intellectual work of planning — deciding what steps are needed, what each step
produces, and how steps depend on each other. The agents do the mechanical work of
implementation — generating scripts, building environments, submitting jobs, and
validating outputs.

Each step in the contract maps to exactly **one script**, **one conda environment**,
**one SLURM job**, and **one output directory**. This one-to-one mapping is what makes
pipeline outputs reproducible and convertible to Snakemake rules for publication.

**If a step requires more than one script to accomplish, it should be split into
multiple steps in the prompt.** The pipeline does not split steps automatically.
The human owns the step boundaries.

---

## Document Structure

A master prompt has two sections: the **Header** and the **Step Definitions**. Everything
in the Header is context shared across all steps. Everything in the Step Definitions is
parsed into exactly one subtask per step.

```
┌─────────────────────────────────────┐
│  HEADER                             │
│  ├── Title                          │
│  ├── Step Manifest (REQUIRED)       │
│  ├── Global Constraints             │
│  ├── Shared Constants               │
│  ├── Context / Background           │
│  └── Reference Material             │
│                                     │
│  STEP DEFINITIONS                   │
│  ├── <<<STEP_1>>> ... <<<END_STEP>>>│
│  ├── <<<STEP_2>>> ... <<<END_STEP>>>│
│  └── ...                            │
└─────────────────────────────────────┘
```

---

## Header

The Header contains all information that is NOT a step. The parser reads the Header
for shared context but does **not** extract any steps from it. All content before the
first `<<<STEP_N>>>` delimiter is treated as Header.

### Title (required)

First line of the document. Plain text, no markdown formatting required.

```
# Phase 1: Data Preparation and Cell Type Annotation
```

### Step Manifest (REQUIRED)

A fenced block labeled `STEP_MANIFEST` that lists every step by number and title.
This is the **single source of truth** for how many steps exist and what they are called.
The parser uses this to validate that the correct number of `<<<STEP_N>>>` blocks appear
in the document.

```
```STEP_MANIFEST
STEP 1: Validate Raw Inputs and Create Output Directories
STEP 2: Create AnnData Objects from DGE Matrices (Per Puck)
STEP 3: QC Filtering (Per Puck)
STEP 4: Run popV Cell Type Annotation
STEP 5: Clustering and UMAP Embedding
STEP 6: Assign Cluster-Level Cell Types
STEP 7: Generate Visualization Figures
STEP 8: Save Individual Processed Pucks
STEP 9: Generate Summary Report and Verify Completion
```​
```

Rules:
- One step per line
- Format: `STEP N: Title Text`
- N must be a positive integer, sequential starting from 1
- No blank lines within the manifest
- The manifest is the ONLY place step count is defined — the parser rejects
  the prompt if the number of `<<<STEP_N>>>` blocks doesn't match

### Global Constraints (recommended)

Constraints that apply to every step. Written as a fenced block labeled
`GLOBAL_CONSTRAINTS`. The parser injects these into every sub-agent's context.

```
```GLOBAL_CONSTRAINTS
- Use Python exclusively — do NOT use R or Seurat
- All outputs go to outputs/step_NN/ (zero-padded to match step number)
- All scripts must print "SUCCESS: Task completed" on successful completion
- All scripts must sys.exit(1) on failure with a descriptive error message
```​
```

### Shared Constants (recommended)

Variables referenced by multiple steps. Written as a fenced block labeled
`SHARED_CONSTANTS`. The parser injects these into every sub-agent's context as
code that should appear at the top of every generated script.

```
```SHARED_CONSTANTS
PUCK_IDS = ['Puck_211214_29', 'Puck_211214_37', 'Puck_211214_40']
PUCK_DIRS = {
    'Puck_211214_29': '2022-01-28_Puck_211214_29',
    'Puck_211214_37': '2022-01-28_Puck_211214_37',
    'Puck_211214_40': '2022-01-28_Puck_211214_40',
}
RAW_BASE = 'data/raw/fastq'
```​
```

### Context / Background (optional)

Free-form prose providing biological or project context. This helps the LLM
make better decisions during script generation but is not parsed for steps.

No special formatting required — just write it as normal prose or markdown.
The parser ignores everything in the Header that isn't inside a labeled
fenced block.

### Reference Material (optional)

Tables, diagrams, AnnData schemas, output file maps, or any other documentation
the human wants to include for their own reference or for the LLM's context.

No special formatting required. The parser ignores all Header content outside
of labeled fenced blocks. You can use any markdown formatting — tables, headers,
bullet points — without risk of triggering step extraction.

---

## Step Definitions

Each step is enclosed in explicit delimiters that the parser uses for extraction.
**The parser extracts steps ONLY from these delimited blocks.** Nothing outside
a step block is treated as a step.

### Step Delimiters

```
<<<STEP_1>>>

(step content here)

<<<END_STEP>>>
```

Rules:
- `<<<STEP_N>>>` must appear on its own line (leading/trailing whitespace OK)
- `<<<END_STEP>>>` must appear on its own line
- N must match the corresponding entry in the STEP_MANIFEST
- Everything between the delimiters is the step's content
- Nothing outside delimiters is parsed as a step

### Required Fields Per Step

Every step MUST contain the following labeled sections. The parser looks for these
exact labels to extract structured data. The order below is the recommended order
but is not enforced.

#### GOAL

One or two sentences stating what this step accomplishes. Focus on the *what*,
not the *how*.

```
GOAL: Load digital expression matrices and spatial coordinates into AnnData format
with proper spatial coordinate mapping. Produces one h5ad file per puck.
```

#### DEPENDS_ON

Machine-parseable dependency declaration. Which steps must complete before this
one can run. Use step IDs in the format `step_N`.

```
DEPENDS_ON: [step_1]
```

For steps with no dependencies:
```
DEPENDS_ON: []
```

For steps with multiple dependencies:
```
DEPENDS_ON: [step_6, step_7]
```

#### INPUT

Exact file paths this step reads. Include critical structural properties the script
should verify before processing. These are cheap runtime assertions that catch
cross-step state corruption early.

```
INPUT:
- data/raw/fastq/2022-01-28_{puck_id}/{puck_id}.matched.digital_expression.txt.gz
  FORMAT: Tab-separated, gzipped. First row is GENE + barcodes. Genes in rows, cells in columns.
- data/raw/fastq/2022-01-28_{puck_id}/barcode_matching/{puck_id}_barcode_matching.txt.gz
  FORMAT: 4 tab-separated columns, no header: observed_barcode, corrected_barcode, x, y
```

For steps that consume output from previous steps, state what properties the
input MUST have:

```
INPUT:
- outputs/step_03/{puck_id}_QC.h5ad
  CONTRACT: adata.X contains raw integer UMI counts (NOT normalized). Verify: adata.X.max() > 100
```

#### OUTPUT

Exact file paths this step produces. These are what the ValidationAgent checks.

```
OUTPUT:
- outputs/step_02/{puck_id}_raw.h5ad  (one per puck, 3 files total)
```

#### ENVIRONMENT

The conda environment specification for this step. Either an inline YAML block
or an explicit pointer to a shared environment.

**Option A: Inline environment (preferred for most steps)**
```
ENVIRONMENT:
```yaml
name: step_02_adata_prep
channels:
  - conda-forge
  - bioconda
dependencies:
  - python>=3.10
  - scanpy>=1.9.0
  - anndata>=0.9.0
  - pandas>=2.0.0
  - numpy>=1.24.0
```​
```

**Option B: Shared environment pointer**

Use when multiple steps genuinely need the same environment (same packages,
tested together). Reference a named environment defined in the Header.

```
ENVIRONMENT: USE_SHARED phase1_celltype
```

When using a shared environment, define it in the Header as a labeled fenced block:
```
```SHARED_ENV:phase1_celltype
name: phase1_celltype
channels:
  - conda-forge
  - bioconda
dependencies:
  - python>=3.10
  - scanpy>=1.9.0
  ...
```​
```

#### APPROACH

A **single cohesive code block** showing the complete logic flow for this step.
This is the primary code hint for the sub-agent. It should be a complete template
that the sub-agent can adapt into the final script.

**Rules for the APPROACH section:**

1. **One code block per step.** Not multiple fragments showing individual operations.
   The sub-agent needs to see the full flow, not pieces it has to assemble.

2. **Write it as a function or complete script.** The code block should be runnable
   (or close to runnable) as-is. Pseudocode is acceptable for complex logic, but
   real Python is preferred.

3. **Do NOT use numbered sub-steps inside the code section.** Numbered items
   (`1. Do this`, `2. Do that`) outside of code blocks risk being parsed as
   separate steps in legacy parsers. Use comments inside the code block instead.

4. **Reference scripts are valuable.** Point to existing working scripts and
   specific functions within them: "Adapt from `ADATA_PREP.py::load_slideseq_puck()`"

```
APPROACH:
Adapt from scripts/example_reference_scripts/ADATA_PREP.py

```python
def create_adata_for_puck(puck_id):
    """Complete pipeline: DGE + coordinates → AnnData with spatial mapping."""
    puck_dir_name = PUCK_DIRS[puck_id]
    dge_file = f'{RAW_BASE}/{puck_dir_name}/{puck_id}.matched.digital_expression.txt.gz'
    bc_file = f'{RAW_BASE}/{puck_dir_name}/barcode_matching/{puck_id}_barcode_matching.txt.gz'

    # Load DGE matrix (genes x cells) and transpose to cells x genes for AnnData
    df = pd.read_csv(dge_file, sep='\t', compression='gzip', index_col=0)
    adata = ad.AnnData(
        X=df.T.values,
        obs=pd.DataFrame(index=df.columns),
        var=pd.DataFrame(index=df.index)
    )

    # Load barcode coordinates and match to DGE barcodes
    coords_df = pd.read_csv(bc_file, sep='\t', compression='gzip', header=None,
                             names=['observed_barcode', 'corrected_barcode', 'x', 'y'])
    coords_df = coords_df.drop_duplicates(subset='corrected_barcode', keep='first')
    coords_df = coords_df.set_index('corrected_barcode')

    # Intersect barcodes and assign spatial coordinates
    common = list(set(adata.obs_names) & set(coords_df.index))
    adata = adata[common, :].copy()
    spatial = np.array([[coords_df.loc[bc, 'x'], coords_df.loc[bc, 'y']]
                        for bc in adata.obs_names])
    adata.obsm['spatial'] = spatial
    adata.obs['x_coord'] = spatial[:, 0]
    adata.obs['y_coord'] = spatial[:, 1]

    # Add metadata
    adata.obs['puck_id'] = puck_id
    adata.uns['puck_id'] = puck_id
    adata.uns['technology'] = 'Slide-TCR-seq'

    # Compute QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None,
                                log1p=False, inplace=True)

    return adata

# Process each puck
for puck_id in PUCK_IDS:
    adata = create_adata_for_puck(puck_id)
    adata.write_h5ad(f'outputs/step_02/{puck_id}_raw.h5ad')
    print(f'{puck_id}: {adata.n_obs} cells, {adata.n_vars} genes')
```​
```

#### SUCCESS_CRITERIA

What the ValidationAgent checks after the script completes. Focus on verifiable
conditions, not aspirational targets.

```
SUCCESS_CRITERIA:
- 3 h5ad files created: outputs/step_02/Puck_211214_{29,37,40}_raw.h5ad
- Each file has adata.obsm['spatial'] with shape (n_cells, 2)
- Each file has adata.X with raw integer counts (max value > 100)
- >95% barcode match rate between DGE and coordinate files
```

#### CONSTRAINTS (recommended)

Per-step guardrails on what the sub-agent should NOT do. These prevent common
LLM over-engineering where the model adds standard-practice steps that conflict
with your pipeline design.

```
CONSTRAINTS:
- Do NOT normalize the data — raw counts must be preserved for popV in Step 4
- Do NOT filter cells or genes — that is Step 3's job
- Do NOT create a separate conda environment — use the shared phase1_celltype env
```

### Optional Fields

#### REFERENCE_SCRIPTS

Pointers to existing scripts the sub-agent should consult.

```
REFERENCE_SCRIPTS:
- scripts/example_reference_scripts/ADATA_PREP.py — load_slideseq_puck() function
```

#### NOTES

Any additional context specific to this step. Free-form prose. Not parsed
for structure.

```
NOTES:
The barcode_matching file's corrected_barcode column has a "-1" suffix that
matches the DGE column headers. Use it directly as the barcode identifier.
```

---

## Formatting Rules — What NOT to Do

These rules prevent the parser from creating spurious steps. They also prevent
legacy parsers (which may still be active in older pipeline versions) from
misinterpreting content.

### NEVER use numbered lists outside of code blocks

Numbered items (`1. Do X`, `2. Do Y`) outside of fenced code blocks are the
single biggest source of over-decomposition. The extraction regex matches ANY
line starting with `N. ` as a potential step.

**BAD — will create spurious steps:**
```markdown
Critical Implementation Details:
1. DGE matrix is genes × cells — must transpose
2. Barcode matching — corrected barcode has -1 suffix
3. Spatial coordinates go into adata.obsm['spatial']
```

**GOOD — safe alternatives:**

Use a single code block with comments:
```python
# DGE matrix is genes × cells — transpose to cells × genes for AnnData
X = df.T.values
# Corrected barcode has -1 suffix matching DGE headers
coords_df = coords_df.set_index('corrected_barcode')
# Spatial coordinates stored in obsm
adata.obsm['spatial'] = spatial_array
```

Or use prose:
```
The DGE matrix is oriented as genes × cells and must be transposed for AnnData.
The corrected barcode column has a -1 suffix that matches DGE column headers.
Spatial coordinates should be stored in adata.obsm['spatial'].
```

### NEVER use markdown headers (##, ###) inside step blocks

Headers trigger Pattern 3 in the legacy extractor. Inside a step block, use
**bold text** or UPPERCASE labels instead.

**BAD:**
```markdown
### Barcode Matching Details
```

**GOOD:**
```markdown
**Barcode Matching Details:**
```

Or simply use the labeled field names defined in this guide (GOAL, INPUT,
OUTPUT, etc.) which are recognized by the parser as field labels, not steps.

### NEVER use checkbox syntax

`- [ ] Task` and `- [x] Task` trigger Pattern 2 in the legacy extractor.
Use the STEP_MANIFEST for task tracking instead.

### NEVER use sequence keywords as line starters outside step blocks

Lines starting with "First,", "Next,", "Then,", "Finally," trigger Pattern 4.
These are safe inside code blocks (as comments) but not in prose outside of
step delimiters.

---

## Complete Minimal Example

```markdown
# Phase 1: Data Preparation

```STEP_MANIFEST
STEP 1: Download Reference Genome
STEP 2: Align Reads with STAR
STEP 3: Generate Count Matrix
```​

```GLOBAL_CONSTRAINTS
- Use Python for all steps
- All outputs to outputs/step_NN/
```​

```SHARED_CONSTANTS
SAMPLES = ['sample_A', 'sample_B', 'sample_C']
GENOME_DIR = 'data/reference/GRCh38'
```​

## Background

This phase processes bulk RNA-seq data from three tumor samples through
alignment and quantification.

<<<STEP_1>>>

GOAL: Download and index the GRCh38 reference genome for STAR alignment.

DEPENDS_ON: []

INPUT:
- None (downloads from Ensembl)

OUTPUT:
- data/reference/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa
- data/reference/GRCh38/star_index/

ENVIRONMENT:
```yaml
name: step_01_reference
channels:
  - conda-forge
  - bioconda
dependencies:
  - star>=2.7.10
  - wget
```​

APPROACH:
```bash
#!/bin/bash
mkdir -p data/reference/GRCh38
wget -O data/reference/GRCh38/genome.fa.gz [ensembl_url]
gunzip data/reference/GRCh38/genome.fa.gz
STAR --runMode genomeGenerate --genomeDir data/reference/GRCh38/star_index/ \
     --genomeFastaFiles data/reference/GRCh38/genome.fa --runThreadN 8
```​

SUCCESS_CRITERIA:
- star_index/ directory contains SA, SAindex, Genome files
- genome.fa file size > 3GB

CONSTRAINTS:
- Do NOT download the full GTF — only the primary assembly FASTA is needed for this step

<<<END_STEP>>>

<<<STEP_2>>>

GOAL: Align RNA-seq reads to GRCh38 using STAR for each sample.

DEPENDS_ON: [step_1]

INPUT:
- data/reference/GRCh38/star_index/
  CONTRACT: Must contain SA, SAindex, Genome files from STAR genomeGenerate
- data/raw/fastq/{sample}_R1.fastq.gz
- data/raw/fastq/{sample}_R2.fastq.gz

OUTPUT:
- outputs/step_02/{sample}_Aligned.sortedByCoord.out.bam  (one per sample)

ENVIRONMENT:
```yaml
name: step_02_alignment
channels:
  - conda-forge
  - bioconda
dependencies:
  - star>=2.7.10
  - samtools>=1.17
```​

APPROACH:
```bash
#!/bin/bash
for SAMPLE in sample_A sample_B sample_C; do
    STAR --runMode alignReads \
         --genomeDir data/reference/GRCh38/star_index/ \
         --readFilesIn data/raw/fastq/${SAMPLE}_R1.fastq.gz \
                       data/raw/fastq/${SAMPLE}_R2.fastq.gz \
         --readFilesCommand zcat \
         --outSAMtype BAM SortedByCoordinate \
         --outFileNamePrefix outputs/step_02/${SAMPLE}_ \
         --runThreadN 16
    samtools index outputs/step_02/${SAMPLE}_Aligned.sortedByCoord.out.bam
done
```​

SUCCESS_CRITERIA:
- 3 BAM files created in outputs/step_02/
- Each BAM file has a corresponding .bai index
- STAR Log.final.out shows >70% uniquely mapped reads

CONSTRAINTS:
- Do NOT run multi-sample mode — process each sample independently
- Do NOT sort by name — sort by coordinate for downstream tools

<<<END_STEP>>>

<<<STEP_3>>>

GOAL: Generate a gene-by-sample count matrix from aligned BAMs using featureCounts.

DEPENDS_ON: [step_2]

INPUT:
- outputs/step_02/{sample}_Aligned.sortedByCoord.out.bam
  CONTRACT: BAM must be coordinate-sorted with .bai index present
- data/reference/GRCh38/gencode.v44.annotation.gtf

OUTPUT:
- outputs/step_03/raw_counts.tsv

ENVIRONMENT:
```yaml
name: step_03_counting
channels:
  - conda-forge
  - bioconda
dependencies:
  - subread>=2.0.6
  - python>=3.10
  - pandas>=2.0.0
```​

APPROACH:
```bash
#!/bin/bash
featureCounts -a data/reference/GRCh38/gencode.v44.annotation.gtf \
              -o outputs/step_03/raw_counts.tsv \
              -T 16 -p --countReadPairs \
              outputs/step_02/*_Aligned.sortedByCoord.out.bam
```​

SUCCESS_CRITERIA:
- raw_counts.tsv exists with >30,000 gene rows
- All 3 samples present as columns
- Assignment rate >50% in featureCounts summary

CONSTRAINTS:
- Do NOT normalize counts — raw counts are needed for DESeq2 downstream
- Use --countReadPairs for paired-end data

<<<END_STEP>>>
```

---

## Tier 2: Complexity Warnings from Expansion

During the expansion phase, the LLM evaluates each step and may flag potential
issues. These are NOT automatic actions — they are warnings surfaced to the human
for review.

The expansion phase adds a `complexity_warnings` field to the expansion output
when it detects any of the following:

- **Multiple languages required:** The step needs both Python and R, or Python
  and bash with complex logic in both.
- **Divergent compute profiles:** Part of the step needs GPU/high-memory and part
  is lightweight.
- **Multiple independent output artifacts:** The step produces files that have no
  dependency on each other and could be generated independently.
- **Exceeds reasonable script length:** The expanded plan suggests the script would
  exceed ~500 lines of complex logic.

These warnings appear in the pipeline logs and in the pre-execution summary. The
human can then choose to revise the prompt, splitting the flagged step into multiple
steps, or proceed as-is if they believe the step is manageable as a single script.

**The pipeline NEVER splits steps automatically.** Step boundaries are always
the human's decision.

---

## Parser Contract

This section defines exactly what the parser extracts and how. Parser developers
should implement against this specification.

### Extraction Algorithm

```
1. Read the entire prompt
2. Find the STEP_MANIFEST block → extract expected step count and titles
3. Find all <<<STEP_N>>> ... <<<END_STEP>>> blocks
4. Validate: block count == manifest count, all N values match
5. For each step block, extract labeled fields:
   - GOAL        → subtask.description
   - DEPENDS_ON  → subtask.dependencies (parsed as list of step IDs)
   - INPUT       → subtask.input_files + input contracts
   - OUTPUT      → subtask.output_files
   - ENVIRONMENT → subtask.env_yaml OR shared env pointer
   - APPROACH    → subtask.code_hints (full code block)
   - SUCCESS_CRITERIA → subtask.success_criteria
   - CONSTRAINTS → subtask.constraints (injected as negative guidance)
   - REFERENCE_SCRIPTS → subtask.reference_scripts (optional)
   - NOTES       → subtask.notes (optional)
6. Extract Header blocks:
   - GLOBAL_CONSTRAINTS → injected into all sub-agent contexts
   - SHARED_CONSTANTS   → injected into all generated scripts
   - SHARED_ENV:*       → named environments for USE_SHARED pointers
```

### What the Parser Must NEVER Do

- Extract steps from content outside `<<<STEP_N>>>` delimiters
- Create additional steps beyond what the STEP_MANIFEST declares
- Split a single step block into multiple subtasks
- Treat numbered items, markdown headers, checkboxes, or keyword bullets
  as step boundaries
- Infer dependencies from prose — use only the DEPENDS_ON field

### Fallback Behavior

If the prompt does not contain `<<<STEP_N>>>` delimiters or a `STEP_MANIFEST`:

1. Log a warning: "Prompt does not use structured format — falling back to
   legacy extraction"
2. Use the existing `_extract_steps_from_prompt()` + `_filter_executable_steps()`
   pipeline (current v3.2.2 behavior)
3. This ensures backward compatibility with older prompts while encouraging
   migration to the structured format

---

## Migration Checklist

When converting an existing prompt (like Phase 1 v3) to this format:

- [ ] Add STEP_MANIFEST block with all step titles
- [ ] Wrap each step in <<<STEP_N>>> ... <<<END_STEP>>> delimiters
- [ ] Add GOAL, DEPENDS_ON, INPUT, OUTPUT, ENVIRONMENT, APPROACH, SUCCESS_CRITERIA
      fields to each step
- [ ] Consolidate code fragments into single cohesive code blocks per step
- [ ] Move numbered implementation details into comments inside code blocks
- [ ] Add CONSTRAINTS (DO NOT) sections where LLM over-engineering is a risk
- [ ] Add INPUT CONTRACT assertions for steps consuming other steps' outputs
- [ ] Move reference tables, schemas, and documentation into the Header
- [ ] Add GLOBAL_CONSTRAINTS block
- [ ] Add SHARED_CONSTANTS block if multiple steps reference the same variables
- [ ] Verify: no numbered lists outside code blocks in step definitions
- [ ] Verify: no markdown headers (##, ###) inside step blocks
- [ ] Verify: step count in manifest matches number of step blocks
