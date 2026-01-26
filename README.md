# Multi-Agent Biotech Analysis System

**Generated**: 2026-01-26 11:07:31
**Total Tasks Completed**: 0
**Total Tasks Failed**: 1
**Dynamic Tools Created**: 0

---

## 📁 Project Structure
```
├── agents
│   ├── __init__.py
│   ├── master_agent.py
│   ├── sub_agent.py
│   └── tool_creator.py
├── config
│   └── config.yaml
├── data
│   ├── inputs

│   └── outputs

├── envs

├── my_project
│   ├── data
│   │   ├── inputs

│   │   └── outputs

│   ├── envs

│   ├── prompts
│   │   ├── prompt_20260125_204258_e184cb21.json
│   │   ├── prompt_20260125_204534_e184cb21.json
│   │   ├── prompt_20260125_204558_e184cb21.json
│   │   ├── prompt_20260125_212005_e184cb21.json
│   │   └── prompt_20260125_213239_e184cb21.json
│   ├── reports
│   │   ├── subtask_subtask_1_report.json
│   │   └── task_decomposition.json
│   ├── scripts
│   │   └── pipeline.py
│   ├── slurm
│   │   └── scripts
│   │       ├── agent_subtask_1_step0_20260125_214118.sbatch
│   │       ├── agent_subtask_1_step0_20260125_214457.sbatch
│   │       ├── agent_subtask_1_step0_20260125_214736.sbatch
│   │       ├── agent_subtask_1_step0_20260125_215001.sbatch
│   │       ├── agent_subtask_1_step0_20260125_215338.sbatch
│   │       ├── agent_subtask_1_step0_20260125_215714.sbatch
│   │       ├── agent_subtask_1_step0_20260125_220051.sbatch
│   │       ├── agent_subtask_1_step0_20260125_220335.sbatch
│   │       ├── agent_subtask_1_step0_20260125_220652.sbatch
│   │       ├── agent_subtask_1_step0_20260125_221049.sbatch
│   │       ├── agent_subtask_1_step0_20260125_221350.sbatch
│   │       ├── agent_subtask_1_step0_20260125_221625.sbatch
│   │       ├── agent_subtask_1_step0_20260125_222211.sbatch
│   │       ├── agent_subtask_1_step0_20260125_222517.sbatch
│   │       ├── agent_subtask_1_step0_20260125_222801.sbatch
│   │       ├── agent_subtask_1_step0_20260125_223110.sbatch
│   │       ├── agent_subtask_1_step0_20260125_223423.sbatch
│   │       ├── agent_subtask_1_step0_20260125_223754.sbatch
│   │       ├── agent_subtask_1_step0_20260125_224114.sbatch
│   │       ├── agent_subtask_1_step0_20260125_224341.sbatch
│   │       ├── agent_subtask_1_step0_20260125_224702.sbatch
│   │       ├── agent_subtask_1_step0_20260125_224935.sbatch
│   │       ├── agent_subtask_1_step0_20260125_225159.sbatch
│   │       ├── agent_subtask_1_step0_20260125_225451.sbatch
│   │       ├── agent_subtask_1_step0_20260125_230131.sbatch
│   │       ├── agent_subtask_1_step0_20260125_230644.sbatch
│   │       ├── agent_subtask_1_step0_20260125_231025.sbatch
│   │       ├── agent_subtask_1_step0_20260125_231409.sbatch
│   │       ├── agent_subtask_1_step0_20260125_231715.sbatch
│   │       ├── agent_subtask_1_step0_20260125_232130.sbatch
│   │       ├── agent_subtask_1_step0_20260125_232605.sbatch
│   │       ├── agent_subtask_1_step0_20260125_232845.sbatch
│   │       ├── agent_subtask_1_step0_20260125_233312.sbatch
│   │       ├── agent_subtask_1_step0_20260125_233648.sbatch
│   │       ├── agent_subtask_1_step0_20260125_234028.sbatch
│   │       ├── agent_subtask_1_step0_20260125_234409.sbatch
│   │       ├── agent_subtask_1_step0_20260125_235035.sbatch
│   │       ├── agent_subtask_1_step0_20260125_235702.sbatch
│   │       ├── agent_subtask_1_step0_20260126_000304.sbatch
│   │       └── agent_subtask_1_step0_20260126_000621.sbatch
│   ├── temp
│   │   └── search_cache

│   ├── work
│   │   └── sdz852
│   │       └── WORKING
│   │           └── AGI
│   │               └── my_project
│   │                   └── data
│   │                       └── inputs
│   └── temp.txt
├── pipeline_run_20260126_104533

├── pipeline_run_20260126_105643
│   ├── data
│   │   ├── inputs

│   │   └── outputs

│   ├── envs

│   ├── prompts
│   │   └── prompt_20260126_105652_e184cb21.json
│   ├── reports
│   │   ├── final_report.md
│   │   ├── subtask_subtask_1_report.json
│   │   └── task_decomposition.json
│   ├── scripts

│   ├── slurm
│   │   └── scripts

│   └── temp
│       └── search_cache

├── prompts

├── reports

├── scripts

├── slurm
│   └── scripts

├── slurm_logs
│   ├── ollama_186412.log
│   └── ollama_186426.log
├── temp

├── test_project
│   ├── data
│   │   ├── inputs

│   │   └── outputs

│   ├── envs

│   ├── reports

│   ├── scripts

│   ├── slurm
│   │   └── scripts

│   └── temp

├── tools
│   ├── dynamic_tools

│   ├── __init__.py
│   ├── base_tools.py
│   ├── conda_tools.py
│   ├── execution_tools.py
│   ├── sandbox.py
│   ├── slurm_tools.py
│   └── web_search_tools.py
├── utils
│   ├── __init__.py
│   ├── documentation.py
│   ├── git_tracker.py
│   └── logging_config.py
├── workflows
│   ├── __init__.py
│   └── langgraph_workflow.py
├── QUICKSTART.md
├── README.md
├── environment.yml
├── example_gpu_ml_task.txt
├── example_prompt.txt
├── example_simple_test.txt
├── main.py
├── requirements.txt
└── setup.sh
```

---

## ✅ Completed Tasks


---

## ❌ Failed Tasks (For Troubleshooting)


### 1. Download and expand the initial list of companies from an external source or create a new CSV file containing the starter list. Perform any necessary cleaning and formatting to prepare the data for analysis.

- **Task ID**: `subtask_1`
- **Agent**: agent_subtask_1
- **Error**: Unknown error
- **Attempts**: 0

**Troubleshooting Notes**: No analysis available



---

## 🚀 Usage

### Running the System
```bash
python main.py --task "Your task description here"
```

### Reviewing Execution History

- **Logs**: Check `logs/` directory for detailed JSON logs
- **Git History**: Use `git log` to see all commits with task context
- **Failed Tasks**: Review git tags starting with `failure-` for debugging
- **State Replay**: Inspect `workflow_state.db` for complete state history

### Configuration

Edit `config/config.yaml` to adjust:
- Maximum retry attempts
- Model selection (Ollama model to use)
- Tool permissions
- Logging verbosity

---

## 📊 Performance Metrics

- **Average Task Duration**: 45.2 seconds
- **Success Rate**: 0.0%
- **Most Used Tools**: file_exploration

---

## 🔍 Troubleshooting

### Common Issues

**Unknown error** (occurred 1 time(s))

### Failed Task Analysis

Use the following command to review all failures:
```bash
git tag -l "failure-*"
```

Then checkout specific failure to review state:
```bash
git show failure-TASK_ID
```

---

## 📝 Notes

- All agent actions are logged to `logs/` in JSON format
- Every task generates a Git commit for full traceability
- Dynamic tools are saved to `tools/dynamic_tools/` and can be reused
- Workflow state is checkpointed to `workflow_state.db` for recovery

**Last Updated**: 2026-01-26T11:07:31.453885
