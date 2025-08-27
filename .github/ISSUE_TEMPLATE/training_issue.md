---
name: 🧠 Training Issue
about: Report problems with neural network training or model performance
title: "[TRAINING] "
labels: ["training", "neural-network", "needs-triage"]
assignees: ""
---

## 🧠 Training Issue Description

A clear and concise description of the training problem you're experiencing.

## 🎯 Training Context

- **Training Mode**: [e.g. Single-threaded, Parallel (3 games), Visual]
- **Model Status**: [e.g. Fresh start, Continue from v1, Continue from v2]
- **Training Stage**: [e.g. Initial training, Incremental training, Model evaluation]

## 📊 Training Parameters

- **Number of Games**: [e.g. 10, 30, 100]
- **Epochs per Game**: [e.g. 3, 5, 10]
- **Learning Rate**: [e.g. 0.001, 0.01, 0.1]
- **Parallel Games**: [e.g. 1, 3, 5]
- **Save Interval**: [e.g. 5, 10, 20]

## 🚨 Specific Problem

Describe the exact issue you're encountering:

- [ ] **Training Hangs**: Process gets stuck and doesn't progress
- [ ] **CUDA Errors**: GPU-related errors or memory issues
- [ ] **Poor Performance**: Model doesn't improve or gets worse
- [ ] **Memory Issues**: Out of memory errors
- [ ] **Model Loading**: Can't load existing models
- [ ] **PGN Generation**: Issues with game history saving
- [ ] **Progress Display**: No progress updates or stuck display
- [ ] **Model Saving**: Models not saved or corrupted
- [ ] **Game Generation**: Games don't complete or are invalid
- [ ] **Other**: [Describe specific issue]

## 🔄 Steps to Reproduce

1. [Step 1]
2. [Step 2]
3. [Step 3]
4. [See the problem]

## 📸 Evidence

- **Screenshots**: [Add relevant screenshots]
- **Log Files**: [Paste relevant log output]
- **Error Messages**: [Copy error messages]
- **Model Files**: [List affected model files]

## 🖥️ System Information

- **OS**: [e.g. Windows 10, macOS 12.0, Ubuntu 20.04]
- **Python Version**: [e.g. 3.8, 3.9, 3.10, 3.11]
- **PyTorch Version**: [e.g. 1.12, 2.0, 2.1]
- **CUDA Version**: [e.g. 11.8, 12.1, None]
- **GPU**: [e.g. NVIDIA RTX 3080, AMD RX 6800, None]
- **RAM**: [e.g. 8GB, 16GB, 32GB]
- **Storage**: [e.g. SSD, HDD, Available space]

## 📁 File Structure

```
models/
├── [List your model files]
games/
├── [List your game history files]
```

## 🧪 Training Behavior

- **Games Started**: [e.g. 5/30 games started]
- **Games Completed**: [e.g. 2/30 games completed]
- **Current Status**: [e.g. "Stuck at game 3", "Hanging after move 15"]
- **Last Successful Action**: [e.g. "Last game completed 2 hours ago"]

## 📊 Performance Metrics

- **Training Loss**: [e.g. "Stuck at 0.85", "Increasing instead of decreasing"]
- **Game Scores**: [e.g. "All negative scores", "No improvement over time"]
- **Training Speed**: [e.g. "Very slow", "Normal until it hangs"]

## 🔧 Troubleshooting Attempted

- [ ] Restarted training from scratch
- [ ] Cleared CUDA cache (`torch.cuda.empty_cache()`)
- [ ] Reduced number of parallel games
- [ ] Reduced epochs per game
- [ ] Adjusted learning rate
- [ ] Checked system resources (RAM, GPU memory)
- [ ] Updated PyTorch/CUDA versions
- [ ] Used different model versions
- [ ] Switched between training modes

## 📋 Error Messages

```
Paste any error messages, warnings, or stack traces here
```

## 🎯 Expected vs Actual

- **Expected**: [e.g. "Training should complete 30 games in 2 hours"]
- **Actual**: [e.g. "Training hangs after 3 games"]

## 📈 Training History

- **Previous Training Sessions**: [e.g. "v1 worked fine, v2 has issues"]
- **Model Evolution**: [e.g. "Started improving then got worse"]
- **Breaking Point**: [e.g. "Issue started when I increased to 100 games"]

## 🔍 Reproduction Consistency

- [ ] **Always**: Issue occurs every time
- [ ] **Sometimes**: Issue occurs randomly
- [ ] **Specific Conditions**: Issue only under certain circumstances
- [ ] **First Time**: Issue never happened before

## 🚀 Workarounds

- **Temporary Solutions**: [e.g. "Reducing parallel games to 1 works"]
- **Alternative Approaches**: [e.g. "Single-threaded mode works fine"]
- **Partial Success**: [e.g. "Can train 10 games but not 30"]

## 📝 Additional Context

- **Recent Changes**: [e.g. "Started happening after updating PyTorch"]
- **Hardware Changes**: [e.g. "Added more RAM", "Changed GPU"]
- **Configuration Changes**: [e.g. "Modified training parameters"]
- **External Factors**: [e.g. "Other programs running", "System updates"]

## 🔗 Related Issues

- **Similar Problems**: [Link to related issues]
- **Known Issues**: [Reference to documented problems]
- **Workarounds**: [Link to solutions or discussions]
