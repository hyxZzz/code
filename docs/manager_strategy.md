# Manager 层强化学习策略改进说明

本次改动围绕“分层强化学习”中的高层 manager 策略展开，核心目标是：

1. 利用深度学习提取战场信息；
2. 在稳定的低层残差控制器之上，通过强化学习微调协同分配策略；
3. 最终提升防守方 D 的整体制导率。

以下内容概述主要改动、使用顺序以及训练策略分析。

## 主要改动

1. **高层策略网络实现**：`src/agents/manager.py` 实现了 `ManagerPolicy`，通过多层感知机编码高维战场特征，并输出每名防守者的目标偏好分布，同时内置价值函数，支持 Actor-Critic 更新。
2. **环境支持学习型分配**：`src/envs/three_d_pursuit.py` 新增 manager 观测、动作掩码及自定义分配接口，可通过 `set_manager_action()` 将策略输出的偏好写回环境，并记录在 `info` 中，便于调试。
3. **PPO 训练管线**：`src/algos/ppo_manager.py` 与 `src/train_manager.py` 构建了专用的 PPO 回合缓冲、优势估计与训练流程，可独立优化并评估 manager 策略。
4. **配置与日志**：`src/configs/default.yaml` 的 `manager` 与 `manager_train` 段提供网络宽度、训练超参、评估频率等开关；其中 `manager_train.controller_ckpt` 指定训练 manager 时要加载的低层控制器权重。所有指标通过 `Logger` 写入 `runs/<name>/manager`，便于对比训练前后的制导率变化。

## 推荐使用顺序

1. **先训练低层残差控制器**：保持 `manager.mode: rule`，运行
   ```bash
   python -m src.train --config src/configs/default.yaml
   ```
   生成 `ckpts/best.pt`（用于后续 manager 训练）以及 `ckpts/latest.pt`。
2. **固定控制器再训练 manager**：确保 `manager_train.controller_ckpt` 指向上一步得到的权重（默认即 `ckpts/best.pt`），然后执行
   ```bash
   python -m src.train_manager --config src/configs/default.yaml
   ```
   脚本会加载冻结的残差控制器输出残差动作，仅优化高层分配策略。日志写入 `runs/<name>/manager`，最优权重保存为 `ckpts/manager_best.pt`。
3. **联合评估分层策略**：运行
   ```bash
   python -m src.eval_hierarchical --config src/configs/default.yaml \
     --controller-ckpt ckpts/best.pt --manager-ckpt ckpts/manager_best.pt --episodes 300
   ```
   同时加载两层策略，评估整体制导率表现。

## 训练策略分析

- **为何先规则 manager 再训练 controller？** 低层策略的收敛依赖稳定的分配逻辑。让 manager 保持规则策略可以减少非平稳性，使控制器专注于残差制导和机动能力提升。
- **为何固定 controller 再训练 manager？** manager 的优化目标是高层协同。如果低层策略同步更新，会导致奖励分布不断漂移、信用分配困难。固定性能达标的控制器后，manager 能专注于学习最优分配。
- **可以同时训练吗？** 理论上可以，但需要额外的稳定化技巧（多时间尺度更新、共享缓冲等），实现和调参复杂度都更高。本项目选择分阶段训练以保证可复现性和易调试性。

## 预期收益

- **信息提取更充分**：manager 观测整合了成本矩阵、存活状态、威胁等级等信息，深度网络可以学习复杂的协同分配模式。
- **分配策略可学习**：高层策略输出偏好，再结合匈牙利算法完成冲突消解，减少目标抖动。
- **指标提升**：训练日志中可观察到成功率、接近速度等指标的提升，为 D 提供更有效的制导支援。

如需进一步分析制导率或成功率变化，可结合 `runs` 目录下的日志文件与其他文档中的流程进行对比评估。

