# CHRISTUS-INTEGRATION-01 实际集成架构

结论：基础运行链已经实现，严格的“核心整合完成”条件尚未满足。不是重新建立一个主体，也未把效果较差的实验适配器转为默认。

唯一入口 `python -m christus` → `ChristusSubject(WPHR.Subject)`。保留原 SparseModularCTRN、事件总线、模块注册表、调度器、记忆层和 NPZ/SQLite 存储。原 WPHR 与 WPHR-P1 源码复制为基线提交，适配代码全部位于 `christus/`，没有覆盖原研究目录。独立 Git 分支 `christus-integration-01`。

实际因果路径：WORLD02 Session 渲染图像/本体位置 → 原 world 观察事件 → 原 Practice 选择动作 → SessionPort 计算行动前身体预测 → Session 执行 → 实测位移及误差 → RecentRelation 更新 → OUTCOME 事件 → 原 Affect 调制和 IntegrationBridge 长期记忆 → 后续 Practice。外部物体预测使用图像检测得到的标签，不把环境内部球坐标作为训练标签。环境 gain 的变化只由评价器施加。

适配器：`session_world.py` 复用 Session/BodyRel/ExternalRel/fit；`runtime.py` 复用 Subject/Practice/Affect/ConceptLanguage/Volition；`__main__.py` 复用 ServiceManager 状态和控制接口。SessionPort 只转发同一个 Subject 的动作，不是第二个自主主体。

safe 默认关闭实验性行动模型、情绪调制、语言纠错；advanced_history、POST03 prior、设备和网络均默认关闭。integration 是显式选择的工程测试配置，不能视作正式通过的算法。既有记忆继续保留，但本轮表现较差，不能宣称优于无记忆。

动作桥把 east/west 映射为左右移动，其他四种网格动作映射为保持。球概念名称来自工程检测标签，不是主体自主发现的概念。身体预测评分即使被计算，也未在本轮改变最终动作；核心 h/c 未进入动作读出，这是真实缺口，不能用架构图掩盖。

安装依赖沿用现有 WPHR Python 3.12 虚拟环境和 numpy；无外部 LLM、无网络调用。完整源码/结果/失败记录/检查点已保留在原路径，另有 3 个 ZIP 及逐文件 SHA256 清单；可重建依赖缓存和 .git 未入 ZIP，旧仓库本身未动。外部论文研究报告保留在原目录，不宣称全部重复归档。
