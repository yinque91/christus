# 模块审计与取舍

|来源/实现|集成状态|依据与限制|
|---|---|---|
|WPHR runtime/subject.py、cognition/smctrn.py|唯一运行时与原核心|原核心被保存、计算，但冻结核心消融动作不变，不能宣称认知核心指导行动|
|WPHR modules/world.py、practice.py|原观察预测、学习与行动|网格动作到接球动作需适配；ball 不在原 Practice 固定 Q 类别表，未见球时的 none 学习仍发生；非完整新对象价值学习|
|WPHR-P1 catcher/session.py、world_cognition.py|真实执行钩子与回归函数接入|只保留最近 40 条经验回归，不启用高级历史候选确认；Session 原保存遗漏字段由适配器补存|
|WORLD01 / WORLD02|WORLD02 Session 作为已存在的物理接口|WORLD02 正式结果不支持默认高级复用；不重跑旧正式实验|
|MR4 / MR8|通过现有 catcher 图像检测和身体接口继承已进入共享代码的部分|未把各独立实验脚本逐个移植；不宣称完整 MR 系列并入|
|POST03|代码及报告保留，机制默认关闭|旧报告的 H1/H2/H3 不支持直接默认采用复杂复用|
|WPHR memory/layers.py、memory/store.py|同一主体长期记忆与适用性检查|实际误差写入 prediction_feedback；动作匹配、来源/范围与最近 40 步约束；无记忆本轮更好|
|modules/affect.py、volition.py|实际误差事件、内部调制与动机|未发现独立 Motivation 类，复用原 Volition；情绪关闭更好，动机消融本轮无动作差异|
|modules/concept_language.py、language/lexicon.py|有证据 ID 的纠错、词语绑定、来源可靠性|合成外部反馈确实修改 belief 和候选评分；没有证明自然语言理解/自主概念形成/实际纠错后的行动改善|
|modules/social.py|原模块与状态保留、可关闭|当前环境无其他社会主体，0 次社会交互，尚未整合社会学习任务|
|AUTONOMY / LIFE01|复用 ServiceManager、同一 Subject 存档与停止控制|不运行其各自旧常驻循环，不引入第二主体；旧 persistence/store.py 独立存储不作为第二套权威状态|
|sensors/camera.py|原代码保留，设备默认关闭|当前虚拟环境无 OpenCV，枚举无可用设备；未获得真实摄像头事件，也未伪称模拟图像为真实摄像头|

旧实验结果参考：`../WPHR_WORLD02_FORMAL_22100/WORLD02_FORMAL_RESULTS.md`、`../WPHR_POST03_20261116/POST03_ALL_EXPERIMENTS_FINAL_REPORT.md`。这里只复用结果，不改变预注册统计口径。

备份源为 `C:/Users/13394/Documents/deepseek-harness/default-workspace/` 下 WPHR、WPHR-P1、WPHR_workspace/subject。此次工程运行从原主体类的标准初始状态启动，**没有导入原 100000 步检查点的已学权重/经验**；这些检查点已备份，跨环境状态迁移尚未验证。
