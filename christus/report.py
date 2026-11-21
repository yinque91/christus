"""Read existing artifacts only; no experiments or training are launched."""
import hashlib, json, sqlite3
from pathlib import Path
import os
from .runtime import ROOT

def main():
    out=ROOT/'results/acceptance01'
    results=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    results['without_core']=json.loads((ROOT/'results/core_ablation01/summary.json').read_text())
    table='|配置|步数|接球/完成回合|总收益|移动成本|相对 full 动作差异|\n|---|---:|---:|---:|---:|---:|\n'
    for name,r in results.items():
        if 'steps' in r:
            table+=f"|{name}|{r['steps']}|{r['caught']}/{r['episodes']}|{r['reward']:.4f}|{r['cost']:.4f}|{r.get('action_differences_vs_full','—')}|\n"
    raw=[json.loads(x) for x in (out/'full/events.jsonl').read_text(encoding='utf-8').splitlines()]
    windows={}
    for label,start,end in [('initial',0,40),('initial_late',350,400),('changed_early',400,440),('changed_late',700,750),('return_early',750,790),('return_late',950,1000)]:
        part=raw[start:end]
        windows[label]={'steps':[start+1,end],'body_mae':sum(r['body_error'] for r in part)/len(part)}
    checks={}
    for name,r in results.items():
        if 'steps' not in r: continue
        p=ROOT/'results/core_ablation01' if name=='without_core' else out/name
        rows=[json.loads(x) for x in (p/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        db=sqlite3.connect(p/'storage/wphr.db')
        checks[name]={'continuous_sequence': [x['sequence'] for x in rows]==list(range(1,1001)),
                     'raw_rows':len(rows),'source_kinds':sorted({x['source_kind'] for x in rows}),
                     'events':db.execute('select count(*),count(distinct event_id) from events').fetchone(),
                     'body_update_count':rows[-1]['body_updates']}
        db.close()
    manifest=json.loads((ROOT/'backup/manifest.json').read_text())
    source=Path(os.environ.get(CHRISTUS_SOURCE_ROOT, ROOT))
    mismatches=[]
    for item in manifest['files']:
        p=source/item['root']/item['path']
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
            mismatches.append(str(p))
    evidence={'run_checks':checks,'error_windows':windows,
              'backup_files':len(manifest['files']),'source_changed_since_backup':mismatches,
              'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (ROOT/'christus').glob('*.py')}}
    (ROOT/'results/delivery_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    docs={
    'CHRISTUS_INTEGRATION_ARCHITECTURE.md':'''# CHRISTUS-INTEGRATION-01 实际集成架构

结论：基础运行链已经实现，严格的“核心整合完成”条件尚未满足。不是重新建立一个主体，也未把效果较差的实验适配器转为默认。

唯一入口 `python -m christus` → `ChristusSubject(WPHR.Subject)`。保留原 SparseModularCTRN、事件总线、模块注册表、调度器、记忆层和 NPZ/SQLite 存储。原 WPHR 与 WPHR-P1 源码复制为基线提交，适配代码全部位于 `christus/`，没有覆盖原研究目录。独立 Git 分支 `christus-integration-01`。

实际因果路径：WORLD02 Session 渲染图像/本体位置 → 原 world 观察事件 → 原 Practice 选择动作 → SessionPort 计算行动前身体预测 → Session 执行 → 实测位移及误差 → RecentRelation 更新 → OUTCOME 事件 → 原 Affect 调制和 IntegrationBridge 长期记忆 → 后续 Practice。外部物体预测使用图像检测得到的标签，不把环境内部球坐标作为训练标签。环境 gain 的变化只由评价器施加。

适配器：`session_world.py` 复用 Session/BodyRel/ExternalRel/fit；`runtime.py` 复用 Subject/Practice/Affect/ConceptLanguage/Volition；`__main__.py` 复用 ServiceManager 状态和控制接口。SessionPort 只转发同一个 Subject 的动作，不是第二个自主主体。

safe 默认关闭实验性行动模型、情绪调制、语言纠错；advanced_history、POST03 prior、设备和网络均默认关闭。integration 是显式选择的工程测试配置，不能视作正式通过的算法。既有记忆继续保留，但本轮表现较差，不能宣称优于无记忆。

动作桥把 east/west 映射为左右移动，其他四种网格动作映射为保持。球概念名称来自工程检测标签，不是主体自主发现的概念。身体预测评分即使被计算，也未在本轮改变最终动作；核心 h/c 未进入动作读出，这是真实缺口，不能用架构图掩盖。

安装依赖沿用现有 WPHR Python 3.12 虚拟环境和 numpy；无外部 LLM、无网络调用。完整源码/结果/失败记录/检查点已保留在原路径，另有 3 个 ZIP 及逐文件 SHA256 清单；可重建依赖缓存和 .git 未入 ZIP，旧仓库本身未动。外部论文研究报告保留在原目录，不宣称全部重复归档。
''',
    'CHRISTUS_MODULE_AUDIT.md':'''# 模块审计与取舍

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
''',
    'CHRISTUS_INTEGRATION_TEST_REPORT.md':'''# 实际工程验收报告

运行种子 31100；每个独立配置 1,000 步，12 个初始配置与单独的核心冻结配置。不是旧研究重复跑，也不是 WORLD02 正式预注册实验。全部使用实际 Session 环境执行，未伪造成功日志。完整数据在 results/acceptance01、results/core_ablation01；汇总读取脚本 `python -m christus.report` 不运行实验。

'''+table+'''
结论：full 接球 41.67%，safe 58.33%；无记忆 91.67%。这是单种子、不同动作引起不同反馈的工程描述，不能解释为稳定总体性能或统计显著性。成本下降伴随更少接球，不是无条件收益。情绪/记忆均发现负面结果；temporal、concept_language、social、volition 和 action_model 零动作差异，不能把它们的健康状态当作因果证据。

|要求|实测判定|
|---|---|
|同一主体连续至少 1000 步|通过：每臂 sequence 1–1000，不人为逐步干预|
|预测误差进入真实记忆|通过：full 1000 次 prediction_feedback 更新、996 次身体回归拟合|
|规律变化后记录、尝试修正|通过连通性：401 步开始 gain=.45，751 步恢复；持续有真实误差和更新。未证明成功适应或高级旧经验加速|
|情绪接收实际事件并影响决策/学习|通过影响测量；关闭后 798 步动作不同且成绩更好，不能判为有益|
|保存恢复继续学习且不重复计入|检查点边界通过：独立进程续行 50 步，轨迹和规范化因果状态一致；任意中途崩溃无损恢复未通过|
|核心模块可关闭与独立消融|已测原 7 模块、WORLD02 学习、记忆、预测行动适配及核心冻结；无效/负面照实报告|
|全部原始日志|已保存 JSONL、SQLite 事件、配置与多个 NPZ 检查点，未删除差结果|
|统一核心因果链完成|未通过：核心冻结与预测行动适配关闭均不改变动作，社会学习无事件|

语言纠错测试：外部合成反馈绑定已观察事件，belief 从 0 到 -0.54；同一反馈 ID 重复提交返回 duplicate，候选评分反转。只验证评分，不冒充实际行动改善。

补充入口验证在 results/cli01：20 步保存后另一个进程追加 10 步，日志连续 30 行。results/uncommitted_tail01 以真实 8 步执行和第 5 步检查点构造未提交尾部，恢复保护返回失败，31 条事件前后保持不变；这是保护性拒绝测试，不是硬断电恢复成功。失败退出保留异常与状态。硬断电、设备、远程常驻和任意中途恢复不在通过项中。
''',
    'CHRISTUS_UNIFIED_STATE_REPORT.md':'''# 统一状态与恢复

唯一权威状态沿用 WPHR 检查点管理器：NPZ 保存 core h/c、权重、随机数、主体时钟、注册模块、shared、WORLD02 环境、回归样本/系数/更新编号、记忆及概念/语言/来源证据。SQLite 保存原始事件、记忆记录、检查点索引及运行审计。没有同时启用 LIFE01 的第二套主体存储。

全局事件 ID 为 subject_id:递增序号，序号、已处理 ID、反馈 ID 都随状态保存；身体/外物模型用单调 transition sequence 防止二次学习。原事件保留墙钟时间，主体另外保留逻辑步数和模拟时间。暂停/离线间隔只登记实际间隔，不填造运行事件。

Session 适配器补存原 Session.state 未覆盖的运行字段、完整环境配置、感知缓存、最近视觉和 RNG。测试在第 127 步保存，以 SQLite online backup 复制持久化工作区；原进程继续 50 步，独立 Python 进程恢复同一检查点后继续 50 步。`results/acceptance01/resume/comparison.json` 的 rows_equal 与 causal_state_equal 均 true。

比较完整模型/核心/模块/记忆证据/环境/RNG；排除墙钟时间、记忆随机 ID、数据库新运行行等非学习元数据。因此这是规范化因果状态一致，不是两个数据库字节一致，也不代表所有未触发机制均已验证。

CLI 同一目录有排他锁、每 100 步检查点、正常退出检查点及 ServiceManager 状态。STOP/PAUSE 文件会保存并退出，继续需清除请求文件再 --resume。状态文件含更新时间，睡眠期间不会宣称推进；远程持续运行未测试。

已知恢复边界：检查点与 SQLite/JSONL 不是跨文件事务。CLI 检测日志尾部与检查点边界不一致便停止，不覆盖/删除未提交日志，也不冒险重复学习。该保护不是任意崩溃恢复实现；缺少预写日志协调、重放和断电故障注入。当前只承诺已验证的正常检查点恢复。CLI 的同目录锁不防止用户故意在不同目录复制并同时运行另一主体。
''',
    'CHRISTUS_END_TO_END_EXPERIMENT.md':'''# 端到端运行记录与复现

结果来源：results/acceptance01/*/events.jsonl、storage/wphr.db、storage/checkpoints/*.npz；首次完整运行控制台保存在 acceptance01.log。单独核心消融位于 results/core_ablation01。所有配置统一种子 31100、1000 步；没有筛选失败种子，没有置信区间或长期学习证明的主张。

环境为真实执行的离线模拟器，不是物理设备。感知来自 Session 图像与本体位置，执行成本来自实际移动。完整配置每步有行动前身体预测、实际后果、误差、模型版本、经验更新计数、内部状态和模块健康记录。外部物体预测也按观测更新，但尚未用于滚动规划。

误差窗口（直接读取旧日志，身体 MAE）：

```json
'''+json.dumps(windows,ensure_ascii=False,indent=2)+'''
```

窗口均值受动作选择和接触边界影响，只说明误差轨迹可观测；不能单独断言适应成功。学习模型与后续动作的特定因果贡献要看消融，本轮预测行动适配未改变动作。

在工程根目录启动（不要覆盖已有输出目录）：

```powershell
.\\run_christus.ps1 -Out results/my_subject -Steps 1000
.\\run_christus.ps1 -Out results/my_subject -Steps 100 -Resume
# 显式开启未获性能验证的组合，仅供实验
.\\run_christus.ps1 -Out results/my_integration -Steps 1000 -Profile integration
```

环境解释器可通过 -Python 指定；默认沿用原 WPHR 虚拟环境。不安装联网依赖。配置和所有输出位于指定目录。完整重新验证命令（仅供需要时复现，**本次不再重跑**）：

```powershell
& 'C:\\Users\\13394\\Documents\\deepseek-harness\\default-workspace\\WPHR\\.venv\\Scripts\\python.exe' -m christus.validate --out results/new_acceptance
```

独立核心冻结测试可调用 christus.validate.run_arm(out, ('core',))；初始12臂报告和新增核心结果分开保留，未改写旧实验。普通生产入口不会运行消融或批量实验。
''',
    'CHRISTUS_REMAINING_GAPS.md':'''# 未完成项与停止边界

**不得宣布 CHRISTUS-INTEGRATION-01 核心整合完成。** 当前交付是可运行、可保存恢复、具备实际因果日志的集成版本，所有负结果保留。

1. 原 SMCTRN 核心 h/c 不进入 Practice 的行动读出。核心冻结 1000 步动作不变。把核心真正接入行动读出是重要架构变更，需要用户决定；本次没有偷偷新增神经决策头或替换核心。
2. 实验性身体预测评分被计算 864 步，但关闭它动作完全相同；原规则导航与其他评分主导。外物模型目前只预测/更新，没有规划读出。尚不能证明学得物理关系指导行动。
3. 记忆关闭本轮成绩更好，情绪关闭也更好；不调参筛选成功，不升级为默认。需先分清原网格 cue/action 与接球环境的不匹配。ball 没有原 Q 表类别，完整配置仅 131 次 Q 更新；当前感知的 ball 标签为工程定义，并非自主概念形成。
4. 社会模块没有真实社会事件；来源可靠性是显式给定并保留，不是自主学会辨别可信度。语言仅绑定事件的简短纠错接口，尚无完整聊天前端、跨来源冲突学习及实际后续行动纠错验证。
5. 标准初始状态运行，不是原长期检查点学到的全部知识迁移。旧档原封备份，跨世界旧档迁移未接通，不能称原记忆已经全部继承。
6. 正常保存恢复已验证；崩溃未提交尾部只会保护性停止，缺事务重放/断电验证。长期 ID 索引增长也未做持续运行容量验收。
7. 未获得摄像头事件（当前环境无 OpenCV），无真实设备、远程持续运行和睡眠/断电期间运行证据。模拟数据都有来源标记。未扩大设备、网络或文件权限。
8. MR/WORLD/POST 独立脚本全部保留，只有共享 Session/回归/感知路径接入；不能宣称各系列所有有效功能都已迁入。

不继续围绕单一成绩反复调参。后续应先决定是否授权核心到行动读出的结构性接入，再以现有真实数据和同一主体推进；暂不新增情绪理论、语言模型或另一运行时。这个决定之前，保留 safe 为工程启动配置，而非已经证明最优的认知主体。
'''}
    for name,body in docs.items(): (ROOT/name).write_text(body,encoding='utf-8')
    print(json.dumps({'reports':list(docs),'backup_files':evidence['backup_files'],
                      'source_changes':len(mismatches),'all_sequences_valid':all(x['continuous_sequence'] for x in checks.values())},ensure_ascii=False))

if __name__=='__main__': main()
