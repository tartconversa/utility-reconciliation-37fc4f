# 汽改电收益的公用工程联合归因

此目录保存用于产品开发的合成业务资料、字段契约和资料包自检工具。资料说明见 docs/资料口径.md，文件目录及完整性摘要见 data/catalog.json。现有程序仅核对文件、字段、类型、记录键和数量，不计算业务指标、不产生分析结论，也不进行业务批准。

## 分析产品（attribution/）

`attribution/` 是面向评审的分析包，回答**全厂公用工程边界**内的能源、费用与排放变化，
方法与本资料下的结论见 [docs/分析方法说明.md](docs/分析方法说明.md)。核心纪律：

- 减汽不乘固定单价记节约；减汽去向由压力网络在仪表误差界与物理容量内重新平衡确定；
- 煤-汽关系由资料受控估计，识别不出即记 0 并保留范围，不用平衡反推替代观测；
- 未计量放空/损失只给容量内可行区间，冲突日带显式松弛，原始观测与推断分列；
- 单项与组合在同一可比工况、同一价格/排放假设下评估，交互项按区间含 0 与否诚实判定；
- 全部分析可按证据可知时间 `as_of` 切片重现，支持改造后承诺检验与补测影响追踪。

命令（隔离模式用脚本路径，普通环境可用 `python -m attribution`）：

    python -I attribution/__main__.py --root . checksum     # 摘要核对（披露 catalog 不一致）
    python -I attribution/__main__.py --root . balance      # 逐日可行平衡与冲突诊断
    python -I attribution/__main__.py --root . review       # 单项+组合评审与联合归因
    python -I attribution/__main__.py --root . verify --timeline
    python -I attribution/__main__.py --root . scenarios    # 外部价格/排放情景对比
    python -I attribution/__main__.py --root . rank|decision|tighten|decompose

## 环境与资料检查

使用 Python 3.11.2，依赖在 requirements.txt 中锁定。可在虚拟环境安装：

    python3 -m pip install -r requirements.txt
    python3 -m sourcepack

资料检查逐文件回报行数、键唯一性、完整性与关系引用是否符合资料目录；资料行中的拒绝、停试、缺测和不确定范围是业务记录，不是读入错误。请勿据输入检查成功推断分析产品已经可用。

**已知资料问题**：catalog.json 登记的 sha256 与实际 CSV 全部不符（行数与字段一致），
`sourcepack` 自检会报“资料摘要不一致”。分析包以原始 CSV 为准并用 `checksum` 命令披露；
分析测试中该预存失败以 expectedFailure 登记。

## 测试与语法构建

    python3 -I -m unittest discover -s tests -t .
    python3 -m compileall -q sourcepack attribution tests

`tests/test_attribution.py` 覆盖多间隔聚合口径、误差界相加、平衡冲突计数、未计量流量
不可识别、补测收紧、煤-汽不可识别性、反事实传播、联合交互区间、排序稳健性、误差收紧
敏感性与 as_of 可重现性。运行无需访问外部业务系统。
