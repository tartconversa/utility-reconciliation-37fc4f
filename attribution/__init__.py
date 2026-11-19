"""全厂公用工程联合归因分析包。

模块分层：
  dataio   只读载入资料，按证据可知时间（as_of）过滤，不修改可信测量
  network  压力节点网络、容量与物料平衡定义
  balance  误差界内的可行平衡（线性规划），区分观测、推断与冲突
  models   煤耗等历史关系的受控估计，明确可识别性
  eval     单项与组合方案的可比工况反事实评估及联合归因
  verify   改造后承诺检验
  report   结果呈现（DataFrame/文本），原始观测与推断分列
"""

__all__ = [
    "dataio",
    "network",
    "balance",
    "models",
    "eval",
    "verify",
    "report",
]
