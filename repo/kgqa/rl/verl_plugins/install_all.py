"""verl 插件总入口: 一个模块装齐所有 OPSD 适配件.

verl 的 `__init__.py` 会在**每个进程** (driver + 全部 ray worker) 导入时调用
`import_external_libs(os.environ["VERL_USE_EXTERNAL_MODULES"])`, 所以:

    export VERL_USE_EXTERNAL_MODULES=kgqa.rl.verl_plugins.install_all
    export PYTHONPATH=.:$PYTHONPATH

即可让 jsd 损失、teacher 注入跳过、KGQA agent loop 注册在 driver 与所有
worker 进程里同时生效 (无需改 site-packages)。

内容:
  - jsd_loss.install()      : 注册 `jsd` 蒸馏损失 + logits processor 分派
  - opsd_teacher.install()  : 预计算 teacher 字段时跳过教师服务
  - opsd_agent_loop         : 导入即注册 `kgqa_opsd_agent`
"""

from __future__ import annotations

_INSTALLED = False


def install_all() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from . import jsd_loss, opsd_no_teacher, opsd_sleep, opsd_teacher
    jsd_loss.install()
    opsd_teacher.install()
    opsd_sleep.install()
    opsd_no_teacher.install()
    from . import opsd_agent_loop  # noqa: F401  (注册副作用)
    _INSTALLED = True


install_all()
