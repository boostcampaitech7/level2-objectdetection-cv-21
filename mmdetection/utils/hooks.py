import warnings
from typing import Dict, Optional, Union

from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.logger.wandb import WandbLoggerHook

@HOOKS.register_module()
class NoInitWandbLoggerHook(WandbLoggerHook):
    def __init__(self, 
                 init_kwargs: Optional[Dict] = None, 
                 interval: int = 10, 
                 ignore_last: bool = True,
                 reset_flag: bool = False,
                 commit: bool = True,
                 by_epoch: bool = True,
                 with_step: bool = True,
                 log_artifact: bool = True,
                 out_suffix: Union[str, tuple] = ('.log.json', '.log', '.py'),
                 define_metric_cfg: Optional[Dict] = None):
        super().__init__(init_kwargs=init_kwargs, interval=interval, ignore_last=ignore_last, reset_flag=reset_flag, commit=commit, by_epoch=by_epoch, with_step=with_step, log_artifact=log_artifact, out_suffix=out_suffix, define_metric_cfg=define_metric_cfg)

    def before_run(self, runner) -> None:
        super(WandbLoggerHook, self).before_run(runner)
        if self.wandb is None:
            self.import_wandb()
        print("Warning: Wandb not initialized in the hook. Ensure initialization before hook execution.")
        summary_choice = ['min', 'max', 'mean', 'best', 'last', 'none']
        if self.define_metric_cfg is not None:
            for metric, summary in self.define_metric_cfg.items():
                if summary not in summary_choice:
                    warnings.warn(
                        f'summary should be in {summary_choice}. '
                        f'metric={metric}, summary={summary} will be skipped.')
                self.wandb.define_metric(  # type: ignore
                    metric, summary=summary)
