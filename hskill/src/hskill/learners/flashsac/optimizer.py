from typing import Any, Optional

import torch
import torch.nn as nn


class Network:
    """
    A bundle class for holding training related components (network, optimizer, lr-scheduler, etc.) in one entity.

    args:
        network: PyTorch nn.Module.
        optimizer: PyTorch optimizer (e.g., torch.optim.Adam).
        scheduler: PyTorch learning rate scheduler
        update_step: Number of update steps taken so far.
        compile_network: Whether to compile network.forward() function.
        compile_mode: Argument for torch.compile.
        use_weight_normalization: Whether to apply weight normalization.
                                  If `compile_network` is `True`, `normalize_fn` will be compiled using `compile_mode`.
        ema_source: Network to EMA. Must have the same nn.Module structure.
                    If `compile_network` is `True`, `ema_fn` will be compiled using `compile_mode`.
        ema_tau: EMA coefficient applied to source, i.e., `target = target*(1-tau) + source*tau`
    """

    def __init__(
        self,
        network: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        update_step: int = 0,
        compile_network: bool = False,
        compile_mode: str = "default",
        use_weight_normalization: bool = False,
        ema_source: Optional["Network"] = None,
        ema_tau: Optional[float] = None,
    ):
        self.network = network
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.update_step = update_step

        if compile_network:
            self.network = torch.compile(self.network, mode=compile_mode)  # type: ignore

        # Prepare weight normalization function
        self._use_weight_normalization = use_weight_normalization
        self._weight_normalize_fn = None
        if self._use_weight_normalization:
            _norm_modules = [m for m in network.modules() if hasattr(m, "normalize_parameters")]

            def _weight_normalize_fn() -> None:
                for m in _norm_modules:
                    m.normalize_parameters()  # type: ignore[operator]

            self._weight_normalize_fn = _weight_normalize_fn
            if compile_network:
                self._weight_normalize_fn = torch.compile(self._weight_normalize_fn, mode=compile_mode)

        # Prepare target EMA function
        self._ema_update_fn = None
        self._use_ema = ema_source is not None
        if self._use_ema:
            assert ema_source is not None
            assert ema_tau is not None
            _param_list: list[torch.Tensor] = list(self.network.parameters())
            _ema_param_list: list[torch.Tensor] = list(ema_source.network.parameters())

            def _ema_update_fn() -> None:
                torch._foreach_lerp_(_param_list, _ema_param_list, ema_tau)

            self._ema_update_fn = _ema_update_fn
            if compile_network:
                self._ema_update_fn = torch.compile(self._ema_update_fn, mode=compile_mode)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        assert self._weight_normalize_fn is not None
        self._weight_normalize_fn()

    @torch.no_grad()
    def ema_update_parameters(self) -> None:
        assert self._ema_update_fn is not None
        self._ema_update_fn()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.network(*args, **kwargs)
