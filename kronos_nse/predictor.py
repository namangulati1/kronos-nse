"""Model loading and Monte-Carlo sampling helpers around ``KronosPredictor``.

Why this module exists: upstream's ``KronosPredictor.predict(..., sample_count=N)``
draws N autoregressive paths and returns their *mean*. For evaluation and for
trading we want the paths themselves - the spread across paths is the model's
uncertainty estimate, and P(up) across paths is a far better signal than a
point forecast. We recover the individual draws by pushing K identical copies
of the same window through ``predict_batch`` with ``sample_count=1``; each
batch row then samples independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .vendor import import_kronos

PRICE_COLS = ["open", "high", "low", "close"]
FEAT_COLS = PRICE_COLS + ["volume", "amount"]


def resolve_device(spec: str = "auto") -> str:
    import torch

    if spec and spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ForecastSamples:
    """Monte-Carlo forecast for a single decision point."""

    timestamps: pd.DatetimeIndex        # (H,) future bar timestamps
    paths: np.ndarray                   # (K, H, 6) sampled OHLCVA paths
    last_close: float                   # close of the final context bar

    @property
    def n_samples(self) -> int:
        return self.paths.shape[0]

    @property
    def close_paths(self) -> np.ndarray:
        """(K, H) sampled close prices."""
        return self.paths[:, :, PRICE_COLS.index("close")]

    def terminal_log_returns(self) -> np.ndarray:
        """(K,) log return from last context close to the horizon close."""
        terminal = np.clip(self.close_paths[:, -1], 1e-9, None)
        return np.log(terminal / self.last_close)

    def path_log_returns(self) -> np.ndarray:
        """(K, H) cumulative log return to each step, vs the last context close."""
        return np.log(np.clip(self.close_paths, 1e-9, None) / self.last_close)

    def summary(self) -> dict:
        term = self.terminal_log_returns()
        # Realised vol implied by each sampled path (bar-to-bar, then averaged).
        step = np.diff(
            np.log(np.clip(
                np.concatenate(
                    [np.full((self.n_samples, 1), self.last_close), self.close_paths], axis=1
                ), 1e-9, None)),
            axis=1,
        )
        return {
            "exp_logret": float(np.mean(term)),
            "median_logret": float(np.median(term)),
            "std_logret": float(np.std(term, ddof=1)),
            "p_up": float(np.mean(term > 0.0)),
            "q05": float(np.quantile(term, 0.05)),
            "q25": float(np.quantile(term, 0.25)),
            "q75": float(np.quantile(term, 0.75)),
            "q95": float(np.quantile(term, 0.95)),
            "pred_realised_vol": float(np.mean(np.std(step, axis=1, ddof=1))),
        }


class KronosRunner:
    """Thin wrapper that owns the model and exposes sampling."""

    def __init__(
        self,
        model_name: str = "NeoQuasar/Kronos-small",
        tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
        device: str = "auto",
        max_context: int = 512,
        clip: int = 5,
        _predictor=None,
    ) -> None:
        self.device = resolve_device(device)
        self.max_context = max_context

        if _predictor is not None:          # injected in tests
            self.predictor = _predictor
            return

        Kronos, KronosTokenizer, KronosPredictor = import_kronos()
        print(f"[model] loading {model_name} + {tokenizer_name} on {self.device}")
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
        model = Kronos.from_pretrained(model_name)
        model.eval()
        tokenizer.eval()
        self.predictor = KronosPredictor(
            model, tokenizer, device=self.device, max_context=max_context, clip=clip
        )

    # -- sampling ----------------------------------------------------------
    def sample(
        self,
        context: pd.DataFrame,
        future_timestamps: pd.DatetimeIndex,
        n_samples: int = 30,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
        batch_size: int = 30,
        verbose: bool = False,
    ) -> ForecastSamples:
        """Draw ``n_samples`` independent forecast paths for one window.

        ``context`` must be an OHLCV(+amount) frame indexed by timestamp; only
        the trailing ``max_context`` bars are used by the model.
        """
        if len(future_timestamps) == 0:
            raise ValueError("future_timestamps is empty")
        if context[PRICE_COLS].isnull().values.any():
            raise ValueError("context contains NaNs in price columns")
        if len(context) > self.max_context:
            context = context.iloc[-self.max_context:]

        pred_len = len(future_timestamps)
        x_ts = pd.Series(context.index)
        y_ts = pd.Series(pd.DatetimeIndex(future_timestamps))
        feats = context[[c for c in FEAT_COLS if c in context.columns]]

        chunks: list[np.ndarray] = []
        remaining = n_samples
        while remaining > 0:
            k = min(batch_size, remaining)
            dfs = [feats] * k
            preds = self.predictor.predict_batch(
                df_list=dfs,
                x_timestamp_list=[x_ts] * k,
                y_timestamp_list=[y_ts] * k,
                pred_len=pred_len,
                T=temperature,
                top_k=top_k,
                top_p=top_p,
                sample_count=1,      # independence comes from the batch dim
                verbose=verbose,
            )
            chunks.append(np.stack([p.values for p in preds], axis=0))
            remaining -= k

        paths = np.concatenate(chunks, axis=0).astype(np.float64)
        return ForecastSamples(
            timestamps=pd.DatetimeIndex(future_timestamps),
            paths=paths,
            last_close=float(context["close"].iloc[-1]),
        )


def set_seed(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
