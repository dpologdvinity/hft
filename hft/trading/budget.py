"""Per-stock dollar budgets and the account-wide loss guard for paper runs."""

from dataclasses import replace
from decimal import Decimal
from functools import partial

from ..account import decimal
from ..risk import RiskConfig, RiskGateway
from ..sizing import make_budget_intent


def budget_risk_config(daily_loss=0.02, max_drawdown=0.05) -> RiskConfig:
    # Exposure is bounded by the dollar budget, so maintenance exposure is the
    # whole stock ledger (no leverage); other limits keep their defaults.
    return replace(
        RiskConfig(), max_daily_loss=daily_loss, max_drawdown=max_drawdown, maintenance_exposure=1.0
    )


def budget_gateway(budget, daily_loss=0.02, max_drawdown=0.05) -> RiskGateway:
    return RiskGateway(budget_risk_config(daily_loss, max_drawdown), entry_cap=budget)


def budget_intent_factory(budget):
    """`PaperEngine(intent_factory=...)` for a stock limited to `budget` dollars."""
    return partial(make_budget_intent, budget=decimal(budget))


class AccountGuard:
    """Daily loss and peak drawdown across every stock in a run."""

    def __init__(self, total_budget, daily_loss=0.02, max_drawdown=0.05):
        self.total_budget = decimal(total_budget)
        if self.total_budget <= 0 or not 0 < daily_loss < 1 or not 0 < max_drawdown < 1:
            raise ValueError("invalid account guard")
        self.daily_loss, self.max_drawdown = decimal(daily_loss), decimal(max_drawdown)
        self.day_start = self.peak = self.total_budget
        self.latched = None
        self.halted = None

    def start_session(self, equity: Decimal):
        self.day_start = decimal(equity)
        self.peak = max(self.peak, self.day_start)
        self.halted = self.latched

    def observe(self, equity: Decimal) -> str | None:
        equity = decimal(equity)
        self.peak = max(self.peak, equity)
        if equity <= self.day_start * (1 - self.daily_loss):
            self.halted = self.halted or "account_daily_loss"
        if equity <= self.peak * (1 - self.max_drawdown):
            self.halted = self.latched = "account_drawdown"
        return self.halted
