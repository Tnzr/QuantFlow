//! Pre-trade risk checks and circuit breakers.

/// Check if an order passes pre-trade risk limits
pub fn check_trade(
    portfolio_equity: f64,
    position_value: f64,
    order_value: f64,
) -> Result<(), RiskError> {
    let max_exposure_pct = 0.25;
    let max_total_exposure_pct = 0.80;

    // Single position cannot exceed 25% of portfolio
    if (position_value + order_value) > portfolio_equity * max_exposure_pct {
        return Err(RiskError::PositionTooLarge);
    }

    // Total exposure cannot exceed 80%
    if order_value > portfolio_equity * max_total_exposure_pct {
        return Err(RiskError::ExposureTooHigh);
    }

    Ok(())
}

pub fn daily_loss_limit(current_pnl: f64, max_daily_loss: f64) -> Result<(), RiskError> {
    if current_pnl < -max_daily_loss {
        return Err(RiskError::DailyLossLimit);
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum RiskError {
    #[error("Position would exceed 25% of portfolio")]
    PositionTooLarge,
    #[error("Total exposure would exceed 80%")]
    ExposureTooHigh,
    #[error("Daily loss limit reached")]
    DailyLossLimit,
}
