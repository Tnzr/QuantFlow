//! HTTP gateway to Python ML inference service.
use serde::{Deserialize, Serialize};

const ML_SERVICE_URL: &str = "http://ml-engine:8000";

#[derive(Debug, Serialize)]
struct PredictRequest {
    ticker: String,
    n_context: u32,
}

#[derive(Debug, Deserialize)]
pub struct PredictResponse {
    pub ticker: String,
    pub predicted_return: f64,
    pub aleatoric_sigma: f64,
    pub direction: String,
    pub confidence: f64,
}

/// Get ML prediction for a ticker (blocking HTTP call).
/// In production, this would use an async HTTP client with connection pooling.
pub async fn get_prediction(ticker: &str) -> anyhow::Result<PredictResponse> {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/predict", ML_SERVICE_URL))
        .json(&PredictRequest { ticker: ticker.to_string(), n_context: 80 })
        .send()
        .await?;

    let prediction: PredictResponse = resp.json().await?;
    Ok(prediction)
}
