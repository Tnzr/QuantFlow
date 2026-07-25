FROM rust:1.85-slim-bookworm AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src/ src/
RUN cargo build --release && strip target/release/quantflow-engine

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/quantflow-engine /usr/local/bin/
EXPOSE 50051
CMD ["quantflow-engine"]
