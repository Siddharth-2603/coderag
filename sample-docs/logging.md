# Logging and Observability

Every request is logged with a unique request ID, the endpoint, the response
status, and the total latency. Logs are structured as JSON so they can be
parsed and queried by a log aggregation system.

Metrics are exported in Prometheus format: request counts, error rates, and
latency percentiles (p50, p95, p99). Traces follow a request across services
using a propagated trace ID, which helps diagnose where time is spent.
