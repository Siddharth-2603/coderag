# Rate Limiting

Each client is limited to a fixed number of requests per minute. The limiter
uses a token bucket algorithm: every client has a bucket that refills at a
steady rate, and each request removes one token. If the bucket is empty, the
request is rejected with a 429 Too Many Requests response.

Limits are configured per API key. Privileged keys can be granted a higher
quota. The remaining quota is returned in the `X-RateLimit-Remaining` header
on every response so clients can back off before being throttled.
