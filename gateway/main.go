// Command gateway is the INGRESS GATEWAY (top layer of the architecture).
//
// It does NOT do any RAG/agent reasoning. Its job is the "edge" concerns Go is
// good at: receive the client request, log it, rate-limit, forward the question
// to the Python agent over HTTP, and relay the answer back.
//
//	[client] --POST /ask--> [gateway :8080] --POST /agent--> [python agent :9000]
//
// Deliberately minimal (learning slice). Productionizing would add: auth,
// response streaming, circuit breaking, and proper per-user rate limiting.
package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"sync"
	"time"
)

// agentURL is where the Python agent server listens.
func agentURL() string {
	if v := os.Getenv("AGENT_URL"); v != "" {
		return v
	}
	return "http://127.0.0.1:9000/agent"
}

// server holds shared dependencies: an HTTP client for talking to the agent,
// and a tiny rate limiter.
type server struct {
	agentURL string
	client   *http.Client
	limiter  *rateLimiter
}

func main() {
	srv := &server{
		agentURL: agentURL(),
		// A client timeout longer than the agent's own timeout, so the agent can
		// finish (it may call the LLM a couple of times) before we give up.
		client:  &http.Client{Timeout: 90 * time.Second},
		limiter: newRateLimiter(30, time.Minute), // 30 requests / minute / client
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", srv.handleHealth)
	mux.HandleFunc("POST /ask", srv.handleAsk)

	addr := ":" + envOr("PORT", "8080")
	httpSrv := &http.Server{
		Addr:         addr,
		Handler:      logging(mux),
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 100 * time.Second, // must exceed client timeout above
	}
	log.Printf("gateway listening on %s -> agent at %s", addr, srv.agentURL)
	if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("gateway error: %v", err)
	}
}

// --- /ask: the main path ---

type askRequest struct {
	Question string `json:"question"`
}

func (s *server) handleAsk(w http.ResponseWriter, r *http.Request) {
	// 1. Rate limit by client IP (demonstrates the gateway protecting the backend).
	clientIP := clientIP(r)
	if !s.limiter.allow(clientIP) {
		writeJSON(w, http.StatusTooManyRequests,
			map[string]string{"error": "rate limit exceeded, slow down"})
		return
	}

	// 2. Parse + validate.
	var req askRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Question == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "question is required"})
		return
	}

	// 3. Forward to the Python agent over HTTP.
	answer, status, err := s.forwardToAgent(req.Question)
	if err != nil {
		log.Printf("agent call failed: %v", err)
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": "agent unavailable"})
		return
	}
	// 4. Relay the agent's response back to the client.
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(answer)
}

// forwardToAgent POSTs the question to the Python agent and returns its raw
// response body + status.
func (s *server) forwardToAgent(question string) ([]byte, int, error) {
	payload, _ := json.Marshal(map[string]string{"question": question})
	resp, err := s.client.Post(s.agentURL, "application/json", bytes.NewReader(payload))
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, 0, err
	}
	return body, resp.StatusCode, nil
}

// --- /healthz: also checks the agent is reachable ---

func (s *server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	// Ping the agent's healthz by swapping the path.
	healthURL := s.agentURL[:len(s.agentURL)-len("/agent")] + "/healthz"
	resp, err := s.client.Get(healthURL)
	agentOK := err == nil && resp.StatusCode == http.StatusOK
	if resp != nil {
		resp.Body.Close()
	}
	writeJSON(w, http.StatusOK, map[string]any{"gateway": "ok", "agent_reachable": agentOK})
}

// --- a tiny fixed-window rate limiter ---

type rateLimiter struct {
	mu      sync.Mutex
	limit   int
	window  time.Duration
	counts  map[string]int
	resetAt map[string]time.Time
}

func newRateLimiter(limit int, window time.Duration) *rateLimiter {
	return &rateLimiter{
		limit:   limit,
		window:  window,
		counts:  make(map[string]int),
		resetAt: make(map[string]time.Time),
	}
}

// allow returns true if this client is under its limit for the current window.
func (rl *rateLimiter) allow(client string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	now := time.Now()
	if now.After(rl.resetAt[client]) {
		rl.counts[client] = 0
		rl.resetAt[client] = now.Add(rl.window)
	}
	if rl.counts[client] >= rl.limit {
		return false
	}
	rl.counts[client]++
	return true
}

// --- helpers ---

func clientIP(r *http.Request) string {
	// RemoteAddr is "ip:port" and the port changes per connection, so we must
	// strip it — otherwise every request looks like a new client and the rate
	// limiter never triggers. Behind a real proxy you'd parse X-Forwarded-For.
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func logging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
	})
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
