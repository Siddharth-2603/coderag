// Command go-service is the RAG serving layer.
//
// Milestone 0: a minimal HTTP server with a health check and a stub /query
// endpoint. Later milestones add real retrieval, streaming (SSE), semantic
// caching, rate limiting, and metrics.
package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"time"
)

func main() {
	addr := ":" + envOr("PORT", "8080")

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", handleHealth)
	mux.HandleFunc("POST /query", handleQuery)

	srv := &http.Server{
		Addr:         addr,
		Handler:      logging(mux),
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 30 * time.Second,
	}

	log.Printf("serving layer listening on %s", addr)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("server error: %v", err)
	}
}

func handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// queryRequest is the shape a client sends to ask a question.
type queryRequest struct {
	Question string `json:"question"`
	TopK     int    `json:"top_k"`
}

// queryResponse is a stub. Milestone 1 fills Chunks from the vector store;
// milestone 2 adds a streamed Answer.
type queryResponse struct {
	Question string   `json:"question"`
	Chunks   []string `json:"chunks"`
	Note     string   `json:"note"`
}

func handleQuery(w http.ResponseWriter, r *http.Request) {
	var req queryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON body"})
		return
	}
	if req.Question == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "question is required"})
		return
	}
	if req.TopK == 0 {
		req.TopK = 5
	}

	// TODO(M1): embed the question, query the vector store, return real chunks.
	resp := queryResponse{
		Question: req.Question,
		Chunks:   nil,
		Note:     "stub: retrieval not wired yet (milestone 1)",
	}
	writeJSON(w, http.StatusOK, resp)
}

// --- helpers ---

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
