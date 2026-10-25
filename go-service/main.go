// Command go-service is the RAG serving layer.
//
// It loads the index.json produced by the Python ingestion pipeline, embeds
// incoming questions via Gemini, and serves the most similar chunks. This is
// the Go serving layer in front of the Python-built index.
package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

// server holds the shared dependencies for request handlers.
type server struct {
	index     *Index
	embedder  *GeminiEmbedder
	generator *GeminiGenerator
}

func main() {
	loadDotEnv("../.env") // best-effort; ignored if absent

	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatal("GEMINI_API_KEY not set (put it in ../.env or the environment)")
	}

	indexPath := envOr("INDEX_PATH", "index.json")
	idx, err := LoadIndex(indexPath)
	if err != nil {
		log.Fatalf("loading index: %v", err)
	}
	log.Printf("loaded index: %d chunks from %s", len(idx.Chunks), indexPath)

	srv := &server{
		index:     idx,
		embedder:  NewGeminiEmbedder(apiKey),
		generator: NewGeminiGenerator(apiKey),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", handleHealth)
	mux.HandleFunc("POST /query", srv.handleQuery)

	addr := ":" + envOr("PORT", "8080")
	httpSrv := &http.Server{
		Addr:         addr,
		Handler:      logging(mux),
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 30 * time.Second,
	}
	log.Printf("serving layer listening on %s", addr)
	if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("server error: %v", err)
	}
}

func handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

type queryRequest struct {
	Question string `json:"question"`
	TopK     int    `json:"top_k"`
}

type queryResponse struct {
	Question string        `json:"question"`
	Answer   string        `json:"answer"`
	Sources  []ScoredChunk `json:"sources"`
}

func (s *server) handleQuery(w http.ResponseWriter, r *http.Request) {
	var req queryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON body"})
		return
	}
	if req.Question == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "question is required"})
		return
	}
	if req.TopK <= 0 {
		req.TopK = 3
	}

	// Embed the question (same model as ingestion), then cosine-search the index.
	queryVec, err := s.embedder.Embed(req.Question)
	if err != nil {
		log.Printf("embed error: %v", err)
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": "embedding failed"})
		return
	}
	results := s.index.Search(queryVec, req.TopK)

	// Generate a grounded answer from the retrieved chunks (the "G" in RAG).
	answer, err := s.generator.Generate(req.Question, results)
	if err != nil {
		log.Printf("generate error: %v", err)
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": "generation failed"})
		return
	}
	writeJSON(w, http.StatusOK, queryResponse{
		Question: req.Question,
		Answer:   answer,
		Sources:  results,
	})
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

// loadDotEnv is a minimal KEY=VALUE reader so we don't need a dependency.
func loadDotEnv(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") || !strings.Contains(line, "=") {
			continue
		}
		key, value, _ := strings.Cut(line, "=")
		key, value = strings.TrimSpace(key), strings.TrimSpace(value)
		if _, exists := os.LookupEnv(key); !exists {
			os.Setenv(key, value)
		}
	}
}
