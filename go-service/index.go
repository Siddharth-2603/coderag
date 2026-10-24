package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
)

// Chunk mirrors one entry in the Python-written index.json. The JSON tags must
// match exactly what the ingestion pipeline writes.
type Chunk struct {
	DocID    string         `json:"doc_id"`
	ChunkID  string         `json:"chunk_id"`
	Text     string         `json:"text"`
	Metadata map[string]any `json:"metadata"`
}

// Index holds the chunks and their parallel pre-computed vectors in memory.
// chunks[i] corresponds to vectors[i].
type Index struct {
	Chunks  []Chunk     `json:"chunks"`
	Vectors [][]float64 `json:"vectors"`
}

// ScoredChunk is a chunk plus its similarity to a query.
type ScoredChunk struct {
	Chunk Chunk   `json:"chunk"`
	Score float64 `json:"score"`
}

// LoadIndex reads index.json produced by the Python ingestion step.
func LoadIndex(path string) (*Index, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read index: %w", err)
	}
	var idx Index
	if err := json.Unmarshal(data, &idx); err != nil {
		return nil, fmt.Errorf("parse index: %w", err)
	}
	if len(idx.Chunks) != len(idx.Vectors) {
		return nil, fmt.Errorf("index corrupt: %d chunks but %d vectors",
			len(idx.Chunks), len(idx.Vectors))
	}
	return &idx, nil
}

// Search returns the topK chunks most similar to queryVec, by cosine similarity.
// Vectors from Gemini are already ~unit length, but we divide by norms to be
// correct regardless, so this is true cosine similarity.
func (idx *Index) Search(queryVec []float64, topK int) []ScoredChunk {
	scored := make([]ScoredChunk, 0, len(idx.Chunks))
	for i, vec := range idx.Vectors {
		scored = append(scored, ScoredChunk{
			Chunk: idx.Chunks[i],
			Score: cosineSimilarity(queryVec, vec),
		})
	}
	sort.Slice(scored, func(a, b int) bool {
		return scored[a].Score > scored[b].Score // highest first
	})
	if topK < len(scored) {
		scored = scored[:topK]
	}
	return scored
}

// cosineSimilarity = dot(a,b) / (||a|| * ||b||). Returns 0 if either is zero.
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
