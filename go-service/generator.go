package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// GeminiGenerator calls Gemini's generateContent endpoint to produce an answer
// grounded in retrieved chunks. This is the "G" in RAG.
type GeminiGenerator struct {
	apiKey string
	model  string
	client *http.Client
}

func NewGeminiGenerator(apiKey string) *GeminiGenerator {
	return &GeminiGenerator{
		apiKey: apiKey,
		model:  "models/gemini-3.1-flash-lite",
		client: &http.Client{Timeout: 25 * time.Second},
	}
}

// buildPrompt assembles a grounded prompt: the model must answer ONLY from the
// provided context, and admit when the answer is not present. This is what
// turns "ask an LLM" into retrieval-augmented generation and fights hallucination.
func buildPrompt(question string, chunks []ScoredChunk) string {
	var b strings.Builder
	b.WriteString("You are a helpful assistant answering questions about a codebase and its documentation.\n")
	b.WriteString("Answer the QUESTION using ONLY the CONTEXT below.\n")
	b.WriteString("If the context does not contain the answer, say you don't know based on the provided documents. Do not invent details.\n")
	b.WriteString("Cite the source document name(s) you used.\n\n")
	b.WriteString("CONTEXT:\n")
	for i, sc := range chunks {
		fmt.Fprintf(&b, "[%d] (source: %s)\n%s\n\n", i+1, sc.Chunk.DocID, sc.Chunk.Text)
	}
	fmt.Fprintf(&b, "QUESTION: %s\n\nANSWER:", question)
	return b.String()
}

// Gemini generateContent request/response shapes (only the fields we use).
type generateRequest struct {
	Contents []genContent `json:"contents"`
}
type genContent struct {
	Parts []genPart `json:"parts"`
}
type genPart struct {
	Text string `json:"text"`
}
type generateResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
}

// Generate produces an answer grounded in the given chunks. It retries on
// transient errors (429/5xx) with exponential backoff, since LLM APIs
// routinely return short-lived 503 "high demand" responses.
func (g *GeminiGenerator) Generate(question string, chunks []ScoredChunk) (string, error) {
	prompt := buildPrompt(question, chunks)
	reqBody := generateRequest{
		Contents: []genContent{{Parts: []genPart{{Text: prompt}}}},
	}
	payload, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}
	url := fmt.Sprintf(
		"https://generativelanguage.googleapis.com/v1beta/%s:generateContent?key=%s",
		g.model, g.apiKey,
	)

	const maxAttempts = 4
	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		answer, status, err := g.doGenerate(url, payload)
		if err == nil {
			return answer, nil
		}
		lastErr = err
		// Retry on transient failures: 429, 5xx, or a client-side timeout/
		// network error (status == 0). Fail fast on other 4xx client errors.
		transient := status == 0 || status == http.StatusTooManyRequests || (status >= 500 && status <= 599)
		if transient && attempt < maxAttempts {
			backoff := time.Duration(attempt) * time.Second // 1s, 2s, 3s
			time.Sleep(backoff)
			continue
		}
		break
	}
	return "", lastErr
}

// doGenerate performs one request. Returns (answer, httpStatus, err).
func (g *GeminiGenerator) doGenerate(url string, payload []byte) (string, int, error) {
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return "", 0, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := g.client.Do(req)
	if err != nil {
		return "", 0, fmt.Errorf("gemini generate request failed: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		snippet := string(body)
		if len(snippet) > 300 {
			snippet = snippet[:300]
		}
		return "", resp.StatusCode, fmt.Errorf("gemini generate error %d: %s", resp.StatusCode, snippet)
	}

	var gr generateResponse
	if err := json.Unmarshal(body, &gr); err != nil {
		return "", resp.StatusCode, fmt.Errorf("parse generate response: %w", err)
	}
	if len(gr.Candidates) == 0 || len(gr.Candidates[0].Content.Parts) == 0 {
		return "", resp.StatusCode, fmt.Errorf("gemini returned no answer")
	}
	return gr.Candidates[0].Content.Parts[0].Text, resp.StatusCode, nil
}
