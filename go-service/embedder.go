package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// GeminiEmbedder embeds text via Google's Gemini embedding REST API.
// Must use the SAME model as the Python ingestion step, or the query vector
// lands in a different space and cosine similarity is meaningless.
type GeminiEmbedder struct {
	apiKey string
	model  string
	client *http.Client
}

func NewGeminiEmbedder(apiKey string) *GeminiEmbedder {
	return &GeminiEmbedder{
		apiKey: apiKey,
		model:  "models/gemini-embedding-001",
		client: &http.Client{Timeout: 30 * time.Second},
	}
}

// embedRequest / embedResponse mirror the Gemini embedContent API shapes.
type embedRequest struct {
	Model   string `json:"model"`
	Content struct {
		Parts []struct {
			Text string `json:"text"`
		} `json:"parts"`
	} `json:"content"`
}

type embedResponse struct {
	Embedding struct {
		Values []float64 `json:"values"`
	} `json:"embedding"`
}

// Embed returns the embedding vector for one piece of text.
func (e *GeminiEmbedder) Embed(text string) ([]float64, error) {
	var reqBody embedRequest
	reqBody.Model = e.model
	reqBody.Content.Parts = []struct {
		Text string `json:"text"`
	}{{Text: text}}

	payload, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	url := fmt.Sprintf(
		"https://generativelanguage.googleapis.com/v1beta/%s:embedContent?key=%s",
		e.model, e.apiKey,
	)
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gemini request failed: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		// Truncate so we never dump a huge error (and never log the key).
		snippet := string(body)
		if len(snippet) > 300 {
			snippet = snippet[:300]
		}
		return nil, fmt.Errorf("gemini API error %d: %s", resp.StatusCode, snippet)
	}

	var er embedResponse
	if err := json.Unmarshal(body, &er); err != nil {
		return nil, fmt.Errorf("parse gemini response: %w", err)
	}
	if len(er.Embedding.Values) == 0 {
		return nil, fmt.Errorf("gemini returned empty embedding")
	}
	return er.Embedding.Values, nil
}
