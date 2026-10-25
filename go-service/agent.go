package main

// This file defines the AGENT layer: tools the LLM can choose to call, and
// (in later steps) the loop that lets the LLM decide which to use.
//
// Step 1 (this file for now): the Tool abstraction. A tool is something the
// agent can invoke. The LLM never runs code itself — it only *asks* to call a
// tool by name with arguments; our code runs the matching Tool and feeds the
// result back.

// ToolParam describes one argument a tool accepts. The LLM reads these so it
// knows what to pass. We keep it simple: a name, a type, and a description.
type ToolParam struct {
	Name        string // e.g. "query"
	Type        string // JSON-schema type, e.g. "string" or "number"
	Description string // what this argument means, so the LLM fills it sensibly
	Required    bool
}

// Tool is one capability the agent can use.
//   - Name:        how the LLM refers to it (e.g. "search_docs")
//   - Description: tells the LLM WHEN to use it — the model picks tools by
//                  reading these, so the wording matters.
//   - Params:      the arguments it accepts.
//   - Run:         the actual Go function executed when the LLM calls this tool.
//                  It takes the arguments the LLM supplied (as a map) and
//                  returns a string result that gets fed back to the LLM.
type Tool struct {
	Name        string
	Description string
	Params      []ToolParam
	Run         func(args map[string]any) (string, error)
}

// Toolbox is the set of tools available to the agent, keyed by name for fast
// lookup when the LLM asks for one.
type Toolbox struct {
	tools map[string]Tool
	order []string // preserve registration order for stable listing
}

func NewToolbox() *Toolbox {
	return &Toolbox{tools: make(map[string]Tool)}
}

// Register adds a tool to the toolbox.
func (tb *Toolbox) Register(t Tool) {
	tb.tools[t.Name] = t
	tb.order = append(tb.order, t.Name)
}

// Get looks up a tool by the name the LLM asked for.
func (tb *Toolbox) Get(name string) (Tool, bool) {
	t, ok := tb.tools[name]
	return t, ok
}

// List returns the tools in registration order (used to describe them to the LLM).
func (tb *Toolbox) List() []Tool {
	out := make([]Tool, 0, len(tb.order))
	for _, name := range tb.order {
		out = append(out, tb.tools[name])
	}
	return out
}
