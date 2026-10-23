# Cost Guardrails

Budget is tight. These rules are non-negotiable.

## Rules
1. **Billing alert before any resource.** On GCP: set a Budget + alert (e.g. alert at a
   low threshold) the moment the account exists, before launching anything.
2. **Smallest viable VM.** 2 vCPU / 4–8 GB is plenty. The heavy LLM work is a hosted API
   or a small model — the VM does not need to be big.
3. **Stop the VM when idle.** Single biggest cost lever. Start it when working, stop it
   when done. Never leave it running overnight.
4. **Free tier / credits first.** GCP new accounts get free credit — lean on it. Pay only
   when a real limit is hit, and flag the cost before spending.

## Rough cost expectations
- VM: a few units of currency per hour; disciplined shutdown keeps monthly spend low.
- Embeddings: negligible (cents to a little more for a project this size).
- Generation: free tier or small local model → near zero.
- Expected total: comfortably within budget, likely a fraction of the cap.

## Habit
Before spinning anything up: "Is this the cheapest option that works?" If unsure, ask first.
