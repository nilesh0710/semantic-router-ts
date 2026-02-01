# semantic-router-ts

Superfast semantic routing for LLMs and AI agents.

[![npm version](https://badge.fury.io/js/semantic-router-ts.svg)](https://www.npmjs.com/package/semantic-router-ts)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🚀 **Fast** - Uses semantic embeddings for sub-100ms routing decisions
- 🔌 **Pluggable Encoders** - OpenAI, local Transformers.js, or bring your own
- 🎯 **Accurate** - Semantic similarity beats keyword matching
- 📦 **Zero Dependencies** - Core package has no deps; encoders are optional
- 🔒 **Type Safe** - Full TypeScript support with strict types
- 🌐 **Offline Support** - LocalEncoder works without API calls

## Installation

```bash
npm install semantic-router-ts

# For local/offline embeddings (recommended)
npm install @xenova/transformers

# For OpenAI embeddings
npm install openai
```

## Quick Start

```typescript
import { SemanticRouter, Route, LocalEncoder } from 'semantic-router-ts';

// Define your routes
const routes: Route[] = [
  {
    name: 'greeting',
    utterances: ['hello', 'hi there', 'hey', 'good morning'],
  },
  {
    name: 'farewell', 
    utterances: ['goodbye', 'bye', 'see you later', 'take care'],
  },
  {
    name: 'help',
    utterances: ['I need help', 'can you assist me', 'support please'],
  },
];

// Create router with local encoder (no API calls)
const router = new SemanticRouter({
  routes,
  encoder: new LocalEncoder(),
  threshold: 0.4, // Minimum confidence to match
});

// Initialize (pre-encodes all routes)
await router.initialize();

// Route queries
const result = await router.route('hey, how are you doing?');
console.log(result.name);       // 'greeting'
console.log(result.confidence); // 0.85

// No match below threshold
const noMatch = await router.route('what is the weather?');
console.log(noMatch.name);      // null
```

## Encoders

### LocalEncoder (Recommended)

Uses [Transformers.js](https://huggingface.co/docs/transformers.js) for fully offline embeddings:

```typescript
import { LocalEncoder } from 'semantic-router-ts';

const encoder = new LocalEncoder({
  model: 'Xenova/all-MiniLM-L6-v2', // Default, 384 dimensions
  normalize: true,
});
```

Supported models:
- `Xenova/all-MiniLM-L6-v2` (384d, fast)
- `Xenova/all-mpnet-base-v2` (768d, more accurate)
- `Xenova/bge-small-en-v1.5` (384d)
- `Xenova/bge-base-en-v1.5` (768d)

### OpenAIEncoder

Uses OpenAI's embedding API:

```typescript
import { OpenAIEncoder } from 'semantic-router-ts';

const encoder = new OpenAIEncoder({
  apiKey: process.env.OPENAI_API_KEY,
  model: 'text-embedding-3-small', // Default
  dimensions: 1536,
});
```

## Configuration

```typescript
const router = new SemanticRouter({
  routes: [...],
  encoder: new LocalEncoder(),
  
  // How many similar utterances to consider
  topK: 5,
  
  // How to aggregate scores ('mean' | 'max' | 'sum')
  aggregation: 'mean',
  
  // Minimum confidence threshold (0-1)
  threshold: 0.4,
  
  // Optional LLM for low-confidence fallback
  llm: myLLMImplementation,
});
```

## Dynamic Routes

Add routes at runtime:

```typescript
await router.addRoute({
  name: 'billing',
  utterances: ['payment issue', 'invoice problem', 'billing question'],
  description: 'Billing and payment related queries',
});
```

## LLM Fallback

For ambiguous queries, provide an LLM to classify:

```typescript
const router = new SemanticRouter({
  routes,
  encoder: new LocalEncoder(),
  threshold: 0.4,
  llm: {
    name: 'claude',
    generate: async (prompt) => {
      // Your LLM call here
      return await callClaude(prompt);
    },
  },
});
```

## API Reference

### SemanticRouter

| Method | Description |
|--------|-------------|
| `initialize()` | Pre-encode all routes (call once) |
| `route(query)` | Route a query, returns `RouteMatch` |
| `classify(query)` | Shorthand, returns route name or null |
| `addRoute(route)` | Add a route dynamically |
| `getStats()` | Get router statistics |
| `isReady()` | Check if initialized |

### RouteMatch

```typescript
interface RouteMatch {
  route: Route | null;    // Full route object
  name: string | null;    // Route name
  confidence: number;     // 0-1 score
  scores?: RouteScore[];  // All route scores
}
```

## License

MIT
