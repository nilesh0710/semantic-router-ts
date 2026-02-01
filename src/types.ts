/**
 * Route - A semantic route definition
 * 
 * Routes define decision paths that the router can choose based on
 * semantic similarity to user queries.
 */

export interface Route {
    /** Unique identifier for this route */
    name: string;

    /** Example utterances that should trigger this route */
    utterances: string[];

    /** Optional description for documentation */
    description?: string;

    /** Optional metadata to attach to route matches */
    metadata?: Record<string, unknown>;

    /** Optional function schema for tool calling */
    functionSchema?: FunctionSchema;
}

export interface FunctionSchema {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
}

/**
 * RouteMatch - Result of routing a query
 */
export interface RouteMatch {
    /** The matched route, or null if no match */
    route: Route | null;

    /** Name of the matched route */
    name: string | null;

    /** Confidence score (0-1) */
    confidence: number;

    /** All scored routes for debugging */
    scores?: RouteScore[];
}

export interface RouteScore {
    name: string;
    score: number;
}

/**
 * RouterConfig - Configuration for SemanticRouter
 */
export interface RouterConfig {
    /** Routes to register */
    routes?: Route[];

    /** Encoder to use for embeddings */
    encoder?: Encoder;

    /** Number of top matches to consider */
    topK?: number;

    /** Score aggregation method */
    aggregation?: 'mean' | 'max' | 'sum';

    /** Minimum score to consider a match */
    threshold?: number;
}

/**
 * Encoder - Interface for embedding providers
 */
export interface Encoder {
    /** Encode a single text */
    encode(text: string): Promise<number[]>;

    /** Encode multiple texts (batch) */
    encodeBatch(texts: string[]): Promise<number[][]>;

    /** Encoder name for logging */
    readonly name: string;

    /** Embedding dimensions */
    readonly dimensions: number;
}

/**
 * Index - Interface for vector storage
 */
export interface Index {
    /** Add embeddings with route names */
    add(embeddings: number[][], routes: string[], utterances: string[]): Promise<void>;

    /** Query for similar embeddings */
    query(embedding: number[], topK: number): Promise<IndexMatch[]>;

    /** Clear all stored data */
    clear(): Promise<void>;

    /** Check if index is ready */
    isReady(): boolean;
}

export interface IndexMatch {
    route: string;
    utterance: string;
    score: number;
}

/**
 * LLM - Interface for LLM fallback
 */
export interface LLM {
    /** Generate a response */
    generate(prompt: string): Promise<string>;

    /** LLM name for logging */
    readonly name: string;
}
