/**
 * LocalIndex - In-memory vector storage
 * 
 * Simple, fast index for development and small-scale use.
 * Port of Python semantic_router.index.local
 */

import type { Index, IndexMatch } from '../types';

interface StoredVector {
    route: string;
    utterance: string;
    embedding: number[];
}

export class LocalIndex implements Index {
    private vectors: StoredVector[] = [];
    private ready = false;

    async add(
        embeddings: number[][],
        routes: string[],
        utterances: string[]
    ): Promise<void> {
        if (embeddings.length !== routes.length || routes.length !== utterances.length) {
            throw new Error('Embeddings, routes, and utterances must have same length');
        }

        for (let i = 0; i < embeddings.length; i++) {
            this.vectors.push({
                route: routes[i],
                utterance: utterances[i],
                embedding: embeddings[i],
            });
        }

        this.ready = true;
    }

    async query(embedding: number[], topK: number): Promise<IndexMatch[]> {
        if (this.vectors.length === 0) {
            return [];
        }

        // Calculate cosine similarity with all vectors
        const scored = this.vectors.map(vector => ({
            route: vector.route,
            utterance: vector.utterance,
            score: this.cosineSimilarity(embedding, vector.embedding),
        }));

        // Sort by score descending and take top K
        scored.sort((a, b) => b.score - a.score);
        return scored.slice(0, topK);
    }

    async clear(): Promise<void> {
        this.vectors = [];
        this.ready = false;
    }

    isReady(): boolean {
        return this.ready;
    }

    /**
     * Get number of stored vectors
     */
    get size(): number {
        return this.vectors.length;
    }

    /**
     * Cosine similarity between two vectors
     */
    private cosineSimilarity(a: number[], b: number[]): number {
        if (a.length !== b.length) return 0;

        let dotProduct = 0;
        let normA = 0;
        let normB = 0;

        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        if (normA === 0 || normB === 0) return 0;
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}
