/**
 * BaseEncoder - Abstract base class for all encoders
 * 
 * Port of Python semantic_router.encoders.base
 */

import type { Encoder } from '../types';

export abstract class BaseEncoder implements Encoder {
    abstract readonly name: string;
    abstract readonly dimensions: number;

    protected initialized = false;

    /**
     * Encode a single text into an embedding vector.
     */
    abstract encode(text: string): Promise<number[]>;

    /**
     * Encode multiple texts in batch.
     * Default implementation calls encode() for each, but subclasses
     * can override for more efficient batch processing.
     */
    async encodeBatch(texts: string[]): Promise<number[][]> {
        return Promise.all(texts.map(text => this.encode(text)));
    }

    /**
     * Normalize an embedding vector to unit length.
     */
    protected normalize(embedding: number[]): number[] {
        const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
        if (norm === 0) return embedding;
        return embedding.map(val => val / norm);
    }
}
