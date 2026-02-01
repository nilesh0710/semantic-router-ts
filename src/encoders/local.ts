/**
 * LocalEncoder - Offline embedding using Transformers.js
 * 
 * Uses Xenova/all-MiniLM-L6-v2 by default (384 dimensions).
 * Runs entirely locally with no API calls.
 */

import { BaseEncoder } from './base';

// Dynamic import types - pipeline function signature
type PipelineFunction = (task: string, model: string) => Promise<any>;

let pipeline: PipelineFunction | null = null;
let loadPromise: Promise<void> | null = null;

async function loadTransformers(): Promise<void> {
    if (pipeline) return;
    if (loadPromise) {
        await loadPromise;
        return;
    }

    loadPromise = (async () => {
        try {
            // Dynamic import to avoid requiring as hard dependency
            const transformers = await import('@xenova/transformers' as string);
            pipeline = transformers.pipeline;
        } catch {
            throw new Error(
                'LocalEncoder requires @xenova/transformers. ' +
                'Install it with: npm install @xenova/transformers'
            );
        }
    })();

    await loadPromise;
}

export interface LocalEncoderConfig {
    /** Model to use (default: Xenova/all-MiniLM-L6-v2) */
    model?: string;
    /** Whether to normalize embeddings (default: true) */
    normalize?: boolean;
}

export class LocalEncoder extends BaseEncoder {
    readonly name = 'LocalEncoder';
    readonly dimensions: number;

    private model: string;
    private embedder: any = null;
    private initPromise: Promise<void> | null = null;
    private shouldNormalize: boolean;

    // Model -> dimensions mapping
    private static readonly MODEL_DIMENSIONS: Record<string, number> = {
        'Xenova/all-MiniLM-L6-v2': 384,
        'Xenova/all-mpnet-base-v2': 768,
        'Xenova/bge-small-en-v1.5': 384,
        'Xenova/bge-base-en-v1.5': 768,
    };

    constructor(config: LocalEncoderConfig = {}) {
        super();
        this.model = config.model || 'Xenova/all-MiniLM-L6-v2';
        this.dimensions = LocalEncoder.MODEL_DIMENSIONS[this.model] || 384;
        this.shouldNormalize = config.normalize ?? true;
    }

    private async ensureInitialized(): Promise<void> {
        if (this.initialized && this.embedder) return;

        if (this.initPromise) {
            await this.initPromise;
            return;
        }

        this.initPromise = this.initialize();
        await this.initPromise;
    }

    private async initialize(): Promise<void> {
        await loadTransformers();

        console.log(`[LocalEncoder] Loading model: ${this.model}...`);
        const start = Date.now();

        // pipeline is guaranteed to be non-null after loadTransformers()
        this.embedder = await pipeline!('feature-extraction', this.model);
        this.initialized = true;

        console.log(`[LocalEncoder] Model loaded in ${Date.now() - start}ms`);
    }

    async encode(text: string): Promise<number[]> {
        await this.ensureInitialized();

        const result = await this.embedder(text, {
            pooling: 'mean',
            normalize: this.shouldNormalize
        });

        return Array.from(result.data as Float32Array);
    }

    async encodeBatch(texts: string[]): Promise<number[][]> {
        await this.ensureInitialized();

        // Process in parallel for better performance
        const results = await Promise.all(
            texts.map(async (text) => {
                const result = await this.embedder(text, {
                    pooling: 'mean',
                    normalize: this.shouldNormalize
                });
                return Array.from(result.data as Float32Array);
            })
        );

        return results;
    }
}
