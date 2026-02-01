/**
 * OpenAIEncoder - Embedding using OpenAI API
 * 
 * Uses text-embedding-3-small by default (1536 dimensions).
 */

import { BaseEncoder } from './base';

// Dynamic import types
type OpenAIClient = {
    embeddings: {
        create(params: { model: string; input: string[]; dimensions?: number }): Promise<{
            data: Array<{ index: number; embedding: number[] }>;
        }>;
    };
};

let OpenAIClass: (new (config: { apiKey: string }) => OpenAIClient) | null = null;

async function loadOpenAI(): Promise<void> {
    if (OpenAIClass) return;

    try {
        // Dynamic import to avoid requiring as hard dependency
        const openaiModule = await import('openai' as string);
        OpenAIClass = openaiModule.default || openaiModule.OpenAI;
    } catch {
        throw new Error(
            'OpenAIEncoder requires openai package. ' +
            'Install it with: npm install openai'
        );
    }
}

export interface OpenAIEncoderConfig {
    /** OpenAI API key */
    apiKey?: string;
    /** Model to use (default: text-embedding-3-small) */
    model?: string;
    /** Embedding dimensions (for ada-3 models that support it) */
    dimensions?: number;
}

export class OpenAIEncoder extends BaseEncoder {
    readonly name = 'OpenAIEncoder';
    readonly dimensions: number;

    private model: string;
    private apiKey: string;
    private client: any = null;

    // Model -> default dimensions mapping
    private static readonly MODEL_DIMENSIONS: Record<string, number> = {
        'text-embedding-3-small': 1536,
        'text-embedding-3-large': 3072,
        'text-embedding-ada-002': 1536,
    };

    constructor(config: OpenAIEncoderConfig = {}) {
        super();
        this.model = config.model || 'text-embedding-3-small';
        this.dimensions = config.dimensions ||
            OpenAIEncoder.MODEL_DIMENSIONS[this.model] || 1536;
        this.apiKey = config.apiKey || process.env.OPENAI_API_KEY || '';

        if (!this.apiKey) {
            console.warn('[OpenAIEncoder] No API key provided. Set OPENAI_API_KEY env var or pass apiKey config.');
        }
    }

    private async ensureClient(): Promise<void> {
        if (this.client) return;

        await loadOpenAI();
        // OpenAIClass is guaranteed to be non-null after loadOpenAI()
        this.client = new OpenAIClass!({ apiKey: this.apiKey });
        this.initialized = true;
    }

    async encode(text: string): Promise<number[]> {
        const results = await this.encodeBatch([text]);
        return results[0];
    }

    async encodeBatch(texts: string[]): Promise<number[][]> {
        await this.ensureClient();

        const response = await this.client.embeddings.create({
            model: this.model,
            input: texts,
            ...(this.model.includes('3-') && { dimensions: this.dimensions }),
        });

        // Sort by index to ensure correct order
        const sorted = response.data.sort((a: any, b: any) => a.index - b.index);
        return sorted.map((item: any) => item.embedding);
    }
}
