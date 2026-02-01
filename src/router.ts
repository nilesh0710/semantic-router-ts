/**
 * SemanticRouter - Fast decision-making layer for LLMs
 * 
 * Port of Python semantic_router.routers.semantic
 * 
 * Uses semantic vector space to route queries to the best matching route
 * based on similarity to pre-encoded utterances.
 */

import type {
    Route,
    RouteMatch,
    RouterConfig,
    Encoder,
    Index,
    IndexMatch,
    LLM
} from './types';
import { LocalEncoder } from './encoders';
import { LocalIndex } from './index/local';

export interface SemanticRouterConfig extends RouterConfig {
    /** Optional LLM for fallback classification */
    llm?: LLM;
}

export class SemanticRouter {
    private routes: Route[] = [];
    private encoder: Encoder;
    private index: Index;
    private llm?: LLM;

    // Configuration
    private topK: number;
    private aggregation: 'mean' | 'max' | 'sum';
    private threshold: number;

    // State
    private initialized = false;
    private initPromise: Promise<void> | null = null;

    constructor(config: SemanticRouterConfig = {}) {
        this.routes = config.routes || [];
        this.encoder = config.encoder || new LocalEncoder();
        this.index = new LocalIndex();
        this.llm = config.llm;

        this.topK = config.topK ?? 5;
        this.aggregation = config.aggregation ?? 'mean';
        this.threshold = config.threshold ?? 0.4;
    }

    /**
     * Initialize the router by encoding all routes.
     * Safe to call multiple times.
     */
    async initialize(): Promise<void> {
        if (this.initialized) return;

        if (this.initPromise) {
            await this.initPromise;
            return;
        }

        this.initPromise = this.doInitialize();
        await this.initPromise;
    }

    private async doInitialize(): Promise<void> {
        if (this.routes.length === 0) {
            console.warn('[SemanticRouter] No routes configured');
            this.initialized = true;
            return;
        }

        console.log(`[SemanticRouter] Encoding ${this.routes.length} routes...`);
        const start = Date.now();

        // Collect all utterances with their route names
        const allUtterances: string[] = [];
        const allRouteNames: string[] = [];

        for (const route of this.routes) {
            for (const utterance of route.utterances) {
                allUtterances.push(utterance);
                allRouteNames.push(route.name);
            }
        }

        // Batch encode all utterances
        const embeddings = await this.encoder.encodeBatch(allUtterances);

        // Add to index
        await this.index.add(embeddings, allRouteNames, allUtterances);

        this.initialized = true;
        console.log(
            `[SemanticRouter] Initialized in ${Date.now() - start}ms ` +
            `(${allUtterances.length} utterances from ${this.routes.length} routes)`
        );
    }

    /**
     * Route a query to the best matching route.
     */
    async route(query: string): Promise<RouteMatch> {
        if (!this.initialized) {
            await this.initialize();
        }

        // Encode the query
        const queryEmbedding = await this.encoder.encode(query);

        // Query the index
        const matches = await this.index.query(queryEmbedding, this.topK);

        if (matches.length === 0) {
            return this.noMatch();
        }

        // Aggregate scores by route
        const routeScores = this.aggregateScores(matches);

        // Sort by score descending
        routeScores.sort((a, b) => b.score - a.score);

        const topMatch = routeScores[0];

        // Check threshold
        if (topMatch.score < this.threshold) {
            // Try LLM fallback if available
            if (this.llm) {
                return await this.llmFallback(query, routeScores);
            }
            return this.noMatch(routeScores);
        }

        // Find the full route object
        const matchedRoute = this.routes.find(r => r.name === topMatch.name);

        return {
            route: matchedRoute || null,
            name: topMatch.name,
            confidence: topMatch.score,
            scores: routeScores,
        };
    }

    /**
     * Shorthand to get just the route name
     */
    async classify(query: string): Promise<string | null> {
        const result = await this.route(query);
        return result.name;
    }

    /**
     * Add a route dynamically
     */
    async addRoute(route: Route): Promise<void> {
        // Remove existing route with same name
        this.routes = this.routes.filter(r => r.name !== route.name);
        this.routes.push(route);

        // Re-encode this route's utterances
        const embeddings = await this.encoder.encodeBatch(route.utterances);
        const routeNames = route.utterances.map(() => route.name);

        await this.index.add(embeddings, routeNames, route.utterances);

        console.log(`[SemanticRouter] Added route: ${route.name} (${route.utterances.length} utterances)`);
    }

    /**
     * Aggregate scores from multiple matches by route
     */
    private aggregateScores(matches: IndexMatch[]): { name: string; score: number }[] {
        const scoresByRoute = new Map<string, number[]>();

        for (const match of matches) {
            const scores = scoresByRoute.get(match.route) || [];
            scores.push(match.score);
            scoresByRoute.set(match.route, scores);
        }

        const aggregated: { name: string; score: number }[] = [];

        for (const [name, scores] of scoresByRoute) {
            let score: number;

            switch (this.aggregation) {
                case 'max':
                    score = Math.max(...scores);
                    break;
                case 'sum':
                    score = scores.reduce((a, b) => a + b, 0);
                    break;
                case 'mean':
                default:
                    score = scores.reduce((a, b) => a + b, 0) / scores.length;
                    break;
            }

            aggregated.push({ name, score });
        }

        return aggregated;
    }

    /**
     * Use LLM to classify when similarity is low
     */
    private async llmFallback(
        query: string,
        scores: { name: string; score: number }[]
    ): Promise<RouteMatch> {
        if (!this.llm) {
            return this.noMatch(scores);
        }

        console.log('[SemanticRouter] Using LLM fallback for low-confidence match');

        const routeDescriptions = this.routes
            .map(r => `- ${r.name}: ${r.description || r.utterances[0]}`)
            .join('\n');

        const prompt = `Classify the user's intent into exactly ONE of these routes:

${routeDescriptions}

User query: "${query}"

Respond with ONLY the route name, nothing else.`;

        try {
            const response = await this.llm.generate(prompt);
            const routeName = response.trim().toLowerCase();

            // Find matching route (case-insensitive)
            const matchedRoute = this.routes.find(
                r => r.name.toLowerCase() === routeName
            );

            if (matchedRoute) {
                console.log(`[SemanticRouter] LLM classified as: ${matchedRoute.name}`);
                return {
                    route: matchedRoute,
                    name: matchedRoute.name,
                    confidence: 0.7, // LLM confidence estimate
                    scores,
                };
            }
        } catch (error) {
            console.warn('[SemanticRouter] LLM fallback failed:', error);
        }

        return this.noMatch(scores);
    }

    /**
     * Return a no-match result
     */
    private noMatch(scores?: { name: string; score: number }[]): RouteMatch {
        return {
            route: null,
            name: null,
            confidence: 0,
            scores,
        };
    }

    /**
     * Get router statistics
     */
    getStats(): {
        routeCount: number;
        routes: string[];
        encoder: string;
        threshold: number;
        initialized: boolean;
    } {
        return {
            routeCount: this.routes.length,
            routes: this.routes.map(r => r.name),
            encoder: this.encoder.name,
            threshold: this.threshold,
            initialized: this.initialized,
        };
    }

    /**
     * Check if router is ready
     */
    isReady(): boolean {
        return this.initialized;
    }
}
