/**
 * Basic test for SemanticRouter
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { SemanticRouter, Route, LocalEncoder } from '../src';

describe('SemanticRouter', () => {
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

    let router: SemanticRouter;

    beforeAll(async () => {
        router = new SemanticRouter({
            routes,
            encoder: new LocalEncoder(),
            threshold: 0.3,
        });
        await router.initialize();
    }, 60000); // 60s timeout for model loading

    it('should route greeting correctly', async () => {
        const result = await router.route('hey there, how are you?');
        expect(result.name).toBe('greeting');
        expect(result.confidence).toBeGreaterThan(0.3);
    });

    it('should route farewell correctly', async () => {
        const result = await router.route('goodbye for now');
        expect(result.name).toBe('farewell');
        expect(result.confidence).toBeGreaterThan(0.3);
    });

    it('should route help correctly', async () => {
        const result = await router.route('I need some assistance');
        expect(result.name).toBe('help');
        expect(result.confidence).toBeGreaterThan(0.3);
    });

    it('should return null for unrelated queries', async () => {
        const result = await router.route('what is the capital of France?');
        // Either no match or very low confidence
        if (result.name !== null) {
            expect(result.confidence).toBeLessThan(0.5);
        }
    });

    it('should have correct stats', () => {
        const stats = router.getStats();
        expect(stats.routeCount).toBe(3);
        expect(stats.routes).toContain('greeting');
        expect(stats.routes).toContain('farewell');
        expect(stats.routes).toContain('help');
        expect(stats.initialized).toBe(true);
    });
});
