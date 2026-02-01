/**
 * @anthropic/semantic-router
 * 
 * Superfast semantic routing for LLMs and AI agents.
 * TypeScript port of aurelio-labs/semantic-router.
 * 
 * @example
 * ```typescript
 * import { SemanticRouter, Route, LocalEncoder } from '@anthropic/semantic-router';
 * 
 * const routes: Route[] = [
 *   { name: 'greeting', utterances: ['hello', 'hi there', 'hey'] },
 *   { name: 'farewell', utterances: ['goodbye', 'bye', 'see you'] },
 * ];
 * 
 * const router = new SemanticRouter({
 *   routes,
 *   encoder: new LocalEncoder(),
 * });
 * 
 * await router.initialize();
 * 
 * const result = await router.route('hey, how are you?');
 * console.log(result.name); // 'greeting'
 * console.log(result.confidence); // 0.87
 * ```
 */

// Core router
export { SemanticRouter, type SemanticRouterConfig } from './router';

// Types
export type {
    Route,
    RouteMatch,
    RouteScore,
    RouterConfig,
    Encoder,
    Index,
    IndexMatch,
    LLM,
    FunctionSchema,
} from './types';

// Encoders
export {
    BaseEncoder,
    LocalEncoder,
    OpenAIEncoder,
    type LocalEncoderConfig,
    type OpenAIEncoderConfig,
} from './encoders';

// Index implementations
export { LocalIndex } from './index/local';
