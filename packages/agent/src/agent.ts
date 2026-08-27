import type {
	AssistantMessage,
	ImageContent,
	Message,
	Model,
	SimpleStreamOptions,
	TextContent,
	ThinkingBudgets,
	ToolResultMessage,
	Transport,
} from "@earendil-works/pi-ai";
import { runAgentLoop, runAgentLoopContinue } from "./agent-loop.ts";
import { getDefaultStreamFn } from "./stream-fn.ts";
import type {
	AfterToolCallContext,
	AfterToolCallResult,
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentLoopTurnUpdate,
	AgentMessage,
	AgentState,
	AgentTool,
	BeforeToolCallContext,
	BeforeToolCallResult,
	PrepareNextTurnContext,
	QueueMode,
	ShouldStopAfterTurnContext,
	StreamFn,
	ToolExecutionMode,
} from "./types.ts";

export type { QueueMode } from "./types.ts";

function defaultConvertToLlm(messages: AgentMessage[]): Message[] {
	return messages.filter(
		(message) => message.role === "user" || message.role === "assistant" || message.role === "toolResult",
	);
}

/** Reason recorded on the turn and tool results a forced abort closes out. */
const ABORTED_MESSAGE = "Operation aborted";

const EMPTY_USAGE = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

const DEFAULT_MODEL = {
	id: "unknown",
	name: "unknown",
	api: "unknown",
	provider: "unknown",
	baseUrl: "",
	reasoning: false,
	input: [],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 0,
	maxTokens: 0,
} satisfies Model<any>;

type MutableAgentState = Omit<AgentState, "isStreaming" | "streamingMessage" | "pendingToolCalls" | "errorMessage"> & {
	isStreaming: boolean;
	streamingMessage?: AgentMessage;
	pendingToolCalls: Set<string>;
	errorMessage?: string;
};

function createMutableAgentState(
	initialState?: Partial<Omit<AgentState, "pendingToolCalls" | "isStreaming" | "streamingMessage" | "errorMessage">>,
): MutableAgentState {
	let tools = initialState?.tools?.slice() ?? [];
	let messages = initialState?.messages?.slice() ?? [];

	return {
		systemPrompt: initialState?.systemPrompt ?? "",
		model: initialState?.model ?? DEFAULT_MODEL,
		thinkingLevel: initialState?.thinkingLevel ?? "off",
		get tools() {
			return tools;
		},
		set tools(nextTools: AgentTool<any>[]) {
			tools = nextTools.slice();
		},
		get messages() {
			return messages;
		},
		set messages(nextMessages: AgentMessage[]) {
			messages = nextMessages.slice();
		},
		isStreaming: false,
		streamingMessage: undefined,
		pendingToolCalls: new Set<string>(),
		errorMessage: undefined,
	};
}

/** Options for constructing an {@link Agent}. */
export interface AgentOptions {
	initialState?: Partial<Omit<AgentState, "pendingToolCalls" | "isStreaming" | "streamingMessage" | "errorMessage">>;
	convertToLlm?: (messages: AgentMessage[]) => Message[] | Promise<Message[]>;
	transformContext?: (messages: AgentMessage[], signal?: AbortSignal) => Promise<AgentMessage[]>;
	streamFn: StreamFn;
	getApiKey?: (provider: string) => Promise<string | undefined> | string | undefined;
	onPayload?: SimpleStreamOptions["onPayload"];
	onResponse?: SimpleStreamOptions["onResponse"];
	beforeToolCall?: (context: BeforeToolCallContext, signal?: AbortSignal) => Promise<BeforeToolCallResult | undefined>;
	afterToolCall?: (context: AfterToolCallContext, signal?: AbortSignal) => Promise<AfterToolCallResult | undefined>;
	shouldStopAfterTurn?: (context: ShouldStopAfterTurnContext, signal?: AbortSignal) => boolean | Promise<boolean>;
	prepareNextTurn?: (
		signal?: AbortSignal,
	) => Promise<AgentLoopTurnUpdate | undefined> | AgentLoopTurnUpdate | undefined;
	prepareNextTurnWithContext?: (
		context: PrepareNextTurnContext,
		signal?: AbortSignal,
	) => Promise<AgentLoopTurnUpdate | undefined> | AgentLoopTurnUpdate | undefined;
	steeringMode?: QueueMode;
	followUpMode?: QueueMode;
	sessionId?: string;
	thinkingBudgets?: ThinkingBudgets;
	transport?: Transport;
	maxRetryDelayMs?: number;
	toolExecution?: ToolExecutionMode;
}

class PendingMessageQueue {
	private messages: AgentMessage[] = [];
	public mode: QueueMode;

	constructor(mode: QueueMode) {
		this.mode = mode;
	}

	enqueue(message: AgentMessage): void {
		this.messages.push(message);
	}

	hasItems(): boolean {
		return this.messages.length > 0;
	}

	drain(): AgentMessage[] {
		if (this.mode === "all") {
			const drained = this.messages.slice();
			this.messages = [];
			return drained;
		}

		const first = this.messages[0];
		if (!first) {
			return [];
		}
		this.messages = this.messages.slice(1);
		return [first];
	}

	clear(): void {
		this.messages = [];
	}
}

type ActiveRun = {
	promise: Promise<void>;
	resolve: () => void;
	abortController: AbortController;
	/** Resolves when the run is abandoned by a forced abort. */
	abandoned: Promise<void>;
	abandon: () => void;
	isAbandoned: boolean;
	/**
	 * Set by `finishRun`. A closed run has handed control back to the caller, so
	 * nothing belonging to it may touch agent state again -- a listener or
	 * executor still unwinding could otherwise overwrite a later run's state.
	 */
	isClosed: boolean;
	/** The turn that closes this run, once one has been built. */
	closingMessage?: AgentMessage;
	/** Set once a closing turn for this run reached agent state. */
	closingTurnEmitted: boolean;
	/** Set once `handleRunFailure` started, so it cannot run twice for one run. */
	failureHandled: boolean;
};

/**
 * Stateful wrapper around the low-level agent loop.
 *
 * `Agent` owns the current transcript, emits lifecycle events, executes tools,
 * and exposes queueing APIs for steering and follow-up messages.
 */
export class Agent {
	private _state: MutableAgentState;
	private readonly listeners = new Set<(event: AgentEvent, signal: AbortSignal) => Promise<void> | void>();
	private readonly steeringQueue: PendingMessageQueue;
	private readonly followUpQueue: PendingMessageQueue;

	public convertToLlm: (messages: AgentMessage[]) => Message[] | Promise<Message[]>;
	public transformContext?: (messages: AgentMessage[], signal?: AbortSignal) => Promise<AgentMessage[]>;
	public streamFunction: StreamFn;
	public getApiKey?: (provider: string) => Promise<string | undefined> | string | undefined;
	public onPayload?: SimpleStreamOptions["onPayload"];
	public onResponse?: SimpleStreamOptions["onResponse"];
	public beforeToolCall?: (
		context: BeforeToolCallContext,
		signal?: AbortSignal,
	) => Promise<BeforeToolCallResult | undefined>;
	public afterToolCall?: (
		context: AfterToolCallContext,
		signal?: AbortSignal,
	) => Promise<AfterToolCallResult | undefined>;
	public shouldStopAfterTurn?: (
		context: ShouldStopAfterTurnContext,
		signal?: AbortSignal,
	) => boolean | Promise<boolean>;
	public prepareNextTurn?: (
		signal?: AbortSignal,
	) => Promise<AgentLoopTurnUpdate | undefined> | AgentLoopTurnUpdate | undefined;
	public prepareNextTurnWithContext?: (
		context: PrepareNextTurnContext,
		signal?: AbortSignal,
	) => Promise<AgentLoopTurnUpdate | undefined> | AgentLoopTurnUpdate | undefined;
	private activeRun?: ActiveRun;
	/** Session identifier forwarded to providers for cache-aware backends. */
	public sessionId?: string;
	/** Optional per-level thinking token budgets forwarded to the stream function. */
	public thinkingBudgets?: ThinkingBudgets;
	/** Preferred transport forwarded to the stream function. */
	public transport: Transport;
	/** Optional cap for provider-requested retry delays. */
	public maxRetryDelayMs?: number;
	/** Tool execution strategy for assistant messages that contain multiple tool calls. */
	public toolExecution: ToolExecutionMode;

	constructor(options: AgentOptions) {
		// Older compiled consumers may omit options or streamFn even though the current API requires them.
		const runtimeOptions: Partial<AgentOptions> = options ?? {};
		this._state = createMutableAgentState(runtimeOptions.initialState);
		this.convertToLlm = runtimeOptions.convertToLlm ?? defaultConvertToLlm;
		this.transformContext = runtimeOptions.transformContext;
		this.streamFunction = runtimeOptions.streamFn ?? getDefaultStreamFn();
		this.getApiKey = runtimeOptions.getApiKey;
		this.onPayload = runtimeOptions.onPayload;
		this.onResponse = runtimeOptions.onResponse;
		this.beforeToolCall = runtimeOptions.beforeToolCall;
		this.afterToolCall = runtimeOptions.afterToolCall;
		this.shouldStopAfterTurn = runtimeOptions.shouldStopAfterTurn;
		this.prepareNextTurn = runtimeOptions.prepareNextTurn;
		this.prepareNextTurnWithContext = runtimeOptions.prepareNextTurnWithContext;
		this.steeringQueue = new PendingMessageQueue(runtimeOptions.steeringMode ?? "one-at-a-time");
		this.followUpQueue = new PendingMessageQueue(runtimeOptions.followUpMode ?? "one-at-a-time");
		this.sessionId = runtimeOptions.sessionId;
		this.thinkingBudgets = runtimeOptions.thinkingBudgets;
		this.transport = runtimeOptions.transport ?? "auto";
		this.maxRetryDelayMs = runtimeOptions.maxRetryDelayMs;
		this.toolExecution = runtimeOptions.toolExecution ?? "parallel";
	}

	/**
	 * Subscribe to agent lifecycle events.
	 *
	 * Listener promises are awaited in subscription order and are included in
	 * the current run's settlement. Listeners also receive the active abort
	 * signal for the current run.
	 *
	 * `agent_end` is the final emitted event for a run, but the agent does not
	 * become idle until all awaited listeners for that event have settled.
	 */
	subscribe(listener: (event: AgentEvent, signal: AbortSignal) => Promise<void> | void): () => void {
		this.listeners.add(listener);
		return () => this.listeners.delete(listener);
	}

	/**
	 * Current agent state.
	 *
	 * Assigning `state.tools` or `state.messages` copies the provided top-level array.
	 */
	get state(): AgentState {
		return this._state;
	}

	/** Controls how queued steering messages are drained. */
	set steeringMode(mode: QueueMode) {
		this.steeringQueue.mode = mode;
	}

	get steeringMode(): QueueMode {
		return this.steeringQueue.mode;
	}

	/** Controls how queued follow-up messages are drained. */
	set followUpMode(mode: QueueMode) {
		this.followUpQueue.mode = mode;
	}

	get followUpMode(): QueueMode {
		return this.followUpQueue.mode;
	}

	/** Queue a message to be injected after the current assistant turn finishes. */
	steer(message: AgentMessage): void {
		this.steeringQueue.enqueue(message);
	}

	/** Queue a message to run only after the agent would otherwise stop. */
	followUp(message: AgentMessage): void {
		this.followUpQueue.enqueue(message);
	}

	/** Remove all queued steering messages. */
	clearSteeringQueue(): void {
		this.steeringQueue.clear();
	}

	/** Remove all queued follow-up messages. */
	clearFollowUpQueue(): void {
		this.followUpQueue.clear();
	}

	/** Remove all queued steering and follow-up messages. */
	clearAllQueues(): void {
		this.clearSteeringQueue();
		this.clearFollowUpQueue();
	}

	/** Returns true when either queue still contains pending messages. */
	hasQueuedMessages(): boolean {
		return this.steeringQueue.hasItems() || this.followUpQueue.hasItems();
	}

	/** Active abort signal for the current run, if any. */
	get signal(): AbortSignal | undefined {
		return this.activeRun?.abortController.signal;
	}

	/**
	 * Abort the current run, if one is active.
	 *
	 * The default abort signals the run and waits for it to unwind, which only
	 * works while every awaited step observes the run signal. A step that ignores
	 * it (a provider call that never settles, an event listener that never
	 * resolves) would otherwise pin the run forever, leaving the agent streaming
	 * with no way back. Pass `force` to abandon such a run: the agent stops
	 * waiting for the executor, records the closing turn, and is idle by the time
	 * this returns -- `state.isStreaming` is already `false` and `prompt()` is
	 * accepted again.
	 *
	 * The abandoned executor is detached, not cancelled. It keeps running, and a
	 * tool or provider call already in flight still completes; only its effect on
	 * the agent is removed. Its events are dropped, it no longer drains the
	 * steering or follow-up queues, and any error it eventually throws is
	 * discarded because there is no longer a run to report it against.
	 *
	 * Returns the messages the closing turn added to `state.messages`, in order,
	 * so a caller that persists the transcript can write them without waiting for
	 * the listeners -- which are notified out of band and may be exactly what
	 * would not settle. Empty unless a run was actually abandoned.
	 */
	abort(options: { force?: boolean } = {}): AgentMessage[] {
		const run = this.activeRun;
		if (!run) {
			return [];
		}
		run.abortController.abort();
		if (!options.force) {
			return [];
		}
		return this.abandonRun(run);
	}

	/**
	 * Resolve when the current run and all awaited event listeners have finished.
	 *
	 * This resolves after `agent_end` listeners settle.
	 */
	waitForIdle(): Promise<void> {
		return this.activeRun?.promise ?? Promise.resolve();
	}

	/** Clear transcript state, runtime state, and queued messages. */
	reset(): void {
		if (this.activeRun) {
			throw new Error("Agent is already processing. Wait for completion before resetting.");
		}

		this._state.messages = [];
		this._state.isStreaming = false;
		this._state.streamingMessage = undefined;
		this._state.pendingToolCalls = new Set<string>();
		this._state.errorMessage = undefined;
		this.clearFollowUpQueue();
		this.clearSteeringQueue();
	}

	/** Start a new prompt from text, a single message, or a batch of messages. */
	async prompt(message: AgentMessage | AgentMessage[]): Promise<void>;
	async prompt(input: string, images?: ImageContent[]): Promise<void>;
	async prompt(input: string | AgentMessage | AgentMessage[], images?: ImageContent[]): Promise<void> {
		if (this.activeRun) {
			throw new Error(
				"Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion.",
			);
		}
		const messages = this.normalizePromptInput(input, images);
		await this.runPromptMessages(messages);
	}

	/** Continue from the current transcript. The last message must be a user or tool-result message. */
	async continue(): Promise<void> {
		if (this.activeRun) {
			throw new Error("Agent is already processing. Wait for completion before continuing.");
		}

		const lastMessage = this._state.messages[this._state.messages.length - 1];
		if (!lastMessage) {
			throw new Error("No messages to continue from");
		}

		if (lastMessage.role === "assistant") {
			const queuedSteering = this.steeringQueue.drain();
			if (queuedSteering.length > 0) {
				await this.runPromptMessages(queuedSteering, { skipInitialSteeringPoll: true });
				return;
			}

			const queuedFollowUps = this.followUpQueue.drain();
			if (queuedFollowUps.length > 0) {
				await this.runPromptMessages(queuedFollowUps);
				return;
			}

			throw new Error("Cannot continue from message role: assistant");
		}

		await this.runContinuation();
	}

	private normalizePromptInput(
		input: string | AgentMessage | AgentMessage[],
		images?: ImageContent[],
	): AgentMessage[] {
		if (Array.isArray(input)) {
			return input;
		}

		if (typeof input !== "string") {
			return [input];
		}

		const content: Array<TextContent | ImageContent> = [{ type: "text", text: input }];
		if (images && images.length > 0) {
			content.push(...images);
		}
		return [{ role: "user", content, timestamp: Date.now() }];
	}

	private async runPromptMessages(
		messages: AgentMessage[],
		options: { skipInitialSteeringPoll?: boolean } = {},
	): Promise<void> {
		await this.runWithLifecycle(async (signal, run) => {
			await runAgentLoop(
				messages,
				this.createContextSnapshot(),
				this.createLoopConfig(run, options),
				(event) => this.processEvents(event, run),
				signal,
				this.streamFunction,
			);
		});
	}

	private async runContinuation(): Promise<void> {
		await this.runWithLifecycle(async (signal, run) => {
			await runAgentLoopContinue(
				this.createContextSnapshot(),
				this.createLoopConfig(run),
				(event) => this.processEvents(event, run),
				signal,
				this.streamFunction,
			);
		});
	}

	private createContextSnapshot(): AgentContext {
		return {
			systemPrompt: this._state.systemPrompt,
			messages: this._state.messages.slice(),
			tools: this._state.tools.slice(),
		};
	}

	private createLoopConfig(run: ActiveRun, options: { skipInitialSteeringPoll?: boolean } = {}): AgentLoopConfig {
		let skipInitialSteeringPoll = options.skipInitialSteeringPoll === true;
		const shouldStopAfterTurn = this.shouldStopAfterTurn;
		// Every hook is scoped to `run`, not to whatever run is active when it fires.
		// An abandoned executor keeps running, so an unscoped hook would hand it the
		// next run's abort signal and let it drain queues the next run needs.
		const signal = run.abortController.signal;
		return {
			model: this._state.model,
			reasoning: this._state.thinkingLevel === "off" ? undefined : this._state.thinkingLevel,
			sessionId: this.sessionId,
			onPayload: this.onPayload,
			onResponse: this.onResponse,
			transport: this.transport,
			thinkingBudgets: this.thinkingBudgets,
			maxRetryDelayMs: this.maxRetryDelayMs,
			toolExecution: this.toolExecution,
			beforeToolCall: this.beforeToolCall,
			afterToolCall: this.afterToolCall,
			// Always supplied, even with no caller hook: this is the loop's only
			// per-turn exit, and an abandoned executor whose model keeps requesting
			// tools would otherwise run turns -- and spend tokens -- forever. Not
			// draining the queues is not enough; a tool-use loop never needs them.
			shouldStopAfterTurn: async (context) => {
				if (run.isAbandoned) {
					return true;
				}
				return shouldStopAfterTurn ? await shouldStopAfterTurn(context, signal) : false;
			},
			prepareNextTurn:
				this.prepareNextTurnWithContext || this.prepareNextTurn
					? async (context) => {
							if (this.prepareNextTurnWithContext) {
								return await this.prepareNextTurnWithContext(context, signal);
							}
							return await this.prepareNextTurn?.(signal);
						}
					: undefined,
			convertToLlm: this.convertToLlm,
			transformContext: this.transformContext,
			getApiKey: this.getApiKey,
			getSteeringMessages: async () => {
				if (skipInitialSteeringPoll || run.isAbandoned) {
					skipInitialSteeringPoll = false;
					return [];
				}
				return this.steeringQueue.drain();
			},
			getFollowUpMessages: async () => (run.isAbandoned ? [] : this.followUpQueue.drain()),
		};
	}

	private async runWithLifecycle(executor: (signal: AbortSignal, run: ActiveRun) => Promise<void>): Promise<void> {
		if (this.activeRun) {
			throw new Error("Agent is already processing.");
		}

		const abortController = new AbortController();
		let resolvePromise = () => {};
		const promise = new Promise<void>((resolve) => {
			resolvePromise = resolve;
		});
		let abandonRun = () => {};
		const abandoned = new Promise<void>((resolve) => {
			abandonRun = resolve;
		});
		const run: ActiveRun = {
			promise,
			resolve: resolvePromise,
			abortController,
			abandoned,
			abandon: abandonRun,
			isAbandoned: false,
			isClosed: false,
			closingTurnEmitted: false,
			failureHandled: false,
		};
		this.activeRun = run;

		this._state.isStreaming = true;
		this._state.streamingMessage = undefined;
		this._state.errorMessage = undefined;

		try {
			// A forced abort settles the run without the executor, which may never
			// return. Losing the race leaves the executor detached, not cancelled.
			await Promise.race([executor(abortController.signal, run), run.abandoned]);
			if (run.isAbandoned) {
				return;
			}
		} catch (error) {
			// The race stays subscribed to the executor, so a detached executor that
			// throws later is caught here rather than becoming an unhandled
			// rejection. It is then deliberately discarded: its run is already closed
			// and reported, and there is nothing left that the failure could change.
			if (run.isAbandoned) {
				return;
			}
			await this.handleRunFailure(error, abortController.signal.aborted, run);
		} finally {
			// An abandoned run was already finished by `abandonRun`; `finishRun` is
			// keyed on run identity, so this is a no-op rather than a second release.
			this.finishRun(run);
		}
	}

	/**
	 * Settle a run whose executor cannot be relied on to unwind.
	 *
	 * The closing turn is reduced into agent state synchronously, so the
	 * transcript is complete and the agent is idle the moment this returns.
	 * Listeners are notified only afterwards, detached, because a listener is one
	 * of the two things that can pin a run in the first place: waiting for one
	 * here would reintroduce the hang this method exists to escape. That is safe
	 * only because state is already final -- `deliverEvent` drops events for a
	 * closed run, so nothing still unwinding can write to a later run's state.
	 */
	private abandonRun(run: ActiveRun): AgentMessage[] {
		if (run.isAbandoned || this.activeRun !== run) {
			return [];
		}
		run.isAbandoned = true;
		const closingEvents = this.buildAbandonedRunEvents(run);
		// Queue policy belongs to the caller, not here. The agent has no post-run
		// loop, and the queued text is the user's input: a caller that force-aborts
		// for its own reasons has no reason to want it destroyed, and only it knows
		// where to put it back.
		const recorded: AgentMessage[] = [];
		for (const event of closingEvents) {
			this.reduceEvent(event, run);
			if (event.type === "message_end") {
				recorded.push(event.message);
			}
		}
		this.finishRun(run);
		run.abandon();
		void this.notifyAbandonedRun(closingEvents, run);
		return recorded;
	}

	/**
	 * Hand the closing events of an abandoned run to listeners, out of band.
	 *
	 * Best effort by construction. Each listener gets its own chain, so one that
	 * never resolves starves only itself, and errors are discarded because there
	 * is no longer a run to fail. Delivery also stops once another run starts: a
	 * listener cannot tell which run an event came from, so a late `agent_end`
	 * arriving mid-stream would look to the UI like the new run had ended.
	 *
	 * Nothing that must not be lost may depend on this dispatch. Callers that need
	 * the closing turn (session persistence) read the final state directly, which
	 * `abandonRun` has already written by the time this starts.
	 */
	private async notifyAbandonedRun(events: AgentEvent[], run: ActiveRun): Promise<void> {
		const signal = run.abortController.signal;
		await Promise.allSettled(
			[...this.listeners].map(async (listener) => {
				// Yield first, so nothing here runs inside the `abort()` call itself.
				await Promise.resolve();
				// All or nothing. Checked once, before the first event rather than
				// before each: a listener that received `message_start` and then no
				// `message_end` is left holding an open message forever, which is worse
				// than never hearing about the turn at all.
				if (this.activeRun !== undefined) {
					return;
				}
				for (const event of events) {
					await listener(event, signal);
				}
			}),
		);
	}

	/**
	 * The events that close out an abandoned run.
	 *
	 * Mirrors how the loop unwinds cooperatively: an aborted result for every tool
	 * call still in flight, then the assistant turn. The tool results are not
	 * cosmetic. Without them the transcript keeps an assistant tool call that
	 * nothing answers, and `transform-messages` papers over the gap with a
	 * synthesized "No result provided" -- telling the model the tool did nothing
	 * while its side effects actually happened.
	 */
	private buildAbandonedRunEvents(run: ActiveRun): AgentEvent[] {
		const events: AgentEvent[] = [];
		const toolResults: ToolResultMessage[] = [];
		for (const { id: toolCallId, name: toolName } of this.findUnansweredToolCalls()) {
			const content = [{ type: "text", text: ABORTED_MESSAGE }] satisfies TextContent[];
			const toolResultMessage = {
				role: "toolResult",
				toolCallId,
				toolName,
				content,
				details: {},
				isError: true,
				timestamp: Date.now(),
			} satisfies ToolResultMessage;
			toolResults.push(toolResultMessage);
			events.push({
				type: "tool_execution_end",
				toolCallId,
				toolName,
				result: { content, details: {} },
				isError: true,
			});
			events.push({ type: "message_start", message: toolResultMessage });
			events.push({ type: "message_end", message: toolResultMessage });
		}

		// A run that already recorded its closing turn -- it finished, or failed, and
		// is pinned only in a listener -- must not be given a second one.
		if (run.closingTurnEmitted) {
			return events;
		}

		const closingMessage = this.buildAbandonedTurnMessage();
		run.closingMessage = closingMessage;
		events.push({ type: "message_start", message: closingMessage });
		events.push({ type: "message_end", message: closingMessage });
		// No tool results on this turn: it requests nothing. Everywhere else
		// `turn_end.toolResults` answers the calls in `turn_end.message`, so a
		// consumer that pairs them would misattribute the previous turn's results.
		// They are already reported by their own `message_end` above.
		events.push({ type: "turn_end", message: closingMessage, toolResults: [] });
		events.push({ type: "agent_end", messages: [closingMessage] });
		return events;
	}

	/**
	 * Tool calls in the transcript that nothing has answered.
	 *
	 * Derived from the transcript rather than from `state.pendingToolCalls`,
	 * because that set is emptied when `tool_execution_end` is reduced -- before
	 * the listeners for it are awaited. A run pinned in exactly that listener,
	 * which is one of the two ways a run becomes unstoppable, would otherwise be
	 * abandoned with the id already gone from the set and the real result not yet
	 * pushed, leaving the call unanswered by both paths and the model later told
	 * the tool produced nothing when it had in fact run.
	 */
	private findUnansweredToolCalls(): Array<{ id: string; name: string }> {
		const answered = new Set<string>();
		for (const message of this._state.messages) {
			if (message.role === "toolResult") {
				answered.add(message.toolCallId);
			}
		}
		const unanswered: Array<{ id: string; name: string }> = [];
		const seen = new Set<string>();
		for (const message of this._state.messages) {
			if (message.role !== "assistant") {
				continue;
			}
			for (const block of message.content) {
				if (block.type === "toolCall" && !answered.has(block.id) && !seen.has(block.id)) {
					seen.add(block.id);
					unanswered.push({ id: block.id, name: block.name });
				}
			}
		}
		return unanswered;
	}

	/**
	 * The assistant turn recorded for an abandoned run.
	 *
	 * Built from the partial message the provider had already streamed, so the
	 * text the user watched arrive is kept along with its usage and the model that
	 * actually produced it; only `stopReason` is overridden. `errorMessage` is
	 * left unset because an interrupt is not a failure, and `turn_end` copies
	 * `errorMessage` into `state.errorMessage` -- setting it would leave the agent
	 * reporting an error for something the user asked for.
	 */
	private buildAbandonedTurnMessage(): AssistantMessage {
		const partial = this._state.streamingMessage;
		if (partial?.role === "assistant") {
			return {
				...partial,
				// A tool call whose arguments were still streaming is truncated and was
				// never executed. Keeping it records a call nothing answers and shows
				// the user half-parsed arguments; the cooperative path fails such calls
				// explicitly (`failToolCallsFromTruncatedMessage`).
				content: partial.content.filter((block) => block.type !== "toolCall"),
				stopReason: "aborted",
				errorMessage: undefined,
			};
		}
		return {
			role: "assistant",
			content: [],
			api: this._state.model.api,
			provider: this._state.model.provider,
			model: this._state.model.id,
			usage: EMPTY_USAGE,
			stopReason: "aborted",
			timestamp: Date.now(),
		};
	}

	private async handleRunFailure(error: unknown, aborted: boolean, run: ActiveRun): Promise<void> {
		// Re-entrant only if a forced abort lands while this is suspended in a
		// listener; a second closing turn for one run would corrupt the transcript.
		if (run.failureHandled || run.closingTurnEmitted) {
			return;
		}
		run.failureHandled = true;
		const failureMessage = {
			role: "assistant",
			content: [{ type: "text", text: "" }],
			api: this._state.model.api,
			provider: this._state.model.provider,
			model: this._state.model.id,
			usage: EMPTY_USAGE,
			stopReason: aborted ? "aborted" : "error",
			errorMessage: error instanceof Error ? error.message : String(error),
			timestamp: Date.now(),
		} satisfies AgentMessage;
		run.closingMessage = failureMessage;
		// Deliver directly: this closing turn is the run's own, not a loop event.
		await this.deliverEvent({ type: "message_start", message: failureMessage }, run);
		await this.deliverEvent({ type: "message_end", message: failureMessage }, run);
		await this.deliverEvent({ type: "turn_end", message: failureMessage, toolResults: [] }, run);
		await this.deliverEvent({ type: "agent_end", messages: [failureMessage] }, run);
	}

	private finishRun(run: ActiveRun): void {
		if (this.activeRun !== run) {
			return;
		}
		run.isClosed = true;
		this._state.isStreaming = false;
		this._state.streamingMessage = undefined;
		this._state.pendingToolCalls = new Set<string>();
		this.activeRun = undefined;
		run.resolve();
	}

	/** Loop events for a run that a forced abort already closed are dropped. */
	private async processEvents(event: AgentEvent, run: ActiveRun): Promise<void> {
		if (run.isAbandoned) {
			return;
		}
		await this.deliverEvent(event, run);
	}

	/**
	 * Reduce internal state for an event, then await listeners.
	 *
	 * `agent_end` only means no further loop events will be emitted. The run is
	 * considered idle later, after all awaited listeners for `agent_end` finish
	 * and `finishRun()` clears runtime-owned state.
	 *
	 * Events belonging to a closed run are dropped. Both the abandonment path and
	 * a cooperative failure suspended in a slow listener can still be mid-chain
	 * when the run is released, and a late reduction would land on whichever run
	 * is active by then.
	 */
	private async deliverEvent(event: AgentEvent, run: ActiveRun): Promise<void> {
		if (run.isClosed) {
			return;
		}
		this.reduceEvent(event, run);
		for (const listener of this.listeners) {
			// Re-checked per listener, not just on entry: a listener can force-abort
			// re-entrantly, and continuing would deliver a pre-abort event to the
			// remaining listeners after the run closed and interleave it into the
			// closing chain.
			if (run.isClosed) {
				return;
			}
			await listener(event, run.abortController.signal);
		}
	}

	/** Apply an event to agent state. Synchronous, so it cannot interleave. */
	private reduceEvent(event: AgentEvent, run: ActiveRun): void {
		switch (event.type) {
			case "message_start":
				this._state.streamingMessage = event.message;
				break;

			case "message_update":
				this._state.streamingMessage = event.message;
				break;

			case "message_end":
				this._state.streamingMessage = undefined;
				this._state.messages.push(event.message);
				if (event.message === run.closingMessage) {
					run.closingTurnEmitted = true;
				}
				break;

			case "tool_execution_start": {
				const pendingToolCalls = new Set(this._state.pendingToolCalls);
				pendingToolCalls.add(event.toolCallId);
				this._state.pendingToolCalls = pendingToolCalls;
				break;
			}

			case "tool_execution_end": {
				const pendingToolCalls = new Set(this._state.pendingToolCalls);
				pendingToolCalls.delete(event.toolCallId);
				this._state.pendingToolCalls = pendingToolCalls;
				break;
			}

			case "turn_end":
				if (event.message.role === "assistant" && event.message.errorMessage) {
					this._state.errorMessage = event.message.errorMessage;
				}
				break;

			case "agent_end":
				this._state.streamingMessage = undefined;
				run.closingTurnEmitted = true;
				break;
		}
	}
}
