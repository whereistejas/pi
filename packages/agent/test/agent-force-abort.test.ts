import { type AssistantMessage, type AssistantMessageEvent, EventStream, getModel } from "@earendil-works/pi-ai/compat";
import { describe, expect, it } from "vitest";
import { Agent, type AgentEvent, type AgentMessage, type StreamFn } from "../src/index.ts";

const EMPTY_USAGE = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function assistantText(message: AgentMessage | undefined): string {
	if (message?.role !== "assistant") {
		return "";
	}
	return message.content
		.filter((block) => block.type === "text")
		.map((block) => block.text)
		.join("");
}

function describeMessages(messages: AgentMessage[]): string[] {
	return messages.map((message) =>
		message.role === "assistant" ? `assistant(${message.stopReason})` : String(message.role),
	);
}

class MockAssistantStream extends EventStream<AssistantMessageEvent, AssistantMessage> {
	constructor() {
		super(
			(event) => event.type === "done" || event.type === "error",
			(event) => {
				if (event.type === "done") return event.message;
				if (event.type === "error") return event.error;
				throw new Error("Unexpected event type");
			},
		);
	}
}

/** Stream that never emits and never reacts to the abort signal. */
const unresponsiveStreamFn: StreamFn = () => new MockAssistantStream();

function createAgent(): Agent {
	return new Agent({
		initialState: { model: getModel("openai", "gpt-5"), systemPrompt: "test", tools: [] },
		streamFn: unresponsiveStreamFn,
	});
}

describe("Agent forced abort", () => {
	it("keeps the run active when a signal-ignoring stream is aborted cooperatively", async () => {
		const agent = createAgent();
		const prompt = agent.prompt("hello");
		let settled = false;
		void prompt.then(() => {
			settled = true;
		});

		agent.abort();
		await new Promise((resolve) => setTimeout(resolve, 50));

		expect(settled).toBe(false);
		expect(agent.state.isStreaming).toBe(true);
	});

	it("settles the run, records an aborted turn, and stays usable after a forced abort", async () => {
		const agent = createAgent();
		const events: AgentEvent[] = [];
		agent.subscribe((event) => {
			events.push(event);
		});

		const prompt = agent.prompt("hello");
		await new Promise((resolve) => setTimeout(resolve, 10));

		agent.abort({ force: true });
		await prompt;

		expect(agent.state.isStreaming).toBe(false);
		expect(events.filter((event) => event.type === "agent_end")).toHaveLength(1);
		const last = agent.state.messages[agent.state.messages.length - 1];
		expect(last?.role).toBe("assistant");
		expect(last?.role === "assistant" ? last.stopReason : undefined).toBe("aborted");

		// The agent must accept work again once the abandoned run is released.
		agent.streamFunction = () => {
			const stream = new MockAssistantStream();
			const message: AssistantMessage = {
				role: "assistant",
				content: [{ type: "text", text: "second" }],
				api: "openai-responses",
				provider: "openai",
				model: "mock",
				usage: {
					input: 0,
					output: 0,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "stop",
				timestamp: Date.now(),
			};
			stream.push({ type: "start", partial: message });
			stream.push({ type: "done", reason: "stop", message });
			stream.end(message);
			return stream;
		};
		await agent.prompt("again");

		const final = agent.state.messages[agent.state.messages.length - 1];
		expect(final?.role === "assistant" ? final.stopReason : undefined).toBe("stop");
	});

	it("drops events emitted by an abandoned run", async () => {
		const stream = new MockAssistantStream();
		const agent = new Agent({
			initialState: { model: getModel("openai", "gpt-5"), systemPrompt: "test", tools: [] },
			streamFn: () => stream,
		});
		const events: AgentEvent[] = [];
		agent.subscribe((event) => {
			events.push(event);
		});

		const prompt = agent.prompt("hello");
		await new Promise((resolve) => setTimeout(resolve, 10));
		agent.abort({ force: true });
		await prompt;

		const countAfterAbort = events.length;
		const late: AssistantMessage = {
			role: "assistant",
			content: [{ type: "text", text: "late" }],
			api: "openai-responses",
			provider: "openai",
			model: "mock",
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: Date.now(),
		};
		stream.push({ type: "done", reason: "stop", message: late });
		stream.end(late);
		await new Promise((resolve) => setTimeout(resolve, 20));

		expect(events).toHaveLength(countAfterAbort);
		// Assert the negative directly: the late message must not reach the transcript.
		expect(agent.state.messages.map((message) => assistantText(message))).not.toContain("late");
	});

	it("settles and records one closing turn when a listener never resolves", async () => {
		const agent = createAgent();
		const events: AgentEvent[] = [];
		let releaseListener = () => {};
		const listenerReleased = new Promise<void>((resolve) => {
			releaseListener = resolve;
		});
		agent.subscribe(async (event) => {
			events.push(event);
			if (event.type === "turn_end") {
				await listenerReleased;
			}
		});

		const prompt = agent.prompt("hello");
		await new Promise((resolve) => setTimeout(resolve, 10));

		const startedAt = Date.now();
		agent.abort({ force: true });
		await prompt;

		// The hung listener must not delay the run: no grace window to wait out.
		expect(Date.now() - startedAt).toBeLessThan(200);
		expect(agent.state.isStreaming).toBe(false);
		const messagesAtIdle = describeMessages(agent.state.messages);
		expect(messagesAtIdle).toEqual(["user", "assistant(aborted)"]);

		// Releasing the listener afterwards must not mutate state, and must not
		// produce more than the single closing turn the run is entitled to.
		releaseListener();
		await new Promise((resolve) => setTimeout(resolve, 30));

		expect(describeMessages(agent.state.messages)).toEqual(messagesAtIdle);
		expect(events.filter((event) => event.type === "agent_end")).toHaveLength(1);
		expect(agent.state.isStreaming).toBe(false);
	});

	it("does not mutate a later run when an abandoned run's listener resolves", async () => {
		const agent = createAgent();
		let releaseListener = () => {};
		const listenerReleased = new Promise<void>((resolve) => {
			releaseListener = resolve;
		});
		let hangOnce = true;
		agent.subscribe(async (event) => {
			if (event.type === "turn_end" && hangOnce) {
				hangOnce = false;
				await listenerReleased;
			}
		});

		const first = agent.prompt("hello");
		await new Promise((resolve) => setTimeout(resolve, 10));
		agent.abort({ force: true });
		await first;

		// Start a second run that is still streaming when the old listener wakes.
		const second = agent.prompt("again");
		await new Promise((resolve) => setTimeout(resolve, 10));
		expect(agent.state.isStreaming).toBe(true);
		const messagesDuringSecondRun = describeMessages(agent.state.messages);

		releaseListener();
		await new Promise((resolve) => setTimeout(resolve, 30));

		// The abandoned run must not append to, or close, the live run.
		expect(describeMessages(agent.state.messages)).toEqual(messagesDuringSecondRun);
		expect(agent.state.isStreaming).toBe(true);

		agent.abort({ force: true });
		await second;
	});

	it("is idle as soon as a forced abort returns", async () => {
		const agent = createAgent();
		const prompt = agent.prompt("hello");
		await new Promise((resolve) => setTimeout(resolve, 10));

		agent.abort({ force: true });

		// No awaits in between: the caller must not observe a half-closed run.
		expect(agent.state.isStreaming).toBe(false);
		expect(agent.signal).toBeUndefined();
		await prompt;

		// `prompt` is async, so a rejection is the failure mode here, not a throw.
		// Accepting a new run is the property that matters; assert it by running one.
		agent.streamFunction = () => {
			const stream = new MockAssistantStream();
			const message: AssistantMessage = {
				role: "assistant",
				content: [{ type: "text", text: "accepted" }],
				api: "openai-responses",
				provider: "openai",
				model: "mock",
				usage: EMPTY_USAGE,
				stopReason: "stop",
				timestamp: Date.now(),
			};
			stream.push({ type: "start", partial: message });
			stream.push({ type: "done", reason: "stop", message });
			stream.end(message);
			return stream;
		};
		await expect(agent.prompt("again")).resolves.toBeUndefined();
		expect(assistantText(agent.state.messages[agent.state.messages.length - 1])).toBe("accepted");
	});

	it("keeps the streamed partial content and leaves no error on the closing turn", async () => {
		const stream = new MockAssistantStream();
		const agent = new Agent({
			initialState: { model: getModel("openai", "gpt-5"), systemPrompt: "test", tools: [] },
			streamFn: () => stream,
		});
		const partial: AssistantMessage = {
			role: "assistant",
			content: [{ type: "text", text: "half a sen" }],
			api: "openai-responses",
			provider: "openai",
			model: "the-model-that-ran",
			usage: { ...EMPTY_USAGE, input: 1234, output: 56 },
			stopReason: "stop",
			timestamp: Date.now(),
		};

		const prompt = agent.prompt("hello");
		await new Promise((resolve) => setTimeout(resolve, 10));
		stream.push({ type: "start", partial });
		await new Promise((resolve) => setTimeout(resolve, 10));

		agent.abort({ force: true });
		await prompt;

		const closing = agent.state.messages[agent.state.messages.length - 1];
		expect(closing?.role).toBe("assistant");
		if (closing?.role !== "assistant") {
			throw new Error("expected an assistant turn");
		}
		expect(assistantText(closing)).toBe("half a sen");
		expect(closing.stopReason).toBe("aborted");
		expect(closing.model).toBe("the-model-that-ran");
		expect(closing.usage.input).toBe(1234);
		// An interrupt is not a failure, and `turn_end` copies errorMessage into state.
		expect(closing.errorMessage).toBeUndefined();
		expect(agent.state.errorMessage).toBeUndefined();
	});

	it("answers tool calls left in flight and leaves the queues to the caller", async () => {
		let releaseTool = () => {};
		const toolReleased = new Promise<void>((resolve) => {
			releaseTool = resolve;
		});
		const stream = new MockAssistantStream();
		// Only the first turn gets the tool call. Later turns -- which the detached
		// executor reaches once the tool is released -- get a stream that never
		// emits, as a real provider call in flight would.
		let turn = 0;
		const agent = new Agent({
			initialState: {
				model: getModel("openai", "gpt-5"),
				systemPrompt: "test",
				tools: [
					{
						name: "slow",
						label: "Slow",
						description: "never finishes on its own",
						parameters: { type: "object", properties: {} },
						execute: async () => {
							await toolReleased;
							return { content: [{ type: "text", text: "done" }], details: {} };
						},
					},
				],
			},
			streamFn: () => (turn++ === 0 ? stream : new MockAssistantStream()),
		});

		const toolCallMessage: AssistantMessage = {
			role: "assistant",
			content: [{ type: "toolCall", id: "call_1", name: "slow", arguments: {} }],
			api: "openai-responses",
			provider: "openai",
			model: "mock",
			usage: EMPTY_USAGE,
			stopReason: "toolUse",
			timestamp: Date.now(),
		};

		const prompt = agent.prompt("hello");
		await new Promise((resolve) => setTimeout(resolve, 10));
		stream.push({ type: "start", partial: toolCallMessage });
		stream.push({ type: "done", reason: "toolUse", message: toolCallMessage });
		stream.end(toolCallMessage);
		await new Promise((resolve) => setTimeout(resolve, 20));
		expect([...agent.state.pendingToolCalls]).toEqual(["call_1"]);

		agent.steer({ role: "user", content: [{ type: "text", text: "steered" }], timestamp: Date.now() });
		agent.abort({ force: true });
		await prompt;

		// Every tool call must be answered, or the next request carries a tool call
		// that nothing responds to.
		const toolResults = agent.state.messages.filter((message) => message.role === "toolResult");
		expect(toolResults).toHaveLength(1);
		expect(toolResults[0]?.role === "toolResult" ? toolResults[0].toolCallId : undefined).toBe("call_1");
		expect(toolResults[0]?.role === "toolResult" ? toolResults[0].toolName : undefined).toBe("slow");
		expect(toolResults[0]?.role === "toolResult" ? toolResults[0].isError : undefined).toBe(true);
		expect(describeMessages(agent.state.messages)).toEqual([
			"user",
			"assistant(toolUse)",
			"toolResult",
			"assistant(aborted)",
		]);

		// Queued text is the caller's to deal with, so the agent must leave it be.
		expect(agent.hasQueuedMessages()).toBe(true);
		releaseTool();
		await new Promise((resolve) => setTimeout(resolve, 30));
		expect(agent.state.isStreaming).toBe(false);
		expect(describeMessages(agent.state.messages)).toEqual([
			"user",
			"assistant(toolUse)",
			"toolResult",
			"assistant(aborted)",
		]);
	});

	it("does not add a second closing turn when the run already finished", async () => {
		const stream = new MockAssistantStream();
		const agent = new Agent({
			initialState: { model: getModel("openai", "gpt-5"), systemPrompt: "test", tools: [] },
			streamFn: () => stream,
		});
		const events: AgentEvent[] = [];
		let releaseListener = () => {};
		const listenerReleased = new Promise<void>((resolve) => {
			releaseListener = resolve;
		});
		agent.subscribe(async (event) => {
			events.push(event);
			if (event.type === "agent_end") {
				await listenerReleased;
			}
		});

		const done: AssistantMessage = {
			role: "assistant",
			content: [{ type: "text", text: "all done" }],
			api: "openai-responses",
			provider: "openai",
			model: "mock",
			usage: EMPTY_USAGE,
			stopReason: "stop",
			timestamp: Date.now(),
		};

		const prompt = agent.prompt("hello");
		await new Promise((resolve) => setTimeout(resolve, 10));
		stream.push({ type: "start", partial: done });
		stream.push({ type: "done", reason: "stop", message: done });
		stream.end(done);
		await new Promise((resolve) => setTimeout(resolve, 20));

		// The turn succeeded; the run is pinned only by the agent_end listener.
		agent.abort({ force: true });
		await prompt;

		// Forcing must not fabricate an aborted turn on top of a successful one.
		expect(describeMessages(agent.state.messages)).toEqual(["user", "assistant(stop)"]);
		expect(events.filter((event) => event.type === "agent_end")).toHaveLength(1);

		releaseListener();
		await new Promise((resolve) => setTimeout(resolve, 20));
		expect(describeMessages(agent.state.messages)).toEqual(["user", "assistant(stop)"]);
		expect(events.filter((event) => event.type === "agent_end")).toHaveLength(1);
	});
	it("stops issuing provider requests once the run is abandoned", async () => {
		// A model that always asks for another tool call: the loop continues on its
		// own, without needing the steering or follow-up queues.
		let calls = 0;
		const streamFn: StreamFn = () => {
			const index = ++calls;
			const stream = new MockAssistantStream();
			setTimeout(() => {
				const message: AssistantMessage = {
					role: "assistant",
					content: [{ type: "toolCall", id: `call_${index}`, name: "quick", arguments: {} }],
					api: "openai-responses",
					provider: "openai",
					model: "mock",
					usage: EMPTY_USAGE,
					stopReason: "toolUse",
					timestamp: Date.now(),
				};
				stream.push({ type: "start", partial: message });
				stream.push({ type: "done", reason: "toolUse", message });
				stream.end(message);
			}, 15);
			return stream;
		};
		const agent = new Agent({
			initialState: {
				model: getModel("openai", "gpt-5"),
				systemPrompt: "test",
				tools: [
					{
						name: "quick",
						label: "Quick",
						description: "returns immediately",
						parameters: { type: "object", properties: {} },
						execute: async () => ({ content: [{ type: "text", text: "ok" }], details: {} }),
					},
				],
			},
			streamFn,
		});

		const prompt = agent.prompt("hello");
		await new Promise((resolve) => setTimeout(resolve, 60));
		agent.abort({ force: true });
		await prompt;

		const callsAtAbort = calls;
		await new Promise((resolve) => setTimeout(resolve, 400));

		// The detached executor is allowed to finish the request in flight, and
		// nothing more. Without a per-turn exit it kept looping indefinitely,
		// billing a request every few milliseconds for as long as the process ran.
		expect(calls - callsAtAbort).toBeLessThanOrEqual(1);
		expect(agent.state.isStreaming).toBe(false);
	});
	it("answers a tool call even when the run is pinned in the tool-result listener", async () => {
		// pendingToolCalls is emptied when tool_execution_end is reduced, which
		// happens before its listeners are awaited. A run pinned in that listener
		// used to be abandoned with the call unanswered, and the provider transform
		// then tells the model "No result provided" for a tool that really ran.
		let releaseListener = () => {};
		const listenerReleased = new Promise<void>((resolve) => {
			releaseListener = resolve;
		});
		let executed = 0;
		let turn = 0;
		const stream = new MockAssistantStream();
		const agent = new Agent({
			initialState: {
				model: getModel("openai", "gpt-5"),
				systemPrompt: "test",
				tools: [
					{
						name: "quick",
						label: "Quick",
						description: "returns immediately",
						parameters: { type: "object", properties: {} },
						execute: async () => {
							executed++;
							return { content: [{ type: "text", text: "real result" }], details: {} };
						},
					},
				],
			},
			streamFn: () => (turn++ === 0 ? stream : new MockAssistantStream()),
		});
		agent.subscribe(async (event) => {
			if (event.type === "tool_execution_end") {
				await listenerReleased;
			}
		});

		const toolCallMessage: AssistantMessage = {
			role: "assistant",
			content: [{ type: "toolCall", id: "call_1", name: "quick", arguments: {} }],
			api: "openai-responses",
			provider: "openai",
			model: "mock",
			usage: EMPTY_USAGE,
			stopReason: "toolUse",
			timestamp: Date.now(),
		};

		const prompt = agent.prompt("hello");
		await new Promise((resolve) => setTimeout(resolve, 10));
		stream.push({ type: "start", partial: toolCallMessage });
		stream.push({ type: "done", reason: "toolUse", message: toolCallMessage });
		stream.end(toolCallMessage);
		await new Promise((resolve) => setTimeout(resolve, 20));

		// The tool ran, and the id has already left pendingToolCalls.
		expect(executed).toBe(1);
		expect([...agent.state.pendingToolCalls]).toEqual([]);

		agent.abort({ force: true });
		await prompt;

		const answered = agent.state.messages.filter(
			(message) => message.role === "toolResult" && message.toolCallId === "call_1",
		);
		expect(answered).toHaveLength(1);
		expect(describeMessages(agent.state.messages)).toEqual([
			"user",
			"assistant(toolUse)",
			"toolResult",
			"assistant(aborted)",
		]);

		releaseListener();
		await new Promise((resolve) => setTimeout(resolve, 20));
		expect(
			agent.state.messages.filter((message) => message.role === "toolResult" && message.toolCallId === "call_1"),
		).toHaveLength(1);
	});

	it("drops a truncated tool call from the closing turn", async () => {
		const stream = new MockAssistantStream();
		const agent = new Agent({
			initialState: { model: getModel("openai", "gpt-5"), systemPrompt: "test", tools: [] },
			streamFn: () => stream,
		});
		const partial: AssistantMessage = {
			role: "assistant",
			content: [
				{ type: "text", text: "calling" },
				{ type: "toolCall", id: "call_partial", name: "quick", arguments: { path: "/etc/pas" } },
			],
			api: "openai-responses",
			provider: "openai",
			model: "mock",
			usage: EMPTY_USAGE,
			stopReason: "stop",
			timestamp: Date.now(),
		};

		const prompt = agent.prompt("hello");
		await new Promise((resolve) => setTimeout(resolve, 10));
		stream.push({ type: "start", partial });
		await new Promise((resolve) => setTimeout(resolve, 10));
		agent.abort({ force: true });
		await prompt;

		// Arguments were still streaming and the call never executed, so recording
		// it would leave a call nothing answers and show half-parsed arguments.
		const closing = agent.state.messages[agent.state.messages.length - 1];
		expect(closing?.role).toBe("assistant");
		expect(assistantText(closing)).toBe("calling");
		expect(closing?.role === "assistant" ? closing.content.some((b) => b.type === "toolCall") : true).toBe(false);
		expect(agent.state.messages.filter((message) => message.role === "toolResult")).toHaveLength(0);
	});
});
