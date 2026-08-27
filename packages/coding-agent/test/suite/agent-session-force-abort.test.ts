import { type AssistantMessage, fauxAssistantMessage } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it } from "vitest";
import { createHarness, type Harness } from "./harness.ts";

async function waitFor(predicate: () => boolean, timeoutMs = 1000): Promise<void> {
	const deadline = Date.now() + timeoutMs;
	while (!predicate()) {
		if (Date.now() > deadline) {
			throw new Error("Timed out waiting for condition");
		}
		await new Promise((resolve) => setTimeout(resolve, 5));
	}
}

/** Provider call that never settles and ignores the abort signal. */
function unresponsiveProvider(): () => Promise<AssistantMessage> {
	return () => new Promise<AssistantMessage>(() => {});
}

/** Roles of the messages actually written to the session transcript, in order. */
function persistedRoles(harness: Harness): string[] {
	return harness.sessionManager
		.getEntries()
		.filter((entry) => entry.type === "message")
		.map((entry) => {
			const message = (entry as { message: { role: string; stopReason?: string } }).message;
			return message.role === "assistant" ? `assistant(${message.stopReason})` : message.role;
		});
}

function liveRoles(harness: Harness): string[] {
	return harness.session.messages.map((message) =>
		message.role === "assistant" ? `assistant(${message.stopReason})` : String(message.role),
	);
}

async function raceTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<"resolved" | "timeout"> {
	let timer: ReturnType<typeof setTimeout> | undefined;
	const outcome = await Promise.race([
		promise.then(() => "resolved" as const),
		new Promise<"timeout">((resolve) => {
			timer = setTimeout(() => resolve("timeout"), timeoutMs);
		}),
	]);
	if (timer !== undefined) {
		clearTimeout(timer);
	}
	return outcome;
}

describe("agent session force abort", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		while (harnesses.length > 0) {
			harnesses.pop()?.cleanup();
		}
	});

	it("stays streaming when a signal-ignoring provider call is aborted", async () => {
		const harness = await createHarness();
		harnesses.push(harness);
		harness.setResponses([unresponsiveProvider()]);

		void harness.session.prompt("hello");
		await waitFor(() => harness.session.isStreaming);

		harness.session.agent.abort();
		await new Promise((resolve) => setTimeout(resolve, 50));

		expect(harness.session.isStreaming).toBe(true);
		expect(harness.eventsOfType("agent_settled")).toHaveLength(0);
	});

	it("settles the run when an extension handler never returns", async () => {
		const harness = await createHarness({
			extensionFactories: [
				(pi) => {
					pi.on("agent_end", async () => {
						await new Promise(() => {});
					});
				},
			],
		});
		harnesses.push(harness);
		harness.setResponses([fauxAssistantMessage("done")]);

		const prompt = harness.session.prompt("hello");
		await waitFor(() => harness.session.isStreaming);
		await new Promise((resolve) => setTimeout(resolve, 50));
		expect(harness.session.isStreaming).toBe(true);

		harness.session.agent.abort({ force: true });
		await prompt;

		expect(harness.session.isStreaming).toBe(false);
		expect(harness.eventsOfType("agent_settled")).toHaveLength(1);
	});

	it("settles the run when the abort is forced", async () => {
		const harness = await createHarness();
		harnesses.push(harness);
		harness.setResponses([unresponsiveProvider()]);

		const prompt = harness.session.prompt("hello");
		await waitFor(() => harness.session.isStreaming);

		harness.session.agent.abort({ force: true });
		await prompt;

		expect(harness.session.isStreaming).toBe(false);
		expect(harness.eventsOfType("agent_settled")).toHaveLength(1);
		const lastMessage = harness.session.messages[harness.session.messages.length - 1];
		expect(lastMessage?.role).toBe("assistant");
		expect(lastMessage?.role === "assistant" ? lastMessage.stopReason : undefined).toBe("aborted");
	});

	it("resolves session.abort({ force: true }) and reports the session idle", async () => {
		const harness = await createHarness();
		harnesses.push(harness);
		harness.setResponses([unresponsiveProvider()]);

		const prompt = harness.session.prompt("hello");
		await waitFor(() => harness.session.isStreaming);

		// The session-level entry point, not agent.abort: this is what the TUI and
		// every teardown path actually call.
		expect(await raceTimeout(harness.session.abort({ force: true }), 2000)).toBe("resolved");
		expect(harness.session.isStreaming).toBe(false);
		expect(harness.session.isIdle).toBe(true);
		await prompt;
	});

	it("resolves session.abort({ force: true }) when an agent_settled handler never returns", async () => {
		const harness = await createHarness({
			extensionFactories: [
				(pi) => {
					pi.on("agent_settled", async () => {
						await new Promise(() => {});
					});
				},
			],
		});
		harnesses.push(harness);
		harness.setResponses([unresponsiveProvider()]);

		void harness.session.prompt("hello");
		await waitFor(() => harness.session.isStreaming);

		// A handler outside the run must not be able to hold waitForIdle open while
		// isIdle already reports true. The dispatch is bounded rather than skipped,
		// so this resolves after the bound, not immediately -- skipping it entirely
		// would let teardown dispose the session under a handler that is merely slow.
		expect(await raceTimeout(harness.session.abort({ force: true }), 10_000)).toBe("resolved");
		expect(harness.session.isIdle).toBe(true);
	});

	it("persists the aborted turn before the next message, despite a slow handler", async () => {
		const harness = await createHarness({
			extensionFactories: [
				(pi) => {
					pi.on("message_end", async () => {
						await new Promise((resolve) => setTimeout(resolve, 600));
					});
				},
			],
		});
		harnesses.push(harness);
		harness.setResponses([unresponsiveProvider()]);

		const prompt = harness.session.prompt("hello");
		await waitFor(() => harness.session.isStreaming);
		// Let the run reach the provider call it will hang in, so the detached
		// executor cannot later wake up and consume the second run's response.
		await waitFor(() => harness.getPendingResponseCount() === 0, 3000);
		// And let the user message's own slow handler finish, so the assertion below
		// is about the aborted turn's ordering rather than about that handler.
		await waitFor(() => persistedRoles(harness).length === 1, 3000);
		await harness.session.abort({ force: true });
		await prompt;

		// The transcript must already hold the aborted turn, whose own message_end
		// handler is still sleeping: the user can type the next message the instant
		// the session reports idle, and a transcript in a different order to the one
		// they saw is what a resumed session replays.
		expect(persistedRoles(harness)).toEqual(["user", "assistant(aborted)"]);
		expect(liveRoles(harness)).toEqual(["user", "assistant(aborted)"]);

		harness.setResponses([fauxAssistantMessage("second")]);
		await harness.session.prompt("next");
		await waitFor(() => !harness.session.isStreaming);

		// And the late listener must not append the aborted turn a second time.
		expect(persistedRoles(harness)).toEqual(["user", "assistant(aborted)", "user", "assistant(stop)"]);
		expect(liveRoles(harness)).toEqual(["user", "assistant(aborted)", "user", "assistant(stop)"]);
	});

	it("does not start a new run from messages queued before a forced abort", async () => {
		const harness = await createHarness();
		harnesses.push(harness);
		harness.setResponses([unresponsiveProvider()]);

		const prompt = harness.session.prompt("hello");
		await waitFor(() => harness.session.isStreaming);
		harness.session.agent.steer({
			role: "user",
			content: [{ type: "text", text: "no, do X instead" }],
			timestamp: Date.now(),
		});

		await harness.session.abort({ force: true });
		await prompt;
		await new Promise((resolve) => setTimeout(resolve, 50));

		// A forced stop that immediately starts another run looks broken, and would
		// run alongside the executor that was just abandoned.
		expect(harness.eventsOfType("agent_start")).toHaveLength(1);
		expect(harness.session.isStreaming).toBe(false);
		expect(liveRoles(harness)).toEqual(["user", "assistant(aborted)"]);
	});

	it("returns queued messages and stops reporting them as pending", async () => {
		const harness = await createHarness();
		harnesses.push(harness);
		harness.setResponses([unresponsiveProvider()]);

		const prompt = harness.session.prompt("hello");
		await waitFor(() => harness.session.isStreaming);
		await harness.session.prompt("queued follow-up", { streamingBehavior: "steer" });
		expect(harness.session.pendingMessageCount).toBe(1);

		const queued = await harness.session.abort({ force: true });
		await prompt;

		// The run that would have consumed them is gone, so they must be handed back
		// rather than silently dropped while the UI still lists them as pending.
		expect([...queued.steering, ...queued.followUp]).toEqual(["queued follow-up"]);
		expect(harness.session.pendingMessageCount).toBe(0);
		expect(harness.session.agent.hasQueuedMessages()).toBe(false);
		expect(harness.eventsOfType("queue_update").length).toBeGreaterThan(0);
	});

	it("does not touch the transcript when forced while idle", async () => {
		const harness = await createHarness();
		harnesses.push(harness);
		harness.setResponses([fauxAssistantMessage("first")]);
		await harness.session.prompt("hello");
		await waitFor(() => !harness.session.isStreaming);

		const before = persistedRoles(harness);
		expect(before).toEqual(["user", "assistant(stop)"]);

		// No run to abandon. A force here must be inert: anything that re-derived
		// the closing turn from the transcript would append the whole conversation
		// to itself, and a resumed session would replay every turn twice.
		await harness.session.abort({ force: true });

		expect(persistedRoles(harness)).toEqual(before);
		expect(liveRoles(harness)).toEqual(["user", "assistant(stop)"]);
	});

	it("does not re-append an existing transcript when forced while idle", async () => {
		const harness = await createHarness();
		harnesses.push(harness);

		// Mirrors how a resumed or imported session is loaded: the agent transcript
		// is assigned wholesale from messages that are already in the file.
		harness.session.agent.state.messages = [
			{ role: "user", content: [{ type: "text", text: "from an earlier session" }], timestamp: Date.now() },
		];

		await harness.session.abort({ force: true });

		expect(persistedRoles(harness)).toEqual([]);
	});
});
