import { describe, expect, test } from "vitest";
import { InteractiveMode } from "../src/modes/interactive/interactive-mode.ts";

type Calls = {
	restore: Array<{ abort?: boolean } | undefined>;
	statuses: string[];
	cleared: string[];
	aborts: Array<{ force?: boolean } | undefined>;
	indicatorMessages: string[];
	editorText: string[];
};

type Context = {
	defaultEditor: { onEscape?: () => void; onAction: (action: string, handler: () => void) => void };
	interruptRequestedAt: number | undefined;
};

/**
 * Interrupt handling built on the real prototype, so the escalation path and its
 * helpers are the shipped implementations rather than stubs. `session`, `agent`
 * and `settingsManager` are prototype getters that read through `runtimeHost`.
 */
function createContext(overrides: { indicatorKind?: string | undefined; hasRun?: boolean; queued?: string[] } = {}): {
	context: Context;
	calls: Calls;
	pressEscape: () => void;
	settled: () => Promise<void>;
} {
	const calls: Calls = { restore: [], statuses: [], cleared: [], aborts: [], indicatorMessages: [], editorText: [] };
	const indicatorKind = "indicatorKind" in overrides ? overrides.indicatorKind : "working";
	const controller = new AbortController();

	const session = {
		isStreaming: true,
		isBashRunning: false,
		abort: async (options?: { force?: boolean }) => {
			calls.aborts.push(options);
			return { steering: overrides.queued ?? [], followUp: [] };
		},
		agent: { signal: overrides.hasRun === false ? undefined : controller.signal },
		settingsManager: { getDoubleEscapeAction: () => "none", getShowTerminalProgress: () => false },
	};

	const context = Object.assign(Object.create(InteractiveMode.prototype), {
		runtimeHost: { session },
		defaultEditor: { onAction: () => {} },
		editor: { getText: () => "", setText: (text: string) => calls.editorText.push(text) },
		ui: { requestRender: () => {} },
		updatePendingMessagesDisplay: () => {},
		activeStatusIndicator:
			indicatorKind === undefined
				? undefined
				: { kind: indicatorKind, setMessage: (message: string) => calls.indicatorMessages.push(message) },
		retryEscapeHandler: undefined,
		isBashMode: false,
		interruptRequestedAt: undefined,
		pendingTools: new Set(),
		// handleEvent short-circuits its lazy init when already initialized.
		isInitialized: true,
		footer: { invalidate: () => {} },
		restoreQueuedMessagesToEditor: (options?: { abort?: boolean }) => {
			calls.restore.push(options);
			return 0;
		},
		clearStatusIndicator: (kind: string) => calls.cleared.push(kind),
		showStatus: (message: string) => calls.statuses.push(message),
		showError: (message: string) => calls.statuses.push(`ERROR: ${message}`),
	}) as Context;

	const setupKeyHandlers = Reflect.get(InteractiveMode.prototype, "setupKeyHandlers") as (this: unknown) => void;
	setupKeyHandlers.call(context);

	return {
		context,
		calls,
		pressEscape: () => context.defaultEditor.onEscape?.(),
		// forceInterrupt fires a floating promise; let its continuations run.
		settled: async () => {
			await Promise.resolve();
			await Promise.resolve();
			await Promise.resolve();
		},
	};
}

describe("InteractiveMode interrupt escalation", () => {
	test("first interrupt aborts cooperatively and warns about escalation", () => {
		const { context, calls, pressEscape } = createContext();

		pressEscape();

		expect(calls.restore).toEqual([{ abort: true }]);
		expect(context.interruptRequestedAt).toBeTypeOf("number");
		expect(calls.indicatorMessages[0]).toContain("again to force stop");
		expect(calls.aborts).toEqual([]);
	});

	test("a rapid second interrupt does not force the run, and says so", () => {
		const { calls, pressEscape } = createContext();

		pressEscape();
		pressEscape();

		// A double-tap is a habit, and a healthy run is still unwinding: forcing
		// here would discard the tool results the cooperative path is recording.
		expect(calls.aborts).toEqual([]);
		expect(calls.cleared).toEqual([]);
		// But the keypress must not vanish, or it reads as "Esc does nothing" again.
		expect(calls.indicatorMessages.at(-1)).toContain("still stopping");
	});

	test("a later second interrupt forces the run", () => {
		const { context, calls, pressEscape } = createContext();

		pressEscape();
		context.interruptRequestedAt = Date.now() - 600;
		pressEscape();

		expect(calls.aborts).toEqual([{ force: true }]);
		expect(calls.cleared).toEqual(["working"]);
		expect(calls.statuses.some((status) => status.includes("Abandoned the run"))).toBe(true);
	});

	test("queued messages discarded by the force are returned to the editor", async () => {
		const { context, calls, pressEscape, settled } = createContext({ queued: ["no, do X instead"] });

		pressEscape();
		context.interruptRequestedAt = Date.now() - 600;
		pressEscape();
		await settled();

		// The abandoned run will never consume them, so losing them silently would
		// destroy text the user typed.
		expect(calls.editorText).toEqual(["no, do X instead"]);
	});

	test("warns even when the working indicator is hidden", () => {
		const { calls, pressEscape } = createContext({ indicatorKind: undefined });

		pressEscape();

		// Arming the force silently would make the next keypress abandon the run
		// with no warning at all.
		expect(calls.statuses.some((status) => status.includes("again to force stop"))).toBe(true);
	});

	test("still forces, but does not claim to abandon a run, in the continuation window", () => {
		const { context, calls, pressEscape } = createContext({ hasRun: false });

		pressEscape();
		context.interruptRequestedAt = Date.now() - 600;
		pressEscape();

		// `isStreaming` is still true between runs, where there is no agent run but a
		// compaction or summary may be in flight -- force stops those too.
		expect(calls.aborts).toEqual([{ force: true }]);
		expect(calls.statuses.some((status) => status.includes("still finishing"))).toBe(true);
		expect(calls.statuses.some((status) => status.includes("Abandoned the run"))).toBe(false);
	});

	test("agent_start re-arms the escalation for the new run", async () => {
		const { context, calls, pressEscape } = createContext();
		const handleEvent = Reflect.get(InteractiveMode.prototype, "handleEvent") as (
			this: unknown,
			event: { type: string },
		) => Promise<void>;

		pressEscape();
		context.interruptRequestedAt = Date.now() - 600;
		await handleEvent.call(context, { type: "agent_start" });

		// A stale timestamp must not let a single interrupt force the next run.
		expect(context.interruptRequestedAt).toBeUndefined();
		pressEscape();
		expect(calls.aborts).toEqual([]);
		expect(calls.restore).toEqual([{ abort: true }, { abort: true }]);
	});
});
