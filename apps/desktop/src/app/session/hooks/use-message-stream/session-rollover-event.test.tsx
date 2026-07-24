import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import {
  $composerAttachments,
  $composerDraft,
  clearSessionDraft,
  type ComposerAttachment,
  stashSessionDraft
} from '@/store/composer'
import { $queuedPromptsBySession, enqueueQueuedPrompt } from '@/store/composer-queue'
import { $gateway } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $activeSessionStoredIdRotation,
  $awaitingResponse,
  $busy,
  $connection,
  $selectedStoredSessionId,
  setActiveSessionId,
  setActiveSessionStoredIdRotation,
  setSessions
} from '@/store/session'
import {
  consumeCompletedSessionRollover,
  releaseSessionRolloverCommit
} from '@/store/session-rollover'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'
import type { RpcEvent, SessionInfo } from '@/types/hermes'

const windowMode = vi.hoisted(() => ({ watch: false }))

vi.mock('@/store/windows', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  isWatchWindow: () => windowMode.watch
}))

import { useMessageStream } from './index'

const RUNTIME_ID = 'runtime-rollover'
const PREDECESSOR_ID = 'stored-before'
const SUCCESSOR_ID = 'stored-after'
const ROOT_ID = 'stored-root'
const TOKEN = 'rollover-token'
const ATTACHMENT: ComposerAttachment = { id: 'file-1', kind: 'file', label: 'notes.txt' }
const SCOPES = [RUNTIME_ID, PREDECESSOR_ID, ROOT_ID] as const

let handleEvent: ((event: RpcEvent) => void) | null = null

const sessionRow = (): SessionInfo => ({
  _lineage_root_id: ROOT_ID,
  ended_at: null,
  id: PREDECESSOR_ID,
  input_tokens: 0,
  is_active: true,
  last_active: 1,
  message_count: 2,
  model: null,
  output_tokens: 0,
  preview: null,
  source: 'desktop',
  started_at: 1,
  title: 'Before rollover',
  tool_call_count: 0
})

function Harness() {
  const activeSessionIdRef = useRef<string | null>(RUNTIME_ID)
  const sessionStateByRuntimeIdRef = useRef(
    new Map<string, ClientSessionState>([[RUNTIME_ID, createClientSessionState(PREDECESSOR_ID)]])
  )
  const queryClientRef = useRef(new QueryClient())

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef,
    updateSessionState: (sessionId, updater, storedSessionId) => {
      const current = sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState()
      const rebound = storedSessionId === undefined ? current : { ...current, storedSessionId }
      const next = updater(rebound)
      sessionStateByRuntimeIdRef.current.set(sessionId, next)
      publishSessionState(sessionId, next)

      return next
    }
  })

  useEffect(() => {
    handleEvent = stream.handleGatewayEvent
  }, [stream.handleGatewayEvent])

  return null
}

const offerEvent = (overrides: Partial<RpcEvent> = {}): RpcEvent => ({
  payload: {
    compression_count: 1,
    final_message_id: 'message-final',
    history_version: 7,
    predecessor_stored_id: PREDECESSOR_ID,
    runtime_id: RUNTIME_ID,
    token: TOKEN,
    turn_generation: 3
  },
  profile: 'default',
  session_id: RUNTIME_ID,
  type: 'session.rollover.offer',
  ...overrides
})

const completeEvent = (overrides: Partial<RpcEvent> = {}): RpcEvent => ({
  payload: {
    predecessor_stored_id: PREDECESSOR_ID,
    runtime_id: RUNTIME_ID,
    successor_stored_id: SUCCESSOR_ID,
    token: TOKEN
  },
  profile: 'default',
  session_id: RUNTIME_ID,
  type: 'session.rollover.complete',
  ...overrides
})

const emit = (event: RpcEvent) => act(() => handleEvent!(event))

async function mountStream(request = vi.fn(async () => ({}))) {
  $gateway.set({ request } as never)
  render(<Harness />)
  await waitFor(() => expect(handleEvent).not.toBeNull())

  return request
}

function deferredCommit() {
  let resolve!: () => void
  const promise = new Promise<Record<string, never>>(settle => {
    resolve = () => settle({})
  })

  return { promise, resolve }
}

function clearScopedState() {
  for (const scope of SCOPES) {
    clearSessionDraft(scope)
  }
  $queuedPromptsBySession.set({})
}

beforeEach(() => {
  handleEvent = null
  windowMode.watch = false
  $activeGatewayProfile.set('default')
  $connection.set({ mode: 'local' } as never)
  setActiveSessionId(RUNTIME_ID)
  $selectedStoredSessionId.set(PREDECESSOR_ID)
  $busy.set(false)
  $awaitingResponse.set(false)
  $composerDraft.set('')
  $composerAttachments.set([])
  setSessions([sessionRow()])
  setActiveSessionStoredIdRotation(null)
  clearAllSessionStates()
  publishSessionState(RUNTIME_ID, createClientSessionState(PREDECESSOR_ID))
  clearScopedState()
  releaseSessionRolloverCommit(TOKEN)
  consumeCompletedSessionRollover(RUNTIME_ID, PREDECESSOR_ID, SUCCESSOR_ID)
})

afterEach(() => {
  cleanup()
  $gateway.set(null)
  $connection.set(null)
  $composerDraft.set('')
  $composerAttachments.set([])
  setActiveSessionId(null)
  $selectedStoredSessionId.set(null)
  setActiveSessionStoredIdRotation(null)
  setSessions([])
  clearScopedState()
  clearAllSessionStates()
  releaseSessionRolloverCommit(TOKEN)
  vi.restoreAllMocks()
})

describe('session rollover offer commit', () => {
  it('commits one eligible offer with the exact runtime and token', async () => {
    const request = await mountStream()

    emit(offerEvent())

    expect(request).toHaveBeenCalledTimes(1)
    expect(request).toHaveBeenCalledWith('session.rollover.commit', {
      session_id: RUNTIME_ID,
      token: TOKEN
    })
  })

  it('deduplicates a pending token and clears it after transient failure', async () => {
    const request = await mountStream(vi.fn(() => Promise.reject(new Error('socket closed'))))

    emit(offerEvent())
    emit(offerEvent())
    expect(request).toHaveBeenCalledTimes(1)

    await act(async () => Promise.resolve())
    emit(offerEvent())
    await waitFor(() => expect(request).toHaveBeenCalledTimes(2))
  })

  it('releases a token when a successful commit settles without a completion event', async () => {
    const commit = deferredCommit()
    const request = await mountStream(vi.fn(() => commit.promise))

    emit(offerEvent())
    emit(offerEvent())
    expect(request).toHaveBeenCalledTimes(1)

    await act(async () => {
      commit.resolve()
      await commit.promise
    })
    emit(offerEvent())

    expect(request).toHaveBeenCalledTimes(2)
  })

  it('releases a token after an ineligible completion and a resolved commit', async () => {
    const commit = deferredCommit()
    const request = await mountStream(vi.fn(() => commit.promise))

    emit(offerEvent())
    $selectedStoredSessionId.set('stored-other')
    emit(completeEvent())
    expect(consumeCompletedSessionRollover(RUNTIME_ID, PREDECESSOR_ID, SUCCESSOR_ID)).toBe(false)

    await act(async () => {
      commit.resolve()
      await commit.promise
    })
    $selectedStoredSessionId.set(PREDECESSOR_ID)
    emit(offerEvent())

    expect(request).toHaveBeenCalledTimes(2)
  })

  it.each([
    ['remote connection', () => $connection.set({ mode: 'remote' } as never), offerEvent()],
    ['profile mismatch', () => undefined, offerEvent({ profile: 'other' })],
    [
      'background runtime',
      () => undefined,
      offerEvent({
        payload: { ...(offerEvent().payload as Record<string, unknown>), runtime_id: 'runtime-background' },
        session_id: 'runtime-background'
      })
    ],
    ['selected predecessor mismatch', () => $selectedStoredSessionId.set('stored-other'), offerEvent()],
    ['busy client', () => $busy.set(true), offerEvent()],
    ['awaiting client', () => $awaitingResponse.set(true), offerEvent()],
    [
      'malformed payload',
      () => undefined,
      offerEvent({ payload: { ...(offerEvent().payload as Record<string, unknown>), history_version: '7' } })
    ],
    [
      'unknown payload field',
      () => undefined,
      offerEvent({ payload: { ...(offerEvent().payload as Record<string, unknown>), unexpected: true } })
    ]
  ])('fails closed for %s', async (_label, arrange, event) => {
    const request = await mountStream()
    arrange()

    emit(event)

    expect(request).not.toHaveBeenCalled()
  })

  it.each([
    ['live draft', () => $composerDraft.set('typing')],
    ['live attachment', () => $composerAttachments.set([ATTACHMENT])],
    ...SCOPES.map(scope => [`stashed draft at ${scope}`, () => stashSessionDraft(scope, 'typing', [])] as const),
    ...SCOPES.map(
      scope => [`stashed attachment at ${scope}`, () => stashSessionDraft(scope, '', [ATTACHMENT])] as const
    ),
    ...SCOPES.map(
      scope =>
        [
          `queued prompt at ${scope}`,
          () => enqueueQueuedPrompt(scope, { attachments: [], text: 'queued' })
        ] as const
    )
  ] as ReadonlyArray<readonly [string, () => unknown]>)('refuses rollover with a %s', async (_label, arrange) => {
    const request = await mountStream()
    arrange()

    emit(offerEvent())

    expect(request).not.toHaveBeenCalled()
  })

  it('never commits from a watch window', async () => {
    windowMode.watch = true
    const request = await mountStream()

    emit(offerEvent())

    expect(request).not.toHaveBeenCalled()
  })
})

describe('session rollover completion identity', () => {
  it('marks a complete-before-response transition and consumes it exactly once', async () => {
    const commit = deferredCommit()
    const request = await mountStream(vi.fn(() => commit.promise))
    emit(offerEvent())
    expect(request).toHaveBeenCalledTimes(1)

    emit(completeEvent())
    expect($activeSessionStoredIdRotation.get()).toBeNull()

    emit({
      payload: { running: false, stored_session_id: SUCCESSOR_ID },
      profile: 'default',
      session_id: RUNTIME_ID,
      type: 'session.info'
    })

    expect($activeSessionStoredIdRotation.get()).toEqual({
      kind: 'rollover',
      nextStoredSessionId: SUCCESSOR_ID,
      previousStoredSessionId: PREDECESSOR_ID,
      runtimeSessionId: RUNTIME_ID
    })
    expect(consumeCompletedSessionRollover(RUNTIME_ID, PREDECESSOR_ID, SUCCESSOR_ID)).toBe(false)

    await act(async () => {
      commit.resolve()
      await commit.promise
    })
    expect(consumeCompletedSessionRollover(RUNTIME_ID, PREDECESSOR_ID, SUCCESSOR_ID)).toBe(false)
  })

  it.each([
    [
      'remote completion',
      () => $connection.set({ mode: 'remote' } as never),
      completeEvent()
    ],
    ['profile-mismatched completion', () => undefined, completeEvent({ profile: 'other' })],
    [
      'background completion',
      () => undefined,
      completeEvent({
        payload: { ...(completeEvent().payload as Record<string, unknown>), runtime_id: 'runtime-background' },
        session_id: 'runtime-background'
      })
    ],
    ['stale predecessor completion', () => $selectedStoredSessionId.set('stored-other'), completeEvent()]
  ])('does not mark a %s', async (_label, arrange, event) => {
    await mountStream()
    emit(offerEvent())
    arrange()

    emit(event)

    expect(consumeCompletedSessionRollover(RUNTIME_ID, PREDECESSOR_ID, SUCCESSOR_ID)).toBe(false)
    expect($activeSessionStoredIdRotation.get()).toBeNull()
  })
})
