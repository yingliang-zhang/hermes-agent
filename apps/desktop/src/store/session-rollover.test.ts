import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

const windowMode = vi.hoisted(() => ({ watch: false }))

vi.mock('@/store/windows', () => ({
  isWatchWindow: () => windowMode.watch
}))

import {
  beginSessionRolloverCommit,
  consumeCompletedSessionRollover,
  registerCompletedSessionRollover,
  releaseSessionRolloverCommit,
  sessionRolloverCapabilityParams
} from './session-rollover'

const TRANSITION = {
  predecessorStoredId: 'stored-before',
  runtimeId: 'runtime-1',
  successorStoredId: 'stored-after',
  token: 'rollover-token'
}

describe('session rollover capability advertisement', () => {
  beforeEach(() => {
    $connection.set(null)
    windowMode.watch = false
  })

  it('advertises local rollover only from a local non-watch window', () => {
    $connection.set({ mode: 'local' } as never)
    expect(sessionRolloverCapabilityParams()).toEqual({ local_rollover_capable: true })

    $connection.set({ mode: 'remote' } as never)
    expect(sessionRolloverCapabilityParams()).toEqual({ local_rollover_capable: false })

    $connection.set({ mode: 'local' } as never)
    windowMode.watch = true
    expect(sessionRolloverCapabilityParams()).toEqual({ local_rollover_capable: false })
  })
})

describe('completed session rollover transitions', () => {
  beforeEach(() => {
    releaseSessionRolloverCommit(TRANSITION.token)
    consumeCompletedSessionRollover(
      TRANSITION.runtimeId,
      TRANSITION.predecessorStoredId,
      TRANSITION.successorStoredId
    )
  })

  it('registers an identity-bound transition and consumes its exact key once', () => {
    expect(beginSessionRolloverCommit(TRANSITION.token)).toBe(true)
    expect(registerCompletedSessionRollover(TRANSITION)).toBe(true)

    expect(
      consumeCompletedSessionRollover('runtime-other', TRANSITION.predecessorStoredId, TRANSITION.successorStoredId)
    ).toBe(false)
    expect(
      consumeCompletedSessionRollover(
        TRANSITION.runtimeId,
        TRANSITION.predecessorStoredId,
        TRANSITION.successorStoredId
      )
    ).toBe(true)
    expect(
      consumeCompletedSessionRollover(
        TRANSITION.runtimeId,
        TRANSITION.predecessorStoredId,
        TRANSITION.successorStoredId
      )
    ).toBe(false)
  })

  it('rejects completion without the matching in-flight token', () => {
    expect(registerCompletedSessionRollover(TRANSITION)).toBe(false)
    expect(
      consumeCompletedSessionRollover(
        TRANSITION.runtimeId,
        TRANSITION.predecessorStoredId,
        TRANSITION.successorStoredId
      )
    ).toBe(false)
  })

  it('evicts the oldest completed transition after the fixed marker cap', () => {
    const markerCap = 32
    const transitions = Array.from({ length: markerCap + 1 }, (_, index) => ({
      predecessorStoredId: `stored-before-${index}`,
      runtimeId: `runtime-${index}`,
      successorStoredId: `stored-after-${index}`,
      token: `rollover-token-${index}`
    }))

    for (const transition of transitions) {
      expect(beginSessionRolloverCommit(transition.token)).toBe(true)
      expect(registerCompletedSessionRollover(transition)).toBe(true)
    }

    const oldest = transitions[0]!
    const newest = transitions.at(-1)!

    expect(
      consumeCompletedSessionRollover(oldest.runtimeId, oldest.predecessorStoredId, oldest.successorStoredId)
    ).toBe(false)
    expect(
      consumeCompletedSessionRollover(newest.runtimeId, newest.predecessorStoredId, newest.successorStoredId)
    ).toBe(true)
    expect(
      consumeCompletedSessionRollover(newest.runtimeId, newest.predecessorStoredId, newest.successorStoredId)
    ).toBe(false)

    for (const transition of transitions.slice(1, -1)) {
      consumeCompletedSessionRollover(
        transition.runtimeId,
        transition.predecessorStoredId,
        transition.successorStoredId
      )
    }
  })
})
