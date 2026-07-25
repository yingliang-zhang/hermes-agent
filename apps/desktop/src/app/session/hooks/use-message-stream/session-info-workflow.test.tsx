import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import {
  $currentCodingWorkflow,
  $currentSessionCodingWorkflow,
  $draftCodingWorkflowOverride,
  $profileCodingWorkflowDefault,
  resetDraftCodingWorkflowOverride,
  setActiveSessionId,
  setCurrentSessionCodingWorkflow,
  setDraftCodingWorkflowOverride,
  setProfileCodingWorkflowDefault
} from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

// session.info must reconcile coding_workflow into the correct session's
// per-runtime cache without clobbering the independent live projection,
// ephemeral draft override, or profile default. A background session's event
// must never overwrite the active session cache (stale-event guard).

const ACTIVE_SID = 'session-active'
const ACTIVE_PROFILE = 'compass'
let handleEvent: ((event: RpcEvent) => void) | null = null
let refreshHermesConfig: ReturnType<typeof vi.fn<() => Promise<void>>>
let refreshSessions: ReturnType<typeof vi.fn<() => Promise<void>>>
let queryClient: QueryClient
let sessionStateByRuntimeIdRef: React.MutableRefObject<Map<string, ClientSessionState>>

function Harness() {
  const activeSessionIdRef = useRef<string | null>(ACTIVE_SID)
  const statesRef = useRef(new Map<string, ClientSessionState>())
  sessionStateByRuntimeIdRef = statesRef

  const stream = useMessageStream({
    activeGatewayProfile: ACTIVE_PROFILE,
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient,
    refreshHermesConfig,
    refreshSessions,
    sessionStateByRuntimeIdRef: statesRef,
    updateSessionState: (sessionId, updater) => {
      const current = statesRef.current.get(sessionId) ?? createClientSessionState()
      const next = updater(current)
      statesRef.current.set(sessionId, next)

      return next
    }
  })

  useEffect(() => {
    handleEvent = stream.handleGatewayEvent
  }, [stream.handleGatewayEvent])

  return null
}

async function mountStream() {
  render(<Harness />)
  await waitFor(() => expect(handleEvent).not.toBeNull())
}

const sessionInfo = (sessionId: string, payload: Record<string, unknown>) =>
  act(() => handleEvent!({ payload, session_id: sessionId, type: 'session.info' }))

beforeEach(() => {
  handleEvent = null
  refreshHermesConfig = vi.fn<() => Promise<void>>(async () => undefined)
  refreshSessions = vi.fn<() => Promise<void>>(async () => undefined)
  queryClient = new QueryClient()
  setActiveSessionId(null)
  setProfileCodingWorkflowDefault('coupled-v1')
  setCurrentSessionCodingWorkflow('coupled-v1')
  resetDraftCodingWorkflowOverride()
})

afterEach(() => {
  cleanup()
  setActiveSessionId(null)
  setProfileCodingWorkflowDefault('coupled-v1')
  setCurrentSessionCodingWorkflow('coupled-v1')
  resetDraftCodingWorkflowOverride()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('session.info coding_workflow reconciliation', () => {
  it('lands the active session workflow in its cache without clobbering draft or profile authorities', async () => {
    setDraftCodingWorkflowOverride('hybrid-v1')

    await mountStream()

    sessionInfo(ACTIVE_SID, { coding_workflow: 'coupled-v1', running: true })

    expect(sessionStateByRuntimeIdRef.current.get(ACTIVE_SID)?.codingWorkflow).toBe('coupled-v1')
    expect($currentSessionCodingWorkflow.get()).toBe('coupled-v1')
    expect($draftCodingWorkflowOverride.get()).toBe('hybrid-v1')
    expect($profileCodingWorkflowDefault.get()).toBe('coupled-v1')
    expect($currentCodingWorkflow.get()).toBe('hybrid-v1')
  })

  it('does not let a background workflow event clobber any foreground authority or the active cache', async () => {
    setProfileCodingWorkflowDefault('hybrid-v1')
    setDraftCodingWorkflowOverride('coupled-v1')
    setCurrentSessionCodingWorkflow('hybrid-v1')
    await mountStream()

    sessionInfo(ACTIVE_SID, { coding_workflow: 'hybrid-v1', running: false })
    expect(sessionStateByRuntimeIdRef.current.get(ACTIVE_SID)?.codingWorkflow).toBe('hybrid-v1')

    sessionInfo('session-background', { coding_workflow: 'coupled-v1', running: true })

    expect(sessionStateByRuntimeIdRef.current.get('session-background')?.codingWorkflow).toBe('coupled-v1')
    expect(sessionStateByRuntimeIdRef.current.get(ACTIVE_SID)?.codingWorkflow).toBe('hybrid-v1')
    expect($currentSessionCodingWorkflow.get()).toBe('hybrid-v1')
    expect($draftCodingWorkflowOverride.get()).toBe('coupled-v1')
    expect($profileCodingWorkflowDefault.get()).toBe('hybrid-v1')
    expect($currentCodingWorkflow.get()).toBe('coupled-v1')
  })

  it('ignores an unknown workflow without clobbering the session cache or workflow authorities', async () => {
    setProfileCodingWorkflowDefault('hybrid-v1')
    setDraftCodingWorkflowOverride('coupled-v1')
    await mountStream()

    sessionInfo(ACTIVE_SID, { coding_workflow: 'hybrid-v1', running: false })
    expect(sessionStateByRuntimeIdRef.current.get(ACTIVE_SID)?.codingWorkflow).toBe('hybrid-v1')

    sessionInfo(ACTIVE_SID, { coding_workflow: 'totally-bogus', running: true })

    expect(sessionStateByRuntimeIdRef.current.get(ACTIVE_SID)?.codingWorkflow).toBe('hybrid-v1')
    expect($currentSessionCodingWorkflow.get()).toBe('coupled-v1')
    expect($draftCodingWorkflowOverride.get()).toBe('coupled-v1')
    expect($profileCodingWorkflowDefault.get()).toBe('hybrid-v1')
    expect($currentCodingWorkflow.get()).toBe('coupled-v1')
  })
})
