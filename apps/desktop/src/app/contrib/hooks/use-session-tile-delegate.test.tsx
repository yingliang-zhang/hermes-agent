import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { getSessionMessages } from '@/hermes'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $connection } from '@/store/session'
import { sessionTileDelegate } from '@/store/session-states'

import type { ClientSessionState } from '../../types'

import { useSessionTileDelegate } from './use-session-tile-delegate'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getSessionMessages: vi.fn(async () => ({ messages: [] }))
}))

function Harness({
  requestGateway
}: {
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}) {
  const runtimeIdByStoredSessionIdRef = useRef(new Map<string, string>())
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())

  useSessionTileDelegate({
    archiveSession: async () => undefined,
    branchStoredSession: async () => undefined,
    executeSlashCommand: async () => undefined,
    removeSession: async () => undefined,
    requestGateway,
    runtimeIdByStoredSessionIdRef,
    sessionStateByRuntimeIdRef,
    updateSessionState: (sessionId, updater, storedSessionId) => {
      const current = sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState(storedSessionId)
      const next = updater(current)
      sessionStateByRuntimeIdRef.current.set(sessionId, next)

      return next
    }
  })

  return null
}

afterEach(() => {
  cleanup()
  $connection.set(null)
  vi.restoreAllMocks()
})

it('advertises local rollover capability on a cold tile resume', async () => {
  $connection.set({ mode: 'local' } as never)
  vi.mocked(getSessionMessages).mockResolvedValue({ messages: [] } as never)

  const requestGateway = vi.fn(async (method: string) => {
    if (method === 'session.resume') {
      return { messages: [], session_id: 'runtime-tile' } as never
    }

    return {} as never
  })

  render(<Harness requestGateway={requestGateway} />)
  await waitFor(() => expect(sessionTileDelegate()).not.toBeNull())

  await act(async () => {
    await sessionTileDelegate()!.resumeTile('stored-tile')
  })

  expect(requestGateway).toHaveBeenCalledWith('session.resume', {
    cols: 96,
    local_rollover_capable: true,
    session_id: 'stored-tile',
    source: 'desktop'
  })
})
