import { QueryClient } from '@tanstack/react-query'
import { cleanup, render, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getGlobalModelInfo } from '@/hermes'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $activeSessionId,
  $currentCodingWorkflow,
  $currentModel,
  $currentProvider,
  $profileCodingWorkflowDefault,
  getComposerSelectionGeneration,
  getCurrentModelSource,
  markComposerSelectionManual,
  resetDraftCodingWorkflowOverride,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider,
  setCurrentSessionCodingWorkflow,
  setDraftCodingWorkflowOverride,
  setProfileCodingWorkflowDefault
} from '@/store/session'
import type * as SessionStates from '@/store/session-states'
import { $sessionStates, setSessionTileDelegate } from '@/store/session-states'

import { useModelControls } from './use-model-controls'

const setGlobalModel = vi.fn()
const notifyError = vi.fn()

vi.mock('@/hermes', () => ({
  getGlobalModelInfo: vi.fn(),
  setApiRequestProfile: vi.fn(),
  setGlobalModel: (...args: Parameters<typeof setGlobalModel>) => setGlobalModel(...args)
}))

vi.mock('@/store/session-states', async importOriginal => {
  const actual = await importOriginal<typeof SessionStates>()

  return { ...actual, sessionTileDelegate: () => actual.sessionTileDelegate() }
})

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: { desktop: { modelSwitchFailed: 'Model switch failed' } }
  })
}))

vi.mock('@/store/notifications', () => ({
  notifyError: (...args: Parameters<typeof notifyError>) => notifyError(...args)
}))

type Controls = ReturnType<typeof useModelControls>

function Harness({
  onReady,
  requestGateway
}: {
  onReady: (controls: Controls) => void
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}) {
  const controls = useModelControls({ queryClient: new QueryClient(), requestGateway })

  onReady(controls)

  return null
}

function resetDraft() {
  $activeGatewayProfile.set('default')
  $activeSessionId.set(null)
  setCurrentModel('')
  setCurrentModelSource('')
  setCurrentProvider('')
  setProfileCodingWorkflowDefault('coupled-v1')
  setCurrentSessionCodingWorkflow('coupled-v1')
  resetDraftCodingWorkflowOverride()
  $sessionStates.set({})
}

describe('useModelControls — coding workflow routing', () => {
  beforeEach(() => {
    resetDraft()
    setSessionTileDelegate({
      updateSession(runtimeId, updater) {
        const current = $sessionStates.get()[runtimeId]

        if (!current) {
          throw new Error(`missing test session: ${runtimeId}`)
        }

        const next = updater(current)

        $sessionStates.set({ ...$sessionStates.get(), [runtimeId]: next })

        return next
      }
    } as SessionStates.SessionTileDelegate)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    resetDraft()
  })

  it('selectHybrid stores a no-session draft (Sol + hybrid-v1) with no gateway write', async () => {
    const requestGateway = vi.fn()
    let controls!: Controls

    render(<Harness onReady={value => (controls = value)} requestGateway={requestGateway} />)

    await expect(controls.selectHybrid()).resolves.toBe(true)

    expect($currentModel.get()).toBe('gpt-5.6-sol')
    expect($currentProvider.get()).toBe('custom:sudo')
    expect($currentCodingWorkflow.get()).toBe('hybrid-v1')
    expect($profileCodingWorkflowDefault.get()).toBe('coupled-v1')
    expect(getCurrentModelSource()).toBe('manual')
    expect(requestGateway).not.toHaveBeenCalled()
    expect(setGlobalModel).not.toHaveBeenCalled()
  })

  it('selectHybrid on a live primary commits one atomic route payload via config.set key route', async () => {
    $activeSessionId.set('session-1')
    const requestGateway = vi.fn(async () => ({ key: 'route' }) as never)
    let controls!: Controls

    render(<Harness onReady={value => (controls = value)} requestGateway={requestGateway} />)

    await expect(controls.selectHybrid()).resolves.toBe(true)

    expect(requestGateway).toHaveBeenCalledWith('config.set', {
      session_id: 'session-1',
      key: 'route',
      value: { provider: 'custom:sudo', model: 'gpt-5.6-sol', coding_workflow: 'hybrid-v1' }
    })
    expect(requestGateway).toHaveBeenCalledTimes(1)
    expect($currentCodingWorkflow.get()).toBe('hybrid-v1')
  })

  it('rolls back provider, model, AND workflow when the route commit fails', async () => {
    $activeSessionId.set('session-1')
    setCurrentModel('prev-model')
    setCurrentProvider('prev-provider')
    setCurrentSessionCodingWorkflow('hybrid-v1')
    markComposerSelectionManual()

    const requestGateway = vi.fn(async () => {
      throw new Error('boom')
    })

    let controls!: Controls

    render(<Harness onReady={value => (controls = value)} requestGateway={requestGateway} />)

    await expect(controls.selectHybrid()).resolves.toBe(false)

    // All three restored to the snapshot taken before the optimistic update.
    expect($currentModel.get()).toBe('prev-model')
    expect($currentProvider.get()).toBe('prev-provider')
    expect($currentCodingWorkflow.get()).toBe('hybrid-v1')
    expect(notifyError).toHaveBeenCalled()
  })

  it('ordinary selectModel forces coupled-v1 on a live session via the route payload', async () => {
    $activeSessionId.set('session-1')
    setCurrentSessionCodingWorkflow('hybrid-v1')
    const requestGateway = vi.fn(async () => ({ key: 'route' }) as never)
    let controls!: Controls

    render(<Harness onReady={value => (controls = value)} requestGateway={requestGateway} />)

    await expect(
      controls.selectModel({ model: 'claude-sonnet-4.6', provider: 'anthropic' })
    ).resolves.toBe(true)

    expect(requestGateway).toHaveBeenCalledWith('config.set', {
      session_id: 'session-1',
      key: 'route',
      value: { provider: 'anthropic', model: 'claude-sonnet-4.6', coding_workflow: 'coupled-v1' }
    })
    expect($currentCodingWorkflow.get()).toBe('coupled-v1')
  })

  it('ordinary selectModel stores a no-session draft that forces coupled-v1 with no gateway write', async () => {
    setDraftCodingWorkflowOverride('hybrid-v1')
    const requestGateway = vi.fn()
    let controls!: Controls

    render(<Harness onReady={value => (controls = value)} requestGateway={requestGateway} />)

    await expect(
      controls.selectModel({ model: 'claude-sonnet-4.6', provider: 'anthropic' })
    ).resolves.toBe(true)

    expect($currentModel.get()).toBe('claude-sonnet-4.6')
    expect($currentProvider.get()).toBe('anthropic')
    expect($currentCodingWorkflow.get()).toBe('coupled-v1')
    expect($profileCodingWorkflowDefault.get()).toBe('coupled-v1')
    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('selectHybrid on a tile updates the tile cache (not primary globals) and scopes the gateway call', async () => {
    $activeSessionId.set('primary-1')
    setCurrentModel('primary-model')
    setCurrentProvider('primary-provider')
    setCurrentSessionCodingWorkflow('coupled-v1')

    const tileState = { ...createClientSessionState('tile-stored'), model: 'tile-model', provider: 'tile-provider' }
    $sessionStates.set({ 'tile-runtime': tileState })

    const requestGateway = vi.fn(async () => ({ key: 'route' }) as never)
    let controls!: Controls

    render(<Harness onReady={value => (controls = value)} requestGateway={requestGateway} />)

    await expect(controls.selectHybrid({ sessionId: 'tile-runtime' })).resolves.toBe(true)

    // Primary draft is untouched — a tile switch must not bleed into the footer.
    expect($currentModel.get()).toBe('primary-model')
    expect($currentProvider.get()).toBe('primary-provider')
    expect($currentCodingWorkflow.get()).toBe('coupled-v1')

    // The tile cache carries the new route.
    expect($sessionStates.get()['tile-runtime']).toMatchObject({
      model: 'gpt-5.6-sol',
      provider: 'custom:sudo',
      codingWorkflow: 'hybrid-v1'
    })

    // The gateway call targets the tile runtime id, not the primary.
    expect(requestGateway).toHaveBeenCalledWith('config.set', {
      session_id: 'tile-runtime',
      key: 'route',
      value: { provider: 'custom:sudo', model: 'gpt-5.6-sol', coding_workflow: 'hybrid-v1' }
    })
  })

  it('selectModel on a tile rolls back the tile cache (all three) on failure', async () => {
    $activeSessionId.set('primary-1')

    const tileState = {
      ...createClientSessionState('tile-stored'),
      model: 'tile-model',
      provider: 'tile-provider',
      codingWorkflow: 'hybrid-v1' as const
    }

    $sessionStates.set({ 'tile-runtime': tileState })

    const requestGateway = vi.fn(async () => {
      throw new Error('boom')
    })

    let controls!: Controls

    render(<Harness onReady={value => (controls = value)} requestGateway={requestGateway} />)

    await expect(
      controls.selectModel({ model: 'claude-sonnet-4.6', provider: 'anthropic', sessionId: 'tile-runtime' })
    ).resolves.toBe(false)

    // Tile cache restored to the pre-switch snapshot (workflow stays hybrid-v1).
    expect($sessionStates.get()['tile-runtime']).toMatchObject({
      model: 'tile-model',
      provider: 'tile-provider',
      codingWorkflow: 'hybrid-v1'
    })
  })

  it('refreshCurrentModel reseeds the workflow draft from global info on a forced profile swap', async () => {
    vi.mocked(getGlobalModelInfo).mockResolvedValue({
      model: 'gpt-5.6-sol',
      provider: 'custom:sudo',
      coding_workflow: 'hybrid-v1'
    })

    const { result } = renderHook(() =>
      useModelControls({ queryClient: new QueryClient(), requestGateway: vi.fn() })
    )

    setProfileCodingWorkflowDefault('coupled-v1')
    await result.current.refreshCurrentModel(true)

    expect($currentCodingWorkflow.get()).toBe('hybrid-v1')
  })

  it('a stale default refresh does not overwrite a manual workflow pick (generation guard)', async () => {
    let resolveInfo!: (value: {
      model: string
      provider: string
      coding_workflow?: string
    }) => void

    const infoPromise = new Promise<Parameters<typeof vi.mocked>[0]>(r => (resolveInfo = r as never))
    vi.mocked(getGlobalModelInfo).mockReturnValueOnce(infoPromise as never)

    const { result } = renderHook(() =>
      useModelControls({ queryClient: new QueryClient(), requestGateway: vi.fn() })
    )

    // Kick off a default refresh, then make a manual Hybrid pick while it's in flight.
    const pending = result.current.refreshCurrentModel()
    await Promise.resolve()

    const generationBefore = getComposerSelectionGeneration()
    await result.current.selectHybrid()
    expect($currentCodingWorkflow.get()).toBe('hybrid-v1')
    expect(getComposerSelectionGeneration()).toBeGreaterThan(generationBefore)

    // The stale default response (carrying the profile default workflow) resolves
    // after the manual pick — it must NOT revert the workflow.
    resolveInfo({ model: 'gpt-5.6-sol', provider: 'custom:sudo', coding_workflow: 'coupled-v1' })
    await pending

    expect($currentCodingWorkflow.get()).toBe('hybrid-v1')
  })
})
