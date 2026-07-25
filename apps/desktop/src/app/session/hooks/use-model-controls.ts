import { type QueryClient } from '@tanstack/react-query'
import { useCallback, useRef } from 'react'

import type { ModelSelection } from '@/app/shell/model-menu-panel'
import { getGlobalModelInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { manualPickRemoved, modelOptionsQueryKey } from '@/lib/model-options'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $activeSessionId,
  $currentCodingWorkflow,
  $currentModel,
  $currentProvider,
  DEFAULT_CODING_WORKFLOW,
  getComposerSelectionGeneration,
  getCurrentModelSource,
  markComposerSelectionManual,
  parseCodingWorkflow,
  resetDraftCodingWorkflowOverride,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider,
  setCurrentSessionCodingWorkflow,
  setDraftCodingWorkflowOverride,
  setProfileCodingWorkflowDefault
} from '@/store/session'
import { $sessionStates, sessionTileDelegate } from '@/store/session-states'
import type { CodingWorkflow, ModelOptionsResponse } from '@/types/hermes'

interface ModelControlsOptions {
  queryClient: QueryClient
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}

export function useModelControls({ queryClient, requestGateway }: ModelControlsOptions) {
  const { t } = useI18n()
  const copy = t.desktop
  const profileRefreshEpochRef = useRef(0)

  // All callbacks here read reactive session state from the store (.get())
  // rather than capturing it as a prop. The actions bag in wiring.tsx mutates
  // in place to keep a stable identity, so memoized surfaces capture these
  // callbacks once and never re-evaluate — a captured prop would be stale
  // forever. The store read is always current.
  const updateModelOptionsCache = useCallback(
    (
      sessionId: null | string,
      provider: string,
      model: string,
      includeGlobal: boolean,
      profile = $activeGatewayProfile.get(),
      codingWorkflow?: CodingWorkflow
    ) => {
      const patch = (prev: ModelOptionsResponse | undefined) => ({
        ...(prev ?? {}),
        provider,
        model,
        ...(codingWorkflow ? { coding_workflow: codingWorkflow } : {})
      })

      queryClient.setQueryData<ModelOptionsResponse>(modelOptionsQueryKey(profile, sessionId), patch)

      if (includeGlobal) {
        queryClient.setQueryData<ModelOptionsResponse>(modelOptionsQueryKey(profile), patch)
      }
    },
    [queryClient]
  )

  // Seed the composer's model state from the profile default. `force` reseeds
  // for a profile swap (the new profile has its own default); otherwise this
  // only fills an EMPTY selection so a user's pick (plain UI state in
  // $currentModel) survives the lifecycle refreshes that fire on boot / fresh
  // draft / session events. A live session owns the footer, so skip entirely.
  const refreshCurrentModel = useCallback(
    async (force = false) => {
      // A forced profile swap opens a new intent epoch; an older in-flight
      // response for a previous profile must stand down when it resolves.
      if (force) {
        profileRefreshEpochRef.current += 1
        resetDraftCodingWorkflowOverride()
      }

      const profileRefreshEpoch = profileRefreshEpochRef.current

      try {
        if ($activeSessionId.get()) {
          return
        }

        // A manual pick stays sticky UNLESS it was removed from the catalog (its
        // model no longer exists on the provider), in which case keeping it would
        // 404 every new chat — fall through to reseed from the profile default.
        // Reads the model-options cache the composer already populated; an
        // unknown/not-yet-loaded catalog conservatively preserves the pick.
        const keepManualPick = () => {
          if (force || !$currentModel.get() || getCurrentModelSource() !== 'manual') {
            return false
          }

          const options = queryClient.getQueryData<ModelOptionsResponse>(
            modelOptionsQueryKey($activeGatewayProfile.get())
          )

          return !manualPickRemoved(options?.providers, $currentProvider.get(), $currentModel.get())
        }


        // Snapshot the selection generation before awaiting so a picker click
        // that lands while getGlobalModelInfo is in flight wins over this older
        // default — value comparisons alone miss re-selecting the same row.
        const selectionGeneration = getComposerSelectionGeneration()
        const result = await getGlobalModelInfo()

        if (profileRefreshEpochRef.current !== profileRefreshEpoch) {
          return
        }

        const workflow = parseCodingWorkflow(result.coding_workflow) ?? DEFAULT_CODING_WORKFLOW
        setProfileCodingWorkflowDefault(workflow)

        if (
          $activeSessionId.get() ||
          getComposerSelectionGeneration() !== selectionGeneration ||
          keepManualPick()
        ) {
          return
        }

        if (typeof result.model === 'string') {
          setCurrentModel(result.model)
        }

        if (typeof result.provider === 'string') {
          setCurrentProvider(result.provider)
        }

        setProfileCodingWorkflowDefault(workflow)

        if (typeof result.model === 'string' || typeof result.provider === 'string') {
          setCurrentModelSource('default')
        }
      } catch {
        // The delayed session.info event still updates this once the agent is ready.
      }
    },
    [queryClient]
  )

  // Returns whether the switch succeeded so callers can await it before applying
  // follow-up changes. The composer model is plain UI state: with no live
  // session it's just stored (and shipped on the next session.create); with one
  // it's scoped to that session via config.set. It NEVER writes the profile
  // default — that lives in Settings → Model — so picking a model here can't
  // silently mutate global config.
  //
  // `selection.sessionId` targets a specific surface (tile). When omitted, the
  // primary `$activeSessionId` is used (overlay / legacy callers). A tile
  // switch must not touch the primary globals — and must not be blocked by a
  // busy primary turn.
  const selectRoute = useCallback(
    async ({
      codingWorkflow,
      model,
      provider,
      sessionId
    }: {
      codingWorkflow: CodingWorkflow
      model: string
      provider: string
      sessionId?: null | string
    }): Promise<boolean> => {
      const primaryRuntimeId = $activeSessionId.get()
      const liveSessionId = sessionId === undefined ? primaryRuntimeId : sessionId
      const touchesPrimary = !liveSessionId || liveSessionId === primaryRuntimeId

      const prevModel = touchesPrimary ? $currentModel.get() : ($sessionStates.get()[liveSessionId!]?.model ?? '')

      const prevProvider = touchesPrimary
        ? $currentProvider.get()
        : ($sessionStates.get()[liveSessionId!]?.provider ?? '')

      const prevCodingWorkflow = touchesPrimary
        ? $currentCodingWorkflow.get()
        : ($sessionStates.get()[liveSessionId!]?.codingWorkflow ?? DEFAULT_CODING_WORKFLOW)

      const prevSource = getCurrentModelSource()
      const liveGatewayProfile = $activeGatewayProfile.get()

      if (touchesPrimary) {
        setCurrentModel(model)
        setCurrentProvider(provider)
        if (liveSessionId) {
          setCurrentSessionCodingWorkflow(codingWorkflow)
        } else {
          setDraftCodingWorkflowOverride(codingWorkflow)
        }
        markComposerSelectionManual()
      } else if (liveSessionId) {
        // Optimistic tile paint — session.info will confirm; rollback on error.
        sessionTileDelegate()?.updateSession(liveSessionId, state => ({
          ...state,
          codingWorkflow,
          model,
          provider
        }))
      }

      updateModelOptionsCache(
        liveSessionId,
        provider,
        model,
        touchesPrimary && !liveSessionId,
        liveGatewayProfile,
        codingWorkflow
      )

      // No live session yet: the pick is pure UI state. session.create reads
      // $currentModel/$currentProvider and applies it as that session's override.
      if (!liveSessionId) {
        return true
      }

      try {
        await requestGateway('config.set', {
          session_id: liveSessionId,
          key: 'route',
          value: { coding_workflow: codingWorkflow, model, provider }
        })

        void queryClient.invalidateQueries({ queryKey: modelOptionsQueryKey(liveGatewayProfile, liveSessionId) })

        return true
      } catch (err) {
        if (touchesPrimary) {
          setCurrentModel(prevModel)
          setCurrentProvider(prevProvider)
          if (liveSessionId) {
            setCurrentSessionCodingWorkflow(prevCodingWorkflow)
          } else {
            setDraftCodingWorkflowOverride(prevCodingWorkflow)
          }
          setCurrentModelSource(prevSource)
        } else if (liveSessionId) {
          sessionTileDelegate()?.updateSession(liveSessionId, state => ({
            ...state,
            codingWorkflow: prevCodingWorkflow,
            model: prevModel,
            provider: prevProvider
          }))
        }

        updateModelOptionsCache(
          liveSessionId,
          prevProvider,
          prevModel,
          touchesPrimary && !liveSessionId,
          liveGatewayProfile,
          prevCodingWorkflow
        )
        notifyError(err, copy.modelSwitchFailed)

        return false
      }
    },
    [copy.modelSwitchFailed, queryClient, requestGateway, updateModelOptionsCache]
  )

  const selectModel = useCallback(
    (selection: ModelSelection): Promise<boolean> =>
      selectRoute({
        codingWorkflow: DEFAULT_CODING_WORKFLOW,
        model: selection.model,
        provider: selection.provider,
        sessionId: 'sessionId' in selection ? (selection.sessionId ?? null) : undefined
      }),
    [selectRoute]
  )

  const selectHybrid = useCallback(
    (selection: { sessionId?: null | string } = {}): Promise<boolean> =>
      selectRoute({
        codingWorkflow: 'hybrid-v1',
        model: 'gpt-5.6-sol',
        provider: 'custom:sudo',
        sessionId: selection.sessionId
      }),
    [selectRoute]
  )

  return { refreshCurrentModel, selectHybrid, selectModel, updateModelOptionsCache }
}
