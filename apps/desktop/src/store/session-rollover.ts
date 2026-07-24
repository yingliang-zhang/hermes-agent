import type { RpcEvent } from '@/types/hermes'

import { $composerAttachments, $composerDraft, takeSessionDraft } from './composer'
import { getQueuedPrompts } from './composer-queue'
import { $activeGatewayProfile, normalizeProfileKey } from './profile'
import {
  $awaitingResponse,
  $busy,
  $connection,
  $selectedStoredSessionId,
  $sessions,
  resolveComposerSessionKey
} from './session'
import { isWatchWindow } from './windows'

export interface SessionRolloverOffer {
  compressionCount: number
  finalMessageId: string
  historyVersion: number
  predecessorStoredId: string
  runtimeId: string
  token: string
  turnGeneration: number
}

export interface CompletedSessionRollover {
  predecessorStoredId: string
  runtimeId: string
  successorStoredId: string
  token: string
}

const MAX_COMPLETED_TRANSITIONS = 32

const inFlightCommitTokens = new Set<string>()
const completedTransitions = new Set<string>()

const offerPayloadKeys = [
  'compression_count',
  'final_message_id',
  'history_version',
  'predecessor_stored_id',
  'runtime_id',
  'token',
  'turn_generation'
] as const

const completePayloadKeys = ['predecessor_stored_id', 'runtime_id', 'successor_stored_id', 'token'] as const

function strictRecord(value: unknown, expectedKeys: readonly string[]): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  const record = value as Record<string, unknown>
  const keys = Object.keys(record)

  if (keys.length !== expectedKeys.length || keys.some(key => !expectedKeys.includes(key))) {
    return null
  }

  return record
}

function nonemptyString(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value : null
}

function nonnegativeInteger(value: unknown): number | null {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : null
}

export function parseSessionRolloverOffer(payload: unknown): SessionRolloverOffer | null {
  const record = strictRecord(payload, offerPayloadKeys)

  if (!record) {
    return null
  }

  const compressionCount = nonnegativeInteger(record.compression_count)
  const finalMessageId = nonemptyString(record.final_message_id)
  const historyVersion = nonnegativeInteger(record.history_version)
  const predecessorStoredId = nonemptyString(record.predecessor_stored_id)
  const runtimeId = nonemptyString(record.runtime_id)
  const token = nonemptyString(record.token)
  const turnGeneration = nonnegativeInteger(record.turn_generation)

  if (
    compressionCount === null ||
    !finalMessageId ||
    historyVersion === null ||
    !predecessorStoredId ||
    !runtimeId ||
    !token ||
    turnGeneration === null
  ) {
    return null
  }

  return {
    compressionCount,
    finalMessageId,
    historyVersion,
    predecessorStoredId,
    runtimeId,
    token,
    turnGeneration
  }
}

export function parseCompletedSessionRollover(payload: unknown): CompletedSessionRollover | null {
  const record = strictRecord(payload, completePayloadKeys)

  if (!record) {
    return null
  }

  const predecessorStoredId = nonemptyString(record.predecessor_stored_id)
  const runtimeId = nonemptyString(record.runtime_id)
  const successorStoredId = nonemptyString(record.successor_stored_id)
  const token = nonemptyString(record.token)

  if (!predecessorStoredId || !runtimeId || !successorStoredId || successorStoredId === predecessorStoredId || !token) {
    return null
  }

  return { predecessorStoredId, runtimeId, successorStoredId, token }
}

export function sessionRolloverCapabilityParams(): { local_rollover_capable: boolean } {
  return {
    local_rollover_capable: $connection.get()?.mode === 'local' && !isWatchWindow()
  }
}

function eventMatchesActiveIdentity(
  event: Pick<RpcEvent, 'profile' | 'session_id'>,
  activeRuntimeId: null | string,
  runtimeId: string,
  predecessorStoredId: string
): boolean {
  return (
    sessionRolloverCapabilityParams().local_rollover_capable &&
    typeof event.profile === 'string' &&
    event.profile.trim().length > 0 &&
    normalizeProfileKey(event.profile) === normalizeProfileKey($activeGatewayProfile.get()) &&
    typeof event.session_id === 'string' &&
    event.session_id.length > 0 &&
    event.session_id === runtimeId &&
    runtimeId === activeRuntimeId &&
    $selectedStoredSessionId.get() === predecessorStoredId
  )
}

function offerScopes(offer: SessionRolloverOffer): string[] {
  const root = resolveComposerSessionKey(offer.predecessorStoredId, $sessions.get())

  return [...new Set([offer.runtimeId, offer.predecessorStoredId, root].filter((key): key is string => Boolean(key)))]
}

function scopeHasPendingComposerState(scope: string): boolean {
  const draft = takeSessionDraft(scope)

  return draft.text.trim().length > 0 || draft.attachments.length > 0 || getQueuedPrompts(scope).length > 0
}

export function eligibleSessionRolloverOffer(
  event: Pick<RpcEvent, 'payload' | 'profile' | 'session_id'>,
  activeRuntimeId: null | string
): SessionRolloverOffer | null {
  const offer = parseSessionRolloverOffer(event.payload)

  if (
    !offer ||
    !eventMatchesActiveIdentity(event, activeRuntimeId, offer.runtimeId, offer.predecessorStoredId) ||
    $busy.get() ||
    $awaitingResponse.get() ||
    $composerDraft.get().trim().length > 0 ||
    $composerAttachments.get().length > 0 ||
    offerScopes(offer).some(scopeHasPendingComposerState)
  ) {
    return null
  }

  return offer
}

export function eligibleCompletedSessionRollover(
  event: Pick<RpcEvent, 'payload' | 'profile' | 'session_id'>,
  activeRuntimeId: null | string
): CompletedSessionRollover | null {
  const completed = parseCompletedSessionRollover(event.payload)

  if (
    !completed ||
    !eventMatchesActiveIdentity(
      event,
      activeRuntimeId,
      completed.runtimeId,
      completed.predecessorStoredId
    )
  ) {
    return null
  }

  return completed
}

export function beginSessionRolloverCommit(token: string): boolean {
  if (!token || inFlightCommitTokens.has(token)) {
    return false
  }

  inFlightCommitTokens.add(token)

  return true
}

export function releaseSessionRolloverCommit(token: string): void {
  inFlightCommitTokens.delete(token)
}

function transitionKey(runtimeId: string, predecessorStoredId: string, successorStoredId: string): string {
  return JSON.stringify([runtimeId, predecessorStoredId, successorStoredId])
}

export function registerCompletedSessionRollover(completed: CompletedSessionRollover): boolean {
  if (!inFlightCommitTokens.delete(completed.token)) {
    return false
  }

  completedTransitions.add(
    transitionKey(completed.runtimeId, completed.predecessorStoredId, completed.successorStoredId)
  )

  if (completedTransitions.size > MAX_COMPLETED_TRANSITIONS) {
    const oldestTransition = completedTransitions.values().next().value

    if (oldestTransition !== undefined) {
      completedTransitions.delete(oldestTransition)
    }
  }

  return true
}

export function consumeCompletedSessionRollover(
  runtimeId: string,
  predecessorStoredId: string,
  successorStoredId: string
): boolean {
  return completedTransitions.delete(transitionKey(runtimeId, predecessorStoredId, successorStoredId))
}
