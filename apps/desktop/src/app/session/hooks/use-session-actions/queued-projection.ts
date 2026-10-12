import type { SessionResumeResult } from '@/types/hermes'

/** One accepted next-turn prompt as the live-session projection carries it. */
export interface QueuedProjectionItem {
  messageId?: string
  /** Accept-time durable row: stored ahead of the running turn's reply. */
  rowId?: number
  submittedAt?: number
  user: string
}

/**
 * The gateway's accepted next-turn FIFO, oldest first. `queued.items` is the
 * whole queue; older gateways send only the head in `queued.user`, and a
 * payload whose items are all unusable falls back to that head.
 */
export function queuedProjectionItems(queued: SessionResumeResult['queued']): QueuedProjectionItem[] {
  const raw: unknown = queued?.items
  const items: QueuedProjectionItem[] = []

  for (const entry of Array.isArray(raw) ? raw : []) {
    if (!entry || typeof entry !== 'object') {
      continue
    }

    const { message_id: messageId, row_id: rowId, submitted_at: submittedAt, user } = entry as Record<string, unknown>

    const text = typeof user === 'string' ? user.trim() : ''

    if (!text) {
      continue
    }

    items.push({
      user: text,
      ...(typeof messageId === 'string' && messageId.trim() ? { messageId: messageId.trim() } : {}),
      ...(typeof rowId === 'number' && Number.isSafeInteger(rowId) && rowId > 0 ? { rowId } : {}),
      ...(typeof submittedAt === 'number' && Number.isFinite(submittedAt) ? { submittedAt } : {})
    })
  }

  if (items.length) {
    return items
  }

  const head = typeof queued?.user === 'string' ? queued.user.trim() : ''

  return head ? [{ user: head }] : []
}

/** ASCII-only, collision-free id fragment: well-formed text keeps its native
 *  URI encoding; a lone UTF-16 surrogate (which `encodeURIComponent` rejects)
 *  becomes `%uXXXX`, which no encoded literal `%` (`%25`) can produce. */
function encodeIdFragment(value: string): string {
  let encoded = ''
  let chunkStart = 0

  for (let index = 0; index < value.length; index++) {
    const unit = value.charCodeAt(index)
    const high = unit >= 0xd800 && unit <= 0xdbff
    const low = unit >= 0xdc00 && unit <= 0xdfff

    if (high && index + 1 < value.length) {
      const next = value.charCodeAt(index + 1)

      if (next >= 0xdc00 && next <= 0xdfff) {
        index++

        continue
      }
    }

    if (high || low) {
      encoded += `${encodeURIComponent(value.slice(chunkStart, index))}%u${unit.toString(16).toUpperCase()}`
      chunkStart = index + 1
    }
  }

  return encoded + encodeURIComponent(value.slice(chunkStart))
}

/**
 * Stable row id for a projected queued prompt. A source-identified prompt keeps
 * its id across reconnects and queue drains; an id-less head keeps the
 * single-slot id older renderer state already painted. `-id-` and `-index-`
 * are disjoint namespaces, so a message id of `"1"` never equals position 1.
 */
export function queuedProjectionRowId(item: QueuedProjectionItem, index: number, sessionId: string): string {
  if (item.messageId) {
    return `user-queued-${sessionId}-id-${encodeIdFragment(item.messageId)}`
  }

  return index === 0 ? `user-queued-${sessionId}` : `user-queued-${sessionId}-index-${index}`
}

/** Whether `id` is a projected queued-prompt row of `sessionId`. */
export function isQueuedProjectionRow(id: string, sessionId: string): boolean {
  const base = `user-queued-${sessionId}`

  return id === base || id.startsWith(`${base}-id-`) || id.startsWith(`${base}-index-`)
}

/**
 * Durable rows of prompts that are still queued. The gateway persists an
 * accepted queued prompt when it accepts it, i.e. BEFORE the running turn's
 * reply exists, so a transcript read during that turn has the prompt ahead of
 * the reply. While the projection still lists it as queued, its stored row is
 * dropped so the projection paints it once, after the live reply; a drained
 * prompt is no longer in the queue and keeps its row.
 */
function queuedAcceptRowIds(queued: SessionResumeResult['queued']): Set<number> {
  return new Set(queuedProjectionItems(queued).flatMap(item => (item.rowId ? [item.rowId] : [])))
}

/** `messages` without the stored user rows of still-queued prompts (see `queuedAcceptRowIds`). */
export function withoutQueuedAcceptRows<T extends { role: string }>(
  messages: T[],
  queued: SessionResumeResult['queued'],
  rowIdOf: (message: T) => number | undefined
): T[] {
  const rowIds = queuedAcceptRowIds(queued)

  if (!rowIds.size) {
    return messages
  }

  return messages.filter(message => {
    const id = message.role === 'user' ? rowIdOf(message) : undefined

    return id === undefined || !rowIds.has(id)
  })
}
