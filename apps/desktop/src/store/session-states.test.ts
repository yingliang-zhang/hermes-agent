import { afterEach, describe, expect, it } from 'vitest'

import { group, split } from '@/components/pane-shell/tree/model'
import { createClientSessionState } from '@/lib/chat-runtime'
import {
  $activeSessionStoredIdRotation,
  setActiveSessionId,
  setActiveSessionStoredIdRotation
} from '@/store/session'
import {
  beginSessionRolloverCommit,
  registerCompletedSessionRollover,
  releaseSessionRolloverCommit
} from '@/store/session-rollover'
import type { SessionTile } from '@/store/session-states'
import {
  clearAllSessionStates,
  orderTilesByTree,
  publishSessionState,
  selectionHomesToWorkspace
} from '@/store/session-states'

const tile = (storedSessionId: string): SessionTile => ({ storedSessionId })
const tilePane = (id: string) => `session-tile:${id}`

describe('orderTilesByTree', () => {
  it('no-ops (null) without a tree or below two tiles', () => {
    expect(orderTilesByTree(null, [tile('a'), tile('b')])).toBeNull()
    expect(orderTilesByTree(group([tilePane('a')]), [tile('a')])).toBeNull()
  })

  it('reorders tiles to layout-tree encounter order across a split', () => {
    const tree = split('row', [group(['workspace', tilePane('b')]), group([tilePane('a')])])

    expect(orderTilesByTree(tree, [tile('a'), tile('b')])).toEqual([tile('b'), tile('a')])
  })

  it('returns null when the array already matches strip order (skip persist)', () => {
    const tree = split('row', [group([tilePane('b')]), group([tilePane('a')])])

    expect(orderTilesByTree(tree, [tile('b'), tile('a')])).toBeNull()
  })

  it('sorts not-yet-adopted tiles after placed ones, stably', () => {
    const tree = group(['workspace', tilePane('b')])

    expect(orderTilesByTree(tree, [tile('a'), tile('b'), tile('c')])).toEqual([tile('b'), tile('a'), tile('c')])
  })
})

describe('selectionHomesToWorkspace', () => {
  const tiles = [tile('a'), tile('b')]

  it('homes for a null selection or a non-tile session', () => {
    expect(selectionHomesToWorkspace(null, tiles)).toBe(true)
    expect(selectionHomesToWorkspace('c', tiles)).toBe(true)
  })

  it('skips homing when the selected id is already an open tile', () => {
    expect(selectionHomesToWorkspace('a', tiles)).toBe(false)
  })
})

describe('active stored-session rotation kind', () => {
  const runtimeId = 'runtime-rollover'
  const predecessorStoredId = 'stored-before'
  const successorStoredId = 'stored-after'
  const token = 'rollover-token'

  afterEach(() => {
    clearAllSessionStates()
    setActiveSessionId(null)
    setActiveSessionStoredIdRotation(null)
    releaseSessionRolloverCommit(token)
  })

  it('consumes an exact completed rollover once and leaves later rotations as compression', () => {
    setActiveSessionId(runtimeId)
    publishSessionState(runtimeId, createClientSessionState(predecessorStoredId))

    expect(beginSessionRolloverCommit(token)).toBe(true)
    expect(
      registerCompletedSessionRollover({ predecessorStoredId, runtimeId, successorStoredId, token })
    ).toBe(true)

    publishSessionState(runtimeId, createClientSessionState(successorStoredId))
    expect($activeSessionStoredIdRotation.get()).toEqual({
      kind: 'rollover',
      nextStoredSessionId: successorStoredId,
      previousStoredSessionId: predecessorStoredId,
      runtimeSessionId: runtimeId
    })

    setActiveSessionStoredIdRotation(null)
    publishSessionState(runtimeId, createClientSessionState(predecessorStoredId))
    setActiveSessionStoredIdRotation(null)
    publishSessionState(runtimeId, createClientSessionState(successorStoredId))

    expect($activeSessionStoredIdRotation.get()).toEqual({
      kind: 'compression',
      nextStoredSessionId: successorStoredId,
      previousStoredSessionId: predecessorStoredId,
      runtimeSessionId: runtimeId
    })
  })
})
