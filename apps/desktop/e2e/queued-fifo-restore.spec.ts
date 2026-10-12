/**
 * Every prompt the gateway accepted behind a running turn must come back after
 * a renderer restart, in FIFO order, once each, while that turn still runs.
 *
 * The gateway keeps the queue as a head (`queued_prompt`) plus later envelopes
 * (`queued_prompts`) and persists each accepted prompt when it accepts it, so
 * the stored row sits AHEAD of the running turn's reply. The live-session
 * projection used to serialize only the head: after a reload the later prompts
 * showed only at their stored position before the reply, and the head showed
 * twice (stored row plus projected copy).
 *
 * Real Electron, backend and mock provider. Two source-identified prompts are
 * queued on the gateway through a second gateway client in the same renderer
 * (`prompt.submit` with `queued: true`, as a queue drain sends them), so the
 * queue under test is the gateway's, not the renderer's local composer queue.
 */

import { MOCK_REPLY } from '../../../tests-js/scripts/mock-server'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, type Page, test } from './test'

const ACTIVE_PROMPT = 'E2E_QUEUE_RESTORE_ACTIVE'
const QUEUED = ['E2E_QUEUE_RESTORE_FIRST', 'E2E_QUEUE_RESTORE_SECOND'] as const
const MARKERS = [ACTIVE_PROMPT, ...QUEUED]

function viewport(page: Page) {
  return page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
}

/** The marker each user bubble starts with (a bubble's text also carries its time label). */
async function userBubbleMarkers(page: Page): Promise<string[]> {
  const texts = await page.evaluate(() =>
    Array.from(document.querySelectorAll<HTMLElement>('[data-slot="aui_thread-viewport"] [data-role="user"]')).map(
      node => node.textContent?.trim() ?? ''
    )
  )

  return texts.flatMap(text => MARKERS.filter(marker => text.startsWith(marker)))
}

/** Queue `texts` on the gateway's busy-turn FIFO through a second gateway client. */
async function queueOnGateway(page: Page, texts: readonly string[]): Promise<string[]> {
  return page.evaluate(async prompts => {
    const desktop = (window as any).hermesDesktop
    const connection = await desktop.getConnection()
    const socket = new WebSocket(connection.wsUrl)
    let nextId = 0
    const pending = new Map<number, (frame: any) => void>()

    socket.addEventListener('message', event => {
      try {
        const frame = JSON.parse(String(event.data))

        if (typeof frame.id === 'number') {
          pending.get(frame.id)?.(frame)
          pending.delete(frame.id)
        }
      } catch {
        /* event frames are not replies */
      }
    })

    await new Promise<void>((resolve, reject) => {
      socket.addEventListener('open', () => resolve(), { once: true })
      socket.addEventListener('error', () => reject(new Error('gateway socket failed')), { once: true })
    })

    const call = (method: string, params: Record<string, unknown>) =>
      new Promise<any>(resolve => {
        const id = ++nextId
        pending.set(id, resolve)
        socket.send(JSON.stringify({ jsonrpc: '2.0', id, method, params }))
      })

    const list = await call('session.active_list', {})
    const sessions: any[] = list.result?.sessions ?? []
    const live = sessions.find(row => row.status !== 'idle') ?? sessions.find(row => row.current)
    const sessionId = live?.id
    const statuses: string[] = []

    for (const [index, text] of prompts.entries()) {
      const reply = await call('prompt.submit', {
        session_id: sessionId,
        text,
        queued: true,
        message_id: `e2e-queued-${index + 1}`,
        submitted_at: Date.now() / 1000
      })

      statuses.push(reply.result?.status ?? `error:${reply.error?.message ?? 'none'}`)
    }

    socket.close()

    return statuses
  }, texts)
}

test.describe('queued FIFO restore', () => {
  let fixture: MockBackendFixture | null = null

  test.beforeEach(async () => {
    fixture = await setupMockBackend({ mockServer: { holdFirstStreamForPrompt: ACTIVE_PROMPT } })
    await waitForAppReady(fixture, 120_000)
  })

  test.afterEach(async () => {
    await fixture?.cleanup()
    fixture = null
  })

  test('a reload during a running turn restores every gateway-queued prompt once, in order', async () => {
    test.setTimeout(180_000)
    const testInfo = test.info()
    const { mock, page } = fixture!

    const composer = page
      .locator('[data-slot="composer-root"] [contenteditable="true"]')
      .filter({ visible: true })
      .first()

    await composer.click()
    await composer.fill(ACTIVE_PROMPT)
    await composer.press('Enter')
    await mock.waitForHeldStream()

    expect(await queueOnGateway(page, QUEUED)).toEqual(['queued', 'queued'])

    // Renderer restart while the active turn is still held open.
    await page.reload()
    await waitForAppReady(fixture!, 120_000)
    await expect(viewport(page)).toContainText(ACTIVE_PROMPT, { timeout: 60_000 })
    await expect(viewport(page)).toContainText(QUEUED[1], { timeout: 30_000 })
    await expect.poll(() => userBubbleMarkers(page), { timeout: 30_000 }).toEqual(MARKERS)
    await page.screenshot({ path: testInfo.outputPath('queued-fifo-after-reload.png') })

    // The provider has seen neither queued prompt yet: they are still waiting.
    expect(mock.receivedPrompts.filter(prompt => (QUEUED as readonly string[]).includes(prompt))).toEqual([])

    // Draining keeps one bubble per prompt, still in order.
    mock.releaseHeldStream()
    await expect
      .poll(() => mock.receivedPrompts.filter(prompt => (QUEUED as readonly string[]).includes(prompt)), {
        timeout: 90_000
      })
      .toEqual([...QUEUED])
    await expect
      .poll(() => page.locator('[data-slot="aui_assistant-message-content"]').filter({ hasText: MOCK_REPLY }).count(), {
        timeout: 90_000
      })
      .toBe(3)
    await expect.poll(() => userBubbleMarkers(page), { timeout: 30_000 }).toEqual(MARKERS)
    await page.screenshot({ path: testInfo.outputPath('queued-fifo-after-drain.png') })
  })
})
