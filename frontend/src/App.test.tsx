import '@testing-library/jest-dom';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, beforeEach, afterEach, expect, it } from 'vitest';

import App from './App';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) => key,
    i18n: { changeLanguage: vi.fn() },
  }),
}));

describe('App pending actions', () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.clearAllMocks();
    vi.clearAllTimers();
    if (originalFetch) {
      global.fetch = originalFetch;
    }
    vi.useRealTimers();
  });

  it('prefills vote defaults for pending human action', async () => {
    const statusPayload = {
      sessionId: 'demo-session',
      completed: false,
      error: null,
      state: {
        round: 1,
        phase: 'VOTE',
        leader: 1,
        teamSize: 2,
        scores: { good: 0, evil: 0 },
        failedProposals: 0,
        currentProposal: { leader: 1, members: [1, 2], approved: null },
        proposals: [],
        missions: [],
        votes: [],
        speeches: [],
        players: [
          { seat: 1, role: null, role_name: null },
          { seat: 2, role: null, role_name: null },
          { seat: 3, role: null, role_name: null },
          { seat: 4, role: null, role_name: null },
          { seat: 5, role: null, role_name: null },
        ],
        winner: null,
      },
      pending: {
        requestId: 'vote-1',
        seat: 3,
        phase: 'VOTE',
        options: {
          teamSize: 2,
          availableSeats: [1, 2, 3, 4, 5],
          currentMembers: [1, 2],
          missionMembers: [1, 2],
          onMission: false,
          canFailMission: false,
        },
        instructions: 'vote please',
        stateSnapshot: {},
        createdAt: Date.now(),
      },
    };

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ sessionId: 'demo-session' }),
      })
      .mockResolvedValue({
        ok: true,
        json: async () => statusPayload,
      });

    global.fetch = fetchMock as unknown as typeof fetch;

    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTimeAsync });
    render(<App />);

    await user.click(screen.getByRole('button', { name: 'button.runHuman' }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));

    const approveButton = await screen.findByRole('button', { name: 'action.form.vote.approve' });
    const rejectButton = screen.getByRole('button', { name: 'action.form.vote.reject' });
    const reflectionField = screen.getByLabelText('action.form.reflection');

    expect(approveButton).toHaveClass('selected');
    expect(rejectButton).not.toHaveClass('selected');
    expect(reflectionField).toHaveValue('');
  });
});
