import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { AccountAccess } from './AccountAccess';
import { apiJson } from '../data/analysisApi';
import { mutate } from '../data/productApi';

vi.mock('../data/analysisApi', () => ({ apiJson: vi.fn() }));
vi.mock('../data/productApi', () => ({ mutate: vi.fn() }));
beforeEach(() => vi.resetAllMocks());
describe('account access', () => {
  it('signs in, rechecks the server session and clears the password', async () => {
    const changed = vi.fn();
    vi.mocked(apiJson).mockResolvedValueOnce({ authenticated: false, required: true, user: null })
      .mockResolvedValueOnce({ authenticated: true, required: true, user: { id: 'u1', username: 'player' } });
    vi.mocked(mutate).mockResolvedValue({ user: { id: 'u1', username: 'player' } });
    render(<AccountAccess onChange={changed} />);
    await screen.findByRole('heading', { name: 'Keep your match private' });
    const user = userEvent.setup();
    await user.type(screen.getByLabelText('Username'), 'player');
    await user.type(screen.getByLabelText('Password'), 'long-password-123');
    await user.click(screen.getByRole('button', { name: 'Sign in' }));
    expect(await screen.findByRole('heading', { name: 'Signed in as player' })).toBeInTheDocument();
    expect(changed).toHaveBeenCalledOnce();
    expect(mutate).toHaveBeenCalledWith('/api/auth/login', { username: 'player', password: 'long-password-123' });
  });
  it('announces a failed sign-in and keeps recovery controls', async () => {
    vi.mocked(apiJson).mockResolvedValue({ authenticated: false, required: true, user: null });
    vi.mocked(mutate).mockRejectedValue(new Error('Username or password is incorrect.'));
    render(<AccountAccess />);
    await screen.findByRole('heading', { name: 'Keep your match private' });
    const user = userEvent.setup();
    await user.type(screen.getByLabelText('Username'), 'player');
    await user.type(screen.getByLabelText('Password'), 'long-password-123');
    await user.click(screen.getByRole('button', { name: 'Sign in' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Username or password is incorrect.');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Sign in' })).toBeEnabled());
  });
});
