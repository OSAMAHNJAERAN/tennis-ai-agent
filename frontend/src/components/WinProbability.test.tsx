import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { WinProbability } from './WinProbability';
import { emptyMetadata, type ProductMatch, type WinPrediction } from '../data/productApi';

const match = (predictions: WinPrediction) => ({ metadata: emptyMetadata(), predictions } as ProductMatch);
describe('prediction evidence states', () => {
  it('labels the neutral prior without attributing performance data', () => {
    render(<WinProbability match={match({ status: 'prior_only', reason: 'No player performance points yet.', probabilities: {
      game: { player_1: .5, player_2: .5 }, set: { player_1: .5, player_2: .5 }, match: { player_1: .5, player_2: .5 } } })} />);
    expect(screen.getByText('Neutral prior + current score')).toBeInTheDocument();
    expect(screen.getAllByText('50.0%')).toHaveLength(2);
    expect(screen.getByText(/has not passed independent calibration/)).toBeInTheDocument();
  });
  it('shows a completed outcome without a stale probability', () => {
    render(<WinProbability match={match({ status: 'confirmed_result', winner_id: 2, reason: 'Confirmed by rules.', probabilities: {
      game: { player_1: .1, player_2: .9 }, set: { player_1: .1, player_2: .9 }, match: { player_1: .1, player_2: .9 } } })} />);
    expect(screen.getByText('Player 2 wins.')).toBeInTheDocument();
    expect(screen.queryByText('90.0%')).not.toBeInTheDocument();
  });
  it('withholds invalid probabilities and an unresolved score', () => {
    render(<WinProbability match={match({ status: 'unavailable', reason: 'A consequential point requires review.' })} />);
    expect(screen.getByText('A consequential point requires review.')).toBeInTheDocument();
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument();
  });
});
