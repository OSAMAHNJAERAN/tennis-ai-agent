import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { NativeForecast } from './NativeForecast';
import { forecastAtFrame, forecastBindingValid, forecastBins, NativeForecastSchema } from '../data/nativeForecast';
import { analysisId, nativeFixture, sourceHash } from '../test/correctionFixtures';

describe('native causal forecast replay', () => {
  it('displays the replay prefix, preserves withheld gaps and seeks the actual event frame', () => {
    const value = nativeFixture(); const onSeek = vi.fn(); const view = render(<NativeForecast value={value} frame={5} onSeek={onSeek} />);
    expect(screen.getByText('Neutral prior · no performance points')).toBeInTheDocument(); expect(screen.getAllByText('P1 50.0%')).toHaveLength(3);
    fireEvent.click(screen.getByRole('button', { name: 'Seek to frame 10: forecast withheld' })); expect(onSeek).toHaveBeenCalledWith(10);
    fireEvent.change(screen.getByRole('slider', { name: 'Forecast history position' }), { target: { value: '1' } }); expect(onSeek).toHaveBeenCalledWith(10);
    view.rerender(<NativeForecast value={value} frame={10} onSeek={onSeek} />); expect(screen.getByText('Forecast withheld')).toBeInTheDocument(); expect(screen.queryByText('P1 50.0%')).not.toBeInTheDocument();
    expect(forecastAtFrame(value.history, -1)).toBeNull(); expect(screen.getByText(/not been validated/)).toBeInTheDocument();
  });
  it('shows confirmed results without a forecast probability', () => {
    const value = nativeFixture(); value.history[2]!.forecast = { ...value.history[2]!.forecast, status: 'confirmed_result', winner_id: 2 }; value.current = value.history[2]!.forecast;
    render(<NativeForecast value={value} frame={12} onSeek={vi.fn()} />); expect(screen.getByText('Confirmed result · Player 2')).toBeInTheDocument(); expect(screen.queryByText(/P1 \d/)).not.toBeInTheDocument();
  });
  it('labels posterior samples as completed points and exposes one keyboard history control', () => {
    const value = nativeFixture(); const posterior = { ...value.history[0]!.forecast, status: 'uncalibrated' as const, sample_count: 3 };
    value.history[0]!.forecast = posterior; render(<NativeForecast value={value} frame={0} onSeek={vi.fn()} />);
    expect(screen.getByText('Posterior estimate · 3 completed points')).toBeInTheDocument();
    expect(screen.getByRole('slider', { name: 'Forecast history position' })).toBeInTheDocument();
    expect(screen.getAllByRole('button').every(button => button.tabIndex === -1)).toBe(true);
  });
  it('rejects foreign sources/revisions, future point frontiers, phase and row clock changes', () => {
    const value = nativeFixture(); expect(forecastBindingValid(value, analysisId, 1, 20, sourceHash)).toBe(true);
    expect(forecastBindingValid(value, analysisId, 0, 20, sourceHash)).toBe(false);
    expect(forecastBindingValid(value, analysisId, 1, 20, 'f'.repeat(64))).toBe(false);
    for (const mutate of [(x: typeof value) => { x.history[1]!.uses_points_through_event_id = 4; },
      (x: typeof value) => { x.history[2]!.source_pts = 'bad'; }, (x: typeof value) => { x.history[2]!.frame_index = 9; }]) {
      const copy = structuredClone(value); mutate(copy); expect(forecastBindingValid(copy, analysisId, 1, 20, sourceHash)).toBe(false);
    }
    const invalid = structuredClone(value); invalid.current.status = 'prior_only'; expect(NativeForecastSchema.safeParse(invalid).success).toBe(false);
  });
  it('bounds long histories to 600 columns while retaining gaps and real frame identities', () => {
    const first = nativeFixture().history[0]!; const rows = Array.from({ length: 2400 }, (_, i) => ({ ...first, frame_index: i }));
    rows[17]!.forecast = nativeFixture().history[2]!.forecast;
    const bins = forecastBins(rows); expect(bins).toHaveLength(600); expect(bins.some(row => row.frame_index === 17 && row.forecast.status === 'unavailable')).toBe(true);
    expect(bins.every(row => rows.includes(row))).toBe(true); expect(forecastAtFrame(rows, 1337)?.frame_index).toBe(1337);
  });
});
