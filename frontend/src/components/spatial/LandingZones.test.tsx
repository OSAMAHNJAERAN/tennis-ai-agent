import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { landingFixture } from '../../test/landingFixture';
import { TelemetryDashboard } from './TelemetryDashboard';

describe('shared landing presentation', () => {
  it('uses the same partition and visibility on the main court and mini court', () => {
    const run = landingFixture();
    const { container } = render(<TelemetryDashboard run={run} selectedEventId={5} onSelectEvent={vi.fn()} />);
    fireEvent.click(screen.getByRole('tab', { name: '2D Court Model' }));
    expect(container.querySelectorAll('[data-landing-zone="near_r0_c0"]')).toHaveLength(2);
    fireEvent.change(screen.getByRole('combobox', { name: 'Landing zone partition' }), { target: { value: 'service' } });
    expect(container.querySelectorAll('[data-landing-zone="near_r0_c0"]')).toHaveLength(0);
    expect(container.querySelectorAll('[data-landing-zone="near_service_left"]')).toHaveLength(2);
    fireEvent.click(screen.getByRole('button', { name: 'Landing zones' }));
    expect(container.querySelectorAll('[data-landing-zone]')).toHaveLength(0);
    expect(screen.getByRole('button', { name: 'Landing zones' })).toHaveAttribute('aria-pressed', 'false');
  });
  it('expires the zone fill after the contact interval while retaining event review', () => {
    const run = landingFixture();
    const { container } = render(<TelemetryDashboard run={run} selectedEventId={5} onSelectEvent={vi.fn()} />);
    fireEvent.click(screen.getByRole('tab', { name: '2D Court Model' }));
    fireEvent.change(screen.getByLabelText('Replay frame'), { target: { value: '70' } });
    expect(container.querySelectorAll('[data-landing-zone]')).toHaveLength(0);
    expect(screen.getAllByRole('button', { name: 'Inspect bounce at frame 50' })).toHaveLength(2);
  });
  it('does not add overlay controls to a run without reviewed zone records', () => {
    const run = { ...landingFixture(), landingZones: [] };
    render(<TelemetryDashboard run={run} selectedEventId={null} onSelectEvent={vi.fn()} />);
    expect(screen.queryByRole('combobox', { name: 'Landing zone partition' })).not.toBeInTheDocument();
  });
});
