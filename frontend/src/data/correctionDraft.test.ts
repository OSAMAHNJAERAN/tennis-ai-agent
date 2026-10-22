import { describe, expect, it } from 'vitest';
import { amendInitial, amendInput, commandBase, initialEditable, initialScoreDraft } from './correctionDraft';
import { CorrectionCommandSchema, RawInitialStateSchema } from './correctionApi';
import { reviewFixture } from '../test/correctionFixtures';

describe('reviewed correction drafts', () => {
  it('preserves all 37 fields and derives side/receiver for reviewed standard scores', () => {
    const source = reviewFixture(); const original = structuredClone(source.initial_state);
    const result = amendInitial(source, { p1: 4, p2: 3, server: 2, serve: 2 });
    expect(Object.keys(result.fields)).toHaveLength(37); expect(RawInitialStateSchema.safeParse(result).success).toBe(true);
    expect(result.fields).toMatchObject({ points_p1: 4, points_p2: 3, server_id: 2, receiver_id: 1,
      serve_attempt: 2, service_side: { value: 'AD' }, point_state: { value: 'SECOND_SERVE_WAITING' }, ball_state: { value: 'DEAD' } });
    expect(result.fields.completed_sets).toEqual(original.fields.completed_sets); expect(source.initial_state).toEqual(original);
  });
  it('derives tie-break service rotation and rejects completed or invalid point scores', () => {
    const source = reviewFixture(); source.initial_state.fields.tie_break_active = true; source.initial_state.fields.games_p1 = source.initial_state.fields.games_p2 = 6;
    expect(amendInitial(source, { p1: 3, p2: 2, server: 1, serve: 1 }).fields).toMatchObject({ server_id: 2, receiver_id: 1, service_side: { value: 'AD' } });
    expect(() => amendInitial(source, { p1: 7, p2: 5, server: 1, serve: 1 })).toThrow('completed tie-break');
    expect(() => amendInitial(source, { p1: .5, p2: 1, server: 1, serve: 1 })).toThrow('whole point counts');
    source.initial_state.fields.tie_break_active = false;
    expect(() => amendInitial(source, { p1: 4, p2: 4, server: 1, serve: 1 })).toThrow('Advantage');
  });
  it('leaves active rallies and completed contexts read-only', () => {
    const source = reviewFixture(); expect(initialScoreDraft(source.initial_state).p1).toBe(0);
    source.initial_state.fields.point_state.value = 'RALLY_LIVE'; expect(initialEditable(source.initial_state)).toBe(false);
    expect(() => amendInitial(source, initialScoreDraft(source.initial_state))).toThrow('read-only');
  });
  it('clones complete event measurements including exact PTS and rejects cross-contract signals', () => {
    const input = reviewFixture().inputs[0]!.source_input; const updated = amendInput(input, 'UNKNOWN_EVENT', null, null);
    expect(updated).toEqual({ ...input, event_type: 'UNKNOWN_EVENT' }); expect(input.event_type).toBe('SERVE_CONTACT');
    expect(updated.context.source_pts).toBe('9007199254740993'); expect(() => amendInput(input, 'SERVICE_LET', null, null)).toThrow('unsupported');
  });
  it('rejects stale generations and requires explicit review evidence', () => {
    const source = reviewFixture(); expect(() => commandBase(source, 'id', '', 'video')).toThrow('reason');
    source.is_current = false; expect(() => commandBase(source, 'id', 'review', 'video')).toThrow('newer analysis');
  });
  it('normalizes textarea line breaks and tabs to the native bounded plain-text contract', () => {
    const command = commandBase(reviewFixture(), 'stable-command', 'Contact missed.\nCompare\tthe return.', 'Original video\r\n00:01–00:02');
    expect(command.reason).toBe('Contact missed. Compare the return.'); expect(command.evidence_ref).toBe('Original video 00:01–00:02');
    expect(CorrectionCommandSchema.safeParse(command).success).toBe(true);
    expect(CorrectionCommandSchema.safeParse({ ...command, reason: 'Contains\ncontrol' }).success).toBe(false);
    expect(() => commandBase(reviewFixture(), 'id', 'Bad\u0000control', 'video')).toThrow('reason for correction contains unsupported control');
    expect(() => commandBase(reviewFixture(), 'id', 'review', 'x'.repeat(2001))).toThrow('video evidence reference must be 2000');
  });
});
