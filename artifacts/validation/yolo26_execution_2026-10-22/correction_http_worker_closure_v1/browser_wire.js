// Actual Number serialization against an explicitly synthetic local source.
const fs = require('node:fs');
const source = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const mode = process.argv[3];
const event = structuredClone(source.inputs[0].source_input);
let initial = null;
if (mode === 'initial_and_binding') {
  event.context.binding.evidence_ref = 'SYNTHETIC_HTTP_NODE_BINDING_REVIEW';
  const state = structuredClone(source.initial_state);
  state.fields.points_p2 = 1;
  state.fields.service_side.value = 'AD';
  initial = { expected_initial_state_sha256: source.initial_state_sha256,
    replacement_initial_state: state };
} else if (mode === 'semantic_unknown') {
  event.event_type = 'UNKNOWN_EVENT';
} else {
  throw new Error('Unrecognized synthetic test mode');
}
process.stdout.write(JSON.stringify({
  schema_version: 1,
  source_sha256: source.source_sha256,
  base_generation: source.base_generation,
  base_analysis_revision: source.base_analysis_revision,
  base_rule_replay_sha256: source.base_rule_replay_sha256,
  correction_id: `synthetic-node-${mode}-${require('node:crypto').randomUUID()}`,
  evidence_ref: 'EXPLICIT SYNTHETIC SOURCE ONLY; NO REAL MATCH ACCURACY',
  reason: `Synthetic HTTP worker regression: ${mode}`,
  initial_state_amendment: initial,
  input_amendments: [{ event_id: event.event_id,
    expected_input_fingerprint: source.inputs[0].input_fingerprint,
    replacement_source_input: event }],
}));
