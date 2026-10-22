"""Render retained real-score metrics; never refit or rerun a final test."""
import base64
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = ROOT / 'data/prediction/livetennisapi_2026_06_snapshot_import_v3_test_evaluation.json'
    if digest(source) != source.with_suffix(source.suffix + '.sha256').read_text().split()[0]:
        raise ValueError('Test evaluation receipt hash mismatch')
    result = json.loads(source.read_text(encoding='utf-8'))
    if result['qualification_status'] != 'UNQUALIFIED_PRIVATE_PUBLISHER_SCORE_ENGINEERING' or result['production_promotion'] is not False:
        raise ValueError('This report is restricted to retained private unqualified evidence')
    output = ROOT / 'artifacts/validation/yolo26_execution_2026-10-09/prediction_snapshot_v3_figure_v2'
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    metrics = result['metrics']['metrics']
    raw, calibrated = metrics['neutral_score_only_baseline'], metrics['development_calibrated_rules']
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), gridspec_kw={'width_ratios': [1.3, 1]})
    fig.patch.set_facecolor('#fafbf8')
    for ax in axes:
        ax.set_facecolor('#fafbf8')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.18, axis='y')
    axes[0].plot([0, 1], [0, 1], '--', color='#87938a', linewidth=1, label='Perfect reliability reference')
    for record, color, label, marker in ((raw, '#9b694e', 'Neutral score-only', 'o'),
                                         (calibrated, '#27664c', 'Development-calibrated (unqualified)', 's')):
        bins = [row for row in record['reliability'] if row['snapshot_count']]
        axes[0].plot([row['mean_probability'] for row in bins], [row['observed_win_frequency'] for row in bins],
                     color=color, marker=marker, markersize=5, linewidth=1.5, label=label)
    axes[0].set(xlim=(0, 1), ylim=(-.03, 1.03), xlabel='Mean forecast P(player 1 wins match)',
                ylabel='Observed player 1 match-win frequency', title='Test reliability · fixed ten bins')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    labels, values = ['Brier', 'Log loss'], [raw, calibrated]
    for offset, record, color, label in ((-.16, values[0], '#9b694e', 'Neutral score-only'),
                                         (.16, values[1], '#27664c', 'Development-calibrated')):
        heights = [record['brier'], record['log_loss']]
        bars = axes[1].bar(np.arange(2) + offset, heights, width=.30, color=color, label=label)
        axes[1].bar_label(bars, labels=[f'{value:.4f}' for value in heights], padding=4, fontsize=9)
    axes[1].set(xticks=np.arange(2), xticklabels=labels, ylabel='Equally match-weighted loss (lower is better)',
                ylim=(0, .44), title='Held-out test loss · no refitting')
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle('Private tennis score forecast evaluation · 19 test matches / 1,979 snapshots', fontsize=13, x=.51)
    fig.text(.06, .025, '114 eligible / 5,380 audited matches. Same-publisher terminal targets; no independent human result review.\n'
             'Each match has equal weight. Reliability is descriptive; substantial bin gaps remain. No vision or deployment qualification.', fontsize=8, color='#44514a')
    fig.tight_layout(rect=(0, .09, 1, .92))
    figure = output / 'calibration.png'
    fig.savefig(figure, dpi=160, facecolor=fig.get_facecolor())
    fig.savefig(output / 'calibration.svg', facecolor=fig.get_facecolor())
    plt.close(fig)
    rows = ''.join(f'<tr><td>{index/10:.1f}–{(index+1)/10:.1f}</td>'
                   f'<td>{a["match_count"]} / {a["snapshot_count"]}</td><td>{a["mean_probability"]:.3f} / {a["observed_win_frequency"]:.3f}</td>'
                   f'<td>{b["match_count"]} / {b["snapshot_count"]}</td><td>{b["mean_probability"]:.3f} / {b["observed_win_frequency"]:.3f}</td></tr>'
                   for index, (a, b) in enumerate(zip(raw['reliability'], calibrated['reliability'])))
    image = base64.b64encode(figure.read_bytes()).decode('ascii')
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Private tennis forecast evaluation</title><style>body{{font:16px system-ui;color:#20352b;max-width:1120px;margin:32px auto;padding:24px;background:#fafbf8}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;text-align:left;border-bottom:1px solid #d6ded8}}.state{{font-weight:700;color:#895723}}@media print{{body{{margin:0;padding:12px}}}}</style>
<h1>Private tennis forecast evaluation</h1><p class="state">Unqualified engineering evidence · 9 October 2026</p>
<p>A frozen development fit was applied to 19 separate test matches. No missing point winners or serve counts were inferred. The raw forecast equals the neutral score-only baseline; only the development-fitted logistic transformation adds a measured loss improvement.</p>
<img alt="Test reliability and equally match-weighted Brier/log-loss comparison" src="data:image/png;base64,{image}">
<p>Test Brier: {raw['brier']:.6f} → {calibrated['brier']:.6f}; log loss: {raw['log_loss']:.6f} → {calibrated['log_loss']:.6f}. The paired match-bootstrap improvement intervals, conditional on fixed calibration parameters, are retained in the linked source receipt.</p>
<p>Only 114 of 5,380 audited groups were eligible (57 development, 38 calibration assessment, 19 test). Targets come from this same publisher's completed score. Independent results, human review and match-specific alternative-format notices are absent. Bin-frequency gaps remain large, and the selection represents a sparse regular June grass-event subset. These results do not qualify uploaded-video scoring, physical measurements, live processing or deployment.</p>
<table><tr><th>Bin</th><th>Raw matches / snapshots</th><th>Raw mean / observed</th><th>Calibrated matches / snapshots</th><th>Calibrated mean / observed</th></tr>{rows}</table>
<p>Protocol SHA256: <code>{escape(result['protocol_sha256'])}</code><br>Development-fit SHA256: <code>{escape(result['development_fit_sha256'])}</code><br>Source evaluation SHA256: <code>{digest(source)}</code></p>
<p><a href="{escape(os.path.relpath(source, output).replace(chr(92), '/'), quote=True)}">Retained evaluation receipt</a> · <a href="{escape(os.path.relpath(ROOT / 'docs/experiments/PUBLISHER_SCORE_SNAPSHOT_V3_RESULTS_2026-10-09.md', output).replace(chr(92), '/'), quote=True)}">Definitions, full coverage and limitations</a></p></html>'''
    (output / 'report.html').write_text(html, encoding='utf-8')
    (output / 'script_snapshot.py').write_bytes(Path(__file__).read_bytes())
    receipt = dict(status='RENDERED_RETAINED_METRICS_NOT_REEVALUATED', created_utc=datetime.now(timezone.utc).isoformat(),
                   source=str(source.relative_to(ROOT)), source_sha256=digest(source), script_sha256=digest(__file__),
                   production_qualified=False, human_independent_review=False,
                   artifacts={path.name: digest(path) for path in output.iterdir() if path.is_file()})
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
