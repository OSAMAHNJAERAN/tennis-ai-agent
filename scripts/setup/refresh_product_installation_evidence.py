"""Append completed restoration evidence without replacing historical receipts."""
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'artifacts/validation/product_installation'


def read(name):
    path = EVIDENCE / name
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def normalized(name):
    return name.lower().replace('_', '-').replace('.', '-')


def refresh(manifest):
    validation = manifest['validation']
    receipts = []
    def receipt(name, report, scope):
        receipts.append({'path': 'artifacts/validation/product_installation/' + name,
                         'verifiedAt': report.get('verifiedAt'), 'scope': scope,
                         'passed': report.get('passed', report.get('checksPassed')),
                         'historicalReceiptPreserved': True})

    pins = {normalized(name): row['version'] for name, row in manifest['python']['runtimeDistributions'].items()}
    clean = read('clean_install.json')
    resolution = read('observed_dependency_resolution.json')
    for name, report, key in [('observed_dependency_resolution.json', resolution, 'observedTransitiveConstraintResolution'),
                              ('clean_install.json', clean, 'cleanInstall')]:
        if report is None:
            continue
        selected = {normalized(row['metadata']['name']): row['metadata']['version'] for row in report['install']}
        if selected != pins:
            raise ValueError(name + ' does not match the manifest runtime versions')
        validation[key] = f'PASSED: {len(selected)} exact recorded runtime distributions; see {name}.'
        receipt(name, report, 'Package resolution/install receipt; no model inference or accuracy qualification.')

    host = read('clean_host_verification.json')
    if host and host['checksPassed']:
        expected_checks = {row['name']: row for row in host['checks']}
        for name, row in manifest['python']['runtimeDistributions'].items():
            if expected_checks['package:' + name]['expected'] != row['version']:
                raise ValueError('Historical package check differs from manifest: ' + name)
        for asset in manifest['assets']:
            if expected_checks['asset:' + asset['path']]['expectedSha256'] != asset['sha256']:
                raise ValueError('Historical asset check differs from manifest: ' + asset['path'])
        validation['apiSmokeWithoutWorker'] = f"PASSED: {len(host['checks'])} isolated installation/API checks; health 200, empty library, invalid upload 415, worker disabled, no model imports."
        receipt('clean_host_verification.json', host, 'Historical isolated host/API receipt; manifest metadata has since been updated. No inference.')

    health = read('clean_dependency_health.json')
    if health and health['passed']:
        validation['cleanDependencyHealth'] = 'PASSED: pip check reports no broken requirements in the isolated environment.'
        receipt('clean_dependency_health.json', health, 'Fresh isolated dependency consistency check; no inference.')

    load = read('offline_model_asset_load.json')
    if load and load['passed']:
        validation['offlineModelLoadInCleanEnvironment'] = f"PASSED: all four retained model assets loaded on CPU in {load['elapsedSeconds']:.2f}s, with zero network attempts; not an inference check."
        receipt('offline_model_asset_load.json', load, 'Historical real four-model CPU load; no inference or accuracy qualification.')
        validation['modelLoadAndInferenceInCleanEnvironment'] = 'MODEL_LOAD_PASSED; real pipeline inference smoke not yet recorded.'

    source = read('wasb_source_restore_check.json')
    if source and source['passed']:
        validation['wasbSourceRestore'] = 'PASSED: exact commit and four required runtime/license file hashes restored independently.'
        receipt('wasb_source_restore_check.json', source, 'Independent source checkout restore; Windows core.autocrlf=true byte policy.')
        for dependency in manifest['sourceDependencies']:
            if dependency['path'] == 'artifacts/research/WASB-SBDT':
                dependency['windowsLineEndingPolicy'] = 'core.autocrlf=true reproduces the pinned required-file bytes.'

    frontend = read('frontend_restore_check.json')
    if frontend and frontend['passed']:
        validation['frontendFrozenInstall'] = f"PASSED: isolated frozen offline restore, {frontend['packageStoreDirectories']} package-store directories; Node {frontend['nodeVersion']}, effective pnpm {frontend['pnpmVersion']} (launcher {frontend['launcherPnpmVersion']}); original manifests unchanged."
        receipt('frontend_restore_check.json', frontend, 'Fresh cached frozen dependency restore; production build and UI tests are separate.')

    smoke = read('real_pipeline_smoke.json')
    for name, scope in [('real_pipeline_smoke_online_probe_20261001.json', 'Initial pipeline completed, but strict smoke failed because a network probe was blocked.'),
                        ('real_pipeline_smoke_disk_failure_20261001.json', 'Offline inference/render completed with zero network attempts, but export failed when disk free space reached zero.'),
                        ('frontend_restore_check_launcher_probe_20261001.json', 'Frozen install succeeded using project pnpm 11.19.0; initial version probe incorrectly inspected the external launcher.')]:
        attempt = read(name)
        if attempt:
            receipt(name, attempt, scope)
    if smoke and smoke['passed']:
        validation['modelLoadAndInferenceInCleanEnvironment'] = f"PASSED_SHORT_SOFTWARE_SMOKE: selected pipeline completed on 30 original-pixel frames at native 60 FPS in {smoke['elapsedSeconds']:.2f}s, all exports parsed, 30 annotated frames decoded, zero network attempts."
        validation['cleanPipelineSmokeScope'] = 'Half-second excerpt only; no full-match performance, independent accuracy, or public deployment qualification.'
        validation['cleanPipelineSmokeLimitations'] = 'Court calibration rejected this excerpt; selected player roles and meaningful pose/racket measurements were withheld. All four assets loaded, and the selected pipeline exported its unavailable states correctly.'
        receipt('real_pipeline_smoke.json', smoke, validation['cleanPipelineSmokeScope'])
        manifest['unresolvedReleaseRequirements'][-1] = 'Full-match operational validation, fresh independent labels, accuracy gates and supported-angle public demos remain required before claiming a qualified production release.'
    else:
        manifest['unresolvedReleaseRequirements'][-1] = 'Clean end-to-end inference, full-match operational validation, fresh independent labels, accuracy gates and supported-angle public demos remain required before claiming a qualified production release.'
    manifest['evidenceUpdatedAt'] = datetime.now(timezone.utc).isoformat()
    manifest['validationEvidence'] = receipts
    return manifest


def main():
    path = ROOT / 'docs/deployment/PRODUCT_RESTORE_MANIFEST.json'
    manifest = refresh(json.loads(path.read_text(encoding='utf-8')))
    path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    report = {'updatedAt': manifest['evidenceUpdatedAt'], 'validation': manifest['validation'],
              'validationEvidence': manifest['validationEvidence'], 'productionQualified': False,
              'publicDeploymentPermissionsVerified': False, 'modelsImported': False, 'inferenceRun': False,
              'note': 'Manifest status refresh only; original clean-host and offline-model-load reports remain historical receipts.'}
    (EVIDENCE / 'restore_validation_refresh.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'updatedAt': report['updatedAt'], 'evidenceReceipts': len(report['validationEvidence']),
                      'productionQualified': False, 'modelsImported': False}))


if __name__ == '__main__':
    main()
