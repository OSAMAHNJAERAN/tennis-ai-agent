$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../../..')).Path
$diagnosticPath = Join-Path $workspaceRoot 'outputs/yolo26_court_reviewed_development_diagnostic_v1/diagnostic.json'
$observationsPath = Join-Path $PSScriptRoot 'observations.json'
$outputPath = Join-Path $PSScriptRoot 'retained_error_decomposition.json'
if (Test-Path -LiteralPath $outputPath) { throw 'Retain the existing review; never overwrite its derived artifact.' }
$diagnosticDigest = (Get-FileHash -LiteralPath $diagnosticPath -Algorithm SHA256).Hash.ToLowerInvariant()
$manual = Get-Content -LiteralPath $observationsPath -Raw | ConvertFrom-Json
if ($diagnosticDigest -ne $manual.source_diagnostic_sha256) { throw 'Saved diagnostic bytes changed.' }
$saved = Get-Content -LiteralPath $diagnosticPath -Raw | ConvertFrom-Json
if ($saved.status -ne 'COMPLETED_FIXED_PROTOCOL_DEVELOPMENT_DIAGNOSTIC_NOT_QUALIFICATION' -or $saved.input_hashes_unchanged -ne $true -or $saved.rows.Count -ne 20) { throw 'Require the exact completed twenty-DEV diagnostic.' }

function Get-ReviewMedian($values) {
    $sorted = @($values | Sort-Object)
    if (-not $sorted.Count) { return $null }
    $middle = [int][Math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2) { return [double]$sorted[$middle] }
    return ([double]$sorted[$middle - 1] + ([double]$sorted[$middle] - [double]$sorted[$middle - 1]) / 2)
}

function Get-PointStatistics($rows) {
    $all = @($rows)
    $visible = @($all | Where-Object { $_.visible_gt })
    $localized = @($visible | Where-Object { $_.has_error })
    $meanDx = $null; $meanDy = $null; $medianError = $null
    if ($localized.Count) {
        $meanDx = ($localized.dx_native_px | Measure-Object -Average).Average
        $meanDy = ($localized.dy_native_px | Measure-Object -Average).Average
        $medianError = Get-ReviewMedian $localized.error_native_px
    }
    return [ordered]@{
        visible_gt_denominator = $visible.Count; unknown_gt_excluded = $all.Count - $visible.Count
        finite_confident_matched_predictions = $localized.Count; missing_visible = $visible.Count - $localized.Count
        within10_px = @($visible | Where-Object { $_.within10_px }).Count
        within20_px = @($visible | Where-Object { $_.within20_px }).Count
        median_error_native_px = $medianError; mean_signed_dx_native_px = $meanDx; mean_signed_dy_native_px = $meanDy
        positive_dx_count = @($localized | Where-Object { $_.dx_native_px -gt 0 }).Count
        negative_dx_count = @($localized | Where-Object { $_.dx_native_px -lt 0 }).Count
    }
}

$pointRows = @(); $imageRows = @()
for ($index = 0; $index -lt 20; $index++) {
    $row = $saved.rows[$index]; $observation = $manual.observations[$index]
    if ($row.id -ne $observation.id -or $observation.index -ne ($index + 1)) { throw 'Manual observations must cover the exact ordered twenty overlay IDs.' }
    if ((Get-FileHash -LiteralPath $row.overlay.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $row.overlay.sha256) { throw 'A viewed overlay changed after the diagnostic.' }
    $imagePoints = @()
    foreach ($point in $row.assessment.points) {
        $dx = $null; $dy = $null
        if ($null -ne $point.error_native_px) {
            $dx = [double]($point.selected_prediction_xy_confidence_px[0]) - [double]($point.gt_xy_visibility_px[0])
            $dy = [double]($point.selected_prediction_xy_confidence_px[1]) - [double]($point.gt_xy_visibility_px[1])
        }
        $item = [PSCustomObject]@{ image_id = $row.id; context_group = $row.context_group; point_index = $point.point_index
            point_name = $point.point_name; visible_gt = $point.visible_gt; has_error = $null -ne $point.error_native_px
            error_native_px = $point.error_native_px; dx_native_px = $dx; dy_native_px = $dy
            within10_px = $point.within_10_px; within20_px = $point.within_20_px; missing_or_exclusion_reason = $point.missing_or_exclusion_reason }
        $imagePoints += $item; $pointRows += $item
    }
    $imageRows += [ordered]@{ index = $index + 1; id = $row.id; context_group = $row.context_group
        original_source_video_id = $row.original_source_video_id; publisher_frame_ordinal = $row.publisher_frame_ordinal
        source_fps = $null; source_pts = $null; source_time_base = $null; overlay = $row.overlay
        native_image = $row.image; label = $row.label; selected_bbox_iou = $row.assessment.selected_bbox_iou
        returned_instances = $row.assessment.returned_instances; instance_missing_reason = $row.assessment.instance_missing_reason
        observation = $observation.observation; failure_tags = $observation.failure_tags; localization = Get-PointStatistics $imagePoints }
}
$landmarks = foreach ($group in ($pointRows | Group-Object point_index | Sort-Object { [int]$_.Name })) {
    [ordered]@{ point_index = [int]$group.Name; point_name = $group.Group[0].point_name; statistics = Get-PointStatistics $group.Group }
}
$contexts = foreach ($group in ($pointRows | Group-Object context_group)) {
    [ordered]@{ context_group = $group.Name; statistics = Get-PointStatistics $group.Group }
}
$sides = foreach ($definition in @(
    @{name = 'left_semantic_slots'; indices = @(0,2,4,5,8,10)},
    @{name = 'right_semantic_slots'; indices = @(1,3,6,7,9,11)},
    @{name = 'service_centers'; indices = @(12,13)})) {
    [ordered]@{ group = $definition.name; point_indices = $definition.indices
        statistics = Get-PointStatistics @($pointRows | Where-Object { $definition.indices -contains $_.point_index }) }
}
$artifact = [ordered]@{ schema = 'tennis.yolo26.court-development-retained-error-review.v1'
    status = 'DERIVED_FROM_EXISTING_PREDICTIONS_AND_ACTUAL_ALL20_OVERLAY_REVIEW_NO_NEW_INFERENCE'
    created_utc = [DateTime]::UtcNow.ToString('o'); qualification_evidence = $false; independent_human_reviewed = $false; final_test_used = $false
    source_diagnostic = @{path = $diagnosticPath; sha256 = $diagnosticDigest}
    manual_review = @{path = $observationsPath; sha256 = (Get-FileHash -LiteralPath $observationsPath -Algorithm SHA256).Hash.ToLowerInvariant()}
    derivation_script_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    input_hashes_unchanged_in_diagnostic = $saved.input_hashes_unchanged; all20_viewed_overlay_hashes_rechecked = $true
    checkpoint = $saved.bindings.checkpoint; run_metadata = $saved.bindings.run_metadata; original_aggregates = $saved.aggregates
    actual_diagnostic_tensor_shape_union = $saved.actual_network_tensor_shapes
    tensor_shape_limitation = 'Hook records a union, not per-image/warmup associations. Square TRAIN previews and rectangular diagnostic tensors are observed; causality for native-point error is unproven.'
    interpretation = 'Post-hoc descriptive grouping only. Still ordinals have no verified FPS/PTS, context IDs are not independent matches, and within-video pairs are correlated.'
    per_image = $imageRows; per_landmark_signed_error = @($landmarks); per_context_signed_error = @($contexts); semantic_side_groups = @($sides); retained_point_errors = $pointRows
    next_repair = 'Root-review a frozen paired rect:true versus rect:false DEV-only transform investigation before considering further optimization or data expansion.'
    long_retrain_authorized = $false; gpu_or_inference_executed_for_this_review = $false; tests_executed_for_this_review = $false }
[IO.File]::WriteAllText($outputPath, ($artifact | ConvertTo-Json -Depth 30), [Text.UTF8Encoding]::new($false))
[PSCustomObject]@{status = $artifact.status; images = $imageRows.Count; point_records = $pointRows.Count; artifact = $outputPath
    sha256 = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash.ToLowerInvariant(); semantic_groups = $sides} | ConvertTo-Json -Depth 8
