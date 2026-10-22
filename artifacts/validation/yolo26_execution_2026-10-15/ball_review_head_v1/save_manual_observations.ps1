$ErrorActionPreference = 'Stop'
$workspace = 'C:\Semester 9\FYP2\tennis_ai_agent_bundle'
$reviewDirectory = Join-Path $workspace 'artifacts\validation\yolo26_execution_2026-10-15\ball_review_head_v1'
$manifestPath = Join-Path $workspace 'outputs\ball_data_repair_review_wave96_v3\review_aids_manifest.json'
$boardsPath = Join-Path $reviewDirectory 'inspection_boards.json'
$outputPath = Join-Path $reviewDirectory 'head_observations.json'
if (Test-Path -LiteralPath $outputPath) { throw 'Refusing to overwrite saved manual observations.' }
$manifestHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($manifestHash -ne 'c6ec5bd28d0a36af85afb5f8018b136f9bc9afbbae65996bf9aa938ab2ab4a17') { throw 'Unexpected reviewed manifest bytes.' }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$boards = Get-Content -LiteralPath $boardsPath -Raw | ConvertFrom-Json

# Literal manual observations made by /root/ball_head_finish after viewing all
# sixteen unmarked native/context boards and all four complete-frame overviews.
# Bounds below are visually selected native-patch pixel edges, not component
# proposals, not publisher-center-radius boxes, and not a model prediction.
# Format: selection index | x1,y1,x2,y2 or null | blur | observation.
$manual = @'
0|44,44,52,51|MILD|Small pale yellow-green horizontal smear below the far service line, separated from line and net; absent from this location in neighboring panels.
1|44,42,51,54|MODERATE|Yellow-green vertical smear in front of the bottom edge of the net; its vertical extent is distinct from the horizontal net tape and neighboring panels show motion onward.
2|45,44,50,51|MILD|Tiny pale moving spot in front of far-player backdrop, separated from player and lettering; same moving ball is visible at different positions in adjacent panels.
3|43,39,52,55|MODERATE|Isolated diagonal green streak over blue court to the right of the center service line; manually include the full visible streak.
4|43,42,53,51|MODERATE|Bright diagonal yellow-green streak against net mesh, separate from the printed lettering; neighboring panels show a different ball position.
5|45,40,53,57|MODERATE|Narrow diagonal yellow streak on the dark advertising backdrop, clear of fixed white sponsor lettering and moving between panels.
6|44,33,51,64|SEVERE|Long faint vertical white ball smear over purple court, left of the service line; low-contrast endpoints require about two native pixels of annotation tolerance.
7|44,44,51,51|MILD|Small yellow-green ball blob over far green run-off, distinct from white ground lettering and far-player limbs.
8|40,37,55,57|SEVERE|Long pale diagonal ball streak below the net over purple court; include the visible blur trail rather than a small center-radius square.
9|null|UNRESOLVED|Current ball proposal merges into the bright horizontal baseline. Neighboring panels support motion, but the current-frame ball boundary cannot be separated reliably from the line.
10|41,42,54,53|MODERATE|Isolated bright yellow diagonal oval over blue court; include the full visible smear and exclude its faint dark compression shadow.
11|44,45,51,52|MILD|Compact yellow-green moving spot above far baseline, distinct from the white SHANGHAI lettering and player/racket.
12|42,44,54,51|MODERATE|Horizontal yellow ball blur between far service line and net; full extent is distinct from both lines.
13|32,43,51,69|SEVERE|Long diagonal yellow blur above the near player's white cap; trail crosses a service line but remains identifiable above and below it. Full 96-pixel current panel, including beyond the central zoom crop, determines the bounds.
14|43,38,51,56|MODERATE|Bright vertical yellow streak over grass in near service box; exclude the detached faint compression halo.
15|44,44,54,52|MODERATE|Small diagonal yellow ball fragment to the left of far player's moving lower leg, visibly separated from the white shoe and shorts.
16|42,42,51,55|MODERATE|Yellow diagonal smear against net mesh; extent is distinguishable from the repetitive vertical net strands.
17|35,37,54,54|SEVERE|Long yellow diagonal streak on net; include the whole visible moving blur rather than net-grid highlights.
18|null|UNRESOLVED|Current publisher-centered patch contains far player's leg and shoe with no separable ball body. Adjacent frames do not justify inventing a current-frame visible box.
19|45,44,51,50|MILD|Small pale green-white ball above far player against dark sponsor backdrop; detached from betway lettering and player head.
20|42,43,53,55|MODERATE|Isolated pale green diagonal ball blur over blue service box; a thin older-looking vertical video trace is excluded from the visible ball extent.
21|43,43,51,51|MODERATE|Pale green compact moving blur against net logo/mesh; distinguishable from the fixed gray lettering by adjacent context panels.
22|null|UNRESOLVED|Target overlaps far player's ankle/shoe. A yellow-green fragment is plausible but its boundary is entangled with the shoe; reject rather than label shoe pixels as ball.
23|44,45,52,52|MILD|Small pale green ball left of the right singles sideline, separated from the line; motion across neighboring panels supports its identity.
24|null|UNRESOLVED|Target lies in net mesh with no clean separable ball outline in current panel; surrounding net highlights and low contrast make a tight box ambiguous.
25|45,43,51,53|MODERATE|Narrow yellow vertical ball blur just right of a diagonal white court line; visible streak remains separable from the line.
26|42,44,55,53|MODERATE|Yellow diagonal oval on grass above the net, below far service line; separate from the fixed line and net.
27|44,43,56,51|MODERATE|Bright yellow-orange diagonal oval over purple sponsor board; moving between panels and clear of lettering/spectators.
28|40,43,54,55|MODERATE|Large bright yellow-green ball blob over near grass left of player; manually follow its asymmetric visible outline.
29|44,43,53,52|MODERATE|Yellow-white diagonal streak left of far player's torso/racket, spatially distinct from yellow shirt and pale racket strings.
30|44,44,52,51|MILD|Small green-white ball at the racket-side edge, visibly brighter and separate from the dark racket frame; nearby racket excluded. This observation alone does not establish active-play event status.
31|44,43,52,51|MILD|Compact green ball immediately above horizontal service line; box stops above the line and excludes its horizontal pixels.
32|42,40,54,58|MODERATE|Long pale green diagonal ball smear over blue court below net and left of service center line; include its visible motion extent.
33|46,43,59,53|MODERATE|Pale yellow ball beside near player's head, visible to the right of dark hair. Full-frame low end-court angle is outside the initial elevated-camera scope; record visual extent but reject primary-scope export.
34|43,42,53,57|MODERATE|Bright pale green diagonal smear against net mesh; full-frame low end-court angle outside initial elevated-camera scope, so record extent but reject primary-scope export.
35|44,44,52,50|MILD|Small pale diagonal moving ball over far green run-off, distinct from far player's leg and baseline; low end-court source is outside initial primary-camera scope.
36|41,42,56,53|SEVERE|Pale diagonal ball streak against dark far backdrop, left of racket/player arm; manually bound the separate visible streak and exclude racket strings.
37|40,45,55,51|MODERATE|Horizontal pale peach ball smear over red clay, clear of nearby white sideline and static clay marks.
38|44,39,51,60|SEVERE|Faint long vertical peach blur against net mesh; current transient streak is distinct from fixed net strands but low-contrast endpoints have roughly two-pixel tolerance.
39|45,44,50,51|MILD|Tiny pale vertical ball over far green run-off to left of player's leg, distinct from limb and baseline.
40|40,43,55,51|MODERATE|Broad yellow-green horizontal oval over blue court right of near sideline; include visible bright extent but not detached dark shadow.
41|45,44,50,51|MILD|Small pale vertical moving spot against far dark advertising backdrop, clear of player/racket and circular sponsor logo.
42|41,46,57,50|MODERATE|Long narrow pale horizontal ball streak between service line and net on blue court; separate from both horizontal lines.
43|42,45,55,51|MODERATE|Pale horizontal ball smear over blue court, well above horizontal service line; adjacent panel confirms movement.
44|44,44,51,51|MODERATE|Faint green moving ball immediately left of far player's thigh, with a small visible gap from skin/shorts. Low contrast retained with two-pixel endpoint tolerance.
45|44,42,52,53|MODERATE|Pale green vertical ball smear over bright blue far run-off, separate from far player and baseline.
46|43,34,54,62|SEVERE|Long green diagonal vertical smear below net over blue court; manually include both bright bottom and faint top visible blur segments.
47|44,41,52,57|SEVERE|Pale green vertical smear crossing the far baseline; its green trail is separately visible on both sides of the white horizontal line, unlike the inseparable line proposal rejected at index 9.
'@

$extraViewedNative = @('positive_0076','positive_0113','positive_0129','positive_0136','positive_0152','positive_0249','positive_0286')
$observations = @()
$viewLog = @()
foreach ($board in $boards.boards) {
    $actual = (Get-FileHash -LiteralPath $board.path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $board.sha256) { throw "Reviewed board changed: $($board.path)" }
    $viewLog += [ordered]@{path=$board.path;sha256=$actual;kind=$board.kind;actually_viewed=$true}
}

foreach ($line in ($manual -split "`r?`n")) {
    if (-not $line.Trim()) { continue }
    $parts = $line -split '\|',4
    $index = [int]$parts[0]
    $item = $manifest.proposals.positive_center_proposals[$index]
    $decoded = @($manifest.decoded_sources | Where-Object { $_.clip -eq $item.clip })
    if ($decoded.Count -ne 1) { throw 'Missing unique source-video binding.' }
    $boardNumber = [math]::Floor($index / 3)
    $overviewNumber = 16 + [math]::Floor($index / 12)
    $itemBoards = @($boards.boards[$boardNumber],$boards.boards[$overviewNumber])
    $viewed = @($itemBoards | ForEach-Object { [ordered]@{path=$_.path;sha256=$_.sha256;kind=$_.kind} })
    $nativeDirect = $extraViewedNative -contains $item.proposal_id
    if ($nativeDirect) {
        $image = $item.images.native_frame
        if ((Get-FileHash -LiteralPath $image.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $image.sha256) { throw 'Native frame changed.' }
        $viewed += [ordered]@{path=$image.path;sha256=$image.sha256;kind='DIRECT_FULL_NATIVE_FRAME_VIEW'}
        $viewLog += [ordered]@{path=$image.path;sha256=$image.sha256;kind='DIRECT_FULL_NATIVE_FRAME_VIEW';actually_viewed=$true}
    }
    $bbox = $null
    $observedBbox = $null
    $center = $null
    $patchBox = $null
    if ($parts[1] -ne 'null') {
        $patchBox = @($parts[1] -split ',' | ForEach-Object { [int]$_ })
        $origin = $item.native_patch_origin_xy_px
        $observedBbox = @(($origin[0]+$patchBox[0]),($origin[1]+$patchBox[1]),($origin[0]+$patchBox[2]),($origin[1]+$patchBox[3]))
        $center = @((($observedBbox[0]+$observedBbox[2])/2.0),(($observedBbox[1]+$observedBbox[3])/2.0))
        $bbox = $observedBbox
    }
    $primaryScope = $item.source_id -ne 'match186'
    $decision = if ($null -eq $bbox) { 'REJECT_AMBIGUOUS' } elseif (-not $primaryScope) { 'REJECT_UNSUPPORTED_VIEW_VISIBLE_EXTENT_RECORDED' } else { 'ACCEPT_VISIBLE_BBOX' }
    if (-not $primaryScope) { $bbox = $null }
    $eventIdentity = switch ($item.source_id) {
        'match212' { 'LIBEMA_OPEN_VISIBLE_BRANDING_CITY_AND_DATE_UNVERIFIED' }
        { $_ -in @('match177','match200','match215','match221','match181','match186') } { 'MIAMI_OPEN_VISIBLE_COURT_AND_SPONSOR_CONTEXT_DATE_UNVERIFIED' }
        'match217' { 'DUBAI_DUTY_FREE_TENNIS_CHAMPIONSHIPS_EXPLICIT_BANNERS_DATE_UNVERIFIED' }
        'match224' { 'NITTO_ATP_FINALS_WITH_50_YEARS_1970_2020_COURT_TEXT_CITY_UNVERIFIED' }
        default { 'FROZEN_EVENT_ALIAS_VISUALLY_PLAUSIBLE_BUT_MATCH_AND_SESSION_UNVERIFIED' }
    }
    $eventConflict = $item.source_id -in @('match212','match177','match200','match215','match221','match217','match224')
    $observations += [ordered]@{
        selection_index=$index
        proposal_id=$item.proposal_id
        source_id=$item.source_id
        clip=$item.clip
        source_video_sha256=$decoded[0].video_sha256
        source_timeline_sha256=$decoded[0].timeline_sha256
        frame_index=$item.frame_index
        source_pts=$item.source_pts
        source_time_base=$item.source_time_base
        timestamp_fraction_seconds=$item.timestamp_fraction_seconds
        timestamp_seconds=$item.timestamp_seconds
        frozen_conservative_event_group=$item.conservative_event_group
        frozen_proposed_split=$item.proposed_split
        frozen_split_approved_for_export=$false
        observed_event_identity=$eventIdentity
        observed_identity_conflicts_with_frozen_alias=$eventConflict
        actual_match_identity_verified=$false
        actual_session_identity_verified=$false
        primary_camera_scope=$primaryScope
        camera_view= $(if ($primaryScope) { 'ELEVATED_END_COURT_SINGLES_BROADCAST' } else { 'LOW_END_COURT_OUTSIDE_INITIAL_ELEVATED_SCOPE' })
        decision=$decision
        accept_visible_bbox_xyxy_native_px=$bbox
        observed_visible_extent_xyxy_native_px=$observedBbox
        observed_center_xy_native_px=$center
        manually_selected_bbox_xyxy_patch_px=$patchBox
        native_patch_origin_xy_px=$item.native_patch_origin_xy_px
        visibility= $(if ($null -eq $observedBbox) { 'AMBIGUOUS_CURRENT_FRAME_NOT_A_NEGATIVE' } else { 'VISIBLE_OR_VISIBLE_BLUR' })
        blur_severity=$parts[2]
        endpoint_annotation_tolerance_native_px= $(if ($null -eq $observedBbox) { $null } else { 2 })
        annotation_tolerance_is_calibrated=$false
        center_definition='MIDPOINT_OF_MANUALLY_REVIEWED_VISIBLE_EXTENT_NOT_PHYSICAL_3D_CENTER'
        reason=$parts[3]
        native_frame=$item.images.native_frame
        context_images_seen_via_unmarked_board=@($item.images.context0,$item.images.context1,$item.images.context2)
        images_actually_viewed=$viewed
        direct_full_native_frame_viewed=$nativeDirect
        reviewer='AGENT:/root/ball_head_finish'
        independent_human_review=$false
        model_used=$false
        component_hypothesis_autoaccepted=$false
        supervised_label_export_approved=$false
    }
}
if ($observations.Count -ne 48) { throw 'Expected exactly 48 manual observations.' }
$result = [ordered]@{
    schema_version='tennis.manual-ball-visible-box-review.v1'
    saved_at_utc=[DateTimeOffset]::UtcNow.ToString('o')
    status='MANUAL_AGENT_VISUAL_OBSERVATIONS_PENDING_ROOT_GROUP_UNION_AND_EXPORT_AUDIT'
    reviewer='AGENT:/root/ball_head_finish'
    independent_human_review=$false
    private_development_only=$true
    qualification_evidence=$false
    manifest=[ordered]@{path=$manifestPath;sha256=$manifestHash}
    frozen_sampler=$manifest.frozen_sampler
    inspection_boards=[ordered]@{path=$boardsPath;sha256=(Get-FileHash -LiteralPath $boardsPath -Algorithm SHA256).Hash.ToLowerInvariant()}
    manual_observation_script=[ordered]@{path=$PSCommandPath;sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()}
    selected_manifest_indices=@(0,47)
    counts=[ordered]@{
        records=48
        accepted_visible_primary_scope_boxes=@($observations | Where-Object { $_.decision -eq 'ACCEPT_VISIBLE_BBOX' }).Count
        rejected_ambiguous=@($observations | Where-Object { $_.decision -eq 'REJECT_AMBIGUOUS' }).Count
        rejected_unsupported_view_with_observed_extent=@($observations | Where-Object { $_.decision -eq 'REJECT_UNSUPPORTED_VIEW_VISIBLE_EXTENT_RECORDED' }).Count
        accepted_negatives=0
        exported_labels=0
    }
    bbox_convention='INTEGER_NATIVE_PIXEL_EDGES_X1_Y1_INCLUSIVE_X2_Y2_EXCLUSIVE_VISUALLY_SELECTED_VISIBLE_BLUR_EXTENT'
    annotation_method='All three full unmarked 96x96 context panels plus nearest-neighbor 8x central zoom actually viewed for each item. Full-frame overviews establish camera/content context. Seven named native frames were additionally viewed directly. Bounds are literal manual observations; publisher centers/component proposals were not adopted as labels.'
    images_actually_viewed=$viewLog
    export_blockers=@('Frozen event aliases contain newly observed Miami/Washington, Dubai/Rome, Libema/Halle and ATP-Finals-year/city inconsistencies. Root must conservatively union all observed source-event edges, including rejected and unsupported items, and resolve train/development assignments before any label export.','Independent human review remains absent. These are private agent-reviewed development observations, not independent test truth.','Bounding boxes have uncalibrated manual endpoint tolerance; no detector, event, physical-speed or final-test accuracy is established.')
    model_used=$false
    gpu_used=$false
    source_or_dataset_writes=$false
    label_export_performed=$false
    observations=$observations
}
$json = $result | ConvertTo-Json -Depth 30
[System.IO.File]::WriteAllText($outputPath, $json + "`n", [System.Text.UTF8Encoding]::new($false))
[PSCustomObject]@{path=$outputPath;sha256=(Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash.ToLowerInvariant();counts=$result.counts} | ConvertTo-Json -Depth 5
