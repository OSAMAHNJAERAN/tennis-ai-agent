# CalTennis camera conventions: primary-source audit

Checked September 10, 2026. Dataset revision: `6c1c10b2e6c16d46ab1a83614f439d2fb91e9d7d`. This note separates published definitions, locally verified representation, and unresolved publication details.

## Camera/world transform

The paper defines projection using `X_camera = R X_world + T` and the inverse homogeneous transform `[R.T | -R.T T]`. It calibrates with court-line intersections and PnP. It does not specify a numbered world-coordinate court template, the court origin, or the mapping from x/y to geographical directions. The paper's stability metric projects onto the xy plane; this supports a z-height interpretation, but does not establish the exact origin or axis signs for each supplied file. [Paper, sections 4 and A.2, equations 14-16](https://arxiv.org/html/2606.20542v1#A1.SS2).

The acquired JSON contains `K`, `R_w2c`, `t_w2c`, `R_c2w`, `t_c2w`, `C`, `Rt_w2c`, `Rt_c2w`, `camera_model`, `image_shape`, and `dist_coeffs`. The existing local audit verifies inverse rotations and `t_w2c = -R_w2c C`. These checks establish algebraic consistency; they cannot identify court-line labels. Example publisher calibration: [September 18 NW01](https://huggingface.co/datasets/demalenk/caltennis/blob/6c1c10b2e6c16d46ab1a83614f439d2fb91e9d7d/camera_calibration/09_18_2025_16_15_court5/09_18_2025_17_25_35_000_court5_NW_01_17_26_16_546/calib.json). Its `camera_model` is `pinhole`, distortion is null, and calibration size is 1920x1080. Its workstation `annotation_file` string is provenance, not an accessible annotation release.

## Timestamp units and synchronization

The paper describes device timestamps with up to one second of inter-camera error. Its evaluation interpolates reconstructed poses and searches a global offset in [-1000,1000] milliseconds to minimize pose disagreement. Therefore, synchrony must not be inferred simply from matching session IDs or array indices. [Paper, section 4 and appendix A.2 temporal calibration](https://arxiv.org/html/2606.20542v1#A1.SS2).

The [pinned dataset card](https://huggingface.co/datasets/demalenk/caltennis/blob/6c1c10b2e6c16d46ab1a83614f439d2fb91e9d7d/README.md) lists timestamp files but does not define their numerical units, epoch, or whether estimated synchronization offsets were already applied. The saved 409-entry dataset inventory contains no path named for synchronization/offsets and no Python or shell source; that inventory is not claimed exhaustive. Existing local timing checks support milliseconds: within-stream relative timestamps agree with decoded presentation time within approximately 2.1-9.6 ms. This is a measured unit hypothesis, not independent verification of cross-view synchronization.

## Why 1080 calibration pixels may correspond to 1088 video rows

All three acquired videos decode to 1920x1088; all three calibrations specify 1920x1080. Nonzero bottom rows do not establish a padding transform. There is a concrete source-based explanation worth testing: the paper identifies `deface` as its anonymizer, and the cited software delegates encoding to ImageIO. [Paper, section 3.1](https://arxiv.org/html/2606.20542v1#S3.SS1).

In [deface commit `09b670db307b970cff6fad1848cf04d5f0810ec4`, lines 149-159](https://github.com/ORB-HD/deface/blob/09b670db307b970cff6fad1848cf04d5f0810ec4/deface/deface.py#L149), `video_detect` copies the supplied FFmpeg configuration, supplies the input FPS if absent, then calls `imageio.get_writer` with that configuration. This function does not itself set a macroblock override. The [deface README](https://github.com/ORB-HD/deface#cli-usage-and-options-summary) lists the default FFmpeg configuration as the libx264 codec alone. These are current upstream behavior, not evidence of the publisher's exact invocation.

In [imageio-ffmpeg commit `ae47d8028c237ca5507ceef1b843ee427b442887`, line 399](https://github.com/imageio/imageio-ffmpeg/blob/ae47d8028c237ca5507ceef1b843ee427b442887/imageio_ffmpeg/_io.py#L399), the writer defaults `macro_block_size` to 16. [Lines 549-560](https://github.com/imageio/imageio-ffmpeg/blob/ae47d8028c237ca5507ceef1b843ee427b442887/imageio_ffmpeg/_io.py#L549) round each incompatible dimension upward and append an FFmpeg scale filter. For 1920x1080, this yields `scale=1920:1088`, a resize rather than blank-row padding. Thus a vertical factor of 1088/1080 is a motivated diagnostic candidate. The exact ImageIO plugin/version, FFmpeg arguments, source video, and original-to-published image correspondence remain unavailable; the candidate must not silently become authoritative calibration. Pixel-center conventions also require care for subpixel claims.

## Code availability and practical consequence

The publisher website is pinned to commit `543c878446b68f150e967f801400fc12cbf8526d`. Its [index.html lines 79-85](https://github.com/ilonadem/caltennis-website/blob/543c878446b68f150e967f801400fc12cbf8526d/index.html#L79) contains `href="#"` for its Code link and the text `Code (coming soon)`. The public website tree has presentation assets, not calibration/export code. A bounded read of the author's public repository list and relevant repository trees did not locate the missing implementation; this does not prove none exists elsewhere.

Saved small source copies are in `artifacts/validation/vision_upgrade/caltennis_source_audit/conventions/`: publisher HTML/diagram, deface implementation, and imageio-ffmpeg writer. No downloaded source was executed. A final optional check of the ImageIO wrapper was rejected by automatic approval review because of a usage limit, so that additional link in the proposed export chain was not independently inspected.

Use the camera matrices for explicitly labeled projection diagnostics. Do not yet claim verified court coordinates, synchronized multiview ground truth, or physical speed accuracy. The paper reports collegiate and recreational participants overall; it does not label the skill level of the three selected clips. [Paper, section 3.1](https://arxiv.org/html/2606.20542v1#S3.SS1).
