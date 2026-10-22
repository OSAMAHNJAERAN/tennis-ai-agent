# Additional independent tennis footage leads

22 October 2026. Metadata-only follow-up. This supplements the [completed source-index audit](INDEPENDENT_TENNIS_MATCH_SOURCES_2026-10-22.md). No video, image, target annotations, archive or reserved TEST frame was downloaded or viewed. No model import or host-resource probe ran. The existing [acquisition ledger](../data/INDEPENDENT_SOURCE_ACQUISITION.md) and source/reservation manifests were consulted first; their exposed matches remain excluded.

## Concrete next acquisition decision

Preserve the **2013 NCAA men's singles final, Blaž Rola–Jarmere Jenkins**, as a new *metadata lead*, with the entire original match kept together. The Commons file identifies ZooTennis.com as the author, YouTube ID `AiM9alNFV2M`, match date 27 May 2013 and publication date 9 June 2013. It reports a 37,419,095-byte VP8/Vorbis WebM, 139.973 seconds, 640×480, SHA-1 `3f55af872e1b8d2418a79a0126421b62a89827de`, and CC BY 3.0, with an external-license review dated 19 February 2016. These are publisher/archive metadata, not local stream measurements. [File metadata and license review](https://commons.wikimedia.org/wiki/File:Bla%C5%BE_Rola_NCAAs_2013.webm)

Next retain the description revision and original-uploader text/license metadata. A later acquisition task may inventory this unchanged file's hash and timing without selecting frames. Native camera FPS/PTS, elevated baseline framing and continuous match coverage remain **unverified**; conversion may have changed timing. Keep it **ineligible for TEST/admission** pending provenance, whole-match overlap and timing/viewpoint checks. It supplies no independent physical-speed reference.

## Other new publisher findings

| Source | Established metadata | Acquisition decision |
| --- | --- | --- |
| OpenSportsLab, `OSL-loc-tennis-dallas` | HF reports five video rows, split 3/1/1, and total size 2.81 GB, rounded. No dataset card or license is displayed. Exact files, event identities, original FPS/PTS and upstream rights were not established; bounded metadata API/tree requests failed. | Keep metadata-only. No media acquisition or reuse of publisher train/test names as our qualification split. [Publisher dataset page](https://huggingface.co/datasets/OpenSportsLab/OSL-loc-tennis-dallas) |
| ACASVA, University of Surrey | Publisher supplies action labels/HOG3D features and states that original video is unavailable. Named tennis material includes Australian Open 2003 and Japan 2009. | Reject as a new raw-video source; do not confuse features with footage. [Publisher downloads](https://cvssp.org/acasva/Downloads.html) |
| Wang et al., tennis player actions for pose estimation | Author paper describes 2,000 extracted JPEGs, four pose classes and COCO labels from self-recordings at Taipei Tennis Center during 2023; original smartphone capture is described as 1280×720 at 30 FPS. The released filtered images have no demonstrated continuous source PTS or full-match identity. | Potential distinct training lead only after dataset-license review, not a qualification match source. This paper is separate from the previously audited NKUST side/top-view shot dataset. [Author data article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11282921/) |
| Bao and Zhang, TennisTV | Current author paper links ModelScope `FDUBay/TennisTV` and describes rally/stroke questions derived from earlier annotations. A usable raw-footage grant and exact original-match mapping were not established in readable publisher metadata. | No acquisition or holdout admission from this audit. [Author paper, v5](https://arxiv.org/html/2509.15602v5), [author-linked dataset](https://modelscope.cn/datasets/FDUBay/TennisTV) |

All prior CalTennis reservations, MUVY/UVY Wimbledon 2019 matches, NKUST practice clips, RacketVision development groups and previously exposed source compilations retain their existing roles. No source was promoted to TRAIN, validation, independent TEST or a public example by this follow-up.
