# Bounded author-text check: clip timing and landing prediction

Checked 22 October 2026, using one primary-source web search for the [author article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11068545/) and one text opening of the [v4 publisher card](https://data.mendeley.com/datasets/75m8vz7jr2/4). The article text result covered its body through Data Availability. No figures, videos, CSV bodies or TEST targets were opened. No inference or training ran.

## Exact source excerpts

Article, **Data Description**, description of the dataset's two components:

> clips of players hitting the ball

Article, **Background**, intended use:

> By integrating computer vision technology, players can predict the landing position of the tennis ball after contact.

Publisher v4 card, **Description**, per-shot segmentation:

> cut the videos to clips of each ball hit by player.

The article separately describes measured landing outcomes in CSV files and physical measurement after impact. This supports a stroke/trajectory-to-landing prediction task. It does not establish a body-motion-only input, a specific prediction horizon, or an observed-bounce requirement. The v4 card also proposes reconstructing motion from corresponding views. The article cites dataset v2, whereas the card is v4.

## Boundary finding and limits

The reviewed article/card text gives no exact start/end trigger, fixed post-strike interval, number of outgoing frames, or guarantee that the measured landing or numbered marker appears before a clip ends. It also gives no explicit general rule that clips end before landing. Therefore it does **not explicitly document the observed IST01/ISS01 truncation as a segmentation policy**. Prediction use is compatible with that DEV observation, but this is an inference, not an author timing specification.

The parent reports a strike bracket and end-before-established-outgoing-landing for the inspected DEV pair. This text-only check did not independently inspect those frames or verify timing, and it cannot generalize their duration or visibility to other clips. No additional sample was acquired. The known camera-pair numbering rule does not prove frame-exact synchronization or the CSV row join.
