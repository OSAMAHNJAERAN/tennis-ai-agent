# NKUST v4 indoor DEV source inventory

This folder retains official public metadata only. It does not contain video or landing CSV bodies. The v4 publisher page applies CC BY 4.0 and requires matching shot-category/view numbers; attribution is retained in the receipt. The folder hierarchy and file metadata bind one indoor straight-shot camera pair, without inventing a CSV row join or camera synchronization.

`requests.json` records exact requested URLs, HTTP status, retrieval UTC, content type, ETag, response byte count and SHA256. The four JSON response files preserve the exact downloaded response bytes. `publisher_bundle.js` is the public script observed on the publisher page; the receipt captures the anonymous file/folder route excerpts and pins the script. No download or view endpoint was followed.

Run `python make_receipt.py` to verify the retained response bytes and derive selected file records and full parent-folder ancestry. The script uses only Python's standard library and does not make network requests. Its receipt keeps baseline/identifier/line-convention readiness unresolved. The selected publisher-listed total is 13,186,692 bytes; actual body acquisition remains a separate indoor DEV step.

Primary sources: [v4 publisher page](https://data.mendeley.com/datasets/75m8vz7jr2/4), [author article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11068545/). The article cites dataset v2, so the retained official v4 inventories provide the current file binding. No reserved outdoor target values or video frames were opened, and no inference or training was run.
