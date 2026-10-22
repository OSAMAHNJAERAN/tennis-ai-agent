"""Actual Chromium owner journeys against the real isolated product server.

All videos, observations and accounts are explicitly synthetic. Request faults
delay or lose real server replies; no product response body or browser intent is
fabricated. This runner starts no worker and performs no inference/resource probe.
Each invocation requires a fresh output directory and retains failure evidence.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import time
import traceback
from urllib.parse import parse_qs, urljoin, urlsplit
import uuid

from playwright.async_api import async_playwright, expect


SCRIPT = Path(__file__).resolve()
ART = SCRIPT.parent
ROOT = SCRIPT.parents[4]
SCOPE = ("SYNTHETIC SOFTWARE TEST: actual Chromium UI, real cookie accounts, "
         "real upload admission/storage and source-bound correction publication; "
         "no original video inference, host RAM/GPU probe, tennis accuracy or capacity claim")
PIN_PATHS = [
    "frontend/src/pages/Upload.tsx", "frontend/src/components/AccountAccess.tsx",
    "frontend/src/data/batchApi.ts", "frontend/src/data/creationIntent.ts",
    "frontend/src/components/AnalysisCorrection.tsx", "frontend/src/data/correctionApi.ts",
    "frontend/src/pages/ProductWorkspace.tsx", "frontend/src/pages/SpatialAnalysis.tsx",
    "frontend/src/app/AnalysisContext.tsx", "frontend/src/components/VideoEvidencePlayer.tsx",
    "src/api/app.py", "src/api/uploads.py", "src/api/creation_requests.py",
    "src/api/corrections.py", "src/api/correction_worker.py", "src/api/analysis_generation.py",
    "src/scoring/corrected_rule_rebuild.py", "src/scoring/corrected_publication.py",
    "frontend/dist/index.html",
]


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_pins():
    paths = [ROOT / name for name in PIN_PATHS]
    paths += sorted((ROOT / "frontend/dist/assets").glob("*.js"))
    paths += sorted((ROOT / "frontend/dist/assets").glob("*.css"))
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


class Journey:
    def __init__(self, output, fixture):
        self.fixture = fixture.resolve()
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "screenshots").mkdir()
        (self.output / "downloads").mkdir()
        (self.output / "upload-inputs").mkdir()
        self.manifest = json.loads((self.fixture / "fixture_manifest.json").read_text())
        self.runtime = json.loads((self.fixture / "runtime.json").read_text())
        self.base = self.runtime.get("base_url") or (
            f"http://{self.runtime['host']}:{self.runtime['port']}")
        assert urlsplit(self.base).hostname == "127.0.0.1", "Loopback fixture only"
        assert self.manifest["label"].startswith("SYNTHETIC SOFTWARE TEST")
        self.analysis = self.manifest["analysis_id"]
        self.api = "/api/analyses/" + self.analysis
        self.store = Path(self.manifest["storage_root"]).resolve()
        assert self.store.is_relative_to(self.fixture), "Fixture store must be task owned"
        self.credentials = json.loads((self.fixture / self.manifest["credentials_file"]).read_text())
        self.owner = self.manifest["owner_id"]
        self.other_credentials = {
            "username": "synthetic_browser_other_" + uuid.uuid4().hex[:10],
            "password": "Synthetic-browser-only-2026!",
        }
        self.receipt = {
            "status": "RUNNING", "started_utc": utc(), "scope": SCOPE,
            "runner_sha256": sha(__file__), "fixture_manifest_sha256": sha(self.fixture / "fixture_manifest.json"),
            "fixture_directory": str(self.fixture),
            "source_before": source_pins(), "base_url": self.base,
            "analysis_id": self.analysis, "server_pid": self.runtime["server_pid"],
            "scenarios": [], "console": [], "page_errors": [], "request_failures": [],
            "screenshots": [], "actual_model_inference": False, "actual_host_ram_gpu_probes": 0,
        }
        self.browser = self.context = self.page = None
        self.clock = time.perf_counter()
        self.run_id = uuid.uuid4().hex[:8]
        self.network_sequence = 0
        self.request_sequences = {}
        self.owner_responses = []
        self.network_changed = asyncio.Event()
        self.upload_boundary = 0
        self.active_owner = None
        self.write()

    def write(self):
        (self.output / "receipt.json").write_text(
            json.dumps(self.receipt, indent=2, allow_nan=False) + "\n")

    def details(self, name, value):
        (self.output / (name + ".json")).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n")

    def generation(self, token):
        assert re.fullmatch(r"[0-9a-f]{64}", token)
        path = self.store / "analyses" / self.analysis / ".result" / (token + ".generation")
        assert sha(path) == token, "Actual immutable generation manifest differs"
        value = json.loads(path.read_text())
        assert value["analysisId"] == self.analysis
        return path.parent / value["filesDirectory"], value

    def fixture_source(self):
        directory, value = self.generation(self.manifest["base_generation"])
        path = directory / "source.mp4"
        assert sha(path) == self.manifest["source_sha256"] == value["files"]["source.mp4"]["sha256"]
        return path

    def upload_file(self, name):
        original = Path(name)
        path = self.output / "upload-inputs" / (original.stem + "-" + self.run_id + original.suffix)
        shutil.copyfile(self.fixture_source(), path)
        os.utime(path, (1700000000, 1700000000))
        return path

    def creation_rows(self, owner, key):
        # Read only the disposable fixture's admission records, never credentials.
        conn = sqlite3.connect("file:" + str(self.store / "analyses.sqlite3") + "?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT descriptor_json FROM upload_creation_requests WHERE owner_id=? AND intent_key=?",
                (owner, key)).fetchall()
            rejected = conn.execute(
                "SELECT COUNT(*) FROM upload_creation_rejections WHERE owner_id=? AND intent_key=?",
                (owner, key)).fetchone()[0]
            descriptors = [json.loads(row[0]) for row in rows]
            counts = []
            for descriptor in descriptors:
                if descriptor.get("id", "").startswith("batch_"):
                    counts.append(conn.execute(
                        "SELECT COUNT(*) FROM upload_sessions WHERE owner_id=? AND batch_id=?",
                        (owner, descriptor["id"])).fetchone()[0])
            return {"accepted_requests": len(rows), "rejected_tombstones": rejected,
                    "descriptors": descriptors, "batch_session_counts": counts}
        finally:
            conn.close()

    async def api_json(self, path, context=None, expected=200):
        response = await (context or self.context).request.get(urljoin(self.base, path))
        assert response.status == expected, (path, response.status, await response.text())
        return await response.json()

    async def shot(self, name):
        path = self.output / "screenshots" / (name + ".png")
        await self.page.screenshot(path=str(path), full_page=True)
        self.receipt["screenshots"].append(str(path.relative_to(self.output)))
        self.write()

    async def intent(self, owner):
        raw = await self.page.evaluate(
            "owner => localStorage.getItem('astra.upload.creation.v1:' + encodeURIComponent(owner))", owner)
        return json.loads(raw) if raw else None

    def observed_request(self, request):
        self.network_sequence += 1
        self.request_sequences[request] = self.network_sequence

    def observed_response(self, response):
        path = urlsplit(response.url).path
        if response.request.method != "GET" or path not in {"/api/auth/session", "/api/analyses"}:
            return
        self.network_sequence += 1
        record = {"path": path, "request_sequence": self.request_sequences.get(response.request, self.network_sequence),
                  "response_sequence": self.network_sequence, "status": response.status}

        async def read():
            try:
                value = await response.json()
                if path == "/api/auth/session":
                    record["owner"] = (value.get("user") or {}).get("id") if value.get("authenticated") else None
                else:
                    record["analysis_ids"] = [row["id"] for row in value] if isinstance(value, list) else None
                self.owner_responses.append(record)
            except Exception as error:
                # An account switch intentionally aborts obsolete fetch bodies.
                record["body_read_error"] = str(error)
                self.receipt.setdefault("owner_observation_errors", []).append(record)
            finally:
                self.network_changed.set()
        asyncio.create_task(read())

    async def goto_upload(self):
        self.upload_boundary = self.network_sequence
        await self.page.goto(self.base + "/upload", wait_until="domcontentloaded")

    async def upload_owner_ready(self):
        """Wait for the product's actual matching-owner/history/owner bracket.

        AccountAccess's signed-in heading precedes Upload's asynchronous owner
        admission. An enabled file button alone therefore cannot prove readiness.
        Observe real page requests, then allow the resulting React paint; never
        mutate browser state or substitute a manufactured health/session reply.
        """
        deadline = time.monotonic() + 30
        while True:
            rows = [r for r in self.owner_responses if r["request_sequence"] > self.upload_boundary and r["status"] == 200]
            sessions = [r for r in rows if r["path"] == "/api/auth/session" and r.get("owner") == self.active_owner]
            histories = [r for r in rows if r["path"] == "/api/analyses" and r.get("analysis_ids") is not None]
            match = next(((opening, history, closing) for opening in sessions for history in histories for closing in sessions
                          if opening["response_sequence"] < history["request_sequence"]
                          and history["response_sequence"] < closing["request_sequence"]), None)
            if match:
                await self.page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
                await expect(self.page.locator('[aria-label="Analysis history"]')).to_be_visible()
                await expect(self.page.get_by_role("button", name="Sign out", exact=True)).to_be_visible()
                self.receipt.setdefault("upload_readiness_brackets", []).append({
                    "owner": self.active_owner, "opening": match[0], "history": match[1], "closing": match[2]})
                return
            remaining = deadline - time.monotonic()
            assert remaining > 0, "No real matching-owner upload history refresh completed"
            self.network_changed.clear()
            await asyncio.wait_for(self.network_changed.wait(), remaining)

    async def sign_in(self, credentials, register=False, goto=True):
        if goto:
            await self.goto_upload()
        await expect(self.page.get_by_label("Username", exact=True)).to_be_visible()
        if register:
            toggle = self.page.get_by_role("button", name="Create an account", exact=True)
        else:
            toggle = self.page.get_by_role("button", name="Use existing account", exact=True)
        # AccountAccess retains its register/login mode across an in-page logout.
        if await toggle.count():
            await toggle.click()
        await self.page.get_by_label("Username", exact=True).fill(credentials["username"])
        await self.page.get_by_label("Password", exact=True).fill(credentials["password"])
        self.upload_boundary = self.network_sequence
        await self.page.get_by_role("button", name="Create account" if register else "Sign in", exact=True).click()
        await expect(self.page.get_by_role("heading", name="Signed in as " + credentials["username"], exact=True)).to_be_visible()
        session = await self.api_json("/api/auth/session")
        assert session["authenticated"] and session["user"]["username"] == credentials["username"]
        self.active_owner = session["user"]["id"]
        return session["user"]["id"]

    async def sign_out(self):
        await self.page.get_by_role("button", name="Sign out", exact=True).click()
        await expect(self.page.get_by_label("Username", exact=True)).to_be_visible()

    async def select_file(self, path):
        await self.upload_owner_ready()
        await self.page.get_by_label("Choose tennis video", exact=True).set_input_files(str(path))
        await expect(self.page.get_by_role("button", name="Analyze match", exact=True)).to_be_enabled()

    async def uploaded(self):
        await expect(self.page.get_by_text("Received · Saved analysis created", exact=True)).to_be_visible(timeout=30000)
        await expect(self.page.locator('[aria-label="Saved batch"]')).to_be_visible()

    @staticmethod
    async def actual_batch_reply(route):
        response = await route.fetch()
        data = await response.json()
        headers = await route.request.all_headers()
        record = {"key": headers.get("idempotency-key"), "owner": headers.get("upload-creation-owner"),
                  "status": response.status, "rejection_header": response.headers.get("upload-creation-rejected"),
                  "response": data}
        return response, record

    async def lost_acknowledgement(self):
        path = self.upload_file("SYNTHETIC-lost-ack.mp4")
        replies = []
        lost = asyncio.Event()

        async def handle(route):
            if route.request.method != "POST":
                await route.continue_(); return
            response, record = await self.actual_batch_reply(route)
            replies.append(record)
            assert response.status == 201
            if len(replies) == 1:
                await route.abort("failed")
                lost.set()
            else:
                await route.fulfill(response=response)

        await self.page.route("**/api/batches", handle)
        await self.select_file(path)
        await self.page.get_by_role("button", name="Analyze match", exact=True).click()
        await asyncio.wait_for(lost.wait(), 30)
        await expect(self.page.get_by_role("button", name="Analyze match", exact=True)).to_be_enabled()
        saved = await self.intent(self.owner)
        assert saved and saved["key"] == replies[0]["key"]
        batch = replies[0]["response"]
        state_before = await self.api_json("/api/batches/" + batch["id"])
        assert len(state_before["uploads"]) == 1 and state_before["uploads"][0]["offset"] == 0
        await self.shot("02-lost-ack-retained")
        self.upload_boundary = self.network_sequence
        await self.page.reload(wait_until="domcontentloaded")
        await expect(self.page.get_by_role("heading", name="Signed in as " + self.credentials["username"], exact=True)).to_be_visible()
        assert (await self.intent(self.owner))["key"] == saved["key"]
        await self.select_file(path)
        await self.page.get_by_role("button", name="Analyze match", exact=True).click()
        await self.uploaded()
        assert len(replies) == 2 and replies[0]["key"] == replies[1]["key"]
        assert replies[0]["response"]["id"] == replies[1]["response"]["id"]
        assert [u["id"] for u in replies[0]["response"]["uploads"]] == [u["id"] for u in replies[1]["response"]["uploads"]]
        assert await self.intent(self.owner) is None
        final = await self.api_json("/api/batches/" + batch["id"])
        assert final["uploads"][0]["status"] == "COMPLETED"
        assert final["uploads"][0]["analysis_id"] is not None
        rows = self.creation_rows(self.owner, saved["key"])
        assert rows["accepted_requests"] == 1 and rows["batch_session_counts"] == [1]
        await self.page.unroute("**/api/batches", handle)
        await self.shot("03-lost-ack-recovered-same-batch")
        self.details("lost_acknowledgement", {"replies": replies, "saved_intent": saved,
                     "durable_rows": rows, "final_batch": final})
        self.first_batch = batch
        self.first_analysis = final["uploads"][0]["analysis_id"]
        return {"same_key_batch_and_sessions": True, "reload_recovered": True,
                "batch_id": batch["id"], "analysis_id": self.first_analysis, "durable_rows": rows}

    async def confirmed_rejection(self):
        await self.goto_upload()
        invalid = self.upload_file("SYNTHETIC-" + "x" * 205 + ".mp4")
        valid = self.upload_file("SYNTHETIC-corrected-filename.mp4")
        replies = []

        async def handle(route):
            if route.request.method != "POST":
                await route.continue_(); return
            response, record = await self.actual_batch_reply(route)
            replies.append(record)
            await route.fulfill(response=response)

        await self.page.route("**/api/batches", handle)
        await self.select_file(invalid)
        await self.page.get_by_role("button", name="Analyze match", exact=True).click()
        await expect(self.page.get_by_role("alert").filter(has_text="valid filename")).to_be_visible()
        assert len(replies) == 1 and replies[0]["status"] == 415
        assert replies[0]["rejection_header"] == replies[0]["key"]
        assert await self.intent(self.owner) is None
        rejected = self.creation_rows(self.owner, replies[0]["key"])
        assert rejected["accepted_requests"] == 0 and rejected["rejected_tombstones"] == 1
        await self.shot("04-confirmed-rejection")
        await self.select_file(valid)
        await self.page.get_by_role("button", name="Analyze match", exact=True).click()
        await self.uploaded()
        assert len(replies) == 2 and replies[1]["status"] == 201
        assert replies[1]["key"] != replies[0]["key"]
        assert await self.intent(self.owner) is None
        accepted = self.creation_rows(self.owner, replies[1]["key"])
        assert accepted["accepted_requests"] == 1 and accepted["batch_session_counts"] == [1]
        await self.page.unroute("**/api/batches", handle)
        await self.shot("05-corrected-submission-accepted")
        self.details("confirmed_rejection", {"replies": replies, "rejected_rows": rejected, "accepted_rows": accepted})
        return {"real_415_with_exact_tombstone": True, "intent_retired": True,
                "new_key_accepted": True, "batch_id": replies[1]["response"]["id"]}

    async def owner_switch(self):
        await self.goto_upload()
        path = self.upload_file("SYNTHETIC-delayed-owner-reply.mp4")
        committed, release, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        replies, delivery = [], []

        async def handle(route):
            if route.request.method != "POST":
                await route.continue_(); return
            response, record = await self.actual_batch_reply(route)
            replies.append(record)
            assert response.status == 201
            if len(replies) == 1:
                committed.set()
                await release.wait()
                try:
                    await route.fulfill(response=response)
                    delivery.append("original_response_released")
                except Exception as error:
                    delivery.append("canceled_by_account_switch: " + str(error))
                finally:
                    settled.set()
            else:
                await route.fulfill(response=response)

        await self.page.route("**/api/batches", handle)
        await self.select_file(path)
        await self.page.get_by_role("button", name="Analyze match", exact=True).click()
        await asyncio.wait_for(committed.wait(), 30)
        saved = await self.intent(self.owner)
        assert saved and saved["key"] == replies[0]["key"]
        a_batch = replies[0]["response"]
        await self.sign_out()
        self.other_owner = await self.sign_in(self.other_credentials, register=True, goto=False)
        assert self.other_owner != self.owner
        release.set()
        await asyncio.wait_for(settled.wait(), 30)
        await expect(self.page.locator('[aria-label="Saved batch"]')).to_have_count(0)
        await expect(self.page.get_by_label("Choose tennis video", exact=True)).to_have_value("")
        assert (await self.intent(self.owner))["key"] == saved["key"]
        file_key = saved["request"]["fileKeys"][0]
        remembered = await self.page.evaluate("([owner,key]) => localStorage.getItem('astra.upload.sessions.v2:' + encodeURIComponent(owner) + ':' + encodeURIComponent(key))", [self.owner, file_key])
        assert remembered is None, "Late A acknowledgement must not adopt its session"
        for api in ("/api/batches/" + a_batch["id"], "/api/uploads/" + a_batch["uploads"][0]["id"],
                    "/api/analyses/" + self.first_analysis, self.api + "/result"):
            response = await self.context.request.get(urljoin(self.base, api))
            assert response.status == 404, (api, response.status)
        history = await self.api_json("/api/analyses")
        assert not any(row["id"] in {self.analysis, self.first_analysis} for row in history)
        await self.shot("06-owner-switch-no-stale-adoption")
        await self.select_file(path)
        await self.page.get_by_role("button", name="Analyze match", exact=True).click()
        await self.uploaded()
        assert len(replies) == 2
        assert replies[1]["owner"] == self.other_owner and replies[0]["owner"] == self.owner
        assert replies[1]["key"] != replies[0]["key"]
        assert replies[1]["response"]["id"] != a_batch["id"]
        assert replies[1]["response"]["uploads"][0]["id"] != a_batch["uploads"][0]["id"]
        assert await self.intent(self.other_owner) is None
        assert (await self.intent(self.owner))["key"] == saved["key"]
        await self.page.unroute("**/api/batches", handle)
        await self.shot("07-other-owner-independent-upload")
        self.details("owner_switch", {"replies": replies, "stale_response_delivery": delivery,
                     "a_uncertain_intent_preserved": True, "a_session_not_adopted": True,
                     "a_durable_rows": self.creation_rows(self.owner, saved["key"]),
                     "b_durable_rows": self.creation_rows(self.other_owner, replies[1]["key"])})
        await self.sign_out()
        assert await self.sign_in(self.credentials, goto=False) == self.owner
        return {"owner_a": self.owner, "owner_b": self.other_owner,
                "stale_response_delivery": delivery, "foreign_access_rejected": True,
                "uncertain_a_intent_preserved": True, "b_independent_creation": True}

    async def video_probe(self, generation, annotated=False):
        await self.page.get_by_role("tab", name="Video Feed", exact=True).click()
        if annotated:
            await self.page.get_by_role("button", name="AI annotated", exact=True).click()
        else:
            await self.page.get_by_role("button", name="Original", exact=True).click()
        video = self.page.get_by_label("Source match video", exact=True)
        await expect(video).to_be_visible()
        await self.page.wait_for_function("() => {const v=document.querySelector('video[aria-label=\"Source match video\"]'); return v && v.readyState >= 2 && v.videoWidth > 0;}")
        await video.evaluate("async v => {v.muted=true; await v.play();}")
        await self.page.wait_for_function("() => document.querySelector('video[aria-label=\"Source match video\"]').currentTime > 0")
        await video.evaluate("v => {v.pause(); v.currentTime=Math.min(0.2, v.duration/2);}")
        await self.page.wait_for_function("() => {const v=document.querySelector('video[aria-label=\"Source match video\"]'); return !v.seeking && v.currentTime > 0;}")
        observation = await video.evaluate("v => ({currentSrc:v.currentSrc, currentTime:v.currentTime, readyState:v.readyState, width:v.videoWidth, height:v.videoHeight, duration:Number.isFinite(v.duration)?v.duration:null, decodedFrames:v.getVideoPlaybackQuality?.().totalVideoFrames ?? v.webkitDecodedFrameCount ?? null, error:v.error?.message ?? null})")
        assert observation["width"] == 1280 and observation["height"] == 960
        assert observation["duration"] and .3 <= observation["duration"] <= .6
        assert observation["decodedFrames"] and observation["decodedFrames"] > 0
        assert observation["error"] is None
        assert parse_qs(urlsplit(observation["currentSrc"]).query).get("generation") == [generation]
        media = await self.context.request.get(observation["currentSrc"])
        assert media.status == 200 and media.headers.get("x-analysis-generation") == generation
        observation["response_sha256"] = hashlib.sha256(await media.body()).hexdigest()
        return observation

    async def correction(self):
        base_generation = self.manifest["base_generation"]
        before = await self.api_json(self.api + "/result?generation=" + base_generation)
        assert before["analysisGeneration"] == base_generation and before["analysisRevision"] == 0
        await self.page.goto(self.base + "/analysis/" + self.analysis + "?generation=" + base_generation,
                             wait_until="domcontentloaded")
        correction = self.page.locator("section.analysis-correction")
        await expect(correction.get_by_role("heading", name="Review and correct analysis", exact=True)).to_be_visible()
        await expect(correction.locator(".correction-badge")).to_have_text("Revision 0")
        original_video = await self.video_probe(base_generation)
        await self.shot("08-owner-original-replay-and-review")
        await correction.get_by_label("Review initial score and serve context", exact=True).check()
        await correction.get_by_role("combobox", name=re.compile(r"^Player 2 initial points\b")).select_option(label="15")
        await expect(correction.get_by_role("combobox", name=re.compile(r"^Server\b"))).to_have_value("1")
        await expect(correction.get_by_role("combobox", name=re.compile(r"^Serve attempt\b"))).to_have_value("2")
        await correction.get_by_label("Reason for correction", exact=True).fill("SYNTHETIC SOFTWARE TEST: review fixture initial receiver score")
        await correction.get_by_label("Video evidence reference", exact=True).fill("SYNTHETIC generated source only, frames 0–11; no real tennis accuracy claim")
        await correction.get_by_label("I reviewed the source video and authorize this correction as the signed-in owner.", exact=True).check()
        await self.shot("09-owner-source-bound-correction-form")
        async with self.page.expect_response(lambda response: response.request.method == "POST" and urlsplit(response.url).path == self.api + "/corrections") as response_info:
            await correction.get_by_role("button", name="Submit correction and rebuild", exact=True).click()
        submitted_response = await response_info.value
        assert submitted_response.status == 202
        submitted = await submitted_response.json()
        self.details("correction_submitted", submitted)
        await self.page.wait_for_url(lambda url: parse_qs(urlsplit(str(url)).query).get("generation", [base_generation])[0] != base_generation,
                                     timeout=180000)
        token = parse_qs(urlsplit(self.page.url).query)["generation"][0]
        await expect(self.page.locator("section.analysis-correction .correction-badge")).to_have_text("Revision 1", timeout=30000)
        status = await self.api_json(self.api + "/corrections/" + submitted["id"])
        assert status["status"] == "COMPLETED" and status["generation"] == token
        result = await self.api_json(self.api + "/result?generation=" + token)
        assert result["analysisRevision"] == 1 and result["analysisGeneration"] == token
        assert result["pointAnalysis"]["summary"]["totalPointsPlayed"] == 1
        assert result["pointAnalysis"]["serveByPlayer"]["player_1"]["doubleFaults"] == 1
        # This native correction fixture deliberately has no legacy matchState.
        # Its canonical source-linked point analysis owns the recomputed score.
        assert result["pointAnalysis"]["summary"]["scoreSummary"] == "Set 1 (0-0) | P1 0 - P2 30 | Server: P1 (DEUCE)"
        assert result["nativeForecast"]["calibrationStatus"] == "not_calibrated"
        assert result["nativeForecast"]["current"]["sample_count"] == 1
        execution = result["analysisCorrection"]["execution"]
        assert execution["new_vision_inference"] is False and execution["pose_prediction_calls"] == 0
        assert result["analysisCorrection"]["authorizationReceipt"]["owner_id"] == self.owner
        assert result["provenance"]["productionQualified"] is False
        after_source = await self.api_json(self.api + "/review-source?generation=" + token)
        before_source = await self.api_json(self.api + "/review-source?generation=" + base_generation)
        assert after_source["initial_state"]["fields"]["points_p2"] == 1
        assert before_source["initial_state"]["fields"]["points_p2"] == 0
        assert after_source["inputs"] == before_source["inputs"], "Score review must not alter physical source events"
        assert after_source["source_sha256"] == before_source["source_sha256"] == self.manifest["source_sha256"]
        await expect(self.page.get_by_text("Unqualified · not calibrated", exact=True)).to_be_visible()
        corrected_video = await self.video_probe(token, annotated=True)
        await self.shot("10-corrected-generation-replay")
        directory, generation_manifest = self.generation(token)
        assert corrected_video["response_sha256"] == generation_manifest["files"]["annotated_web.mp4"]["sha256"]
        downloads = []
        for key, label, filename in (
            ("readableReport", "Download report", "report.html"),
            ("annotatedVideo", "Download annotated video", "annotated.mp4"),
            ("reportJson", "Analysis JSON", "report.json"),
            ("csv", "Movement CSV", "movement.csv"),
            ("originalVideo", "Original video", "original.mp4"),
        ):
            anchor = self.page.get_by_role("link", name=label, exact=True)
            href = await anchor.get_attribute("href")
            assert href == result["exports"][key]
            assert parse_qs(urlsplit(href).query).get("generation") == [token]
            reference = await self.context.request.get(urljoin(self.base, href))
            assert reference.status == 200
            payload = await reference.body()
            async with self.page.expect_download() as download_info:
                await anchor.click()
            download = await download_info.value
            assert await download.failure() is None
            path = self.output / "downloads" / filename
            await download.save_as(str(path))
            digest = sha(path)
            assert digest == hashlib.sha256(payload).hexdigest()
            if key == "reportJson":
                assert json.loads(path.read_text()) == result
            if key == "readableReport":
                assert "Owner-authorized correction review" in path.read_text()
            if key == "annotatedVideo":
                assert digest == generation_manifest["files"]["annotated_web.mp4"]["sha256"]
                assert reference.headers.get("x-analysis-generation") == token
            if key == "originalVideo":
                assert digest == generation_manifest["files"]["original.mp4"]["sha256"]
            downloads.append({"key": key, "href": href, "bytes": path.stat().st_size, "sha256": digest,
                              "actual_browser_download": True, "suggested_filename": download.suggested_filename})
        source_reply = await self.context.request.get(urljoin(self.base, result["videoUrl"]))
        assert source_reply.status == 200 and source_reply.headers.get("x-analysis-generation") == token
        assert hashlib.sha256(await source_reply.body()).hexdigest() == generation_manifest["files"]["source.mp4"]["sha256"]
        self.details("corrected_result", result)
        self.details("correction_downloads", downloads)
        self.details("correction_source_preservation", {"before": before_source, "after": after_source})
        await self.page.goto(self.base + "/analysis/" + self.analysis + "?generation=" + base_generation,
                             wait_until="domcontentloaded")
        await expect(self.page.get_by_text(re.compile("This generation has been superseded"))).to_be_visible()
        await expect(self.page.locator("section.analysis-correction .correction-badge")).to_have_text("Revision 0")
        assert parse_qs(urlsplit(self.page.url).query).get("generation") == [base_generation]
        await expect(self.page.get_by_role("button", name="Submit correction and rebuild", exact=True)).to_be_disabled()
        assert await self.api_json(self.api + "/result?generation=" + base_generation) == before
        await self.shot("11-historical-generation-readonly")
        await self.page.locator("section.analysis-correction details > summary").click()
        await self.page.get_by_role("button", name="Open saved result", exact=True).click()
        await self.page.wait_for_url(lambda url: parse_qs(urlsplit(str(url)).query).get("generation") == [token])
        await expect(self.page.locator("section.analysis-correction .correction-badge")).to_have_text("Revision 1")
        await self.shot("12-explicit-corrected-generation-reopened")
        self.corrected_result = result
        return {"base_generation": base_generation, "generation": token, "revision": 1,
                "correction_id": status["id"], "source_bound": True, "receiver_score": "0–30",
                "actual_original_video": original_video, "actual_corrected_video": corrected_video,
                "downloads": downloads, "historical_generation_readonly": True,
                "explicit_successor_selection": True, "new_vision_inference": False}

    async def private_downloads(self):
        # Independent browser contexts exercise actual cookie owner/guest access.
        result = self.corrected_result
        foreign = await self.browser.new_context(base_url=self.base)
        guest = await self.browser.new_context(base_url=self.base)
        foreign_page = await foreign.new_page()
        prior_page, prior_context = self.page, self.context
        try:
            self.page, self.context = foreign_page, foreign
            assert await self.sign_in(self.other_credentials) == self.other_owner
            await self.shot("13-other-owner-private-history")
            targets = [self.api + "/result", self.api + "/review-source", result["annotatedVideoUrl"],
                       result["videoUrl"], *result["exports"].values()]
            statuses = []
            for path in dict.fromkeys(targets):
                other = await foreign.request.get(urljoin(self.base, path))
                anonymous = await guest.request.get(urljoin(self.base, path))
                assert other.status == 404 and anonymous.status == 401, (path, other.status, anonymous.status)
                statuses.append({"path": path, "foreign_owner": other.status, "guest": anonymous.status})
            self.details("private_downloads", statuses)
            return {"actual_independent_cookie_contexts": True, "denials": statuses}
        finally:
            self.page, self.context = prior_page, prior_context
            await foreign.close()
            await guest.close()

    async def scenario(self, name, function):
        row = {"name": name, "started_utc": utc(), "status": "RUNNING"}
        self.receipt["scenarios"].append(row)
        self.write()
        print(json.dumps({"scenario": name, "status": "RUNNING"}), flush=True)
        try:
            row["observations"] = await function()
            row["status"] = "PASS"
        except BaseException as error:
            row["status"] = "FAIL"
            row["error"] = type(error).__name__ + ": " + str(error)
            raise
        finally:
            row["completed_utc"] = utc()
            self.write()
            print(json.dumps({"scenario": name, "status": row["status"]}), flush=True)

    async def run(self):
        async with async_playwright() as playwright:
            try:
                self.browser = await playwright.chromium.launch(
                    executable_path="/usr/bin/chromium", headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage", "--autoplay-policy=no-user-gesture-required"])
                self.receipt["browser_version"] = self.browser.version
                self.context = await self.browser.new_context(base_url=self.base, accept_downloads=True,
                                                              viewport={"width": 1440, "height": 1080})
                self.page = await self.context.new_page()
                self.page.set_default_timeout(30000)
                self.page.on("request", self.observed_request)
                self.page.on("response", self.observed_response)
                self.page.on("console", lambda message: self.receipt["console"].append({"type": message.type, "text": message.text}))
                self.page.on("pageerror", lambda error: self.receipt["page_errors"].append(str(error)))
                self.page.on("requestfailed", lambda request: self.receipt["request_failures"].append(
                    {"method": request.method, "url": request.url, "failure": request.failure}))
                assert await self.sign_in(self.credentials) == self.owner
                await self.shot("01-actual-owner-signin")
                await self.scenario("lost_creation_acknowledgement_reload", self.lost_acknowledgement)
                await self.scenario("confirmed_rejection_corrected_submission", self.confirmed_rejection)
                await self.scenario("cross_owner_delayed_acknowledgement", self.owner_switch)
                await self.scenario("source_bound_correction_replay_exports_and_history", self.correction)
                await self.scenario("private_owner_media_and_exports", self.private_downloads)
                assert not self.receipt["page_errors"], self.receipt["page_errors"]
                self.receipt["status"] = "PASS"
            except BaseException as error:
                self.receipt["status"] = "FAIL"
                self.receipt["error"] = type(error).__name__ + ": " + str(error)
                (self.output / "failure.txt").write_text(traceback.format_exc())
                if self.page is not None:
                    try:
                        await self.shot("failure")
                        (self.output / "failure-visible-text.txt").write_text((await self.page.locator("body").inner_text())[:100000])
                    except Exception as diagnostic_error:
                        self.receipt["failure_capture_error"] = str(diagnostic_error)
                raise
            finally:
                if self.browser is not None:
                    await self.browser.close()
                self.receipt["source_after"] = source_pins()
                self.receipt["source_stable"] = self.receipt["source_before"] == self.receipt["source_after"]
                self.receipt["runner_sha256_after"] = sha(__file__)
                self.receipt["runner_stable"] = self.receipt["runner_sha256"] == self.receipt["runner_sha256_after"]
                if not self.receipt["source_stable"] or not self.receipt["runner_stable"]:
                    self.receipt["status"] = "FAIL"
                self.receipt["elapsed_seconds"] = time.perf_counter() - self.clock
                self.receipt["completed_utc"] = utc()
                self.write()
                print(json.dumps({key: self.receipt[key] for key in
                                  ("status", "source_stable", "runner_stable", "elapsed_seconds")}), flush=True)
        assert self.receipt["status"] == "PASS", "Current run did not pass; retain its failure receipt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Fresh evidence directory; existing data is never overwritten")
    parser.add_argument("--fixture", default=ART, type=Path, help="Prepared isolated fixture directory, independent of evidence output")
    arguments = parser.parse_args()
    journey = Journey(arguments.output, arguments.fixture)
    asyncio.run(journey.run())


if __name__ == "__main__":
    main()
