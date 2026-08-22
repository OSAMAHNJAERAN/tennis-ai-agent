import * as Dialog from "@radix-ui/react-dialog";
import { flexRender } from "@tanstack/react-table";
import { getCoreRowModel, legacyCreateColumnHelper, useLegacyTable, type LegacyColumnDef } from "@tanstack/react-table/legacy";
import { ArrowRight, Faders, Funnel, Play, X } from "@phosphor-icons/react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAnalysis } from "../app/AnalysisContext";
import type { MatchEvent } from "../data/contracts";
import { Card, EmptyState, PageHeader, StatusBadge } from "../components/ui";
import { formatConfidence, humanize } from "../lib/format";

const columnHelper = legacyCreateColumnHelper<MatchEvent>();

export default function Events() {
  const { run, selectedEventId, selectEvent } = useAnalysis();
  const [typeFilter, setTypeFilter] = useState("ALL");
  const [minimumConfidence, setMinimumConfidence] = useState(0);
  const navigate = useNavigate();
  const events = run?.events;
  const eventTypes = Array.from(new Set((events ?? []).map((event) => event.event_type)));
  const filteredEvents = useMemo(
    () => (events ?? []).filter((event) => (typeFilter === "ALL" || event.event_type === typeFilter) && event.confidence >= minimumConfidence),
    [events, minimumConfidence, typeFilter],
  );
  const columns = useMemo(() => [
    columnHelper.accessor("event_id", { header: "ID", cell: (info) => `E${info.getValue()}` }),
    columnHelper.accessor("event_type", { header: "Event", cell: (info) => humanize(info.getValue()) }),
    columnHelper.accessor("frame", { header: "Frame", cell: (info) => info.getValue() }),
    columnHelper.accessor("timestamp_s", { header: "Time", cell: (info) => `${info.getValue().toFixed(3)} s` }),
    columnHelper.accessor("player_id", { header: "Participant", cell: (info) => info.getValue() ? `Player ${info.getValue()}` : "Ball" }),
    columnHelper.accessor("trajectory_state", { header: "Tracker", cell: (info) => <StatusBadge>{humanize(info.getValue())}</StatusBadge> }),
    columnHelper.accessor("confidence", { header: "Confidence", cell: (info) => formatConfidence(info.getValue()) }),
    columnHelper.display({ id: "open", header: "", cell: (info) => <button className="table-open" type="button" onClick={() => selectEvent(info.row.original.event_id)} aria-label={`Open event ${info.row.original.event_id}`}><ArrowRight size={16} /></button> }),
  ], [selectEvent]);
  const table = useLegacyTable({
    data: filteredEvents,
    columns: columns as unknown as LegacyColumnDef<MatchEvent, unknown>[],
    getCoreRowModel: getCoreRowModel(),
  });
  const selectedEvent = (events ?? []).find((event) => event.event_id === selectedEventId) ?? null;

  if (!run) return null;

  const replay = (event: MatchEvent) => navigate(`/analysis?run=${run.summary.id}&event=${event.event_id}`);

  return (
    <Dialog.Root open={Boolean(selectedEvent)} onOpenChange={(open) => { if (!open) selectEvent(null); }}>
      <div className="page events-page">
        <PageHeader eyebrow="Semantic detection log" title="Events" description="Filter, inspect, and replay every contact or bounce with frame, time, confidence, player attribution, and tracker provenance." />
        <Card className="event-filter-bar">
          <span className="filter-label"><Faders size={17} /> Filters</span>
          <label><span>Event type</span><select value={typeFilter} onChange={(event) => setTypeFilter(event.target.value)}><option value="ALL">All events</option>{eventTypes.map((type) => <option value={type} key={type}>{humanize(type)}</option>)}</select></label>
          <label><span>Minimum confidence</span><select value={minimumConfidence} onChange={(event) => setMinimumConfidence(Number(event.target.value))}><option value={0}>Any confidence</option><option value={0.8}>80% and above</option><option value={0.9}>90% and above</option><option value={0.95}>95% and above</option></select></label>
          <span className="filter-result"><Funnel size={15} /> {filteredEvents.length} of {run.events.length}</span>
        </Card>
        {!run.events.length ? <EmptyState title="No semantic events" detail="This generation predates match_events.json. Select Phase 3 or later." /> : !filteredEvents.length ? <EmptyState title="No events match" detail="Reduce the confidence threshold or select another event type." /> : <>
          <Card className="event-table-card desktop-events-table">
            <table className="data-table">
              <caption className="sr-only">Semantic match events for {run.summary.label}</caption>
              <thead>{table.getHeaderGroups().map((headerGroup) => <tr key={headerGroup.id}>{headerGroup.headers.map((header) => <th scope="col" key={header.id}>{header.isPlaceholder ? null : flexRender(header.column.columnDef.header, header.getContext())}</th>)}</tr>)}</thead>
              <tbody>{table.getRowModel().rows.map((row) => <tr key={row.id} className={row.original.event_id === selectedEventId ? "table-row-selected" : ""}>{row.getVisibleCells().map((cell) => <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>)}</tr>)}</tbody>
            </table>
          </Card>
          <div className="mobile-event-records">{filteredEvents.map((event) => <Card className="mobile-event-card" key={event.event_id}><button type="button" onClick={() => selectEvent(event.event_id)}><div><span>E{event.event_id} · F{event.frame}</span><StatusBadge>{humanize(event.trajectory_state)}</StatusBadge></div><h2>{humanize(event.event_type)}</h2><dl><div><dt>Time</dt><dd>{event.timestamp_s.toFixed(3)} s</dd></div><div><dt>Confidence</dt><dd>{formatConfidence(event.confidence)}</dd></div><div><dt>Participant</dt><dd>{event.player_id ? `Player ${event.player_id}` : "Ball"}</dd></div></dl></button></Card>)}</div>
        </>}
      </div>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="detail-drawer" aria-describedby="event-drawer-description">
          <div className="drawer-title-row"><div><p className="eyebrow">Event evidence</p><Dialog.Title>{selectedEvent ? humanize(selectedEvent.event_type) : "Event"}</Dialog.Title></div><Dialog.Close asChild><button className="icon-button" type="button" aria-label="Close event details"><X size={20} /></button></Dialog.Close></div>
          {selectedEvent ? <>
            <p id="event-drawer-description">Recorded at frame {selectedEvent.frame}, {selectedEvent.timestamp_s.toFixed(3)} seconds into the source clip.</p>
            <div className="drawer-event-signal"><span>E{selectedEvent.event_id}</span><strong>{formatConfidence(selectedEvent.confidence)}</strong><small>Detection confidence</small></div>
            <dl className="drawer-facts"><div><dt>Participant</dt><dd>{selectedEvent.player_id ? `Player ${selectedEvent.player_id}` : "Ball bounce"}</dd></div><div><dt>Tracker state</dt><dd>{humanize(selectedEvent.trajectory_state)}</dd></div><div><dt>Pixel coordinate</dt><dd>{selectedEvent.ball_position_px?.map((value) => value.toFixed(1)).join(", ") ?? "Unavailable"}</dd></div><div><dt>Court coordinate</dt><dd>{selectedEvent.court_position_m?.map((value) => value.toFixed(2)).join(", ") ?? "Unavailable"} m</dd></div></dl>
            <div className="drawer-evidence"><h3>Recorded evidence</h3>{Object.entries(selectedEvent.evidence ?? {}).map(([key, value]) => <div key={key}><span>{humanize(key)}</span><strong>{typeof value === "number" ? value.toFixed(3) : String(value)}</strong></div>)}</div>
            <button className="button button-primary drawer-replay" type="button" onClick={() => replay(selectedEvent)}><Play size={16} weight="fill" /> Replay at this frame</button>
          </> : null}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
