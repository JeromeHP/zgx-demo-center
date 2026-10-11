# Presenter script

## Timing reality check -- read this first

This script was written as a 5-minute live talk track. Where you can run it live now depends
on the hardware:

**On the HP ZGX Nano (GB10): run it live.** Measured 2026-09-18, quick scope, 3 reps:
**median 189.9 s (~3.2 minutes)**, 3/3 completed, with the budget-over-cap, fringe-mismatch and
honest-prior-outcomes beats hitting 3/3 and the stale-letter beat 2/3
(`reports/e2e_baseline_nano.json`). That is inside the original under-6-minute design target,
so the talk track below works as written against a real run.

**On the ZBook (or comparable laptop APU): default to Replay mode.** A live quick-scope run
there takes ~15-22 minutes -- a real, measured, documented finding (`PROGRESS.md` Phase 10),
not a bug being ignored. A successful run is committed at
`data/replays/quick_zbook_20260917.jsonl`; select it in Replay mode and it plays back at a
natural pace with the amber "REPLAY -- recorded on..." banner visible throughout, so nobody is
misled about what they are watching.

The timings below apply to either -- a Nano live run, or the replay's pacing.

## Talk track — Story mode (the built-in scenario, ~3 min live on a Nano)

1. **0:00 Hook (30s).** "Grants offices are one or two people. This proposal is due in two
   weeks. Watch an AI agent do the first pass, entirely on this device."
2. **0:30 Start the quick-scope run** (on a Nano; elsewhere press play on the replay). Point to
   the trace: requirements loaded, budget check running.
3. **1:00 First catch.** The budget-over-cap and fringe-mismatch issues appear. "Deterministic
   checks. The agent didn't guess; it computed."
4. **1:30 The unplug** (live runs only). Disconnect the network. Point at the agent still
   working. "Their proposal vault, budgets and reviewer comments never leave this box."
5. **2:15 The agent at work.** Searches, reads, drafts. Open a citation chip to show the actual
   cited document text.
6. **3:00 Self-review.** The v1 score fails, the must-fix list appears, v2 revises. "An agent
   that checks its own work, with guardrails the harness enforces, not just the prompt."
7. **3:45 Honesty moment.** Results from Prior Support reports 84 completers against a target of
   120, with the documented reasons. "It won't spin your outcomes."
8. **4:15 The meter.** Token count. "That's where per-token cloud bills grow, and why flat-cost
   local compute matters."
9. **4:45 Download packet (PDF).** A real document, not a screenshot.

**If they ask "does this only work for community colleges?"** — change the Scenario picker
and run it again. Six institutions ship: Harrow Creek (community college), Calderwick
(research institute), Vantcross (R1), Tierra Sandoval (Hispanic-Serving Institution),
Ashgrove (academic medical centre) and Sable Creek (rural consortium). Each has its own
vault, its own solicitation, its own requirements and its own section names — Ashgrove
drafts *Specific Aims*, Harrow Creek drafts *Results from Prior NSF Support*. Beat 7
changes with the scenario: every one has its own documented shortfall to report honestly.

Only Harrow Creek applies to a real solicitation (NSF 24-584). The other five apply to
generated ones, labelled synthetic in their own front matter — say so if asked. The real
announcements are in Explore and Grade, where "nobody configured this for NIH" is the point.

## The other three modes

Story is the scripted one. These are what a research office actually asks for, and each is a
better answer to "but what about *our* grants?" than any slide.

### Discover — "find grants we could win"

Upload their documents (or the two sample ones). The agent writes a **cited** capability
profile, then searches the live federal catalogue.

**The line that lands:** open the **Egress** tab. "Here is every request this box made, and
every term it was refused permission to send. Their project name, their PI's name, their award
number — refused, with the reason, before it left. What did leave is `robotics`,
`workforce development`. Words you could read off a funding announcement."

Then: "and it cannot recommend a grant it didn't actually retrieve. That's a code path, not an
instruction."

*Requires network.* Without one it falls back to cached results. General web search needs the
optional SearXNG container; the catalogue search does not.

### Explore — "what about a solicitation you've never seen?"

Upload any call for proposals. It reads it, extracts the requirements, and **discards every
requirement it cannot quote verbatim** from the document — on a real NIH announcement it kept 78
and dropped 16. It then derives that solicitation's own sections rather than assuming NSF's.

**The line:** "Nobody configured this for NIH. It read the document."

Takes 2-3 minutes on first upload, instant on re-run. Point at the per-section progress bar
rather than letting it look hung.

### Grade — "mark a proposal we already wrote"

Upload a draft. It scores every section, rules on every requirement, runs the budget checks, and
cross-checks the figures against the institution's own records.

**The line that lands hardest:** the claim check. "This proposal claims a placement rate their
own evaluation data contradicts. A panel would have found that. It found it first — and it
cannot finish leaving a single requirement unruled, because a requirement nobody looked at reads
as compliant."

## Timing and what to avoid

- Story quick-scope on a Nano: ~3 min. Discover: ~2 min. Grade: ~3 min. Explore: ~4-5 min
  including extraction.
- Only one run at a time. Hit **Reset** between demos — it also deletes uploaded documents,
  which is worth saying out loud if a visitor uploaded their own.
- Section planning under Explore varies run to run. Don't promise a specific section list.
- Claim checking catches overstatement but not every instance. Don't claim it is exhaustive.

## Kiosk / unstaffed booth

Explore and Discover both accept visitor uploads. Session files live in `runtime/sessions/` and
are deleted on Reset. For an unattended booth, consider defaulting to Replay, with the live
modes driven by someone who can explain what is being uploaded and where it goes.
