# Spectrum Duel — Executive Talking Points
 
**The one-liner:** Two AI agents fight over the radio spectrum while a neural network on the GB10 decides who's winning in a third of a millisecond — and shows the room every time it gets it wrong.
 
## The Four Numbers
 
- **0.29 ms** per classification, host-to-host, measured on-device and shown in the header. ~3,400/sec available; the demo uses 20.
- **86% live accuracy** — self-graded on screen against ground truth, not a lab number.
- **3 AI workloads concurrent** on one box: the CNN, two policy agents, and a 70B LLM.
- **Zero network.** No cloud, no API key, no license server. Reps should pull the Ethernet mid-demo.
## What It Proves to a Customer
 
1. **Inference isn't the bottleneck.** The headroom is the pitch — whatever the customer's sensor rate is, the AI isn't the constraint.
2. **A 70B model and a 144 KB model, resident together.** ~42 GB and sub-megabyte, side by side in unified memory. On a discrete-GPU workstation, that pairing is where the conversation stops.
3. **Train where you deploy.** Nothing ships pre-built — no PyTorch in the env, no ONNX in the repo. The Nano generates its own training data, trains the network, and compiles a TensorRT engine in ~6 minutes at install. For any customer whose data can't leave the building, that's the whole value proposition demonstrated end-to-end.
4. **It's genuinely disconnected.** All of the above with the cable out.
## The Differentiator
 
Most AI demos are built so they can't visibly fail. This one is built so it can, and grades itself out loud. The classifier is load-bearing — a misread produces a genuinely wrong countermeasure and the datalink drops in front of the customer. A panel freezes that moment: what it saw, what it said, what it should have said, what the mistake cost. Technical buyers trust a system that shows its errors far more than one that claims not to have any.
 
## The Numbers as the Demo Reports Them
 
| Metric | Value |
|---|---|
| Settled accuracy | 98.5% — one technique fills the window |
| Mid-technique-change | 69.3% — genuinely ambiguous, and labelled as such |
| Overall, live | 86.0% rolling 600 samples |
| Link uptime under jamming | 74.4% |
| Median time to counter | 2.4 s |
| Engagements this session | DEF 30 – JAM 8 (the jammer still wins some — it's a real contest) |
 
## Talk Track — Three Sentences Carry It
 
- "The network has never been told the answer — it's reading the picture."
- "That decision took a third of a millisecond, on this box, with the network unplugged."
- When it slips: "Watch — it just got one wrong, and it's telling you so."
When it misreads, lean in. It happens every few engagements, almost always in the ~2 s after the jammer switches. Don't talk over it. Point at the frozen panel, note the model knew it was in an ambiguous window. That moment closes more technical buyers than the accuracy number does.
 
## Honesty Boundary — Say It Before You're Asked
 
Annotation Mode tags every element **REAL** — measured on this device — or **SIMULATED** — synthetic RF.
 
- **Simulated:** the radio physics and engagement geometry.
- **Real:** the network, its GPU execution, the latency, the LLM briefs, the GPU telemetry.
**Do not claim:** this is not a certified EW product, not built against real threat libraries, not validated on captured RF. It's a compute-platform demonstration using a physically-grounded simulation. Positioned that way it's strong; positioned as an EW capability it won't survive the first knowledgeable question.
 
## Questions You'll Get
 
- **"Is this real EW?"** — No, the RF is simulated; the AI isn't. Annotation Mode marks which is which.
- **"Couldn't this run on a laptop?"** — The classifier, yes. The 70B alongside it, no — that's ~42 GB resident.
- **"How accurate really?"** — 86%, and it's on screen because the demo grades itself. Turn it around: ask what their model scores against ground truth, live, in front of a customer.
- **"What's the model?"** — Custom 5-class CNN, 144 KB, trained from scratch here at install, TensorRT FP16. Nothing proprietary, nothing downloaded.
- **"What if it breaks on stage?"** — Degrades GPU → CPU → heuristic, active tier always named in the header. LLM unreachable falls back to a template. No state where the screen goes blank.
## Status
 
Running and demonstrable, port 8089, autonomous — safe on an unattended table. ~6 min install after deps, idempotent. ~3,100 lines, no Docker, no external model download. The train-on-device → TensorRT → serve pattern transfers to any vertical with a sensor stream.
 
Recent work took live accuracy 83% → 86%, corrected seven inaccurate or stale on-screen claims, and added the self-grading scorecard.