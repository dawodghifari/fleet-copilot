# Workshop playbook — acting on APS failure predictions

## When a truck is flagged

A flagged vehicle means the model estimates the fault is APS-related
with probability above the cost-tuned threshold. Because the threshold
is deliberately low (missed failures cost 50x more than checks), roughly
half of flags may be false alarms — that is by design, not a model error.

## Triage sequence

1. **Pressure build-up test**: with full reservoirs drained, time the
   compressor from 85 to 100 psi at governed RPM. Build-up slower than
   the manufacturer's spec (typically under 45 seconds) points to
   compressor wear or a supply-side leak.
2. **Leak-down test**: engine off, full pressure, brakes released —
   pressure drop should stay under ~2 psi/min for straight trucks
   (~3 psi/min with brakes applied). Faster loss means leaks at
   fittings, hoses, or chamber diaphragms.
3. **Air dryer check**: open the wet-tank drain. Water or oil discharge
   means the dryer cartridge is saturated — replace it and inspect
   downstream valves for contamination.
4. **Governor verification**: confirm cut-in/cut-out pressures against
   spec. Out-of-range settings stress the compressor and dryer.
5. **Valve function**: check relay and quick-release valves for slow
   release or dragging brakes.

## Prioritization guidance

- Flags with high predicted probability (well above threshold) and
  histogram features showing heavy high-load operation should be seen
  first.
- Vehicles whose flag persists across consecutive prediction runs are
  higher priority than one-off flags.
- A cleared vehicle that is re-flagged within a short interval warrants
  a deeper inspection than the standard triage.

## Record keeping

Log the inspection outcome (confirmed APS fault vs. false alarm and the
component found) against the vehicle record. Confirmed outcomes are the
ground truth that future model retraining depends on; unlogged false
alarms silently degrade the next training cycle.
