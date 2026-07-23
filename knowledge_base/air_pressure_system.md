# Air Pressure System (APS) in heavy trucks

## What the APS does

The Air Pressure System generates, stores, and distributes compressed air
in heavy vehicles. Its most safety-critical consumer is the service brake:
air brakes on heavy trucks are applied by compressed air and held OFF by
spring pressure in the parking-brake chambers. The APS also supplies air
for gear changing, clutch actuation, suspension height control (air
suspension), and auxiliary functions such as the driver's seat and doors
on some vehicles.

## Main components

- **Compressor**: engine-driven pump that produces compressed air. Runs
  continuously with the engine; a governor controls when it builds
  pressure (cut-in around 100 psi / 6.9 bar, cut-out around 125 psi /
  8.6 bar in typical North American systems).
- **Air dryer**: removes moisture and oil aerosols from compressed air
  before storage. A saturated dryer cartridge lets water into the tanks,
  which corrodes valves and can freeze in winter — a common root cause of
  downstream APS faults.
- **Reservoirs (wet and dry tanks)**: store compressed air. The wet tank
  receives air first and collects residual moisture; drain valves must be
  exercised regularly.
- **Foot (treadle) valve**: meters air to the brake chambers in
  proportion to pedal force.
- **Relay and quick-release valves**: speed up brake application and
  release at axles distant from the foot valve.
- **Brake chambers and slack adjusters**: convert air pressure into
  mechanical force on the foundation brakes.
- **Low-pressure warning**: activates below roughly 60 psi; spring brakes
  self-apply if pressure falls to the 20–45 psi range.

## Common APS failure modes

- Compressor wear: falling build-up rate, oil carry-over into the system.
- Air dryer saturation: moisture contamination, valve corrosion, freezing.
- Leaks at fittings, hoses, chamber diaphragms: slow build-up, frequent
  compressor cycling, audible hiss.
- Governor faults: over- or under-pressurization.
- Stuck relay/quick-release valves: dragging or slow-releasing brakes.

## Why predictive maintenance matters for the APS

An APS failure on the road is expensive and dangerous: a vehicle with
insufficient air pressure will have its spring brakes apply automatically,
immobilizing it wherever it stands. Roadside recovery of a loaded heavy
truck, cargo delay, and potential safety exposure make an undetected
imminent failure far more costly than a scheduled workshop inspection.
This asymmetry is encoded in the Scania APS dataset's cost metric: a
missed failure (false negative) costs 500, an unnecessary check (false
positive) costs 10 — a 50:1 ratio.
