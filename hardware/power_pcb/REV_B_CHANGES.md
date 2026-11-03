# Rev B: Change List and Order Gate

Rev B fixes every Rev A issue listed in [README.md](README.md#known-issues-rev-a). I don't order boards until every gate at the bottom of this file passes.

The circuit stays the same. The buck design (5.0 V feedback divider, catch diode, inductor, compensation, layout) checked out on Rev A. Every fix is a connection change.

## Changes

| # | Change | In KiCad | New parts? |
|---|---|---|---|
| 1 | Connect J3 pin 1 to the VIN / input-cap net | Schematic: wire J3.1 to U1 VIN (pin 2) and C2, C4, C5, C8. Layout: route it as a real connection, no dangling stub. | No |
| 2 | Take EN off VIN | Schematic: break the U1 pin 2 to pin 3 link. Leave EN floating; the datasheet's internal pull-up enables the part. Also add two DNP 0402 footprints for an optional UVLO divider, so I can set a start voltage later without another respin. | No (the divider is optional and not populated) |
| 3 | Put the shunt in series | L1 output and C6, C7, C9 go to INA219 IN+ and R7 pin 2. R7 pin 1 and IN- become the load-side 5 V net, feeding J2 and the INA219 VS pin. | No |
| 4 | Move Q1 to low-side switching | J1 pin 1 (LED+) goes to the LED supply rail. J1 pin 2 (LED-) goes to Q1 drain. Q1 source goes to GND. Gate network R2 and R3 stays as is. | No, same IRLZ44N |
| 5 | Add a signal header J4: LED_PWM, SDA, SCL, GND | 4-pin connector near U2 and R2, cable exit on an edge | **Yes: one 4-pin connector, plus its mating housing and crimps** |
| 6 | Add mounting holes | M3 (3.2 mm NPTH) at top-right, bottom-left, and bottom-right. The top-left is blocked by J3. | No board parts. M3 standoffs for the enclosure. |
| 7 | Settle the input-cap value | Match the schematic and BOM to the part I actually use | **Maybe**, see below |
| 8 | Clean up the BOM | Re-export from the routed project. D1 becomes SMC (not SOD-123). Fix the C2 to C8 values. | No |
| 9 | Recover or rebuild the routed `.kicad_pcb` | Rev B starts from the layout that produced the Rev A Gerbers, so placement carries over | No |

## Do I need new components?

Mostly no. Every Rev A part carries over: U1, U2, Q1, L1, D1, R1 to R7, the compensation parts, C1, C6, C7, C9, C10, and J1 to J3.

| Item | Answer |
|---|---|
| 4-pin signal connector J4, plus its mating housing and crimps | **Yes, new.** Rev A has nowhere for these signals to go. |
| M3 standoffs and screws | **Yes, new** mechanical hardware, but not board components |
| Input caps C2, C4, C5, C8 | **Confirm before ordering.** I ordered 1 uF 25 V X5R 0402 parts. At 12 V, small X5R parts lose a lot of their rated capacitance, so the four may add up to well under 4 uF in practice. I'll check the TPS54560B datasheet's recommended input capacitance and the part's DC-bias curve. If it falls short, I'll buy larger caps (0805, higher voltage) before ordering. |
| LED supply rail for change #4 | **No part,** but I need one fact: is the LED panel rated for 5 V or for the input voltage? That decides which rail J1 pin 1 connects to. If the panel has no current limiting of its own, I'll also need a series resistor. |
| Q1 at 3.3 V gate drive | **No part if it checks out.** The IRLZ44N is a logic-level FET. I'll check its transfer curve at my LED current and 3.3 V gate drive before keeping it. |

## Order of work

1. **Schematic:** apply changes 1 to 5 and 7, then run ERC to zero errors.
2. **Layout:** apply the changes, add the holes and J4, then run DRC to 0 errors and 0 unconnected pads. Silkscreen warnings are acceptable.
3. **Plot:** export Gerbers and drill files with the extended X2 format on.
4. **Netlist gate:** run `python3 hardware/power_pcb/tools/check_rev_b.py <new gerber dir>`. It has to report 15/15. On the Rev A Gerbers it reports 2/15, which shows it catches every Rev A defect.
5. **Human review:** Dr. Ryberg or the UConn PCB lab looks over the schematic and the checker output before I order.
6. **Order:** Gerbers plus any new parts (J4, its mating side, and caps if #7 calls for them).

## Gates before I place the order

- [ ] ERC 0 errors
- [ ] DRC 0 errors, 0 unconnected pads
- [ ] `check_rev_b.py` 15/15 on the exact Gerbers being uploaded
- [ ] Input cap value confirmed against the datasheet and the DC-bias curve
- [ ] LED supply rail and Q1 gate drive confirmed
- [ ] Reviewed by Dr. Ryberg or the PCB lab
- [ ] BOM re-exported, and any new parts are in the cart with the boards
