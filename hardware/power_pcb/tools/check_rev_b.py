"""Rev B gate: check the fab Gerbers' embedded netlist against every Rev A defect.

I run this on the Gerbers I'm about to order, not on the KiCad file. The Gerbers
are what the fab builds. KiCad writes each pad's component, pin and net into
the copper layers as X2 attributes, so the netlist the board will actually have
can be read straight from them.

Usage:
    python3 hardware/power_pcb/tools/check_rev_b.py <gerber_dir>

It exits 0 only if every check passes. Run against the Rev A Gerbers in
fab/gerbers/, it should fail on every Rev A issue. That is how I know the checks
can see each defect.
"""
import glob
import os
import re
import sys
from collections import defaultdict

# Reference designators and pin numbers. Update these if Rev B renumbers anything.
INPUT_CONN = ("J3", "1")          # supply input, + pin
U1_VIN, U1_EN = ("U1", "2"), ("U1", "3")   # TPS54560B DDA: pin 2 VIN, pin 3 EN
L1_OUT = ("L1", "2")              # buck output side of the inductor
INA_INP, INA_INN = ("U2", "8"), ("U2", "7")  # INA219 SOIC-8: 8 IN+, 7 IN-
SHUNT = "R7"
PI_OUT = ("J2", "1")              # 5 V to the Pi
Q_DRAIN, Q_SOURCE = ("Q1", "2"), ("Q1", "3")  # IRLZ44N TO-220: 1 G, 2 D, 3 S
LED_CONN = "J1"
SIGNALS = ("LED_PWM", "SDA", "SCL")
MIN_MOUNT_HOLES, MIN_MOUNT_DIA_MM = 3, 3.0


def read_nets(gerber_dir):
    """Return {(ref, pin): net} from the copper layers' X2 pad attributes."""
    pin_net = {}
    for path in glob.glob(os.path.join(gerber_dir, "*Cu.gbr")):
        pin = net = None
        for line in open(path, errors="replace"):
            m = re.match(r"%TO\.P,([^,*]+),([^,*]+)", line)
            if m:
                pin = (m.group(1), m.group(2))
                continue
            m = re.match(r"%TO\.N,([^*]+)\*%", line)
            if m:
                net = m.group(1)
                continue
            if line.startswith("%TD*%"):
                pin = net = None
                continue
            if pin and net and re.match(r"X-?\d+Y-?\d+D03", line):
                pin_net[pin] = net
    return pin_net


def mount_holes(gerber_dir):
    """Count drill hits at or above MIN_MOUNT_DIA_MM across PTH and NPTH files."""
    count = 0
    for path in glob.glob(os.path.join(gerber_dir, "*.drl")):
        tools, current = {}, None
        for line in open(path, errors="replace"):
            m = re.match(r"T(\d+)C([\d.]+)", line)
            if m:
                tools[m.group(1)] = float(m.group(2))
                continue
            m = re.match(r"T(\d+)\s*$", line)
            if m:
                current = tools.get(m.group(1))
                continue
            if current and current >= MIN_MOUNT_DIA_MM and re.match(r"X-?[\d.]+Y-?[\d.]+", line):
                count += 1
    return count


def main(gerber_dir):
    pin_net = read_nets(gerber_dir)
    if not pin_net:
        sys.exit(f"No X2 netlist attributes found in {gerber_dir}. Re-plot with 'Use extended X2 format' on.")
    members = defaultdict(set)
    for p, n in pin_net.items():
        members[n].add(p)
    net = pin_net.get

    checks = []

    def check(name, ok, detail):
        checks.append((name, bool(ok), detail))

    vin = net(U1_VIN)
    check("1. Input reaches VIN", vin and net(INPUT_CONN) == vin,
          f"J3.1 on {net(INPUT_CONN)!r}, U1 VIN on {vin!r}")
    check("2. EN not tied to VIN", net(U1_EN) != vin,
          f"U1 EN on {net(U1_EN)!r}, VIN on {vin!r}")

    shunt_ends = {net((SHUNT, "1")), net((SHUNT, "2"))}
    inp, inn = net(INA_INP), net(INA_INN)
    check("3a. Shunt sits between IN+ and IN-", shunt_ends == {inp, inn} and inp != inn,
          f"R7 ends {sorted(map(str, shunt_ends))}, IN+ {inp!r}, IN- {inn!r}")
    check("3b. Buck output feeds IN+", net(L1_OUT) == inp, f"L1.2 on {net(L1_OUT)!r}")
    check("3c. Pi load is on IN- side", net(PI_OUT) == inn, f"J2.1 on {net(PI_OUT)!r}")
    check("3d. IN+ net carries more than the stub", len(members.get(inp, ())) >= 3,
          f"IN+ net members: {sorted(members.get(inp, ()))}")

    led_pins = {pin_net.get((LED_CONN, "1")), pin_net.get((LED_CONN, "2"))}
    check("4a. Q1 source on GND", net(Q_SOURCE) == "GND", f"Q1.3 on {net(Q_SOURCE)!r}")
    check("4b. Q1 drain goes to an LED pin", net(Q_DRAIN) in led_pins and net(Q_DRAIN) != "GND",
          f"Q1.2 on {net(Q_DRAIN)!r}, J1 pins on {sorted(map(str, led_pins))}")
    check("4c. Other LED pin is a supply, not GND", "GND" not in led_pins,
          f"J1 pins on {sorted(map(str, led_pins))}")

    for sig in SIGNALS:
        nets_for_sig = [n for n in members if n.strip("/").upper() == sig]
        on_conn = any(ref.startswith("J") for n in nets_for_sig for ref, _ in members[n])
        check(f"5. {sig} reaches a connector", on_conn,
              f"{sig} members: {sorted(p for n in nets_for_sig for p in members[n])}")
    check("5b. A GND pin on a connector other than power", any(
        ref.startswith("J") and ref not in ("J1", "J2", "J3") for ref, _ in members.get("GND", ())),
        "GND on signal header")

    holes = mount_holes(gerber_dir)
    check(f"6. At least {MIN_MOUNT_HOLES} mounting holes >= {MIN_MOUNT_DIA_MM} mm", holes >= MIN_MOUNT_HOLES,
          f"{holes} found")

    singles = sorted(n for n, m in members.items() if len(m) == 1 and n not in ("GND",))
    check("7. No single-pad nets (dangling connections)", not singles, f"single-pad nets: {singles}")

    width = max(len(c[0]) for c in checks)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {detail}")
    failed = sum(not ok for _, ok, _ in checks)
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
