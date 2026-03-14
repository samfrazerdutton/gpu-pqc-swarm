### Demo 7: Quantum-Safe Command & Control (C2)
Fusing **ML-KEM-768** (Confidentiality) with **ML-DSA-65** (Authenticity).

```text
[ Step 5 ] Authority signs command with ML-DSA-65...
  Command: EXECUTE:MISSION_ALPHA|WAYPOINT:53.48,-2.24|ALT:120|ROE:HOLD
[ Step 6 ] 1000 peers verify signature on GPU...
  [DSA Verify] 1000 verifications in 0.379ms — 1000/1000 valid
[ Step 7 ] Tamper test — injecting forged command...
  [DSA Verify] 1000 verifications in 0.323ms — 0/1000 valid
  Forged command rejected by: 1000/1000 peers ✓

TOTAL PIPELINE TIME (1000 Nodes): 29.77ms

