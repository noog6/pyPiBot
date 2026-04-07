# Tool Inventory

`config/default.yaml -> governance.tool_specs` is the canonical source of tool governance metadata.
This document is a derived representation for human review and should stay in parity with config.

> **Status (2026-04-03): Derived reference (not canonical).**
> Update this table in the same PR whenever `governance.tool_specs` changes.
> If values differ, `config/default.yaml` wins.

| Tool | Governance Tier | Side Effects | Sensitivity | Default Confirmation |
|---|---|---|---|---|
| `read_battery_voltage` | SAFE | NONE | LOW | NEVER |
| `read_environment` | SAFE | NONE | LOW | NEVER |
| `read_runtime_diagnostics` | SAFE | NONE | LOW | NEVER |
| `read_imu_data` | SAFE | NONE | LOW | NEVER |
| `gesture_idle` | GUARDED | REVERSIBLE_MOTION | LOW | ASK |
| `gesture_nod` | GUARDED | REVERSIBLE_MOTION | LOW | ASK |
| `gesture_no` | GUARDED | REVERSIBLE_MOTION | LOW | ASK |
| `gesture_look_around` | GUARDED | REVERSIBLE_MOTION | LOW | ASK |
| `gesture_look_up` | GUARDED | REVERSIBLE_MOTION | LOW | ASK |
| `gesture_look_left` | GUARDED | REVERSIBLE_MOTION | LOW | ASK |
| `gesture_look_right` | GUARDED | REVERSIBLE_MOTION | LOW | ASK |
| `gesture_look_down` | GUARDED | REVERSIBLE_MOTION | LOW | ASK |
| `gesture_look_center` | GUARDED | REVERSIBLE_MOTION | LOW | ASK |
| `gesture_curious_tilt` | GUARDED | REVERSIBLE_MOTION | LOW | ASK |
| `gesture_attention_snap` | GUARDED | REVERSIBLE_MOTION | LOW | ASK |
| `update_user_profile` | PRIVILEGED | PERSISTENT_WRITE | HIGH | ALWAYS |
| `get_output_volume` | SAFE | NONE | LOW | NEVER |
| `set_output_volume` | GUARDED | REVERSIBLE_AUDIO | LOW | ASK |
| `remember_memory` | GUARDED | PERSISTENT_WRITE | MEDIUM | ASK |
| `recall_memories` | SAFE | NONE | MEDIUM | NEVER |
| `forget_memory` | PRIVILEGED | PERSISTENT_DELETE | HIGH | ALWAYS |
| `perform_research` | GUARDED | NETWORK_IO | MEDIUM | ASK |

## Availability outcomes convention

Tools may be callable even when temporarily unavailable. In those cases, the tool should return a structured unavailable result instead of disappearing from the harness contract:

- `status` (`disabled`, `unavailable`, `blocked`, or `deferred`)
- `reason_code`
- `message`
- `retryable`
- optional `user_action`, `provider`, and `details`

`perform_research` is the reference implementation for this pattern.
