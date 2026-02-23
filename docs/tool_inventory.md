# Tool Inventory

| Tool | Governance Tier | Side Effects | Sensitivity | Default Confirmation |
|---|---|---|---|---|
| `read_battery_voltage` | SAFE | NONE | LOW | NEVER |
| `read_environment` | SAFE | NONE | LOW | NEVER |
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
