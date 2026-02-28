# Run log 413 excerpt (server_auto + assistant_message double-response)

- 09:51:14 speech ended, Theo entered thinking, and `response.created: origin=server_auto` was received.
- `response_binding ... response_key=synthetic_server_auto_2 turn_id=turn_2 origin=server_auto`.
- `lifecycle_event ... decision=response_created_defer:awaiting_transcript_final`.
- Transcript finalized for `item_DEFwrndyEkMGVAp8RhuPE` and key was rebound (`decision=transition_replaced`).
- `response_schedule_marker ... origin=assistant_message input_event_key=synthetic_preference_recall_3 ... mode=queued`.
- `response_create_scheduled turn_id=turn_2 origin=assistant_message reason=active_response`.
- Before cancellation could win, server-auto audio started: multiple `lifecycle_event ... origin=server_auto ... decision=audio_delta_allow:transitioned=audio_started`.
- `response_cancel_noop ... reason=no_active_response` appears only after `response.done`.
- Later, after playback restart, a second response is created: `response.created: origin=assistant_message` with `input_event_key=synthetic_preference_recall_3`.
- This second assistant response also reaches audio started and completes, increasing diagnostics responses count from 2 to 3 for one utterance.
