# Checkpoints

Place fine-tuned lip-recognition checkpoints in this directory. Checkpoint binary
files are intentionally ignored by Git.

End-to-end inference requires a checkpoint containing all three trained parts:

- visual lip encoder;
- CTC projection head;
- attention decoder.

The Hand Prompt Decoding Agent does not have a separate checkpoint. It adds the
frame-aligned hand prompt to the trained CTC logits during beam search.
