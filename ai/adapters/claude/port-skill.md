---
name: port
description: Port a PyTorch script to flodl (Rust deep learning). Reads the source, maps PyTorch patterns to flodl equivalents, generates a complete Rust project, validates with cargo check.
argument-hint: <pytorch-file.py>
allowed-tools: Bash(fdl *) Bash(cargo check *) Bash(cargo build *)
---

# Port PyTorch to flodl

You are porting `$ARGUMENTS` from PyTorch to flodl.

## Bootstrap

First, get the flodl API reference and porting guide:

```!
fdl api-ref 2>/dev/null | head -200 || echo "fdl not available, will explore source"
```

Read the full porting guide:

```!
cat ai/skills/port/guide.md
```

Read the agent instructions:

```!
cat ai/skills/port/instructions.md
```

Now read the PyTorch source file and follow the instructions exactly.

## Process

1. Read the PyTorch file
2. Classify all blocks by intent (list them)
3. Design the flodl architecture (FlowBuilder vs manual, tags, data flow)
4. Generate the complete Rust project
5. Run `cargo check` and fix until clean
6. Report any missing flodl APIs encountered
