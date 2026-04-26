---
title: "impl Drop for Tensor"
subtitle: "How Rust replaced five layers of GPU memory management"
date: 2026-03-18
description: "What happens when you try to manage GPU memory with a garbage collector, and what happens when you stop trying."
---

I'm not a systems programmer. I spent twenty years writing PHP and JavaScript.
But I had a deep learning project that needed a custom framework, and Python's
overhead was killing the architectures I wanted to explore. So I learned Go
and built one.

It worked. Then I rebuilt it in Rust. Not because Rust is trendy, but because
Go's garbage collector cannot manage GPU memory, and the infrastructure I
built to work around that was becoming the project.

This is the story of what I learned. And yes, AI helped me write the code.
I'll be honest about that throughout. But understanding *why* Go's GC fails
at GPU memory, and recognizing that Rust's ownership model is the structural
answer, required staring at the problem long enough that no amount of
autocomplete would have helped.

## The problem

When you train a neural network on a GPU, tensors live in VRAM. They're
allocated by libtorch's C++ memory allocator, which knows nothing about your
language's runtime. As far as Go is concerned, a tensor is a pointer. 8 bytes
on the heap. The fact that it points to 50 MB of GPU memory is invisible.

This matters because Go's garbage collector makes decisions based on heap
pressure. A million 8-byte pointers exert almost no GC pressure. The GC sees a
clean, tidy heap and does nothing. Meanwhile, VRAM fills up silently until
CUDA returns an out-of-memory error and your training run crashes.

This is not a Go bug. It is a fundamental mismatch between garbage collection
and foreign memory.

## What we built to fix it

Five layers of infrastructure, each solving a problem created by the one
before. I say "we" because AI did the heavy lifting on implementation. But
each layer exists because I hit a wall, understood why, and described what
was needed. The pattern repeated five times. That should tell you something
about how deep the problem goes.

### Layer 1: Atomic reference counting

Every tensor gets a manual reference count. `Retain()` increments it.
`Release()` decrements it. At zero, the C++ handle is freed. A GC finalizer
acts as a safety net for tensors that escape explicit lifecycle management.

```go
type Tensor struct {
    raw  *libtorch.Tensor
    refs int64
    err  error
}

func wrap(raw *libtorch.Tensor) *Tensor {
    t := &Tensor{raw: raw, refs: 1}
    activeTensors.Add(1)
    runtime.SetFinalizer(t, (*Tensor).release)
    libtorch.EnforceVRAMBudget()
    return t
}

func (t *Tensor) Retain() {
    atomic.AddInt64(&t.refs, 1)
}

func (t *Tensor) release() {
    if atomic.AddInt64(&t.refs, -1) == 0 {
        t.raw.Free()
        t.raw = nil
        activeTensors.Add(-1)
        runtime.SetFinalizer(t, nil)
    }
}
```

This is manual memory management written in a garbage-collected language.
It works, but every operation that touches a tensor must get the lifecycle
right. Miss a `Release()` and you leak VRAM. Call it twice and you
use-after-free.

### Layer 2: Autograd scope

The backward pass creates hundreds of intermediate tensors per batch.
Waiting for the GC to find them is not an option. So I built a scope
that tracks every intermediate and releases them all at once:

```go
scope := autograd.NewScope()
// ... forward, backward, step ...
scope.Close()  // frees all intermediates instantly
```

Behind the scenes: an atomic pointer to the active scope, a mutex-protected
list, and careful bookkeeping to not release leaf parameters that must
survive across batches.

### Layer 3: GC callback for CUDA OOM

When the CUDA allocator runs out of memory, it fires a callback. I registered
a Go function that triggers `runtime.GC()` to reclaim unreachable tensors.
But there's a trap: the callback runs on the same thread as the CGo call
that triggered the allocation. You cannot free tensor handles inside the
callback because the C++ code above you on the stack is still using them.

So I added a pending-free queue:

```go
var pendingFreeHandles []unsafe.Pointer

//export goTriggerGC
func goTriggerGC() {
    gcCallbackActive.Add(1)
    runtime.GC()
    time.Sleep(time.Millisecond)
    gcCallbackActive.Add(-1)
}
```

When the GC callback is active, `Free()` queues handles instead of freeing
them. The queue drains lazily on the next `Free()` call outside the callback
context. Getting the threading right took days of debugging
use-after-free crashes.

### Layer 4: VRAM budget with proactive GC

The GC callback only fires at true OOM. By then it's often too late. So I
added a proactive check: every 100 tensor allocations, read the C++ allocation
counter. If it exceeds 90% of VRAM, trigger `runtime.GC()` preemptively.

```go
func EnforceVRAMBudget() {
    if vramBudget <= 0 {
        return
    }
    if wrapCount.Add(1)%checkEveryN != 0 {
        return
    }
    if CUDAAllocatedBytes() > vramBudget {
        runtime.GC()
    }
}
```

Three layers of defense: proactive at 90%, allocator cap at 95%, OOM
callback as last resort.

### Layer 5: runtime.KeepAlive everywhere

Go's GC can collect a `*Tensor` wrapper while its C++ handle is being passed
to a CGo function. If the GC fires between the CGo call and the return, the
finalizer can queue the handle for freeing. Next drain: use-after-free.

The fix: `runtime.KeepAlive(t)` after every CGo call. Every single one.
Fifty-plus sites in the tensor operations file alone:

```go
func (t *Tensor) Add(other *Tensor) *Tensor {
    raw, err := libtorch.Add(t.raw, other.raw)
    runtime.KeepAlive(t)
    runtime.KeepAlive(other)
    if err != nil {
        return errTensor(err)
    }
    return wrap(raw)
}
```

Miss one and you have a race condition that manifests as a CUDA crash under
load, three hours into a training run. Good luck debugging that.

## The cost

All of this works. [goDl](https://github.com/fab2s/goDl) trains real models.
But the memory management infrastructure became a project within the project.
Hundreds of lines of `Retain()`/`Release()`, `runtime.KeepAlive`,
`runtime.SetFinalizer`, pending-free queues, VRAM budgets, and GC callbacks.
Each layer added because the previous one wasn't enough. Each one a new
surface area for bugs that only manifest under GPU pressure.

And the cognitive overhead is worse than the code overhead. Every time I wrote
a new operation, I had to think: does the caller own this tensor? Who releases
it? What if the scope is active? What if the GC callback fires here? The
language that was supposed to let me focus on the math was forcing me to think
about memory on every line.

## Then Rust

```rust
impl Drop for Tensor {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            LIVE_TENSOR_COUNT.fetch_sub(1, Ordering::Relaxed);
            unsafe { ffi::flodl_free_tensor(self.handle) };
        }
    }
}
```

That's it. That is the entire VRAM management story.

When a `Tensor` goes out of scope, `Drop` fires and the C++ handle is freed.
Deterministically. On the same thread. At the exact point where the tensor
is no longer needed. No reference counting. No finalizers. No GC callbacks.
No pending-free queues. No VRAM budgets. No `KeepAlive`.

The five layers collapse into a language guarantee. Rust's ownership model does
not manage VRAM by tracking it. It manages VRAM by making it impossible to
forget to free it. The compiler rejects code where a tensor's lifetime is
ambiguous. There is no runtime check because there is no runtime question.

### Clone is explicit

```rust
impl Clone for Tensor {
    fn clone(&self) -> Self {
        // Shallow clone: bumps libtorch's internal refcount.
        // Both handles point to the same storage.
        // Each one drops independently.
    }
}
```

No `Retain()`/`Release()` pairs to get right. If you want two handles to the
same data, you `clone()`. Both are owned. Both drop. libtorch's internal
refcount handles the storage. The Rust side doesn't need to know or care.

### Backward works for free

In Go, the backward pass had explicit `Release()` calls for saved tensors,
accumulated gradient replacement, and scope integration. In Rust, the
autograd engine delegates to libtorch's native C++ backward. Tensors created
during the backward pass are owned by the Rust variables that hold them.
When those variables go out of scope, the tensors drop. No scope needed. No
explicit release. No lifecycle tracking.

### No KeepAlive

Rust's borrow checker guarantees that a reference to a tensor is valid for
the duration of the FFI call. There is no garbage collector that might
collect the wrapper while C++ is reading the handle. The failure mode that
required fifty `runtime.KeepAlive` annotations in Go does not exist as a
concept in Rust.

## The graph builder didn't come from nowhere

A side note on design lineage. flodl's fluent graph builder, where you write
`FlowBuilder::from(encoder).through(decoder).also(residual).build()`, looks
like it appeared during the Rust rewrite. It didn't.

Years ago, working in PHP, I kept bumping into the same wall: orchestrating
complex task flows without drowning in boilerplate. I wrote
[NodalFlow](https://github.com/fab2s/NodalFlow), a generic directed graph
execution engine where nodes compose into flows and flows nest into larger
flows. I wrote on its README: "NodalFlow could easily become Turing complete
after all." That was a PHP workflow library. The ambition was already there.

NodalFlow became the foundation for [YaEtl](https://github.com/fab2s/YaEtl),
an ETL framework where extractors, transformers, loaders, and joiners compose
as graph nodes. Same pattern: fluent construction, data flows through, nodes
don't know about each other, the graph handles routing.

When I designed goDl's graph engine, and then flodl's, the same architecture
carried forward. `from/through/split/merge/gate/switch/loop_body` is the
deep learning descendant of a PHP workflow library. The nodes are neural
network layers now. The data flowing through is tensors. But the composition
model, the idea that complex behavior emerges from simple nodes in a
well-structured graph, that's been evolving in my head for a decade.

The irony is not lost on me: a framework that could train models capable of
beating the Turing test, descended from a PHP library I once mused could
become Turing complete.

## What it cost, what it bought

Rebuilding the framework in Rust took ten epic days. I didn't know Rust when I
started. AI wrote most of the code, just as it had for Go. But here's the
thing: in Go, AI also had to write the five layers of memory management. In
Rust, there was nothing to write. The code I didn't have to ask for is the
point. No reference counting. No scope tracking. No GC callbacks. No VRAM
budgets. No KeepAlive barriers. No pending-free queues. No finalizers. No
three-layer OOM defense.

All of those were solutions to the same problem: a language runtime that
cannot see GPU memory. Rust doesn't solve this problem. Rust makes it not
exist.

The result is [flodl](https://github.com/flodl-labs/flodl), a deep learning
framework where the memory management story is seven lines long and the rest
is math. On a real training workload (recurrent attention with a 9-component
loss stack), it runs
[19% faster than PyTorch](https://github.com/flodl-labs/flodl/blob/main/docs/benchmark.md)
on the same GPU. Not because Rust is faster at matrix multiplication. The CUDA
kernels are identical. But the space between kernel launches is where Rust's
ownership model pays off: no GC pauses, no interpreter overhead, no per-op
dispatch cost. The GPU never starves for work.

I didn't set out to beat PyTorch. I set out to train a model I couldn't
train in Python. I just needed a language that would let me think about the
model instead of the memory. Rust is that language. AI is what let a PHP
developer get there in ten epic days. But the insight that made it work was
neither mine nor AI's. It's Bjarne Stroustrup's, and Graydon Hoare's, and
everyone who understood that resource management belongs in the type system.

---

*flodl is open source:
[GitHub](https://github.com/flodl-labs/flodl) |
[crates.io](https://crates.io/crates/flodl) |
[docs](https://docs.rs/flodl) |
[benchmark](https://flodl.dev/benchmark)*

*The Go predecessor:
[goDl](https://github.com/fab2s/goDl)*
