# Locals

A local inference server for Apple Silicon. This fork also lets Qwen3.5 and Qwen3-VL models read images.

## Language

**Image prompt**:
A prompt that holds one or more images, each written as its own run of image tokens, read with every token at its image-aware position.
_Avoid_: vision prompt, embedded prompt, image request

**Cache key**:
The token sequence a prompt is stored under in the prompt cache. For a text prompt it is the prompt itself; for an image prompt, each image's tokens are replaced by that image's identity, so two different images never share an entry. An image prompt with an image of unknown identity has no cache key and is not cached.
_Avoid_: lookup tokens, key tokens

**Cache budget**:
The most memory the prompt cache may hold right now. It starts at the start budget, shrinks when other apps need memory, grows back when they let it go, and never passes the size the user set. When less than the memory floor is free, it can be zero.
_Avoid_: cache size, cache limit, cache cap

**Start budget**:
The cache budget the server starts with: the size the user set, or else what the GPU limit leaves free at start. The memory other apps use does not change it. The cache budget never goes above it.
_Avoid_: initial size, max cache

**Memory floor**:
The share of the Mac's memory the server leaves free for other apps. When less than that is free, the prompt cache gives memory back. A new request also waits while others run if starting it would leave less than that free.
_Avoid_: memory margin, reserve, headroom

**Draft**:
Tokens proposed ahead of the model and checked in the same pass as the next token. Only the tokens the model would have chosen itself are kept, so a draft never changes a reply.
_Avoid_: speculation, guess tokens

**Draft source**:
Where a draft comes from: prompt lookup (what followed an earlier occurrence of the last few tokens) or the MTP layer.
_Avoid_: drafter, draft model

**MTP layer**:
The model's own extra layer that predicts the token after next from the main model's state at the current token. Qwen3.5 checkpoints have one; their 4-bit MLX conversions drop it.
_Avoid_: MTP head, draft head, guess layer

**Hole**:
A position in a request's cache whose content later positions must not see: it is masked out of attention. The MTP layer's entry has one at each position the layer did not see; a stretch leaves one in the main cache at each rejected draft, until its end.
_Avoid_: gap, masked slot, padding

**Stretch**:
The run of passes a request alone makes with one MTP draft each, one pass launched before the last one is read. It starts when the request has its draft and ends at a stop, a forced span or a request arriving, with its holes compacted away.
_Avoid_: pipelined run, draft session, burst

**Tuning file**:
This Mac's record of which kernel each matmul shape uses, with the timings that chose it. Each shape records the MLX version it was timed under.
_Avoid_: tune cache, kernel cache

**Stale shape**:
A shape in the tuning file timed under another MLX version than the running one. It keeps its choice until it is timed again. A shape with no recorded version, or one pinned after it stopped locals twice, is never stale.
_Avoid_: outdated shape, old timing

**Seed**:
Another tuning file, for this GPU, that a start under an MLX version with no file of its own begins from: a first start, or one whose file did not parse and was set aside. Its shapes keep their recorded versions, so only those timed under another version are stale.
_Avoid_: base file, starter file
