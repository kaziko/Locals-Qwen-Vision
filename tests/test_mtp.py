"""MTP drafts (issue #18, docs/adr/0001-mtp-kv-in-the-cache-list.md): the MTP layer's KV is one more entry at the end
of a request's cache list, its offset equal to the main cache's wherever the cache goes, and its drafts never change a
reply. Runs on a tiny random Qwen3.5 (four layers, the fused kernels' head sizes) and a random MTP layer, so it needs
no download; the real model's checks are in test_mtp_model.py."""

import mlx.core as mx
import mlx.nn as nn
import pytest

EOS = 1


@pytest.fixture(scope="module")
def tiny():
    from mlx_lm.models import qwen3_5 as Q

    from locals.fuse import fuse_model

    args = Q.TextModelArgs(model_type="qwen3_5", hidden_size=256, intermediate_size=512, num_hidden_layers=4,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=128, vocab_size=512,
                           linear_num_value_heads=2, linear_num_key_heads=1, linear_key_head_dim=128,
                           linear_value_head_dim=128, tie_word_embeddings=False)
    mx.random.seed(0)
    m = Q.TextModel(args)
    nn.quantize(m, group_size=64, bits=4)
    assert fuse_model(m) == 4
    return m


@pytest.fixture(scope="module")
def mtp_file(tiny, tmp_path_factory):
    """A random MTP layer for the tiny model, saved as the checkpoint stores it (``mtp.`` names, norms as w)."""
    from locals.mtp import Layer

    mx.random.seed(1)
    layer = Layer(tiny.args)
    path = tmp_path_factory.mktemp("mtp") / "mtp.safetensors"
    from mlx.utils import tree_flatten

    flat = {f"mtp.{k}": (v - 1.0 if v.ndim == 1 else v) for k, v in tree_flatten(layer.parameters())}
    mx.save_safetensors(str(path), flat)
    return path


class Tok:
    eos_token_ids = [EOS]

    def decode(self, t):
        return " ".join(map(str, t))


def _free():
    from locals.engine import Controller

    class Free(Controller):
        eos_ids = [EOS]

    return Free()


PROMPT = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]


def _engine(tiny, mtp=None):
    from locals.hybrid_batch import HybridBatchEngine

    return HybridBatchEngine(tiny, Tok(), n_slots=4, share_prefix=False, mtp=mtp)


def _serve(eng, reqs, lru=None, **kw):
    from locals.packed import Request

    return _serve_reqs(eng, [Request(list(p), _free(), n, **kw) for p, n in reqs], lru)


def _serve_reqs(eng, reqs, lru=None):
    """Serve the requests ``reqs``, all arriving at once."""
    waiting = list(reqs)

    def source(block):
        out = waiting[:]
        waiting.clear()
        return out

    eng.serve(source, lru=lru)
    return reqs


def _lru():
    from locals.hybrid_batch import chain_cache

    return chain_cache(max_size=16, max_bytes=1 << 30)


def _offsets(cache):
    """Positions each attention entry of a single-row cache holds."""
    from mlx_lm.models.cache import KVCache

    return [c.offset for c in cache if isinstance(c, KVCache)]


def test_mtp_off_keeps_the_cache_list_of_the_model(tiny):
    from mlx_lm.models.cache import make_prompt_cache

    lru = _lru()
    (r,) = _serve(_engine(tiny), [(PROMPT, 8)], lru)
    cache = lru.fetch_nearest_cache("locals", PROMPT + r.out + [0] * 20)[0]
    assert len(cache) == len(make_prompt_cache(tiny))


def test_mtp_on_adds_one_entry_with_the_main_offset(tiny, mtp_file):
    from mlx_lm.models.cache import make_prompt_cache

    from locals.mtp import MTP

    lru = _lru()
    (r,) = _serve(_engine(tiny, MTP.load(tiny, mtp_file)), [(PROMPT, 8)], lru)
    cache = lru.fetch_nearest_cache("locals", PROMPT + r.out + [0] * 20)[0]
    assert len(cache) == len(make_prompt_cache(tiny)) + 1
    offs = _offsets(cache)
    assert len(set(offs)) == 1, offs


def _real(entry):
    """Which slots of a single-row MTP entry hold the layer's KV (1) and which are holes (0)."""
    return [int(v > 0) for v in entry.keys[0, 0, : entry.offset, -1].tolist()]


def test_a_read_prompt_is_in_the_mtp_entry_up_to_its_last_token(tiny, mtp_file):
    """Every prompt position the read covers gets the layer's KV: the token after each is known, the last one's too
    (the prompt's last token, which the first decode step feeds)."""
    from locals.mtp import MTP

    lru = _lru()
    (r,) = _serve(_engine(tiny, MTP.load(tiny, mtp_file)), [(PROMPT, 3)], lru)
    cache = lru.fetch_nearest_cache("locals", PROMPT + r.out + [0] * 20)[0]
    assert _real(cache[-1])[: len(PROMPT) - 1] == [1] * (len(PROMPT) - 1)


def _oracle(tiny, mtp_file, truth, wrong, drafts):
    """The MTP layer, run as it is, but with its draft tokens taken from ``truth`` (the reply with MTP off) and the
    ones ``wrong(pass, i)`` names changed: the passes then keep all, some or none of the draft. Each draft records
    the entry's written positions next to the positions the request has fed (its own bookkeeping)."""
    from locals.mtp import MTP, _written

    class Oracle(MTP):
        seen = []

        def draft(self, entry, out, r, n):
            got = super().draft(entry, out, r, n)
            assert len(got) == n
            k = len(Oracle.seen)
            Oracle.seen.append((_written(entry), len(r.prompt) - 1 + len(r.fed) - 1))
            m = len(r.out)
            return [(truth[m + i] + 1) % 500 + 2 if wrong(k, i) else truth[m + i] for i in range(n) if m + i < len(truth)]

    return _as_oracle(Oracle, tiny, mtp_file, drafts)


def _as_oracle(cls, tiny, mtp_file, drafts):
    """The MTP layer of ``mtp_file``, loaded as usual, wrapped as the oracle class ``cls``."""
    from locals.mtp import MTP

    base = MTP.load(tiny, mtp_file, drafts=drafts)
    o = cls(tiny, base.layer, drafts)
    o.dtype = base.dtype
    return o


LONG = [5, 9, 3, 8, 2, 7, 4, 6, 33, 41, 57, 12, 99, 70, 18, 64, 23, 81, 45, 30]


@pytest.mark.parametrize("drafts,wrong", [
    (1, lambda k, i: False),          # every draft kept
    (1, lambda k, i: k % 2 == 1),     # every other draft rejected outright
    (2, lambda k, i: False),
    (2, lambda k, i: i == 1),         # the second draft rejected: part kept
    (2, lambda k, i: k % 3 == 2),     # both rejected every third pass
])
def test_mtp_drafts_keep_the_reply_and_the_entry_at_the_main_offset(tiny, mtp_file, drafts, wrong):
    """The loop that reads every pass before the next (2 drafts, and the reference loop for 1; the pipelined one:
    test_the_pipelined_loop_gives_the_greedy_reply_of_mtp_off_and_a_dense_cache)."""
    (off,) = _serve(_engine(tiny), [(LONG, 40)])
    mtp = _oracle(tiny, mtp_file, off.out, wrong, drafts)
    lru = _lru()
    eng = _engine(tiny, mtp)
    eng.mtp_pipelined = False
    (on,) = _serve(eng, [(LONG, 40)], lru)
    assert on.out == off.out
    assert mtp.seen and all(a == b for a, b in mtp.seen), mtp.seen
    assert eng.counters["draft_mtp_tokens"] > 0
    assert 0 < eng.counters["draft_mtp_accepted"] <= eng.counters["draft_mtp_tokens"]
    cache = lru.fetch_nearest_cache("locals", LONG + on.out + [0] * 20)[0]
    assert len(set(_offsets(cache))) == 1, _offsets(cache)
    assert sum(_real(cache[-1])[len(LONG):]) > len(on.out) // 2  # (the solo passes wrote the layer's KV)


def test_batched_rows_and_a_restored_prefix_keep_the_entry_at_the_main_offset(tiny, mtp_file):
    """Two requests decode together (batched steps leave holes), then one starts from a cached prefix of the first
    (the prompt cache's copy, its entry with it): the same replies as with MTP off, every kept cache at one offset."""
    from locals.mtp import MTP

    other = [40, 41, 42, 43, 44, 45, 46, 47, 48]

    def run(mtp):
        lru = _lru()
        eng = _engine(tiny, mtp)
        a, b = _serve(eng, [(LONG, 24), (other, 30)], lru)
        batched = dict(eng.counters)
        (c,) = _serve(eng, [(LONG + a.out + [77, 78], 20)], lru)
        return lru, batched, (a, b, c)

    _, _, ref = run(None)
    lru, batched, got = run(MTP.load(tiny, mtp_file))
    assert [r.out for r in got] == [r.out for r in ref]
    assert got[2].cached_tokens > len(LONG)  # (it did start from the first one's cache)
    assert batched["steps"] > batched.get("solo_steps", 0)  # (batched steps ran)
    for r in got[1:]:  # (the first one's entry went: the third one's extends it, chain_cache)
        cache = lru.fetch_nearest_cache("locals", r.prompt + r.out + [0] * 20)[0]
        assert cache is not None
        assert len(set(_offsets(cache))) == 1, _offsets(cache)


def test_a_prompt_read_in_mixed_passes_fills_the_entry(tiny, mtp_file):
    """A prompt read a chunk at a time beside a decoding row (locals/mixed.py): its entry gets the layer's KV at every
    position the read covers, and the replies are MTP off's."""
    from locals.mtp import MTP

    longp = [(7 * i) % 400 + 3 for i in range(90)]

    def run(mtp):
        lru = _lru()
        eng = _engine(tiny, mtp)
        eng.mix_chunk = 16
        from locals.packed import Request

        first = Request(list(LONG), _free(), 60)
        second = Request(list(longp), _free(), 10)
        sent = []

        def source(block):
            if not sent:
                sent.append(1)
                return [first]
            if len(sent) == 1 and len(first.out) >= 2:
                sent.append(1)
                return [second]
            return []

        eng.serve(source, lru=lru)
        return lru, eng, first, second

    _, _, f0, s0 = run(None)
    lru, eng, f1, s1 = run(MTP.load(tiny, mtp_file))
    assert (f1.out, s1.out) == (f0.out, s0.out)
    assert eng.counters.get("mixed_steps", 0) > 0
    row = lru.fetch_nearest_cache("locals", LONG + f1.out + [0] * 20)[0]  # (the row that decoded beside the read)
    assert len(set(_offsets(row))) == 1, _offsets(row)
    cache = lru.fetch_nearest_cache("locals", longp + s1.out + [0] * 20)[0]
    assert len(set(_offsets(cache))) == 1, _offsets(cache)
    assert _real(cache[-1])[: len(longp) - 1] == [1] * (len(longp) - 1)


def test_the_memory_guard_counts_the_mtp_entry(tiny, mtp_file):
    """One more attention layer next to the tiny model's one: twice the KV bytes a token (Qwen3.5-4B: 9 / 8)."""
    from locals.mtp import MTP

    off, on = _engine(tiny)._kv_bytes_per_token(), _engine(tiny, MTP.load(tiny, mtp_file))._kv_bytes_per_token()
    assert on == 2 * off


def test_a_saved_cache_keeps_its_entry_and_holes(tiny, mtp_file, tmp_path):
    """The disk cache saves and loads the list as it is: the entry, its offset and its hole marks."""
    from locals.hybrid_batch import DiskPromptCache
    from locals.mtp import MTP

    lru = _lru()
    eng = _engine(tiny, MTP.load(tiny, mtp_file))
    (r,) = _serve(eng, [(PROMPT, 6)], lru)
    key = PROMPT + r.out
    cache = lru.fetch_nearest_cache("locals", key + [0] * 20)[0]
    tokens = PROMPT[:-1] + r.fed
    disk = DiskPromptCache(tmp_path / "d")
    disk.save(tokens, cache)
    back = _lru()
    assert DiskPromptCache(tmp_path / "d").restore(back, "locals", tokens + [0, 0]) == len(tokens)
    loaded = back.fetch_nearest_cache("locals", tokens + [0, 0])[0]
    assert len(loaded) == len(cache)
    assert _offsets(loaded) == _offsets(cache)
    assert _real(loaded[-1]) == _real(cache[-1]) and 0 in _real(cache[-1])  # (the last fed token's slot: a hole)


def test_the_single_request_engine_gets_no_entry(tiny, mtp_file):
    from mlx_lm.models.cache import make_prompt_cache

    from locals.engine import make_spec_cache
    from locals.mtp import MTP

    _engine(tiny, MTP.load(tiny, mtp_file))
    assert len(make_spec_cache(tiny)) == len(make_prompt_cache(tiny)) == len(tiny.layers)


def test_another_model_keeps_mtp_off_with_one_line(tiny, mtp_file, monkeypatch, capsys):
    from locals.service import _load_mtp

    monkeypatch.setenv("LOCALS_MTP", str(mtp_file))
    assert _load_mtp(tiny, False) is None
    assert _load_mtp(tiny, True) is None
    lines = [l for l in capsys.readouterr().err.splitlines() if "LOCALS_MTP" in l]
    assert len(lines) == 2
    assert "not Qwen3.5-4B" in lines[0] and "Qwenputer" in lines[1]
    monkeypatch.delenv("LOCALS_MTP")
    assert _load_mtp(tiny, False) is None and capsys.readouterr().err == ""


def test_mtp_on_and_off_never_share_disk_cache_files(mtp_file, tmp_path):
    from locals.service import _disk_folder

    off = _disk_folder(tmp_path, "m/x", "float16")
    off.mkdir()
    on = _disk_folder(tmp_path, "m/x", "float16", mtp_file)
    on.mkdir()
    assert on != off
    assert _disk_folder(tmp_path, "m/x", "float16") == off and on.exists()  # (neither one's start deletes the other)
    assert _disk_folder(tmp_path, "m/x", "float16", mtp_file) == on and off.exists()


def _image_row(tiny, prompt, spans, n):
    """An image prompt on the tiny model: made-up features for its image tokens (no tower), as test_vision_batch."""
    from locals import vision
    from locals.engine import Engine
    from locals.packed import Request

    feats = [mx.random.normal((16, 256), key=mx.random.key(2)).astype(mx.float32)]
    eng = Engine(tiny, Tok())
    r = Request(list(prompt), _free(), n)
    enc = type("Enc", (), {"encode": lambda self, pics: (feats, []), "mrope_section": (6, 5, 5)})()
    r.image = vision.ImagePrompt(prompt, spans, [vision.Picture(None, (8, 8), key=77)], lambda: enc,
                                 eng.prefill_embedded, lambda: False)
    return r


def test_an_image_prompt_gets_no_mtp_drafts_and_keeps_its_reply(tiny, mtp_file):
    """The M-RoPE swap covers only the model's own layers: the image row's entry is holes past the text before its
    image, its reply is MTP off's, and the layer drafts nothing for it."""
    from locals.mtp import MTP

    prompt = [3, 17, 40, 41, 42] + [99] * 16 + [60, 61, 62, 63, 64, 65, 66, 67, 68]

    def run(mtp):
        lru = _lru()
        eng = _engine(tiny, mtp)
        r = _image_row(tiny, prompt, [(5, 21)], 30)
        waiting = [r]
        eng.serve(lambda block: [waiting.pop()] if waiting else [], lru=lru)
        return lru, eng, r

    _, _, ref = run(None)
    lru, eng, got = run(MTP.load(tiny, mtp_file))
    assert got.out == ref.out and len(got.out) == 30
    assert eng.counters.get("draft_mtp_tokens", 0) == 0
    cache = lru.fetch_nearest_cache("locals", got.cache_key + got.out)[0]
    assert len(set(_offsets(cache))) == 1, _offsets(cache)
    assert 1 not in _real(cache[-1])[5:]  # (holes from the image on)


def test_a_draft_cut_to_nothing_backs_off(tiny, mtp_file):
    """An MTP draft cut to nothing (here: every draft is the end-of-sequence token) counts as a miss: the row backs off
    to pipelined steps, as after a draft rejected outright, instead of draining the pipeline for every token."""
    (off,) = _serve(_engine(tiny), [(LONG, 40)])
    mtp = _oracle(tiny, mtp_file, off.out, lambda k, i: False, 1)
    type(mtp).draft = lambda self, entry, out, r, n, _d=type(mtp).draft: [EOS] * len(_d(self, entry, out, r, n))
    eng = _engine(tiny, mtp)
    eng.mtp_pipelined = False  # (the loop that reads every pass; the pipelined one is below)
    eng._solo_draft = lambda r, cache: []  # (no prompt-lookup drafts: the tiny model's reply repeats itself)
    (on,) = _serve(eng, [(LONG, 40)])
    assert on.out == off.out
    assert 0 < len(mtp.seen) <= 12, len(mtp.seen)  # (1, 2, 4, 8, ... tokens between tries: not one try a token)
    mtp = _oracle_at(tiny, mtp_file, off.out, lambda m: "eos")
    eng, (on,) = _stretch_serve(tiny, mtp, [(LONG, 40)], True)
    assert on.out == off.out
    assert eng.counters.get("draft_mtp_tokens", 0) == 0  # (a draft cut to nothing is never fed)
    assert 0 < eng.counters["draft_mtp_cut"] <= 12  # (the back-off, inside the stretch: 1, 2, 4, 8, ... tokens between tries)
    assert eng.counters["stretch_passes"] >= 40  # (every token of the reply came from a pass of the stretch)


# -- the pipelined loop (issue #37, docs/adr/0002-pipelined-draft-loop.md) ---------------------------------------------


def _oracle_at(tiny, mtp_file, truth, bad):
    """The MTP layer, run as it is, with its draft at reply position ``m`` (the token after the m-th committed one) taken
    from ``truth`` and changed where ``bad(m)`` says: True, a wrong token (the pass rejects it); "eos", the end-of-sequence
    token (the draft is cut to nothing). Position-keyed, so the old loop (``draft``) and the pipelined one
    (``draft_on_device``) see the same drafts at the same positions."""
    from locals.mtp import MTP

    n = len(truth) + 4
    t = [truth[m] if m < len(truth) else EOS for m in range(n)]
    alt = [EOS if bad(m) == "eos" else ((t[m] + 1) % 500 + 2 if bad(m) else t[m]) for m in range(n)]
    dev = mx.array(alt, dtype=mx.int32)

    class Oracle(MTP):
        def draft(self, entry, out, r, k):
            assert k == 1
            return [alt[min(len(r.out), n - 1)]]

        def draft_on_device(self, out, r, n_out):
            return dev[mx.minimum(n_out, n - 1)].reshape(1)

    return _as_oracle(Oracle, tiny, mtp_file, 1)


def _stretch_engine(tiny, mtp, pipelined=True, backoff=None):
    """An engine with MTP drafts, through the pipelined loop or the one that reads every pass, the back-off after
    rejected drafts as given (None: each loop's default), and no prompt-lookup drafts."""
    eng = _engine(tiny, mtp)
    eng.mtp_pipelined, eng.mtp_backoff = pipelined, backoff
    eng._solo_draft = lambda r, cache: []  # (no prompt-lookup drafts: the tiny model's reply repeats itself)
    return eng


def _stretch_serve(tiny, mtp, reqs, pipelined, lru=None, backoff=True, **kw):
    """Serve ``reqs`` on a _stretch_engine; ``kw`` go to each Request (a temperature, a seed)."""
    eng = _stretch_engine(tiny, mtp, pipelined, backoff)
    out = _serve(eng, reqs, lru, **kw)
    return eng, out


def _arriving(first, second, after):
    """A serve source: ``first`` at once, ``second`` once the first wrote ``after`` tokens."""
    sent = []

    def source(block):
        if not sent:
            sent.append(1)
            return [first]
        if len(sent) == 1 and len(first.out) >= after:
            sent.append(1)
            return [second]
        return []

    return source


PATTERNS = {
    "all kept": lambda m: False,
    "every other rejected": lambda m: m % 2 == 1,
    "rejected in runs": lambda m: m % 7 in (2, 3, 4),  # (the back-off doubles: 1, 2, 4 tokens without a draft)
    "cut drafts": lambda m: "eos" if m % 5 == 3 else False,
    "mixed": lambda m: "eos" if m % 11 == 6 else m % 3 == 1,
}


@pytest.mark.parametrize("pattern", list(PATTERNS))
@pytest.mark.parametrize("backoff", [True, False])
def test_the_pipelined_loop_gives_the_greedy_reply_of_mtp_off_and_a_dense_cache(tiny, mtp_file, pattern, backoff):
    (off,) = _serve(_engine(tiny), [(LONG, 40)])
    mtp = _oracle_at(tiny, mtp_file, off.out, PATTERNS[pattern])
    lru = _lru()
    eng, (on,) = _stretch_serve(tiny, mtp, [(LONG, 40)], True, lru, backoff=backoff)
    assert on.out == off.out
    assert eng.counters.get("stretch_passes", 0) > 0
    assert 0 < eng.counters["draft_mtp_accepted"] <= eng.counters["draft_mtp_tokens"]
    cache = lru.fetch_nearest_cache("locals", LONG + on.out + [0] * 20)[0]
    assert len(set(_offsets(cache))) == 1, _offsets(cache)
    assert _offsets(cache)[0] == len(LONG) - 1 + len(on.fed)  # dense: one slot per token fed
    assert _real(cache[-1])[-1] == 0  # (the last fed token's slot: a hole, as the old loop leaves it)


@pytest.mark.parametrize("pattern", list(PATTERNS))
@pytest.mark.parametrize("seed", [3, 11])
@pytest.mark.parametrize("backoff", [True, False])
def test_a_seeded_request_gets_the_tokens_of_the_loop_that_reads_every_pass(tiny, mtp_file, pattern, seed, backoff):
    """The ADR's regression test: the same seeded request through the loop that reads every pass before the next and
    through the pipelined one, token for token, over kept drafts, rejected drafts, cut drafts and the back-off (the
    patterns), with the back-off after rejected MTP drafts on and off in both. MTP off is not the reference (the two
    rules draw different variates from the same key)."""
    (off,) = _serve(_engine(tiny), [(LONG, 40)])
    kw = {"temp": 0.9, "seed": seed, "backoff": backoff}
    old_mtp = _oracle_at(tiny, mtp_file, off.out, PATTERNS[pattern])
    _, (old,) = _stretch_serve(tiny, old_mtp, [(LONG, 40)], False, **kw)
    new_mtp = _oracle_at(tiny, mtp_file, off.out, PATTERNS[pattern])
    eng, (new,) = _stretch_serve(tiny, new_mtp, [(LONG, 40)], True, **kw)
    assert new.out == old.out
    # the pipelined loop gives a discarded pass's index back, so it ends at 40 draws for 40 tokens; the loop that reads
    # every pass ends at 40 or 41, one more when its last step was launched by the main loop before the stop was read
    # (the tokens above prove the indices agree up to the end)
    assert new.draws == len(new.out) and old.draws - new.draws in (0, 1)
    assert eng.counters.get("stretch_passes", 0) > 0 and eng.counters.get("draft_mtp_tokens", 0) > 0
    assert new.out != off.out  # (sampled, not greedy)


def test_a_request_arriving_during_a_stretch_joins_and_both_replies_are_mtp_off(tiny, mtp_file):
    """A request arriving while the pipelined loop runs ends the stretch: the row's cache is compacted, the new row
    joins, and both replies are MTP off's; the first row's kept cache is dense."""
    other = [40, 41, 42, 43, 44, 45, 46, 47, 48]

    def run(mtp):
        from locals.packed import Request

        lru = _lru()
        eng = _stretch_engine(tiny, mtp)
        first, second = Request(list(LONG), _free(), 40), Request(list(other), _free(), 20)
        eng.serve(_arriving(first, second, 12), lru=lru)
        return lru, eng, first, second

    _, _, f0, s0 = run(None)
    lru, eng, f1, s1 = run(_oracle_at(tiny, mtp_file, f0.out, PATTERNS["mixed"]))
    assert (f1.out, s1.out) == (f0.out, s0.out)
    assert eng.counters.get("stretch_passes", 0) > 0
    for r in (f1, s1):
        cache = lru.fetch_nearest_cache("locals", r.prompt + r.out + [0] * 20)[0]
        assert len(set(_offsets(cache))) == 1, _offsets(cache)
        assert _offsets(cache)[0] == len(r.prompt) - 1 + len(r.fed)


def test_a_turn_continues_from_the_cache_a_stretch_left(tiny, mtp_file):
    """The compacted cache is right, not only dense: a second request that extends the first one's prompt and reply
    starts from its cache and gives MTP off's reply, with and without the pipelined loop."""
    def run(mtp, pipelined):
        lru = _lru()
        eng = _stretch_engine(tiny, mtp, pipelined)
        (a,) = _serve(eng, [(LONG, 30)], lru)
        (b,) = _serve(eng, [(LONG + a.out + [77, 78, 79], 24)], lru)
        return a, b

    a0, b0 = run(None, True)
    a1, b1 = run(_oracle_at(tiny, mtp_file, a0.out, PATTERNS["every other rejected"]), True)
    assert (a1.out, b1.out) == (a0.out, b0.out)
    assert b1.cached_tokens == len(LONG) + len(a0.out) - 1  # (it did start from the first one's cache: all but its last token)


def test_a_stop_at_a_kept_draft_ends_where_the_loop_that_reads_every_pass_ends(tiny, mtp_file):
    """A controller that finishes the reply at a token (a stop string, an end the grammar reaches): found one pass
    late, the stop still ends the reply at that token, the pass launched after it is discarded, and the kept cache
    holds the tokens fed and nothing after the stop, as with the loop that reads every pass."""
    from locals.engine import Controller
    from locals.packed import Request

    class StopAt(Controller):
        eos_ids = [EOS]

        def __init__(self):
            self.n = 0

        def accept(self, token):
            self.n += token == 257  # (the tiny model's reply repeats this token)

        def finished(self):
            return self.n >= 9

    def run(mtp, pipelined):
        lru = _lru()
        eng = _stretch_engine(tiny, mtp, pipelined)
        r = Request(list(LONG), StopAt(), 40)
        waiting = [r]
        eng.serve(lambda block: [waiting.pop()] if waiting else [], lru=lru)
        return r, eng, lru.fetch_nearest_cache("locals", r.prompt + r.out + [0] * 20)[0]

    off, _, _ = run(None, True)
    assert off.finish == "stop" and 9 <= len(off.out) < 40
    old, _, cache_old = run(_oracle_at(tiny, mtp_file, off.out, PATTERNS["all kept"]), False)
    new, eng, cache_new = run(_oracle_at(tiny, mtp_file, off.out, PATTERNS["all kept"]), True)
    assert (new.out, new.finish) == (off.out, "stop") == (old.out, old.finish)
    assert new.fed == old.fed and len(new.fed) == len(new.out)  # (the prompt's last token in, the stop token never fed)
    assert eng.counters.get("stretch_discarded", 0) == 1
    assert _offsets(cache_new) == _offsets(cache_old) and _offsets(cache_new)[0] == len(LONG) - 1 + len(new.fed)
    assert _real(cache_new[-1])[-1] == 0 == _real(cache_old[-1])[-1]


def test_a_request_arriving_as_the_stop_is_read_is_still_served(tiny, mtp_file):
    """A request that arrives in the pass whose picks end the first row (the stretch's two exits at once) joins after
    the row retires and gets its reply."""
    from locals.packed import Request

    (off,) = _serve(_engine(tiny), [(LONG, 40)])
    (other_off,) = _serve(_engine(tiny), [([40, 41, 42, 43, 44, 45, 46, 47, 48], 12)])
    eng = _stretch_engine(tiny, _oracle_at(tiny, mtp_file, off.out, PATTERNS["all kept"]))
    first, second = Request(list(LONG), _free(), 40), Request([40, 41, 42, 43, 44, 45, 46, 47, 48], _free(), 12)
    eng.serve(_arriving(first, second, 38))  # (the pass read next commits tokens 39 and 40: the stop)
    assert first.out == off.out and first.finish == "length"
    assert second.out == other_off.out and second.finish == "length"


def test_the_back_off_is_off_in_the_pipelined_loop_and_on_in_the_other_by_default(tiny, mtp_file):
    """Decided on issue #37: with ``mtp_backoff`` unset, the pipelined loop drafts after every pass and the loop that
    reads every pass (2 drafts, or ``mtp_pipelined`` off) keeps its back-off; set, both loops follow it."""
    from locals.mtp import MTP

    eng = _engine(tiny, MTP.load(tiny, mtp_file))
    assert eng.mtp_backoff is None and not eng._mtp_backoff_on()
    eng.mtp_pipelined = False
    assert eng._mtp_backoff_on()
    eng2 = _engine(tiny, MTP.load(tiny, mtp_file, drafts=2))
    assert eng2._mtp_backoff_on()
    eng2.mtp_backoff = False
    assert not eng2._mtp_backoff_on()
    eng.mtp_backoff = True
    eng.mtp_pipelined = True
    assert eng._mtp_backoff_on()
    assert _engine(tiny)._mtp_backoff_on()  # (MTP off: the setting is moot, and reads as the loop that reads every pass)


def _with_checkpoint(prompt, n, cp, kind="user"):
    """A request whose prompt cache is also kept at ``cp`` (a checkpoint)."""
    from locals.packed import Request

    r = Request(list(prompt), _free(), n)
    r.checkpoints = [(cp, kind)]
    return r


def _slot(entry, i):
    """The layer's keys and values at slot ``i`` of a single-row MTP entry (the keys without their flag column)."""
    return entry.keys[0, :, i, :-1].astype(mx.float32), entry.values[0, :, i, :].astype(mx.float32)


def _same_slot(a, b, i):
    """Slot ``i`` of two entries holds the same KV, up to the rounding of reads cut into other chunks."""
    for x, y in zip(_slot(a, i), _slot(b, i)):
        assert mx.allclose(x, y, rtol=2e-2, atol=2e-2).item(), mx.abs(x - y).max().item()


def _cold_row(tiny, mtp_file, prompt, n):
    """The prompt cache a request leaves read with no checkpoint, from nothing cached: the slots as one read fills
    them."""
    from locals.mtp import MTP

    lru = _lru()
    (r,) = _serve(_engine(tiny, MTP.load(tiny, mtp_file)), [(prompt, n)], lru)
    return lru.fetch_nearest_cache("locals", prompt + r.out + [0] * 20)[0]


def test_a_read_that_goes_on_past_a_checkpoint_has_no_hole_there(tiny, mtp_file):
    """The kept cache at a checkpoint holds nothing about the token after its end (its last slot is a hole), but the
    read that goes on knows that token: the request's own cache has the layer's KV there (issue #29, item 2)."""
    from locals.mtp import MTP

    cp = 12
    (off,) = _serve_reqs(_engine(tiny), [_with_checkpoint(LONG, 3, cp)], _lru())
    lru = _lru()
    (r,) = _serve_reqs(_engine(tiny, MTP.load(tiny, mtp_file)), [_with_checkpoint(LONG, 3, cp)], lru)
    assert r.out == off.out
    kept = lru.fetch_nearest_cache("locals", LONG[:cp])[0]
    assert _offsets(kept) == [cp, cp]
    assert _real(kept[-1]) == [1] * (cp - 1) + [0]
    row = lru.fetch_nearest_cache("locals", LONG + r.out + [0] * 20)[0]
    assert _real(row[-1])[: len(LONG) - 1] == [1] * (len(LONG) - 1)
    _same_slot(row[-1], _cold_row(tiny, mtp_file, LONG, 3)[-1], cp - 1)  # (the hidden state there, and the token after)


@pytest.mark.parametrize("more", [[77, 78, 79], [77]], ids=["tokens to read", "only the token fed next"])
def test_a_turn_from_a_kept_checkpoint_has_no_hole_where_it_ended(tiny, mtp_file, more):
    """A prompt-cache hit (an agent turn from the checkpoint the turn before kept): the entry gets a real slot at the
    position where the cached prompt ended, and the reply is MTP off's (issue #29, item 2)."""
    from locals.mtp import MTP

    cp = 12

    def run(mtp):
        lru = _lru()
        eng = _engine(tiny, mtp)
        _serve_reqs(eng, [_with_checkpoint(LONG, 3, cp)], lru)
        (b,) = _serve(eng, [(LONG[:cp] + more, 6)], lru)
        return lru, b

    _, b0 = run(None)
    lru, b1 = run(MTP.load(tiny, mtp_file))
    assert b1.out == b0.out
    assert b1.cached_tokens == cp
    row = lru.fetch_nearest_cache("locals", b1.prompt + b1.out + [0] * 20)[0]
    assert _real(row[-1])[: len(b1.prompt) - 1] == [1] * (len(b1.prompt) - 1)
    _same_slot(row[-1], _cold_row(tiny, mtp_file, b1.prompt, 6)[-1], cp - 1)


def test_a_prompt_read_in_mixed_passes_past_a_checkpoint_has_no_hole_there(tiny, mtp_file):
    """The same as a read that goes on past a checkpoint in one go, a chunk at a time beside a decoding row
    (locals/mixed.py): the kept cache's last slot is a hole, the request's own is not, and the replies are MTP off's."""
    from locals.mtp import MTP
    from locals.packed import Request

    longp = [(7 * i) % 400 + 3 for i in range(90)]
    cp = 40

    def run(mtp):
        lru = _lru()
        eng = _engine(tiny, mtp)
        eng.mix_chunk = 16
        first = Request(list(LONG), _free(), 60)
        second = _with_checkpoint(longp, 10, cp)
        sent = []

        def source(block):
            if not sent:
                sent.append(1)
                return [first]
            if len(sent) == 1 and len(first.out) >= 2:
                sent.append(1)
                return [second]
            return []

        eng.serve(source, lru=lru)
        return lru, eng, first, second

    _, _, f0, s0 = run(None)
    lru, eng, f1, s1 = run(MTP.load(tiny, mtp_file))
    assert (f1.out, s1.out) == (f0.out, s0.out)
    assert eng.counters.get("mixed_steps", 0) > 0
    kept = lru.fetch_nearest_cache("locals", longp[:cp])[0]
    assert _real(kept[-1]) == [1] * (cp - 1) + [0]
    cache = lru.fetch_nearest_cache("locals", longp + s1.out + [0] * 20)[0]
    assert _real(cache[-1])[: len(longp) - 1] == [1] * (len(longp) - 1)


def test_rows_read_on_from_a_shared_prefix_snapshot_have_no_hole_there(tiny, mtp_file):
    """Two requests with one prefix: the shared-prefix snapshot's last slot is a hole, each row's is not, and the
    replies are MTP off's (issue #29, item 2)."""
    from locals.hybrid_batch import HybridBatchEngine
    from locals.mtp import MTP

    a, b = LONG + [60, 61, 62], LONG + [70, 71]

    def run(mtp):
        lru = _lru()
        eng = HybridBatchEngine(tiny, Tok(), n_slots=4, share_prefix=True, mtp=mtp)
        eng.min_share = 8
        got = _serve(eng, [(a, 4), (b, 4)], lru)
        return lru, eng, got

    _, _, ref = run(None)
    lru, eng, got = run(MTP.load(tiny, mtp_file))
    assert [r.out for r in got] == [r.out for r in ref]
    (snap,) = eng.snapshots.values()
    assert _real(snap[-1]) == [1] * (len(LONG) - 1) + [0]
    for r in got:
        cache = lru.fetch_nearest_cache("locals", r.prompt + r.out + [0] * 20)[0]
        assert _real(cache[-1])[: len(r.prompt) - 1] == [1] * (len(r.prompt) - 1)


def test_a_reply_checkpoint_keeps_its_hole_and_the_next_turn_fills_it(tiny, mtp_file):
    """A "reply" checkpoint's entry is a view of the finished row cut to the checkpoint: it still holds nothing about
    the token after its end, and the next turn, which goes on from it with other tokens, gets the layer's KV there for
    its own token (issue #29, item 2)."""
    from locals.mtp import MTP

    n = 12
    nxt = LONG[:n] + [90, 91, 92]

    def run(mtp):
        lru = _lru()
        eng = _engine(tiny, mtp)
        (a,) = _serve_reqs(eng, [_with_checkpoint(LONG, 3, n, "reply")], lru)
        kept = lru.fetch_nearest_cache("locals", LONG[:n])[0]
        (b,) = _serve(eng, [(nxt, 6)], lru)
        return lru, kept, a, b

    _, _, a0, b0 = run(None)
    lru, kept, a1, b1 = run(MTP.load(tiny, mtp_file))
    assert (a1.out, b1.out) == (a0.out, b0.out)
    assert _offsets(kept) == [n, n]
    assert _real(kept[-1])[-1] == 0
    assert b1.cached_tokens == n
    row = lru.fetch_nearest_cache("locals", b1.prompt + b1.out + [0] * 20)[0]
    assert _real(row[-1])[: len(nxt) - 1] == [1] * (len(nxt) - 1)
    _same_slot(row[-1], _cold_row(tiny, mtp_file, nxt, 6)[-1], n - 1)
