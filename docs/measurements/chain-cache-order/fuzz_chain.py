"""Issue #58 review: ChainCache drops a conversation's stale copies before its insert (the new order); mlx-lm's
LRUPromptCache inserts first and the old ChainCache dropped them after (Old). Random inserts of fake entries into
both, from the same state, count where the orders end differently. Run: .venv/bin/python
docs/measurements/chain-cache-order/fuzz_chain.py SEED (seeds 0 to 5 in docs/prompt-cache/README.md)."""
import copy, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repo root
from mlx_lm.models.cache import LRUPromptCache, can_trim_prompt_cache
from locals.hybrid_batch import chain_cache

class Fake:
    def __init__(self, n): self.nbytes = n
    def is_trimmable(self): return False

class Old(LRUPromptCache):
    def insert_cache(self, model, tokens, prompt_cache, *, cache_type="assistant"):
        super().insert_cache(model, tokens, prompt_cache, cache_type=cache_type)
        if cache_type == "system" or can_trim_prompt_cache(prompt_cache):
            return
        node = self._trie._trie.get(model)
        for t in tokens:
            node = node.get(t) if node is not None else None
        if node is None or "__value__" not in node:
            return
        for n, entry in self._trie.pop_prefixes(model, tokens):
            if not (entry.cache_type == "assistant" or (entry.cache_type == "user" and cache_type == "user")):
                self._trie.add(model, tokens[:n], entry)
                continue
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
            self._lru.remove(model, tokens[:n])

def keys(c):
    return {(t, tuple(k)) for t, d in c._lru._lrus.items() for _, k in d}

def is_stale(k, ktype, tokens, ctype):
    k = list(k)
    return len(k) < len(tokens) and tokens[:len(k)] == k and (ktype == "assistant" or (ktype == "user" and ctype == "user"))

rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
stats = {"steps": 0, "N_kept_old_not_new": 0, "lost_more": 0, "new_better": 0, "same": 0, "replace": 0}
for trial in range(3000):
    maxb = rng.randint(3, 12)
    new = chain_cache(max_size=rng.randint(3, 10), max_bytes=maxb)
    for step in range(25):
        L = rng.randint(1, 5)
        tokens = [rng.randint(0, 2) for _ in range(L)]
        ctype = rng.choice(["assistant", "user", "system"])
        size = rng.randint(1, 4)
        if rng.random() < 0.15:
            new.max_bytes = rng.randint(1, 12)  # a pressure-trim style budget change
        old = copy.deepcopy(new); old.__class__ = Old
        before = keys(new)
        old_before = {(t, k): new._trie.get("m", list(k)).nbytes for t, k in before}
        if any(tuple(tokens) == k for _, k in before): stats["replace"] += 1
        new.insert_cache("m", tokens, [Fake(size)], cache_type=ctype)
        old.insert_cache("m", tokens, [Fake(size)], cache_type=ctype)
        stats["steps"] += 1
        kn, ko = keys(new), keys(old)
        assert new.nbytes == sum(new._trie.get("m", list(k)).nbytes for _, k in kn)
        Nk = (ctype, tuple(tokens))
        if Nk in ko and Nk not in kn:
            stats["N_kept_old_not_new"] += 1
        # survivors other than N and stale prefixes
        def others(ks): return {(t, k) for t, k in ks if (t, k) != Nk and not is_stale(k, t, tokens, ctype)}
        on, oo = others(kn), others(ko)
        if len(oo - on) > len(on - oo) or (Nk in ko and Nk not in kn):
            stats.setdefault("strict", 0); stats["strict"] += 1
            print("STRICT", ctype, tokens, size, "maxb", new.max_bytes, "max_size", new.max_size, "before", [(t, k, old_before[(t,k)]) for t, k in sorted(before)], "old", sorted(ko), "new", sorted(kn))
        if not oo <= on:
            stats["lost_more"] += 1
            0 and print("LOST", ctype, size, "maxb", new.max_bytes, "N old,new", Nk in ko, Nk in kn, "stale", sorted(t for t,k in before if is_stale(k,t,tokens,ctype)), "old-only", sorted(oo - on), "new-only", sorted(on - oo), "before", sorted(before))
        elif on != oo or (Nk in kn) != (Nk in ko): stats["new_better"] += 1
        else: stats["same"] += 1
print(stats)
