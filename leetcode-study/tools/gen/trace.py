"""Run each solution for real and record every executed line with a snapshot of the variables."""
import json, sys, random
from collections import *
from heapq import *
from bisect import *
from math import *
from itertools import *
from functools import *
from typing import *

class ListNode:
    def __init__(self, val=0, next=None): self.val, self.next = val, next
    def __repr__(self): return "ListNode(%r)" % (self.val,)
class TreeNode:
    def __init__(self, val=0, left=None, right=None): self.val, self.left, self.right, self.next = val, left, right, None
    def __repr__(self): return "TreeNode(%r)" % (self.val,)
class Node:
    def __init__(self, val=0, isLeaf=None, topLeft=None, topRight=None, bottomLeft=None, bottomRight=None):
        self.val, self.isLeaf, self.topLeft, self.topRight, self.bottomLeft, self.bottomRight = val, isLeaf, topLeft, topRight, bottomLeft, bottomRight
        self.neighbors, self.next, self.random = [], None, None
    def __repr__(self): return "Node(%r)" % (self.val,)

def mk_list(a):
    d = cur = ListNode()
    for x in a: cur.next = ListNode(x); cur = cur.next
    return d.next
def mk_tree(a):
    if not a or a[0] is None: return None
    root = TreeNode(a[0]); q = deque([root]); i = 1
    while q and i < len(a):
        n = q.popleft()
        if i < len(a) and a[i] is not None: n.left = TreeNode(a[i]); q.append(n.left)
        i += 1
        if i < len(a) and a[i] is not None: n.right = TreeNode(a[i]); q.append(n.right)
        i += 1
    return root
def nexts(root):
    out, cur = [], [root] if root else []
    while cur:
        out.append([(n.val, n.next.val if n.next else None) for n in cur])
        cur = [c for n in cur for c in (n.left, n.right) if c]
    return out
def find(root, v):
    if not root: return None
    if root.val == v: return root
    return find(root.left, v) or find(root.right, v)

import types
SKIP_TYPES = (types.FunctionType, types.MethodType, types.BuiltinFunctionType, types.ModuleType, type)
def short(v):
    r = repr(v)
    return r if len(r) <= 24 else r[:23] + "…"
def enc(v, d=0):
    if v is None or isinstance(v, bool): return v
    if isinstance(v, int): return v
    if isinstance(v, float):
        if v != v: return ["x", "nan"]
        if v in (float("inf"), float("-inf")): return ["x", "∞" if v > 0 else "-∞"]
        return round(v, 6)
    if isinstance(v, str): return ["s", v if len(v) <= 40 else v[:40] + "…"]
    if isinstance(v, (set, frozenset)):
        try: items = sorted(v)
        except Exception: items = list(v)
        return ["S", [enc(x, d + 1) for x in items[:30]]]
    if isinstance(v, dict): return ["d", [[enc(k, d + 1), enc(x, d + 1)] for k, x in list(v.items())[:30]]]
    if isinstance(v, (list, tuple, deque)):
        return ["l", [enc(x, d + 1) if d < 3 else ["o", short(x)] for x in list(v)[:40]], 1 if isinstance(v, tuple) else 0]
    if isinstance(v, ListNode):
        vals, seen, n = [], [], v
        while n is not None and len(vals) < 30:
            if id(n) in seen: return ["L", vals, seen.index(id(n))]
            seen.append(id(n)); vals.append(enc(n.val, d + 1)); n = n.next
        return ["L", vals, -1]
    if isinstance(v, TreeNode):
        levels, cur = [], [v]
        while any(x is not None for x in cur) and len(levels) < 5:
            levels.append([enc(x.val) if x is not None else None for x in cur])
            cur = [c for x in cur for c in ((x.left, x.right) if x is not None else (None, None))]
        return ["T", levels]
    return ["o", short(v)]

def snapshot(frame):
    loc = frame.f_locals; out = {}
    names = list(frame.f_code.co_varnames) + [n for n in loc if n not in frame.f_code.co_varnames]
    for n in names:
        if n not in loc or n.startswith("__") or n == "self" and False: continue
        v = loc[n]
        if n == "self":
            if type(v).__name__ == "Solution": continue
            for k, x in vars(v).items(): out["self." + k] = enc(x)
            continue
        if isinstance(v, SKIP_TYPES): continue
        out[n] = enc(v)
    return out


def fmt(x):
    if x is None: return "None"
    if isinstance(x, bool): return "True" if x else "False"
    if isinstance(x, float): return ("%g" % round(x, 5))
    if isinstance(x, str): return repr(x)
    if isinstance(x, ListNode):
        vals = []
        while x is not None and len(vals) < 20: vals.append(fmt(x.val)); x = x.next
        return " → ".join(vals)
    if isinstance(x, TreeNode):
        out, cur = [], [x]
        while any(n is not None for n in cur):
            out += [n.val if n is not None else None for n in cur]
            cur = [c for n in cur if n is not None for c in (n.left, n.right)]
        while out and out[-1] is None: out.pop()
        return "[" + ", ".join(fmt(v) for v in out) + "]"
    if isinstance(x, (list, tuple)): return "[" + ", ".join(fmt(v) for v in x) + "]"
    if isinstance(x, (set, frozenset)): return "{" + ", ".join(fmt(v) for v in sorted(x)) + "}"
    if isinstance(x, dict): return repr(x)
    return repr(x)
def pretty_args(driver):
    first = driver.strip().split("\n")
    if len(first) == 1:
        m = re.match(r"Solution\(\)\.\w+\((.*)\)$", first[0])
        if m:
            a = m.group(1)
            a = re.sub(r"mk_(?:tree|list)\((\[[^\]]*\])\)", r"\1", a)
            return a.replace("None", "null")
    return None

MAX_STEPS = 900
import re
def prep(driver):
    if "RESULT" in driver: return driver
    lines = driver.strip().split("\n"); last = lines[-1]
    if not re.match(r"^\s*(\w+(\.\w+)*\s*(=|\+=)|for |while |if |def |class )", last): lines[-1] = "RESULT = " + last
    return "\n".join(lines)
def run(code, driver):
    steps = []; state = {"depth": 0, "fid": 0, "fids": {}, "trunc": False}
    ns = {k: v for k, v in globals().items() if not k.startswith("_")}
    exec(compile(code, "<sol>", "exec"), ns)
    def tracer(frame, event, arg):
        if frame.f_code.co_filename != "<sol>": return None
        if frame.f_code.co_name.startswith("<") : return None
        if event == "call":
            state["depth"] += 1; state["fid"] += 1; state["fids"][id(frame)] = state["fid"]; return tracer
        fid = state["fids"].get(id(frame), 0)
        if len(steps) >= MAX_STEPS: state["trunc"] = True; return None
        if event == "line":
            steps.append({"l": frame.f_lineno, "d": state["depth"], "f": fid, "v": snapshot(frame)})
        elif event == "return":
            steps.append({"l": frame.f_lineno, "d": state["depth"], "f": fid, "v": snapshot(frame), "ret": enc(arg)})
            state["depth"] -= 1
        return tracer
    ns["_run_driver"] = None
    ns["RESULT"] = None
    dcode = compile(prep(driver), "<driver>", "exec")
    sys.settrace(tracer)
    try: exec(dcode, ns)
    finally: sys.settrace(None)
    return steps, state["trunc"], ns.get("RESULT")

if __name__ == "__main__":
    from drivers import D
    from drivers2 import D2
    ex2 = {}
    probs = json.load(open("problems.json")); out = {}; bad = []; results = {}; results2 = {}
    for pid, codes in probs.items():
        drv = D.get(int(pid))
        if drv is None: bad.append(pid + " (no driver)"); continue
        res = []
        e2 = D2[int(pid)]; drv2 = e2[0]
        for vi, code in enumerate(codes):
            random.seed(1)
            try:
                st, tr, rv = run(code, drv); res.append({"steps": st, "trunc": tr}); results.setdefault(pid, []).append(rv)
                random.seed(1)
                st2, tr2, rv2 = run(code, drv2); res[-1]["t2"] = {"steps": st2, "trunc": tr2}; results2.setdefault(pid, []).append(rv2)
                if vi == 0: ex2[pid] = {"input": (e2[2] if len(e2) > 2 else (pretty_args(drv2) or "")), "output": fmt(rv2), "why": e2[1]}
            except Exception as e:
                bad.append("%s v%d: %s" % (pid, vi, repr(e)[:90])); res.append(None)
        out[pid] = res
    UNORD = {15,49,17,46,77,39,216,22,2215,212,30,373,128}
    SKIP = {210, 450, 162}
    def canon(x, unord):
        if isinstance(x, (list, tuple)):
            c = [canon(i, unord) for i in x]
            return sorted(c, key=lambda z: json.dumps(z, sort_keys=True, default=str)) if unord else c
        if isinstance(x, (set, frozenset)): return sorted(canon(i, unord) for i in x)
        if isinstance(x, dict): return {str(k): canon(v, unord) for k, v in x.items()}
        if isinstance(x, float): return int(x) if x == int(x) else round(x, 5)
        return enc(x)
    for pid, rs in results.items():
        if int(pid) in SKIP or len(rs) < 2 or rs[0] is None: continue
        base = json.dumps(canon(rs[0], int(pid) in UNORD), sort_keys=True, default=str)
        for vi, r in enumerate(rs[1:], 1):
            if json.dumps(canon(r, int(pid) in UNORD), sort_keys=True, default=str) != base: bad.append("%s v%d: result differs from the best approach: %s vs %s" % (pid, vi, str(canon(r, int(pid) in UNORD))[:60], str(canon(rs[0], int(pid) in UNORD))[:60]))
    for pid, rs in results2.items():
        if int(pid) in SKIP or len(rs) < 2 or rs[0] is None: continue
        base = json.dumps(canon(rs[0], int(pid) in UNORD), sort_keys=True, default=str)
        for vi, r in enumerate(rs[1:], 1):
            if json.dumps(canon(r, int(pid) in UNORD), sort_keys=True, default=str) != base: bad.append("%s v%d (edge case): result differs: %s vs %s" % (pid, vi, str(canon(r, int(pid) in UNORD))[:60], str(canon(rs[0], int(pid) in UNORD))[:60]))
    json.dump(ex2, open("ex2.json", "w"), ensure_ascii=False)
    json.dump(out, open("traces.json", "w"), separators=(",", ":"))
    print("traced", len(out), "problems; failures:", len(bad))
    for b in bad: print("  ", b)
    print("steps:", sum(len(v["steps"]) for r in out.values() for v in r if v), "max:", max(len(v["steps"]) for r in out.values() for v in r if v))
