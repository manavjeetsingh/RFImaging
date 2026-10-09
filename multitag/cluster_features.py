import os, sys, argparse, itertools, json, math
import numpy as np
import pandas as pd

# Usage: python cluster_features.py K P [--input link_features.csv] [--output cluster_features_K{K}_P{P}.csv] [--seed 0]
#   Builds sub-cluster feature rows from linkFeatures.py output. Each experiment (dataset) is a cluster
#   of N tags; for every one of the N-choose-K tag combinations, P distinct orderings (permutations) of
#   the K tags are sampled at random (all K! of them if P >= K!). The ordering is the tag naming of the
#   sub-cluster: position i in the permutation is tag i, and links are listed in ascending order of
#   these names, i.e. (t1,t2), (t1,t3), ..., (t1,tK), (t2,t3), ..., (tK-1,tK).
#
#   One output row per (experiment, combination, permutation, run, frequency) with list-valued cells:
#     tags        tag names in permutation order              (K)
#     tag_nums    tag numbers in permutation order            (K)
#     tag_xyz     [x, y, z] of each tag in permutation order  (K x 3)
#     links       [tagA, tagB] of each link in link order     (K(K-1)/2 x 2)
#     theta, alpha, avgV_A, avgV_B   one value per link in link order (K(K-1)/2)
#   theta and alpha are symmetric; avgV_A/avgV_B are oriented so A is the first tag of the link in
#   the permuted order (the stored A/B values are swapped when the permutation reverses a link).
#   List cells are written as JSON strings in CSV; an output ending in .pkl keeps them as Python lists.

FREQ, RUN = "Frequency (MHz)", "Run Exp Num"
FEATS = ["theta", "alpha", "avgV_A", "avgV_B"]


def sample_permutations(tags, P, rng):
    """Return min(P, K!) distinct orderings of tags, chosen uniformly at random."""
    K = len(tags)
    total = math.factorial(K)
    if P >= total:
        return list(itertools.permutations(tags))
    if total <= 50000:
        perms = list(itertools.permutations(tags))
        return [perms[i] for i in rng.choice(total, size=P, replace=False)]
    seen = {}
    while len(seen) < P:
        p = tuple(rng.permutation(tags))
        seen.setdefault(p, None)
    return list(seen)


def experiment_tensors(g):
    """Per experiment: feature tensor F[feat, a, b, run, freq] over directed tag pairs, tag info, runs, freqs."""
    nums = sorted(set(g.tagA_num) | set(g.tagB_num))
    N = max(nums) + 1
    runs, freqs = np.sort(g[RUN].unique()), np.sort(g[FREQ].unique())
    ri, fi = np.searchsorted(runs, g[RUN]), np.searchsorted(freqs, g[FREQ])
    a, b = g.tagA_num.to_numpy(), g.tagB_num.to_numpy()

    F = np.full((len(FEATS), N, N, len(runs), len(freqs)), np.nan)
    for k, f in enumerate(FEATS):
        # Reverse direction: theta/alpha unchanged, avgV_A and avgV_B swap roles.
        rev = {"avgV_A": "avgV_B", "avgV_B": "avgV_A"}.get(f, f)
        F[k, a, b, ri, fi] = g[f].to_numpy()
        F[k, b, a, ri, fi] = g[rev].to_numpy()

    names, xyz = {}, {}
    for side in ("A", "B"):
        t = g.drop_duplicates(f"tag{side}_num")
        for _, r in t.iterrows():
            n = int(r[f"tag{side}_num"])
            names[n] = r[f"tag{side}"]
            xyz[n] = [float(r[f"tag{side}_{ax}"]) for ax in "xyz"]
    return F, nums, names, xyz, runs, freqs


def build(df, K, P, rng):
    rows = []
    for exp, g in df.groupby("experiment", sort=True):
        F, nums, names, xyz, runs, freqs = experiment_tensors(g)
        if K > len(nums):
            raise ValueError(f"experiment {exp}: K={K} > N={len(nums)} tags")
        for ci, combo in enumerate(itertools.combinations(nums, K)):
            for pi, perm in enumerate(sample_permutations(list(combo), P, rng)):
                perm = [int(t) for t in perm]
                pairs = list(itertools.combinations(perm, 2))
                I, J = np.array(pairs).T
                V = F[:, I, J]  # (feat, link, run, freq)
                if np.isnan(V).any():
                    raise ValueError(f"experiment {exp}: missing link data for tags {perm}")
                common = {
                    "experiment": exp, "N": len(nums), "K": K,
                    "combination_id": ci, "permutation_id": pi,
                    "combination": sorted(perm),
                    "tags": [names[t] for t in perm],
                    "tag_nums": perm,
                    "tag_xyz": [xyz[t] for t in perm],
                    "links": [[names[i], names[j]] for i, j in pairs],
                }
                for r, run in enumerate(runs):
                    for q, freq in enumerate(freqs):
                        row = dict(common)
                        row[RUN], row[FREQ] = int(run), int(freq)
                        for k, f in enumerate(FEATS):
                            row[f] = V[k, :, r, q].tolist()
                        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("K", type=int, help="tags per sub-cluster")
    ap.add_argument("P", type=int, help="random permutations sampled per combination")
    ap.add_argument("--input", default=os.path.join(here, "link_features.csv"))
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.K < 2 or args.P < 1:
        sys.exit("need K >= 2 and P >= 1")
    out_path = args.output or os.path.join(here, f"cluster_features_K{args.K}_P{args.P}.csv")

    df = pd.read_csv(args.input)
    out = build(df, args.K, args.P, np.random.default_rng(args.seed))
    if out_path.endswith(".pkl"):
        out.to_pickle(out_path)
    else:
        list_cols = ["combination", "tags", "tag_nums", "tag_xyz", "links"] + FEATS
        out[list_cols] = out[list_cols].map(json.dumps)
        out.to_csv(out_path, index=False)
    print(f"{df.experiment.nunique()} experiments, K={args.K}, P={args.P} -> {len(out)} rows saved to {out_path}")
