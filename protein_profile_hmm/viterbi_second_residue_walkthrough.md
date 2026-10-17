# Viterbi Algorithm: consuming the second residue

Continue the [first-residue walkthrough](viterbi_first_residue_walkthrough.md):

```python
rows = ["AGGCE", "A..CE", "A...E"]
seq = "AAAA"
i = 2
aa = x[i - 1]  # x[1] = 0, encoding the second A
```

`i` counts query residues consumed; `k` identifies the profile position.
The inner loop visits `k = 0, 1, 2, 3`. Match and insert updates read row
`i - 1 = 1` and consume the second A. Delete updates read the current row
`i = 2` and consume nothing.

## Continue from the first iteration

These are the scores already computed for one consumed A:

| Table entry | k = 0 | k = 1 | k = 2 | k = 3 |
|---|---:|---:|---:|---:|
| `M[1, k]` | NEG_INF | 0.833 | -4.102 | -4.752 |
| `Id[1, k]` | 0 | NEG_INF | NEG_INF | NEG_INF |
| `D[1, k]` | NEG_INF | NEG_INF | -0.752 | -2.336 |

The diagram keeps the first walkthrough's layout by profile position. Yellow
nodes are the reachable entries from `i = 1`; blue, green, and orange nodes
are the new match, insert, and delete scores at `i = 2`. Solid arrows show
the winning predecessor for each new cell. Dotted arrows show reachable
candidates that lose a comparison. Unreachable predecessors are included in
the calculations below instead of adding more arrows.

An edge adds `log_trans[slot, T_...] = log2(p)`. An emitting destination also
adds the emission score shown inside its node. All scores are in bits and
are rounded for display; they are log2 odds scores, not probabilities.

```mermaid
flowchart LR
    subgraph K0["k = 0: before the profile"]
        direction TB
        I10["Id[1, 0] = 0<br/>First A emitted at I0"]
        I20["Id[2, 0] = -1<br/>I0 emits second A; emission score = 0"]
        U0["M[2, 0], D[2, 0]<br/>NEG_INF"]
    end

    subgraph K1["k = 1: first match position"]
        direction TB
        M11["M[1, 1] = 0.833<br/>First A emitted at M1"]
        M21["M[2, 1] = 0.155<br/>M1 emits second A; emission score = +1.155"]
        I21["Id[2, 1] = -0.889<br/>I1 emits second A; emission score = -0.138"]
        U1["D[2, 1]<br/>NEG_INF"]
    end

    subgraph K2["k = 2: second match position"]
        direction TB
        M12["M[1, 2] = -4.102<br/>First A emitted at M2"]
        D12["D[1, 2] = -0.752<br/>First A emitted at M1; position 2 skipped"]
        M22["M[2, 2] = -1.532<br/>M2 emits second A; emission score = -0.780"]
        I22["Id[2, 2] = -6.424<br/>I2 emits second A; emission score = 0"]
        D22["D[2, 2] = -1.430<br/>Skip position 2; no emission"]
    end

    subgraph K3["k = 3: third match position"]
        direction TB
        M13["M[1, 3] = -4.752<br/>First A emitted at M3"]
        D13["D[1, 3] = -2.336<br/>No outgoing path for the second A"]
        M23["M[2, 3] = -2.181<br/>M3 emits second A; emission score = -0.845"]
        I23["Id[2, 3] = -4.752<br/>I3 emits second A; emission score = 0"]
        D23["D[2, 3] = -3.015<br/>Skip position 3; no emission"]
    end

    I10 -->|"log_trans[0, T_II] = log2(1/2) = -1"| I20
    I10 -->|"log_trans[0, T_IM] = log2(1/2) = -1"| M21
    M11 -->|"log_trans[1, T_MI] = log2(1/3) = -1.585"| I21
    M11 -->|"log_trans[1, T_MM] = log2(1/3) = -1.585"| M22
    M21 -->|"log_trans[1, T_MD] = log2(1/3) = -1.585"| D22
    M12 -->|"log_trans[2, T_MI] = log2(1/5) = -2.322"| I22
    M12 -.->|"log_trans[2, T_MM] = log2(3/5) = -0.737; loses"| M23
    D12 -->|"log_trans[2, T_DM] = log2(2/3) = -0.585; wins"| M23
    M13 -->|"log_trans[3, T_MI] = log2(1) = 0"| I23
    M22 -.->|"log_trans[2, T_MD] = log2(1/5) = -2.322; loses"| D23
    D22 -->|"log_trans[2, T_DD] = log2(1/3) = -1.585; wins"| D23

    classDef previous fill:#fff3cd,stroke:#856404,color:#222;
    classDef match fill:#dbeafe,stroke:#2563eb,color:#222;
    classDef insert fill:#dcfce7,stroke:#16a34a,color:#222;
    classDef deletion fill:#ffedd5,stroke:#ea580c,color:#222;
    classDef unreachable fill:#f3f4f6,stroke:#9ca3af,color:#555;
    class I10,M11,M12,D12,M13,D13 previous;
    class M21,M22,M23 match;
    class I20,I21,I22,I23 insert;
    class D22,D23 deletion;
    class U0,U1 unreachable;
```

`D[1, 3]` has no outgoing arrow: it has reached the last profile position after
consuming only one residue. There is no M4 and this model has no D3-to-I3
transition, so that path cannot consume the second A.

## Working out the numbers

### Transition probabilities

Each transition count starts with one pseudocount. Probabilities are normalized
within the listed group; for example, M1's outgoing counts `[2, 2, 2]` give
`2/6 = 1/3` each. The following table includes transitions whose source will
turn out to be unreachable at this iteration.

| Source and slot | Transitions, in order | Observed counts | Counts plus one each | Probabilities | `log_trans` values |
|---|---|---|---|---|---|
| Begin, 0 | `T_BM`, `T_BD` | 3, 0 | 4, 1 | 4/5, 1/5 | -0.322, -2.322 |
| Virtual M0, 0 | `T_MI` | 0 | 1 | 1 | 0 |
| I0, 0 | `T_IM`, `T_II` | 0, 0 | 1, 1 | 1/2, 1/2 | -1, -1 |
| M1, 1 | `T_MM`, `T_MI`, `T_MD` | 1, 1, 1 | 2, 2, 2 | 1/3, 1/3, 1/3 | -1.585, -1.585, -1.585 |
| I1, 1 | `T_IM`, `T_II` | 1, 1 | 2, 2 | 1/2, 1/2 | -1, -1 |
| D1, 1 | `T_DM`, `T_DD` | 0, 0 | 1, 1 | 1/2, 1/2 | -1, -1 |
| M2, 2 | `T_MM`, `T_MI`, `T_MD` | 2, 0, 0 | 3, 1, 1 | 3/5, 1/5, 1/5 | -0.737, -2.322, -2.322 |
| I2, 2 | `T_IM`, `T_II` | 0, 0 | 1, 1 | 1/2, 1/2 | -1, -1 |
| D2, 2 | `T_DM`, `T_DD` | 1, 0 | 2, 1 | 2/3, 1/3 | -0.585, -1.585 |
| M3, 3 | `T_MI` | 0 | 1 | 1 | 0 |
| I3, 3 | `T_II` | 0 | 1 | 1 | 0 |



M3's `T_MI` and I3's `T_II` are each normalized in a singleton group in this
implementation, giving probability 1. Begin's `T_BM` is still 4/5, but
`M[1, 0]` is unreachable: a path cannot return to Begin after consuming a residue.

### Emission scores for the second A

The residue is A again, so the match emission scores are unchanged. Let
`b = BACKGROUND_FREQ[aa]`, approximately `0.078078`.

| Code call | Probability of A | Emission score `log2(P(A given state) / b)` |
|---|---|---:|
| `_match_odds(0, aa)` | M1: 4/23 | +1.155 |
| `_match_odds(1, aa)` | M2: 1/22 | -0.780 |
| `_match_odds(2, aa)` | M3: 1/23 | -0.845 |
| `_insert_odds(0, aa)` | I0: b | 0 |
| `_insert_odds(1, aa)` | I1: (20 * b) / 22 | -0.138 |
| `_insert_odds(2, aa)` | I2: b | 0 |
| `_insert_odds(3, aa)` | I3: b | 0 |

I1 becomes reachable in this iteration. It observed two G residues and no A
in training. Its A count remains `20 * b`, but its total count is `20 + 2 = 22`.
Thus `_insert_odds(1, aa) = log2(((20 * b) / 22) / b) = log2(20/22)`.

### i = 2, k = 0: another leading insertion

```text
cands_i = [M[1, 0] + log_trans[0, T_MI],
           Id[1, 0] + log_trans[0, T_II]]
        = [NEG_INF + 0, 0 - 1]
best_i = -1
Id[2, 0] = _insert_odds(0, aa) + best_i = 0 - 1 = -1
```

The path is `B -> I0(first A) -> I0(second A)`. Both residues occur before the
first match position. `M[2, 0]` and `D[2, 0]` remain unreachable.

### i = 2, k = 1: match at M1 or insert after M1

For the match update, the predecessor is I0:

```text
cands = [M[1, 0] + log_trans[0, T_BM],
         Id[1, 0] + log_trans[0, T_IM]]
      = [NEG_INF - 0.322, 0 - 1]
best = -1
M[2, 1] = _match_odds(0, aa) + best = 1.155 - 1 = 0.155
```

This is `B -> I0(first A) -> M1(second A)`: the first A was inserted before
the profile, and the second A matches its first position.

For the insert update, the predecessor is M1:

```text
cands_i = [M[1, 1] + log_trans[1, T_MI],
           Id[1, 1] + log_trans[1, T_II]]
        = [0.833 - 1.585, NEG_INF - 1]
best_i = -0.752
Id[2, 1] = _insert_odds(1, aa) + best_i = -0.138 - 0.752 = -0.889
```

This is `B -> M1(first A) -> I1(second A)`. Unlike `Id[1, 1]` in the first
iteration, this cell is reachable: one residue has already been consumed at M1.

```text
Path ending at M[2, 1]:     I0  M1
Query residues:             A   A

Path ending at Id[2, 1]:    M1  I1
Query residues:             A   A
```

`D[2, 1]` remains unreachable. Entering D1 from Begin consumes zero residues,
and the model does not allow I0-to-D1 transitions.

### i = 2, k = 2: match, insert, or delete at position 2

The match update uses row 1:

```text
cands = [M[1, 1] + log_trans[1, T_MM],
         Id[1, 1] + log_trans[1, T_IM],
         D[1, 1] + log_trans[1, T_DM]]
      = [0.833 - 1.585, NEG_INF - 1, NEG_INF - 1]
best = -0.752
M[2, 2] = _match_odds(1, aa) + best = -0.780 - 0.752 = -1.532
```

The path matches both residues: `B -> M1(first A) -> M2(second A)`.

The insert update also uses row 1:

```text
cands_i = [M[1, 2] + log_trans[2, T_MI],
           Id[1, 2] + log_trans[2, T_II]]
        = [-4.102 - 2.322, NEG_INF - 1]
best_i = -6.424
Id[2, 2] = _insert_odds(2, aa) + best_i = 0 - 6.424 = -6.424
```

The path is `B -> D1 -> M2(first A) -> I2(second A)`.

The delete update uses the newly computed row 2:

```text
cands_d = [M[2, 1] + log_trans[1, T_MD],
           D[2, 1] + log_trans[1, T_DD]]
        = [0.155 - 1.585, NEG_INF - 1]
D[2, 2] = max(cands_d) = -1.430
```

The path is `B -> I0(first A) -> M1(second A) -> D2`. Both A residues have
already been consumed; deleting position 2 adds no emission score.

### i = 2, k = 3: compare two routes into M3

```text
cands = [M[1, 2] + log_trans[2, T_MM],
         Id[1, 2] + log_trans[2, T_IM],
         D[1, 2] + log_trans[2, T_DM]]
      = [-4.102 - 0.737, NEG_INF - 1, -0.752 - 0.585]
      = [-4.839, NEG_INF, -1.336]
best = -1.336
M[2, 3] = _match_odds(2, aa) + best = -0.845 - 1.336 = -2.181
```

The route through `D[1, 2]` wins. It matched the first A at M1 and skipped
position 2: `B -> M1(first A) -> D2 -> M3(second A)`. The other reachable
route, `B -> D1 -> M2(first A) -> M3(second A)`, has predecessor score
`-4.839` and final score `-5.684` after the same M3 emission contribution.

For insertion after M3:

```text
cands_i = [M[1, 3] + log_trans[3, T_MI],
           Id[1, 3] + log_trans[3, T_II]]
        = [-4.752 + 0, NEG_INF + 0]
best_i = -4.752
Id[2, 3] = _insert_odds(3, aa) + best_i = 0 - 4.752 = -4.752
```

The path is `B -> D1 -> D2 -> M3(first A) -> I3(second A)`.

For deletion, compare two paths that have already consumed both A residues:

```text
cands_d = [M[2, 2] + log_trans[2, T_MD],
           D[2, 2] + log_trans[2, T_DD]]
        = [-1.532 - 2.322, -1.430 - 1.585]
        = [-3.854, -3.015]
D[2, 3] = max(cands_d) = -3.015
```

The winning path is `B -> I0(first A) -> M1(second A) -> D2 -> D3`.
The path `B -> M1(first A) -> M2(second A) -> D3` loses this comparison.

## Scores after the second consumption

| Table entry | k = 0 | k = 1 | k = 2 | k = 3 |
|---|---:|---:|---:|---:|
| `M[2, k]` | NEG_INF | 0.155 | -1.532 | -2.181 |
| `Id[2, k]` | -1 | -0.889 | -6.424 | -4.752 |
| `D[2, k]` | NEG_INF | NEG_INF | -1.430 | -3.015 |

These are nine separate reachable endpoints, each retaining its own best path.
There is no overall winner yet: two query A residues remain. A dotted arrow
losing a comparison does not erase its source cell; that cell can still supply
other transitions. As with `D[1, 3]`, `D[2, 3]` has no way to consume another
residue, so it cannot lead to a complete alignment of this four-residue query.
