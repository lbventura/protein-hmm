# Viterbi Algorithm: consuming the second residue

For the first part, see the [first-residue walkthrough](viterbi_first_residue_walkthrough.md). The starting point here is:

```python
i = 2
rows = ["AGGCE", "A..CE", "A...E"]
seq = "AAAA"
aa = x[i - 1]  # x[1] = 0, encoding the second A
```

`i` counts query residues consumed; `k` identifies the profile position.
Match and Insert updates read row `i - 1 = 1` and consume the second A.
Delete updates read the current row `i = 2` and consume nothing.

## Continuing from the first iteration

The scores computed for the first iteration, where one A was consumed, are:

| Table entry | k = 0 | k = 1 | k = 2 | k = 3 |
|---|---:|---:|---:|---:|
| `M[1, k]` | NEG_INF | 0.570 | -4.365 | -5.015 |
| `Id[1, k]` | -2.585 | NEG_INF | NEG_INF | NEG_INF |
| `D[1, k]` | NEG_INF | NEG_INF | -1.015 | -2.600 |

The chart should be read from left to right as `k` increases. Yellow nodes contain scores from `i = 1`; the other state nodes contain scores
at `i = 2`. Match and insert arrows consume the second A; deletion arrows
stay at the same `i` and consume nothing.

Green and red boxes show retained and rejected candidates for a particular
destination, before its emission score is added. Dotted arrows mark rejected
candidates. These colors describe the comparison, not whether a source state
is usable elsewhere. All boxes use dark text for contrast.

Arrow labels give the `log_trans` entry and the score added, in bits.
The probability-to-score computations remain [below](#transition-probabilities). Unreachable predecessors are explained below.

```mermaid
flowchart LR
    subgraph K0["k = 0: before the profile"]
        direction TB
        I10["Id[1, 0] = -2.585<br/>First A emitted at I0"]
        I20["Id[2, 0] = -3.585<br/>I0 emits A; emission = 0"]
        U0["M[2, 0], D[2, 0]<br/>NEG_INF"]
    end

    subgraph K1["k = 1: first match position"]
        direction TB
        M11["M[1, 1] = 0.570<br/>First A emitted at M1"]
        M21["M[2, 1] = -2.430<br/>M1 emits A; emission = +1.155"]
        I21["Id[2, 1] = -1.152<br/>I1 emits A; emission = -0.138"]
        U1["D[2, 1]<br/>NEG_INF"]
    end

    subgraph K2["k = 2: second match position"]
        direction TB
        M12["M[1, 2] = -4.365<br/>First A emitted at M2"]
        D12["D[1, 2] = -1.015<br/>A emitted at M1; skip position 2"]
        M22["M[2, 2] = -1.795<br/>M2 emits A; emission = -0.780"]
        I22["Id[2, 2] = -6.687<br/>I2 emits A; emission = 0"]
        D22["D[2, 2] = -4.015<br/>Skip position 2; no emission"]
    end

    subgraph K3["k = 3: third match position"]
        direction TB
        M13["M[1, 3] = -5.015<br/>First A emitted at M3"]
        D13["D[1, 3] = -2.600<br/>Cannot consume another residue"]
        CM["Via M[1, 2]: -5.102"]
        CD["Via D[1, 2]: -1.600"]
        M23["M[2, 3] = -2.444<br/>M3 emits A; emission = -0.845"]
        I23["Id[2, 3] = -7.336<br/>I3 emits A; emission = 0"]
        CM2["Via M[2, 2]: -4.117"]
        CD2["Via D[2, 2]: -5.600"]
        D23["D[2, 3] = -4.117<br/>Skip position 3; no emission"]
    end

    I10 -->|"log_trans[0, T_II]: -1"| I20
    I10 -->|"log_trans[0, T_IM]: -1"| M21
    M11 -->|"log_trans[1, T_MI]: -1.585"| I21
    M11 -->|"log_trans[1, T_MM]: -1.585"| M22
    M21 -->|"log_trans[1, T_MD]: -1.585"| D22
    M12 -->|"log_trans[2, T_MI]: -2.322"| I22
    M12 -.->|"log_trans[2, T_MM]: -0.737"| CM
    D12 -->|"log_trans[2, T_DM]: -0.585"| CD
    CM -.-> M23
    CD --> M23
    M13 -->|"log_trans[3, T_MI]: -2.322"| I23
    M22 -->|"log_trans[2, T_MD]: -2.322"| CM2
    D22 -.->|"log_trans[2, T_DD]: -1.585"| CD2
    CM2 --> D23
    CD2 -.-> D23

    classDef initial fill:#fff3cd,stroke:#856404,color:#111827;
    classDef match fill:#dbeafe,stroke:#2563eb,color:#111827;
    classDef insert fill:#ede9fe,stroke:#7c3aed,color:#111827;
    classDef deletion fill:#ffedd5,stroke:#ea580c,color:#111827;
    classDef unreachable fill:#f3f4f6,stroke:#9ca3af,color:#111827;
    classDef selected fill:#dcfce7,stroke:#15803d,color:#111827;
    classDef rejected fill:#fee2e2,stroke:#b91c1c,color:#111827;
    class I10,M11,M12,D12,M13,D13 initial;
    class M21,M22,M23 match;
    class I20,I21,I22,I23 insert;
    class D22,D23 deletion;
    class U0,U1 unreachable;
    class CD,CM2 selected;
    class CM,CD2 rejected;
```

For example, the arrow from `M[2, 1]` goes to `D[2, 2]`: it skips profile
position 2 without consuming another A. It does not lead to `Id[2, 1]`.

`D[1, 3]` has no outgoing arrow: there is no M4 and this model has no D3-to-I3
transition, so that path cannot consume the second A.

## Working out the numbers

### Transition probabilities

Each transition count starts with one pseudocount. Probabilities are normalized
within the listed group; for example, M1's outgoing counts `[2, 2, 2]` give
`2/6 = 1/3` each. The following table includes transitions whose source will
turn out to be unreachable at this iteration.

| Source and slot | Transitions, in order | Observed counts | Counts plus one each | Probabilities | `log_trans` values |
|---|---|---|---|---|---|
| Begin, 0 | `T_BM`, `T_MI`, `T_BD` | 3, 0, 0 | 4, 1, 1 | 4/6, 1/6, 1/6 | -0.585, -2.585, -2.585 |
| I0, 0 | `T_IM`, `T_II` | 0, 0 | 1, 1 | 1/2, 1/2 | -1, -1 |
| M1, 1 | `T_MM`, `T_MI`, `T_MD` | 1, 1, 1 | 2, 2, 2 | 1/3, 1/3, 1/3 | -1.585, -1.585, -1.585 |
| I1, 1 | `T_IM`, `T_II` | 1, 1 | 2, 2 | 1/2, 1/2 | -1, -1 |
| D1, 1 | `T_DM`, `T_DD` | 0, 0 | 1, 1 | 1/2, 1/2 | -1, -1 |
| M2, 2 | `T_MM`, `T_MI`, `T_MD` | 2, 0, 0 | 3, 1, 1 | 3/5, 1/5, 1/5 | -0.737, -2.322, -2.322 |
| I2, 2 | `T_IM`, `T_II` | 0, 0 | 1, 1 | 1/2, 1/2 | -1, -1 |
| D2, 2 | `T_DM`, `T_DD` | 1, 0 | 2, 1 | 2/3, 1/3 | -0.585, -1.585 |
| M3, 3 | `T_MI`, `T_ME` | 0, 3 | 1, 4 | 1/5, 4/5 | -2.322, -0.322 |
| I3, 3 | `T_II`, `T_IE` | 0, 0 | 1, 1 | 1/2, 1/2 | -1, -1 |
| D3, 3 | `T_DE` | 0 | 1 | 1 | 0 |



M3's insertion competes with End, giving `T_MI = 1/5` and `T_ME = 4/5`.
I3's loop competes with End, giving `T_II = T_IE = 1/2`.
Begin's `T_BM` is 4/6, but
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

I1 becomes reachable in this iteration. Only the two G residues in `AGGCE`
are assigned to I1 during training (see [training paths and repair](viterbi_first_residue_walkthrough.md#transition-probabilities));
the A residues are assigned to M1.
I1's A count therefore remains `20 * b`, while its total count is `20 + 2 = 22`.
Thus `_insert_odds(1, aa) = log2(20/22) = -0.138`.

### i = 2, k = 0: another leading insertion

```text
cands_i = [M[1, 0] + log_trans[0, T_MI],
           Id[1, 0] + log_trans[0, T_II]]
        = [NEG_INF - 2.585, -2.585 - 1]
best_i = -3.585
Id[2, 0] = _insert_odds(0, aa) + best_i = 0 - 3.585 = -3.585
```

The path is `B -> I0(first A) -> I0(second A)`. Both residues occur before the
first match position. `M[2, 0]` and `D[2, 0]` remain unreachable.

```text
Profile position:    I0
Training row 1:      no residues
Training row 2:      no residues
Training row 3:      no residues
Query assignment:    first A, second A
```

The diagrams below show only positions reached by each path. A query entry
such as `gap (D1)` means that profile position was skipped without consuming A.

### i = 2, k = 1: match at M1 or insert after M1

For the match update, the predecessor is I0:

```text
cands = [M[1, 0] + log_trans[0, T_BM],
         Id[1, 0] + log_trans[0, T_IM]]
      = [NEG_INF - 0.585, -2.585 - 1]
best = -3.585
M[2, 1] = _match_odds(0, aa) + best = 1.155 - 3.585 = -2.430
```

This is `B -> I0(first A) -> M1(second A)`: the first A was inserted before
the profile, and the second A matches the first match position M1. This is easier to visualize with the following diagram:

```text
Profile position:   I0       M1
Training data:               A   (all three rows)
Query assignment: first A  second A
```

For the insert update, the predecessor is M1:

```text
cands_i = [M[1, 1] + log_trans[1, T_MI],
           Id[1, 1] + log_trans[1, T_II]]
        = [0.570 - 1.585, NEG_INF - 1]
best_i = -1.015
Id[2, 1] = _insert_odds(1, aa) + best_i = -0.138 - 1.015 = -1.152
```

This is `B -> M1(first A) -> I1(second A)`. Unlike `Id[1, 1]` in the first
iteration, this cell is reachable: one residue has already been consumed at M1.

```text
Profile position:    M1          I1
Training row 1:       A          G G
Training row 2:       A       no residues
Training row 3:       A       no residues
Query assignment: first A     second A
```

`D[2, 1]` remains unreachable. Entering D1 from Begin consumes zero residues,
and the model does not allow I0-to-D1 transitions.

### i = 2, k = 2: match, insert, or delete at position 2

In this instance, there are three possible predecessors for the match state:

```text
cands = [M[1, 1] + log_trans[1, T_MM],
         Id[1, 1] + log_trans[1, T_IM],
         D[1, 1] + log_trans[1, T_DM]]
      = [0.570 - 1.585, NEG_INF - 1, NEG_INF - 1]
best = -1.015
M[2, 2] = _match_odds(1, aa) + best = -0.780 - 1.015 = -1.795
```

The path matches both residues: `B -> M1(first A) -> M2(second A)`.

```text
Profile position:    M1          M2
Training row 1:      A           C
Training row 2:      A           C
Training row 3:      A           gap
Query assignment:    first A     second A
```

The insert update also uses row 1:

```text
cands_i = [M[1, 2] + log_trans[2, T_MI],
           Id[1, 2] + log_trans[2, T_II]]
        = [-4.365 - 2.322, NEG_INF - 1]
best_i = -6.687
Id[2, 2] = _insert_odds(2, aa) + best_i = 0 - 6.687 = -6.687
```

The path is `B -> D1 -> M2(first A) -> I2(second A)`.

```text
Profile position:    M1          M2          I2
Training row 1:      A           C           no residues
Training row 2:      A           C           no residues
Training row 3:      A           gap         no residues
Query assignment:    gap (D1)    first A     second A
```

The delete update uses the `M[2,1]` value computed at `i = 2, k = 1`:

```text
cands_d = [M[2, 1] + log_trans[1, T_MD],
           D[2, 1] + log_trans[1, T_DD]]
        = [-2.430 - 1.585, NEG_INF - 1]
D[2, 2] = max(cands_d) = -4.015
```

The path is `B -> I0(first A) -> M1(second A) -> D2`. Both A residues have
already been consumed; deleting position 2 adds no emission score.

```text
Profile position:    I0           M1          M2
Training row 1:      no residues  A           C
Training row 2:      no residues  A           C
Training row 3:      no residues  A           gap
Query assignment:    first A      second A    gap (D2)
```

### i = 2, k = 3: compare two routes into M3

```text
cands = [M[1, 2] + log_trans[2, T_MM],
         Id[1, 2] + log_trans[2, T_IM],
         D[1, 2] + log_trans[2, T_DM]]
      = [-4.365 - 0.737, NEG_INF - 1, -1.015 - 0.585]
      = [-5.102, NEG_INF, -1.600]
best = -1.600
M[2, 3] = _match_odds(2, aa) + best = -0.845 - 1.600 = -2.444
```

The route through `D[1, 2]` wins. It matched the first A at M1 and skipped
position 2: `B -> M1(first A) -> D2 -> M3(second A)`. The other reachable
route, `B -> D1 -> M2(first A) -> M3(second A)`, would have a final score `-5.947 < -2.444` after the M3 emission contribution.

Both candidates reach M3; their assignments differ at positions 1 and 2:

```text
Profile position:    M1          M2          M3
Training row 1:      A           C           E
Training row 2:      A           C           E
Training row 3:      A           gap         E
Query (retained):    first A     gap (D2)    second A
Query (rejected):    gap (D1)    first A     second A
```

For insertion after M3:

```text
cands_i = [M[1, 3] + log_trans[3, T_MI],
           Id[1, 3] + log_trans[3, T_II]]
        = [-5.015 - 2.322, NEG_INF - 1]
best_i = -7.336
Id[2, 3] = _insert_odds(3, aa) + best_i = 0 - 7.336 = -7.336
```

The path is `B -> D1 -> D2 -> M3(first A) -> I3(second A)`.

```text
Profile position:    M1          M2          M3          I3
Training row 1:      A           C           E           no residues
Training row 2:      A           C           E           no residues
Training row 3:      A           gap         E           no residues
Query assignment:    gap (D1)    gap (D2)    first A     second A
```

For deletion, compare two paths that have already consumed both A residues:

```text
cands_d = [M[2, 2] + log_trans[2, T_MD],
           D[2, 2] + log_trans[2, T_DD]]
        = [-1.795 - 2.322, -4.015 - 1.585]
        = [-4.117, -5.600]
D[2, 3] = max(cands_d) = -4.117
```

Of the two paths, `B -> I0(first A) -> M1(second A) -> D2 -> D3` and `B -> M1(first A) -> M2(second A) -> D3`, the latter wins.

Rejected alignment at `D[2, 3]`:

```text
Profile position:    I0           M1          M2          M3
Training row 1:      no residues  A           C           E
Training row 2:      no residues  A           C           E
Training row 3:      no residues  A           gap         E
Query assignment:    first A      second A    gap (D2)    gap (D3)
```

Retained alignment at `D[2, 3]`:

```text
Profile position:    M1          M2          M3
Training row 1:      A           C           E
Training row 2:      A           C           E
Training row 3:      A           gap         E
Query assignment:    first A     second A    gap (D3)
```

## Scores after the second consumption

| Table entry | k = 0 | k = 1 | k = 2 | k = 3 |
|---|---:|---:|---:|---:|
| `M[2, k]` | NEG_INF | -2.430 | -1.795 | -2.444 |
| `Id[2, k]` | -3.585 | -1.152 | -6.687 | -7.336 |
| `D[2, k]` | NEG_INF | NEG_INF | -4.015 | -4.117 |

These are nine separate reachable endpoints, each retaining its own best path.
There is no overall winner yet as two query A residues remain.  As with `D[1, 3]`, `D[2, 3]` has no way to consume another
residue, so it cannot lead to a complete alignment of this four-residue query.
