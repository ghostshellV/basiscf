## How to use it on **any** time-series dataset

### A) Regression example (like RUL, health score, cost, risk score)

```python
# x_query: (T, D) normalized sequence tensor
schema = TSFeatureSchema(
    feature_names=feature_names,
    roles=roles,  # e.g. ["immutable","action","state",...]
    min_vals=feature_mins_norm,    # shape (D,)
    max_vals=feature_maxs_norm,    # shape (D,)
    mad_inv=mad_inv,               # shape (D,), optional
    static_mask=static_mask,       # shape (D,), 1 for age/sex repeated across time
    time_mutable_mask=time_mask,   # optional (T,D) or (T,)
    action_groups={"medication":[2,3], "diet":[4], "exercise":[5]}
)

target = TargetSpec(task_type="regression", target_value=0.2)  # or target_range=(0.1, 0.3)

gen = BasisGenerator(
    model=model,
    sequence_length=T,
    feature_dim=D,
    basis_type="bspline",
    num_basis=8,
    device="cuda",
)

cfs, info = gen.generate(
    query_instance=x_query,
    target=target,
    schema=schema,
    num_cfs=4,
)
```

---

### B) Binary classification example (e.g., class 0 → class 1)

```python
target = TargetSpec(task_type="binary", target_class=1, margin=1.0)
cfs, info = gen.generate(query_instance=x_query, target=target, schema=schema, num_cfs=3)
```

If your model outputs 2 logits `(N,2)` it is handled. If it outputs one logit `(N,)` / `(N,1)`, it is also handled.

---

### C) Multiclass classification example (e.g., diabetes type labels)

```python
target = TargetSpec(task_type="multiclass", target_class=2, margin=0.5)
cfs, info = gen.generate(query_instance=x_query, target=target, schema=schema, num_cfs=5)
```

---

### D) Model with **time-series + static features** (dict input)

If your model accepts something like:

```python
model({"x_ts": x_ts_batch, "x_static": x_static_batch})
```

then do:

```python
query = {
    "x_ts": x_ts_query,          # (T,D)
    "x_static": x_static_query,  # e.g. (S,) or model-specific
}

gen = BasisGenerator(
    model=model,
    sequence_length=T,
    feature_dim=D,
    basis_type="rbf",
    num_basis=10,
    device="cuda",
    sequence_key="x_ts",   # tells generator which tensor is perturbable sequence
)

cfs, info = gen.generate(query_instance=query, target=target, schema=schema)
```

This is the key step for making it generic beyond predictive maintenance.


