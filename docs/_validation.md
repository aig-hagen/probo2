# Validation
- Validating the solver solutions with a specified reference solutions of the benchmark
- The path to the reference solutions is specified in the YAML file of the corresponding benchmark

```yaml
name: MyBenchmark
path: /path/to/my/benchmarks/MyBenchmark
solution_path: /path/to/solutions
query_arg_format: af.arg
format: i23
```

- The solutions have to follow a naming pattern: <instance_name>_<task>.sol
- example for task DS-PR: my_instances.tgf -> my_instances_DC-PR.sol

- Instance name is the name of the instance without the extension like apx, tgf, af or i23

- Callbacks for validation are implented in src.callbacks.validation_callbacks

-