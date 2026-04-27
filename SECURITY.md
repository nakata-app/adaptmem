# Security policy

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security-sensitive
findings. Instead, email the maintainer at
**ataknakbaba@gmail.com** with:

- A description of the issue.
- Steps to reproduce (a minimal repro is enough).
- The version / commit you tested against.
- Optionally, your proposed fix.

We aim to acknowledge a report within 72 hours and to ship a fix in
the next minor release where applicable.

## Scope

The package itself is the in-scope surface. The training pipeline,
search API, the `[server]` HTTP daemon, the bundled benchmarks, and
the CLI are all in scope.

Out of scope:
- Bugs in third-party encoders / cross-encoders we depend on. Report
  those to the upstream library (`sentence-transformers`,
  `transformers`, `torch`).
- Performance issues without a security impact (file regular issues
  instead).

## Threat model — daemon mode

`adaptmem serve` is a **localhost-only, single-user** daemon by
default. It does not implement authentication or authorisation. Do
not bind it to a public network interface. If you need cross-host
access, run it behind a reverse proxy that handles auth.
