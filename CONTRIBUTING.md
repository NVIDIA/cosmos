# Contributing

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Setup](#setup)
- [Test](#test)
  - [Run Linting and Formatting](#run-linting-and-formatting)
  - [Run Tests](#run-tests)
  - [Run a Single Test](#run-a-single-test)
- [Code Reviews](#code-reviews)
- [Signing Your Work](#signing-your-work)

______________________________________________________________________

<!--TOC-->

We'd love to receive your patches and contributions. Please keep your PRs as draft until such time that you would like us to review them.

## Setup

Install system dependencies:

[just](https://just.systems/man/en/pre-built-binaries.html#pre-built-binaries)

```shell
uv tool install -U rust-just
```

To see all available `just` commands, run

```shell
just
```

## Test

### Run Linting and Formatting

```shell
just lint
```

This will also run auto-fixes and linting. We recommend that you commit your changes first.

### Run Tests

```shell
just test
```

Test levels (`--levels`):

0. Smoke tests. Requires >= 1 GPU.
1. Partial E2E tests. Requires >= 8 GPUs.
2. Full E2E tests. Requires >= 8 GPUs.

Test outputs are saved to `outputs/pytest/<test_name>`. To monitor a test, open `console.log`/`debug.log`.

### Run a Single Test

```shell
# List tests to get the test name
just test-list
# Run the test
just test-single <test_name> [--pdb]
```

## Code Reviews

All submissions, including submissions by project members, require review. We use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more information on using pull requests.

## Signing Your Work

- We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  - Any contribution which contains commits that are not Signed-Off will not be accepted.

- To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

  ```bash
  git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```text
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the DCO:

  ```text
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```text
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```
