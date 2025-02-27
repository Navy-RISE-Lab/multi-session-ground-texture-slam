# Contributing to this Project #

Thanks for thinking about using or contributing to this software ("Project") and its documentation!

- [Policy](#policy)
- [Submitting an Issue](#submitting-an-issue)
- [Submitting Code](#submitting-code)

## Policy ##

### 1. Introduction ###

The project maintainer for this Project will only accept contributions using the Developer's Certificate of Origin 1.1
located at [developercertificate.org](https://developercertificate.org) ("DCO"). The DCO is a legally binding statement
asserting that you are the creator of your contribution, or that you otherwise have the authority to distribute the
contribution, and that you are intentionally making the contribution available under the license associated with the
Project ("License").

### 2. Developer Certificate of Origin Process ###

Before submitting contributing code to this repository for the first time, you'll need to sign a Developer Certificate
of Origin (DCO) (see below). To agree to the DCO, add your name and email address to the
[CONTRIBUTORS.md](https://github.com/Code-dot-mil/code.mil/blob/master/CONTRIBUTORS.md) file. At a high level, adding
your information to this file tells us that you have the right to submit the work you're contributing and indicates that
you consent to our treating the contribution in a way consistent with the license associated with this software (as
described in [LICENSE.md](https://github.com/Code-dot-mil/code.mil/blob/master/LICENSE.md)) and its documentation
("Project").

### 3. Important Points ###

Pseudonymous or anonymous contributions are permissible, but you must be reachable at the email address provided in the
Signed-off-by line.

U.S. Federal law prevents the government from accepting gratuitous services unless certain conditions are met. By
submitting a pull request, you acknowledge that your services are offered without expectation of payment and that you
expressly waive any future pay claims against the U.S. Federal government related to your contribution.

If you are a U.S. Federal government employee and use a `*.mil` or `*.gov` email address, we interpret your
Signed-off-by to mean that the contribution was created in whole or in part by you and that your contribution is not
subject to copyright protections.

### 4. DCO Text ###

The full text of the DCO is included below and is available online at
[developercertificate.org](https://developercertificate.org):

```txt
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Submitting an Issue ##

Feel free to submit an issue or bug directly within GitHub. When submitting, please describe in as much detail:

- Steps to reproduce the problem,
- What you expected to happen,
- What actually happened (or didn't happen)

If you have insight into why or suggested fixes, feel free to add those as well. The same procedure applies to feature
requests as well, except there is no steps to reproduce the problem.

**Note:** I will try to address issues when able, but may be limited based on my workload.

## Submitting Code ##

You are welcome to submit additions to the code. Please see the previous section on [Policy](#policy) for important
information. Then, fork and make a pull request to this project. Your pull request should describe what is being changed
and why.

**Note:** I will try to address any contributions when able, but may be limited based on my workload.

### Code Style ###

Please match the existing code style. The project information should already define all the style checks used by the
project. If you are using VS Code and a Docker container, these checks and formatting should be done automatically.

If not using those tools, make sure to run Ruff to check for issues.

### Note on "Source of Truth" ###

GitHub is not the primary location of this code. It is maintained on an internal code repository. As such, contributions
may not follow the usual pull request path. They may get added internally and show up in a new release. If so, I will
ensure contributions are correctly attributed.
