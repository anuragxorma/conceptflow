# Security policy

## Supported versions

Security fixes go into the latest release. ConceptFlow is in beta (0.x), so
fixes are not backported to older minor versions.

## Reporting a vulnerability

Please do not report security problems in public issues. Report them
privately instead, either

- through GitHub's
  [private vulnerability reporting](https://github.com/anuragxorma/conceptflow/security/advisories/new),
  or
- by email to [sharma@cs.uni-kassel.de](mailto:sharma@cs.uni-kassel.de).

Please describe the problem, the affected versions, and how to reproduce
it. You will get an answer within a week; once the problem is confirmed, a
fix is released and the problem is described in the changelog.

ConceptFlow reads files (`.cxt`, `.csv`, JSON) and writes HTML pages that
load D3 and jsPDF from a CDN. Problems such as code execution when reading
a file, or script injection through object or attribute names in the HTML
output, are security problems.
