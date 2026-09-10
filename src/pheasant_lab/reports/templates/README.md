# Report templates

The reports are built in code (`reports/summary.py` and its siblings) rather
than rendered through a template engine. Two reasons:

* every number in a report carries its denominator, its formula, its
  substituted calculation and its limitation, and threading those through a
  template makes it easy to drop one — the code path makes it a type error
  instead;
* a template engine would put the report's structure behind a dependency, and
  these documents are the artifact a reviewer reads when they are checking
  whether a claim is defensible.

`summary.md.tmpl` is kept as a readable statement of the intended shape.
