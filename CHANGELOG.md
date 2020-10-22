reb's CHANGELOG
================

0.1.1
-----
*(October 22nd, 2020)*

+ Speed up vm by cython
+ Change default implement to cython vm

0.1.0
-----
*(October 9th, 2020)*

+ A better README
+ Fix bug:
  - match error about greediness of plain engine

0.0.4
-----
*(October 3rd, 2020)*

+ Add a new match engine, implemented by re virtual machine in pure python
+ Add a new option to `reb` command to select match engine
+ Fix bug:
  - `PTNode.children` should not return all descendants

0.0.3
-----
*(September 4th, 2020)*

+ Add command line entry `reb`
+ New syntax P.n(, exact=x)
+ New pattern primitive P.onceeach
+ Lazy-extraction is possible by Pattern.extractiter
+ Fix bugs:
  - pattern match error when P.n _to == _from
  - tags under P.n are all dropped

0.0.2
-----
*(August 24, 2020)*

+ Implement example feature
+ Add new patterns: `P.STARTING`, `P.ENDING` and `P.NEWLINE`
+ Simulate re interface, adding `Pattern.finditer` and `Pattern.findall` methods

0.0.1
-----
*(August 22, 2020)*

+ Impelement basic features and pass simple tests
