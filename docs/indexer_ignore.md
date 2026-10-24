# Indexer Ignore Policy (`.subconsciousignore`)

Subconscious indexes the directories you attach to a workspace so their contents
can be searched (RAG) and traversed via the knowledge graph. You can exclude
files and directories from indexing with a `.subconsciousignore` file — a
gitignore-style list of patterns.

This is useful for keeping secrets, build output, dependencies, large binaries,
or private notes out of the index (and therefore out of anything the assistant
can retrieve).

## Where ignore files live

There are two places an ignore file can be placed. All applicable files are
combined.

- **Global** — `<data_dir>/.subconsciousignore`
  Applies to every workspace and every attached directory. Use it for rules you
  always want (e.g. `*.pem`, `node_modules/`).
  `<data_dir>` is the Subconscious application data directory (for example, on
  Windows `%APPDATA%\Subconscious`; in dev mode the folder is suffixed `-dev`).

- **Per directory (cascading)** — a `.subconsciousignore` in an attached
  directory's root, or in *any* subdirectory beneath it.
  A file's rules apply to the directory it sits in and everything below it. A
  `.subconsciousignore` deeper in the tree adds to (and can override, via
  negation) the rules inherited from above.

You do not need any ignore file to index a directory. Without one, only the
built-in skips (below) apply.

## Built-in skips (always applied)

Regardless of any `.subconsciousignore`, the indexer never descends into common
noise directories:

```text
.subconscious   .git   .hg   .svn   __pycache__   node_modules
.venv   venv   .mypy_cache   .pytest_cache   .hypothesis   .idea   .vscode
```

The indexer's own sidecar store (`.subconscious/`) is always skipped, so it can
never recursively index itself.

Only files with recognised text/document extensions are indexed, and files
larger than 20 MB are skipped. Ignore rules apply on top of these limits.

## Pattern syntax

The syntax is a pragmatic subset of `.gitignore`:

- `# comment` — comment line; ignored. Blank lines are ignored too.
- `name` — matches `name` (file or directory) at **any** depth.
- `*.log` — glob on the basename at any depth (e.g. `a/b/app.log`).
- `build/` — trailing slash: matches **directories only**.
- `/secret.txt` — leading slash: **anchored** to the ignore file's directory (does not match `sub/secret.txt`).
- `docs/*.tmp` — a slash makes the pattern **path-relative** (anchored to this directory); `*` does not cross `/`.
- `a/b` — anchored path; also excludes everything beneath it (e.g. `a/b/c.txt`).
- `**` — in an anchored pattern, matches any remaining path segments.
- `!keep.log` — leading `!` **re-includes** a path excluded by an earlier rule.

### Precedence

Rules are evaluated in order and **the last matching rule wins** (gitignore
semantics). This is what makes negation (`!`) work: a later `!pattern` can
rescue a file that an earlier pattern excluded. Rules loaded from files closer
to a given file (deeper in the tree) are applied after inherited rules, so a
subdirectory can override a parent's decision.

### Anchoring, in short

- A pattern **without** a slash matches by basename at any depth.
- A pattern **with** a slash (or a leading `/`) is anchored to the directory of
  the `.subconsciousignore` file that declared it, and `*` will not cross `/`.

## Examples

Exclude build artifacts, dependencies, secrets, and logs everywhere:

```text
# .subconsciousignore (global or at a directory root)

# Dependencies & build output
node_modules/
dist/
build/
*.min.js

# Secrets & credentials
*.pem
*.key
.env
/secrets/

# Noise
*.log
*.tmp
```

Exclude a folder but keep one file inside it (negation):

```text
logs/
!logs/IMPORTANT.md
```

Anchored vs. unanchored:

```text
# Anchored: only the config.json at this directory's root
/config.json

# Unanchored: every config.json anywhere below
config.json
```

Restrict a pattern to one subtree:

```text
# Only .tmp files directly under docs/ (not docs/sub/x.tmp)
docs/*.tmp

# Every .tmp under docs/ at any depth
docs/**/*.tmp
```

## Notes and limitations

- Ignore rules affect **what gets indexed**, not files already indexed. After
  editing a `.subconsciousignore`, re-index the directory (use the refresh
  button next to the directory, or re-attach it) to drop newly-excluded files
  from the index. Removed/excluded files are pruned on the next scan.
- This is a deliberately small subset of gitignore. Advanced constructs (e.g.
  character classes beyond basic globbing, or the full range of `**` placements)
  may behave differently than Git. When in doubt, prefer explicit patterns.
- Ignore files themselves are not indexed (their extension isn't recognised
  content), so they won't appear in search results.

## Implementation

The matcher lives in `src/subconscious/rag/ignore.py` (`IgnoreMatcher`), and is
applied by the workspace indexer while walking each attached directory. Built-in
skip directories are defined in `src/subconscious/indexing.py`.
