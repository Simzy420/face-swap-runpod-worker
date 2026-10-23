# Source repositories

The combined codebase was assembled from these active `main` revisions:

| Repository | Revision | Location in this repository |
|------------|----------|-----------------------------|
| [face-swap-runpod-worker](https://github.com/Simzy420/face-swap-runpod-worker) | `f8b76cea7db7d570aadc252da67c49652dbdb489` | Root Python worker, Dockerfile, requirements, and image workflow; original README at `docs/face-swap.md` |
| [wan22-templates](https://github.com/Simzy420/wan22-templates) | `5f006eedf38222438360e3f9ea052e78ab7e362a` | Root web app, APIs, tests, hosting configuration, and `space/`; README at `docs/templates.md` |
| [wan22-extend-home](https://github.com/Simzy420/wan22-extend-home) | `12c88cd9aea42eac78e3a812968248871eb8dae2` | `public/extend/` |

Templates and Extend were imported without nested Git repositories. The new
shared navigation, Face Swap launcher, development routes, and consolidation
documentation are maintained here. Extend icon and manifest links use the new
`/extend/` location. The Templates lockfile updates Vite and source-map-js to
compatible patched versions. The existing GPU worker, API handlers, model
settings, and generation flows retain the imported behavior.

The source repositories remain available for their earlier commit history.
